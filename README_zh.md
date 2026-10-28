# Mini-vLLM：Qwen 混合架构推理与 DFlash2 推测解码

[English](README.md) | [简体中文](README_zh.md)

本项目基于 Mini-vLLM，研究 **Qwen3.5/Qwen3.8 混合架构模型与 DFlash2 推测解码**。当前重点是真实模型推理、贪心输出一致性、混合缓存事务，以及不同并发下 draft、verify、commit 的性能优化。

主要实验使用 Qwen3.8-27B target、配套 DFlash2 draft 和 NVIDIA A800 80GB。这里是独立实现，官方 vLLM 使用独立环境，作为对照和诊断参考。

## 已实现的主要内容

- **混合架构 target**：Full Attention 与 Gated DeltaNet（GDN）、权重加载、RoPE、缓存解码，以及供 draft 使用的 hidden taps。
- **DFlash2 proposal**：最多七个候选 token 的块预测、target 条件上下文和批量草稿生成。
- **验证与提交**：贪心接受和纠正、KV/循环状态事务、接受前缀恢复与拒绝回滚。
- **性能优化**：并行块验证、GDN 递推及前缀恢复融合、追加式 KV 复用、验证采样批处理、减少中间复制。
- **实验工具**：target-only/DFlash2 配对基准、严格 token 比较、接受率、分阶段计时、显存跟踪，以及与官方 vLLM 的固定前缀对照。
- **在线运行时**：并发请求准入、取消、流式基础设施与逐请求指标；具体入口和支持范围见[在线服务文档](docs/online_serving.md)。

原始 attention benchmark 和教学示例仍保留，但不再作为项目的主要运行入口。

## 当前结果与适用范围

性能取决于任务、并发、数值配置和实现版本。LM head 变快，不等于端到端解码一定同比变快。

| 实验 | 观测结果 | 适用范围 |
|---|---|---|
| C1 真实模型配对 | Target-only 8.80 tok/s；DFlash2 18.80 tok/s；接受率 29.57% | 四请求各生成 3,000 token；默认未合并头；12,000 个输出 token 完全一致；单次观测 |
| C24 LM head A/B | 累计 verify 从 651.38 降至 634.48 秒，下降 2.59% | 两边均为 DFlash2；24×3,000 输出一致；不是 target-only 对照 |
| B4 同输入诊断 | LM head GPU 活动累计从 11.89 降至 1.48 ms；官方为 1.48 ms | 相同 prompt 和候选，K=7；预热后首轮验证；插桩数据，非吞吐指标 |

证据：[C1 数据](reports/batch1_default_comparison_20261013.json)、[C24 数据](reports/merged_lm_head_real_comparison_20261013.json)、[同输入诊断](reports/matched_profile_20261028/README.md)。历史实验见[报告索引](reports/README.md)，不同版本和负载的数据不能直接拼成并发扩展曲线。

**数值边界：** BF16 的 kernel 形状变化可能改变贪心决策。`--merged-lm-head` 仍是默认关闭的实验选项：另一轮 C1 长输出测试出现了 token 分歧，B4 首轮通过不能覆盖该失败记录。见[合并头 C1 数据](reports/batch1_merged_comparison_20261013.json)。

## 安装

真实模型路径需要 Linux、NVIDIA GPU、兼容的 CUDA 驱动和本地 target/draft 权重。已测大模型环境为 A800 80GB；目标并发的显存容量需要实测。

```bash
git clone https://github.com/WenyueD/Mini-vLLM.git
cd Mini-vLLM
uv sync --python 3.11 --group dev
```

项目固定 PyTorch 2.9.1、Transformers 4.57.3、Linux x86-64 Triton 3.5.1。官方 vLLM 对照使用**独立环境**，运行 Mini-vLLM 本身不需要官方 vLLM；可选 `oracle` 依赖也不等于复现某个历史官方版本。

## 运行真实模型配对基准

仓库不包含权重。指定本地目录后，先运行 C1 smoke 配对：

```bash
export MINIVLLM_QWEN38_PATH=/path/to/Qwen3.8-27B
export MINIVLLM_DFLASH2_PATH=/path/to/Qwen3.8-27B-DFlash2

uv run python scripts/benchmark_qwen38_dflash2_concurrency.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --variants target_only dflash2 \
  --workload short --concurrency 1 \
  --requests-per-run 4 --repeats 1 \
  --warmup-requests 1 --warmup-output-tokens 16 \
  --num-speculative-tokens 7 \
  --parallel-target-verification --parallel-verification-chunk-size 8 \
  --vectorized-target-numerics --fused-gdn-target \
  --cached-target-micro-batch-size 1 \
  --fused-prefix-commit --reuse-verification-samples \
  --compact-verification-copies --short-prefill-attention \
  --lightweight-live-stats --timeout 3600 \
  --run-id qwen_dflash2_c1_smoke
```

该命令是短负载 smoke，**不是**表中的 C1 3,000-token 实验。`--workload tasks4096` 使用四个长文本任务，输出预算为 4,096 token。先完成 C1，再提升并发；`--cached-target-micro-batch-size` 会改变执行形状，需要明确设置。最大准入并发与实际验证 batch 是两个不同指标。

结果位于 `results/qwen38_dflash2_perf/<run-id>/`，日志位于 `logs/qwen38_dflash2_perf/<run-id>/`。解释吞吐前，应检查 fairness/token-match、失败数量、实际 batch 和预热配置。原始日志、trace、模型权重及环境目录不提交到 Git。

## 关键开关

| 参数 | 作用与边界 |
|---|---|
| `--num-speculative-tokens 7` | 七个候选；完整验证输入还包含一个 anchor |
| `--parallel-target-verification` | 按因果块验证，避免每个位置独立调用 target |
| `--vectorized-target-numerics` | 实验性的共享 BF16 向量化 target 路径 |
| `--fused-gdn-target`、`--fused-prefix-commit` | 分别融合递推和接受前缀恢复 |
| `--short-prefill-attention` | 直接访问 KV 的缓存 attention；需要向量化 target |
| `--merged-lm-head` | 将验证位置合成一次 LM-head GEMM；显式开启，已有 BF16 一致性失败记录 |
| `--stop-verification-at-mismatch` | 可选的首错早停，默认关闭；实际节省取决于验证模式及分块大小 |

全部参数见 `--help`。修改数值路径后仍需严格贪心输出回归。本项目不宣称通用随机拒绝采样、不宣称所有输出与官方 vLLM 一致，也不宣称所有并发下都有加速。

## 测试与诊断

```bash
uv run pytest -q
python scripts/validate_matched_evidence.py reports/matched_profile_20261028
```

测试包括小模型、缓存、回滚，以及有 CUDA 时运行的 GPU 测试。真实权重测试还需额外显式开启；跳过不能视为真实模型验证通过。[profile_qwen35_verifier.py](scripts/profile_qwen35_verifier.py) 用于算子分析，插桩运行应与吞吐基准分开。同输入 profile 脚本依赖实验目录布局，见[复现说明](docs/matched_profile_reproduction.md)。

## 代码结构

| 目录 | 内容 |
|---|---|
| `src/myvllm/models/` | Qwen 模型、混合 attention、GDN 内核 |
| `src/myvllm/spec_decode/` | DFlash2、proposal/verify 协调和贪心接受 |
| `src/myvllm/engine/` | Runner、序列、调度、混合缓存事务 |
| `src/myvllm/online/` | 并发在线运行时 |
| `scripts/` | 基准、官方 fixture、诊断工具 |
| `tests/` | 功能、数值、回滚和 GPU 测试 |
| `reports/` | 按日期记录的实验、失败证据和结论 |

进一步阅读：[模型说明](docs/qwen35_target.md)、[推测解码设计](docs/speculative_decoding_design.md)、[服务器命令](docs/autodl_qwen38_validation.md)、[版本统一记录](docs/version_reconciliation_20261028.md)。

## 来源与许可证

本项目扩展自原始 [MiniVLLM](https://github.com/Wenyueh/MinivLLM)，后者基于 Nano-vLLM，提供独立 attention benchmark。保留[原学习指南](HowToApproachvLLM.md)和[历史中文介绍](docs/legacy_minivllm_readme_zh.md)；历史文档中的能力边界不代表当前版本。

仓库保留 [Apache-2.0 许可证](LICENSE)。模型权重和外部参考实现遵循各自许可证。
