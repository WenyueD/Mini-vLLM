# Qwen3.8 DFlash2 高并发基准报告 - 2026-09-07

## 当前状态

2026-09-08，A800 服务器的新连接端点恢复可用。现已完成试运行、短负载矩阵、
资源清理验证、可复现性验证和目标模型决策诊断。所有性能测试单元均可正常执行，
但 C=4 下存在一个可复现的 BF16 近似并列分歧，因此尚不能判定为严格无损。

- 本地回归测试：`150 passed, 2 skipped`。
- 远端在提交 `525f7cf` 上运行 Qwen 定向回归测试：`19 passed`。
- 远端运行环境：NVIDIA A800-SXM4-80GB、Python 3.11.16、
  PyTorch 2.9.1+cu128、CUDA 12.8、BF16。
- Target-only 可复现性：在 C=4 下进行两次独立模型加载，8 个请求产生的全部
  512 个 token 均完全一致。
- 短负载矩阵：C=1、C=2 和 C=8 通过严格 token 一致性校验；C=4 未通过。
- 由于 C=4 尚未实现严格 target-lossless，正式多负载矩阵与稳态压力测试仍处于
  门禁阻塞状态。

当前基准比较对象仍为 Mini-vLLM target-only 与 Mini-vLLM DFlash2。官方
vLLM 0.28 V2 Oracle 的语义对齐限制仍然存在：Oracle runner 会产生非无损行为，
因此本报告不能被解读为已经完成与官方 DFlash2 的全面对齐。

## 远端修复

本次远端测试发现并修复了数个相互独立的实现与基准工具问题：

- `0531f1b`：在 `torch.inference_mode()` 下执行目标模型推理，并确保构造器失败时
  能够清理分布式状态。
- `62f7143`：预热后重置统计数据时，保留所有已有运行时计数器字段。
- `46776ba`：引擎关闭时释放 speculative proposer 及其持有的 model runner，
  消除了跨测试单元残留的 53.7 GiB GPU 显存泄漏。
- `de89110`：固定 Qwen 缓存解码时的 attention 上下文宽度。
- `f3534e2`：通过虚拟请求状态固定 Qwen 缓存解码的 batch 大小。
- `525f7cf`：为普通解码和 speculative 路径增加有界的目标模型 top-2 决策诊断。

资源清理修复完成后，所有远端测试单元均可在同一进程内依次加载 target-only
和 DFlash2，未再发生 OOM 或 NCCL 关闭异常。

## 基准工具

新增脚本 `scripts/benchmark_qwen38_dflash2_concurrency.py`。

主要特性：

- 两种模式均使用 `OnlineEngine`，稳态请求延迟不包含模型加载和启动时间。
- target-only 与 DFlash2 依次运行，并使用相同的 tokenizer、prompt token ID、
  负载定义、准入路径、数据类型、缓存设置、`0.0` 温度和随机种子。
- 支持闭环并发、可选的开环泊松到达、固定间隔突发到达、混合变长负载、
  正式测量前预热，以及稳态压力测试中的取消边界场景。
- 在 `results/qwen38_dflash2_perf/<run_id>/` 下写入逐请求 JSONL、监控 JSONL、
  单次运行汇总 JSON、聚合 JSON 和 CSV；原始日志写入
  `logs/qwen38_dflash2_perf/<run_id>/`。
- 记录提交、准入、首 token、逐 token 和完成时间戳，以及吞吐量、TTFT、TPOT、
  ITL、E2E p50/p90/p95/p99、有效吞吐 SLO、队列与运行中请求采样、阶段耗时、
  DFlash2 proposal profile、按位置与解码步统计的接受率、接受长度直方图、
  batch 大小分桶、CUDA 显存快照和失败记录。

默认有效吞吐 SLO 明确定义如下，并且可以通过参数调整：

- TTFT <= 5.0s
- TPOT <= 0.25s

新增 `OnlineRequestSpec` 和 `OnlineEngine.submit_requests()`，使混合负载可以使用
逐请求 `SamplingParams` 进行原子化准入。现有 `submit()` 和 `submit_batch()`
行为保持不变。

## 测试设计

本地测试覆盖：

- 百分位插值和延迟汇总；
- 吞吐量、TTFT、TPOT、ITL、E2E 和有效吞吐聚合；
- 按解码步、proposal 位置统计的接受率，以及 batch 大小分桶；
- 汇总数据结构校验和 token 分歧诊断；
- `OnlineEngine.submit_requests()` 使用不同逐请求生成长度时，仍能进入共享的
  batch 解码路径。

基准工具会对抽样的 target-only 与 DFlash2 输出执行严格 token ID 比较。
启用 `--debug-metadata` 后，还会记录请求 ID、输出位置、来源路径、query 偏移、
活跃 batch 大小、输入 token、实际 argmax token、top-2 token ID 与 logit，
以及 top-2 margin。公平性校验始终采用严格 token ID 比较；诊断信息只用于解释
失败，不会把失败重新标记为通过。

## 短负载结果

C=1、C=2 和 C=8 数据来自提交 `de89110` 上固定 attention 宽度后的短负载矩阵；
C=4 数据来自提交 `f3534e2` 上更新后的固定 batch 门禁测试。每行均包含 8 个请求，
每个请求输入 128 个 token、输出 64 个 token，K=1，并执行一次正式测量。

| 并发数 C | Target 输出 tok/s | DFlash2 输出 tok/s | DFlash/Target | 接受率 | Target 峰值 GiB | DFlash 峰值 GiB | 严格一致 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 4.275 | 3.454 | 0.808x | 25.19% | 55.17 | 58.07 | 是 |
| 2 | 5.381 | 4.436 | 0.824x | 24.87% | 55.18 | 58.07 | 是 |
| 4 | 5.825 | 5.118 | 0.879x | 25.19% | 55.17 | 58.07 | 否 |
| 8 | 6.272 | 5.586 | 0.891x | 27.04% | 55.18 | 58.91 | 是 |

在所有已测并发度下，DFlash2 均慢于 target-only。随着并发度升高，其相对吞吐量
有所改善，但本次试运行没有观察到加速。在 C=4 下，DFlash2 的 TTFT 中位数更好
（34.07s 对 37.81s），但 E2E 延迟中位数更差（49.86s 对 43.94s），TPOT 中位数
也更差（0.251s 对 0.084s）。

结果文件：

- `results/qwen38_dflash2_perf/qwen38_dflash2_short_matrix_fixedpad_k1_20260908`
- `results/qwen38_dflash2_perf/qwen38_dflash2_fixedbatch_c4_gate_20260908`
- `logs/qwen38_dflash2_perf/qwen38_target_repro_c4_r2_20260908`

## 公平性诊断

固定 attention 宽度后，短负载矩阵只剩一个 C=4 分歧。固定 batch 大小消除了该
分歧，但 `short-000000` 又在输出位置 61 出现了另一个分歧。提交 `525f7cf` 上的
诊断复现了相同的 token 对，并确认输出位置 0 至 60 的前缀完全一致：

| 路径 | Token 25 logit | Token 13 logit | Margin | 贪心结果 |
|---|---:|---:|---:|---:|
| target-only 普通解码 | 22.375 | 22.250 | 0.125 | 25 |
| DFlash2 verifier，query 偏移 0 | 22.250 | 22.250 | 0.000 | 13 |

DFlash2 verifier 在 BF16 下产生了精确并列，`torch.argmax` 因而选择较小的 token
ID 13。这与接受 speculative token 后累积的低精度状态敏感性相符，但与
target-only 基线不稳定不符。分歧发生时也没有伴随请求失败、缓存异常、OOM 或
关闭错误。

诊断文件：

- `results/qwen38_dflash2_perf/qwen38_dflash2_target_decision_diag_c4_20260908`
- `logs/qwen38_dflash2_perf/qwen38_dflash2_target_decision_diag_c4_20260908`

## 远端复现命令

代码部署至 AutoDL 后的试运行命令：

```bash
cd /root/autodl-tmp/Mini-vLLM
export MINI_PY=/root/autodl-tmp/Mini-vLLM/.venv/bin/python
export MINIVLLM_QWEN38_PATH=/root/autodl-tmp/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export MINIVLLM_DFLASH2_PATH=/root/autodl-tmp/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/<snapshot>

$MINI_PY scripts/benchmark_qwen38_dflash2_concurrency.py \
  --concurrency 1 2 4 \
  --workload short medium mixed \
  --repeats 1 \
  --requests-per-run 8 \
  --num-speculative-tokens 1 \
  --target-cache-memory-fraction 0.05 \
  --profile-dflash2-stages \
  --run-id qwen38_dflash2_pilot_20260907
```

试运行日志通过公平性与显存校验后的正式矩阵命令：

```bash
$MINI_PY scripts/benchmark_qwen38_dflash2_concurrency.py \
  --concurrency 1 2 4 8 16 32 \
  --workload short medium long mixed \
  --repeats 3 \
  --requests-per-run 128 \
  --num-speculative-tokens 1 \
  --target-cache-memory-fraction 0.05 \
  --profile-dflash2-stages \
  --run-id qwen38_dflash2_matrix_20260907
```

稳态压力测试命令：

```bash
$MINI_PY scripts/benchmark_qwen38_dflash2_concurrency.py \
  --concurrency 8 \
  --workload mixed \
  --repeats 1 \
  --requests-per-run 1000 \
  --soak \
  --soak-requests 1000 \
  --num-speculative-tokens 1 \
  --target-cache-memory-fraction 0.05 \
  --profile-dflash2-stages \
  --run-id qwen38_dflash2_soak_20260907
```

发生 OOM 后不得静默降低并发度。应保留失败的汇总 JSON，并使用其中记录的显存与
失败字段作为容量证据。

## 当前结论

A800 上的执行流程和显存生命周期目前均正常，高并发基准工具能够生成完整的性能
产物。在 C=1/2/4/8 下，K=1 DFlash2 仍慢于 target-only，草稿 token 接受率约为
25% 至 27%。

当前实现在 C=4 下尚不能宣称严格 target-lossless。正式矩阵应继续保持阻塞，直到
缓存目标状态在普通调度与已接受 speculative token 的调度之间实现数值不变性，
或者为两条路径实现并明确记录更高精度的目标决策策略。不得使用 logit 容差将当前
分歧重新标记为严格通过。
