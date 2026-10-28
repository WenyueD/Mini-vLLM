# 同前缀、同候选的 verify / commit 诊断

日期：2026-10-28；服务器 connect.bjb1.seetacloud.com:38750，A800 80GB。

## 结论

本次固定输入实测排除了接受率不同：官方、Mini 默认头、Mini 合并头在首轮均接受候选 `[3,3,3,7]`，提交长度（含 anchor）为 `[4,4,4,8]`。合并 LM head 已把该部分 GPU 时间降至官方水平，但主体线性投影及缓存/状态管理仍是差距来源。commit 主要由逐请求上下文投影和 GDN 前缀状态恢复组成。

## 方法与边界

- 当前本地源码快照上传至独立 candidate 目录；未覆盖服务器生产源码。服务器原有快照没有本地最新合并头选项。
- 官方服务器安装版 vLLM 0.28.0 / g2cf0a6915，V2 runner，BF16，eager；Mini 同样 eager。
- 实际 B4、K7，每请求 anchor + 7 个候选，共32个验证位置。四个原始任务 prompt 长度187、194、190、201。
- 官方先运行相同任务的16 token预热；Mini 也运行4请求×16 token在线预热。跳过 Mini 启动时最大形状预热。
- Mini 首轮候选替换为官方实际 verification 输入；代码逐请求断言完整 prompt token IDs 和 anchor 相同。之后正常生成至16 token。
- Mini 保留 attention pad length=3203、数值一致性线性层、fused GDN、fused prefix commit、compact copies 等原配置。官方使用自己的缓存/attention实现，二者输入一致但内部计算布局并不一致。
- 初始官方 C24 尝试实际首轮只有11请求、88位置，不能算作严格 C24。本报告仅作 B4 诊断，不外推 C24 比例。
- 以下 GPU 时间为 torch profiler 中 kernel/memcpy/memset 时长之和，按 CUDA correlation 映射到 CPU 发射作用域；不是端到端时延。CPU 插桩墙钟受 profiler、Python annotation、同步影响，不作为吞吐基准。
- 单轮、短前缀诊断，不能证明长输出/所有batch的数值一致性或长期收益。日志有 profiler worker-thread callback 警告，但两轮均完整导出 CPU/CUDA trace、成功匹配 correlation 且 complete=true。

## 测量结果

| 项目 | Mini 默认头 | Mini 合并头 | 官方 |
|---|---:|---:|---:|
| LM head GPU活动时间 | 11.894 ms | 1.477 ms | 1.485 ms |
| 主体线性投影 GPU活动时间 | 45.613 ms | 45.601 ms | 34.467 ms |
| Mini整个verify GPU活动时间 | 80.929 ms | 70.522 ms | 见下方口径说明 |
| Mini整个verify GPU操作数 | 5368 | 5352 | — |
| Mini整个commit GPU活动时间 | 5.200 ms | 5.194 ms | 无同边界独立阶段 |
| Mini整个commit GPU操作数 | 904 | 904 | — |
| Mini verify 插桩墙钟 | 360.29 ms | 336.72 ms | 无同边界独立阶段 |
| Mini commit 插桩墙钟 | 48.26 ms | 47.09 ms | 无同边界独立阶段 |

官方首轮 execute_model GPU活动49.087 ms、3028个操作，插桩墙钟137.39 ms；后续 sample_and_next_draft GPU活动6.766 ms，包含 LM head、采样、上下文处理和下一次draft。取 target forward + LM head + sample 三个部分为50.643 ms，**不含所有输出后处理或下一轮状态处理，因此不是与Mini verify严格同边界的“官方verify时延”**。

合并头的 GPU活动下降10.417 ms（87.6%），整个verify GPU活动下降10.407 ms（12.9%）；插桩墙钟下降约6.5%，此比例不能当成真实无插桩收益。首轮接受/纠正token一致；不表示该可选优化已通过所有数值回归。

最终验证脚本确认完整prompt、候选、接受后的纠正token一致，而且两轮Mini的自然候选也与强制输入相同。最初采集脚本挂错proposer入口，已修正为RunnerDFlash2Proposer并重跑；本报告和JSON均为断言通过后的最终数据。

## 差距定位

### 1. LM head 的问题在本轮已消除

默认路径逐位置进行8次词表GEMM；合并后32行一起计算，只有1次词表GEMM，耗时与官方相当。

源码：`src/myvllm/models/qwen3_8.py:1414`。仍为显式可选项；此前其他条件曾发现 BF16 输出差异，本轮一致不足以据此默认开启。

### 2. 主体线性层是合并头后的最大GPU开销

Mini `_qwen35_vectorized_bf16_linear` 累计496次kernel、45.601 ms，占合并后verify GPU活动64.7%。官方对应 MergedColumnParallelLinear + RowParallelLinear + QKVParallelLinear 累计34.467 ms。两者模块分组不同，但都覆盖本轮主体投影；差距约11.13 ms。

Mini为数值一致性采用自定义固定配置Triton投影；官方使用合并投影和优化GEMM。不能把Mini主体解释成仍在逐token循环F.linear：本次已开启vectorized路径。

源码：`src/myvllm/models/qwen3_8.py:90`。

### 3. verify包含大量缓存复制、拼接和小操作

默认头trace中仅显式DtoD memcpy即905次、3.229 ms，另有copy/cat kernels；合并头只减少16个GPU操作。GDN周边、缓存事务和Python组织结构基本未变。合并后verify共有5352个GPU操作，GPU活动70.52 ms而插桩墙钟336.72 ms。

该差值不能全部称作生产环境Python开销：它还包含profiler扰动、同步及GPU空隙。可靠结论是小操作数量大，必须继续融合/批处理，并用无profiler计时验证收益。

### 4. commit不是再跑一次完整target，但仍逐请求恢复和投影

默认头首轮commit分解（GPU活动）：

| 部分 | 时间 | 占比 | GPU操作数 |
|---|---:|---:|---:|
| 上下文投影与安装 | 3.322 ms | 63.9% | 596 |
| GDN接受前缀恢复 | 1.708 ms | 32.9% | 288 |
| hidden history | 0.170 ms | 3.3% | 20 |

三个部分的GPU事件相加正好覆盖本轮commit。合并头结果基本相同。

- `model_runner.py:2128`：遍历请求提交；全接受请求复用最终状态，三个部分接受请求执行restore。
- `hybrid_cache.py:362`：逐请求、逐GDN层从原状态和rank-one更新恢复接受前缀，随后复制conv状态。虽然使用fused prefix kernel，仍按请求×层发射。本轮3请求×48层，共144个恢复kernel及对应复制等操作。
- `model_runner.py:1295`：每请求分别combine_hidden_taps、project_combined_context_kv、install_projected_context；本轮4次独立上下文投影。

官方的GDN kernel用 `num_accepted_tokens` 和 `ssm_state_indices` 选择已存储的接受位置状态，状态处理与下一轮forward结合。服务器代码位置：

```
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1445
vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:106
vllm/model_executor/layers/mamba/ops/causal_conv1d.py:875
```

因此，Mini的“恢复前缀”与官方的状态布局存在结构差异。官方也有状态/上下文处理成本，本次没有提取一个可与Mini commit直接相减的总耗时，不能说官方commit为零。

## 建议的后续优化顺序

1. commit上下文投影跨请求批处理，减少4次独立组合/投影/安装及其小操作；先保持各请求数值语义并做token回归。
2. GDN prefix restore将请求维与层维的工作批量发射，减少大量逐请求kernel和conv状态复制；或研究保存可直接选择的验证位置状态，评估显存代价。
3. 优化主体线性层的GEMM与合并投影；这是GPU计算侧最大项，但需重新验证BF16决策一致性。
4. 对通过数值回归的实现再跑无profiler的C1/C4/C24长输出配对基准；不要用本报告插桩墙钟作最终加速比。

## 文件

- `inputs.json`：四个完整prompt token IDs及模型路径。
- `official_evidence.json`：官方实际query、position、采样结果。
- `mini_evidence.json`：默认头完整输入断言证据、target决策与commit长度。
- `merged/mini_evidence.json`：合并头同样证据。
- `trace_summary.json`、`merged/trace_summary.json`：按发射作用域归属的GPU统计。CPU inclusive字段有嵌套同名作用域，禁止直接累加解释阶段耗时。
- `validation.json`：运行 `scripts/validate_matched_evidence.py` 后的严格输入与决策校验结果。
- 原始Chrome traces和日志保存在服务器 `/root/autodl-tmp/Mini-vLLM/results/qwen38_dflash2_perf/matched_profile_20261028/`，合并版本在其 `merged/`。
- 复现脚本：`scripts/diagnose_official_matched_profile.py`、`scripts/diagnose_mini_matched_profile.py`、`scripts/analyze_matched_traces.py`。Mini脚本用环境变量 `DIAG_MERGED_HEAD=1`切换合并版本。
