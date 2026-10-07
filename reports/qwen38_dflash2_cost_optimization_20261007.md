# Qwen3.8 DFlash2：降低 verifier / commit 成本

日期：2026-10-07。初版 checkpoint：`3a5de90`，消除恢复重打包版本：`a10bfa1`；基线为 `fcf7cf5`，
生产模型数学路径沿用 `b2081ac`。单张 A800-SXM4-80GB，K=7、C4，验证块长4，
固定 micro-batch 4，四个自然代码／推理任务各输出4096 tokens。

本阶段已完成。最终版本的单轮观测：相对本次基线，verifier墙钟减少4.68%，
commit减少7.02%，吞吐提高4.06%。三个正式版本全部逐token一致；
本次没有三轮重复，也没有重新配对target-only，因此不宣称稳定投机加速。

## 实现

三个独立开关默认关闭，便于同代码版本下做单变量或组合对照：

| benchmark 开关 | runtime config | 改动 |
|---|---|---|
| `--fused-prefix-commit` | `qwen35_fused_prefix_commit` | 用 Triton 融合已验证前缀的 GDN 状态恢复 |
| `--reuse-verification-samples` | `qwen35_reuse_verification_samples` | 复用分块 verifier 已计算的 greedy token，避免再次拼接完整 logits 并采样 |
| `--compact-verification-copies` | `qwen35_compact_verification_copies` | 按请求、层、字段打包紧凑更新，避免逐 token 分别 clone，并去掉一次重复 final-state clone |

融合恢复仍按逐位置顺序执行 `state*decay + key*delta`，显式保留每次
FP32 乘法与加法的 round-to-nearest，关闭 FP fusion；不做重新结合。
输入原状态不可变，异常时仍通过事务回滚。最终版本直接传入已有更新的指针元组，
仅分配一个输出状态，不重新打包更新，也不增加逐 token 完整 recurrent-state 快照。

更新记录的每个字段每层每请求只拥有一个打包缓冲，各 token 记录是其互不重叠的视图。
没有保留 padding 请求行。测试比较底层 storage 总字节与统计字节完全一致；
短生命周期的输出状态由 CUDA allocator 的峰值统计覆盖。

target attention、MLP、RoPE、draft proposal、候选精度和固定计算形状保持本轮基线。
本次不混入 selector FP32 对齐修改，从而要求 proposal 接受数和分布也保持一致。

## 正确性验证

- 首批融合恢复／采样复用聚焦测试：139 passed。
- 初版及直接读取版本分别执行 `python -m pytest -q tests`，均为 **255 passed、2 skipped**。
- 包含 FP32/BF16、不同 head dim、所有0–8缓存提交前缀的逐位对照，
  K=7接受长度0–7、异长batch、EOS、verifier异常和commit中途异常rollback。
- 原始 recurrent state 不变；完整新状态与逐步 target 状态逐位一致；
  `transaction_snapshot_state_bytes == 0`。
- `git diff --check` 通过。

全量测试最初误从仓库根目录收集结果目录中的诊断启动脚本，导致递归启动；
已终止该次独立进程组、加入入口保护、明确只收集 `tests/`，确认旧进程组全部退出。
有效全量结果来自修正后的独立运行。正式性能实验在此之后启动。

## 初版算子诊断（3a5de90）

保留优化前后第129、130次 verifier/commit 的完整 Chrome trace。
查询长度均为 `[8,8,8,8]`；两个commit前缀分别为 `[8,1,2,8]` 和 `[6,1,6,2]`。

| 周期／阶段 | 基线 GPU kernel/copy 总时间 ms | 优化 ms | 基线→优化 kernel/copy 次数 |
|---|---:|---:|---:|
| 129 verify | 226.127 | 217.892 | 15172 → 11609 |
| 130 verify | 248.486 | 241.886 | 15171 → 11608 |
| 129 commit | 6.980 | 5.553 | 1132 → 1084 |
| 130 commit | 18.815 | 8.053 | 2955 → 1564 |

其中前缀恢复为 3.274→1.842 ms、15.097→4.330 ms。
局部GPU时间不等于请求墙钟，不能将两个样本当作稳定端到端加速。
profiler会显著扰动CPU与同步开销；正式A/B使用 `--profile-calls 0`，
关闭细分DFlash profiling，保留相同内存记录设置。

## 实验设计与证据

服务器目录：`results/qwen38_dflash2_perf/cost_optimization_20261007/`。

- `baseline_source.tar.gz`、`baseline_diff.patch`、`baseline_hashes.json`：修改前完整备份。
- `patch_files.tar.gz`、`patch_files_v2.tar.gz`：两阶段源码及测试；
  安装前逐文件确认远端内容没有变化，不覆盖未知改动。
- `baseline_profile/`、`optimized_profile/`：算子统计与完整trace。
- `operator_comparison.json`：相同周期的算子对照。
- `focused_tests.log`、`full_tests.log`、`full_tests_exit.json`：测试证据。
- `experiment_driver.py`、`experiment_status.json`：每阶段精确命令与状态。
  每阶段完成后独立读取request JSONL，与既有target-only的4×4096输出逐token比较；
  接受数与直方图也必须相同才继续下一阶段。

正式A/B使用相同源码与相同四个prompt token序列，greedy、ignore_eos、seed不变。
两模式都跳过耗时的初始化最大窗口预热，均使用4个在线预热请求、每个64输出，
预热不计入正式计时。正式请求始终各4096输出。加载、初始化与在线预热独立记录。
因此本次配对之间可比较；不直接与采用不同预热流程的历史吞吐混合统计。

所有阶段已完成；原始JSONL审计、配置与源码差异核查记录在
`cost_results_analysis.json` 和 `profile_output_audit.json`。

## 第一版正式 A/B 与继续修正的依据

关闭算子 profiler 和细分 DFlash profiling，同设置按基线、初版的顺序运行：

| 指标 | 基线 | 初版3a5de90 |
|---|---:|---:|
| 输出吞吐 tok/s | 33.2559 | 34.6218 |
| verifier 秒 | 357.5997 | 338.3102 |
| commit 秒 | 47.0416 | 47.5845 |
| proposal 秒 | 27.4425 | 27.4291 |
| 峰值分配 GiB | 59.4788 | 59.4762 |

初版单轮吞吐提高4.11%，verifier下降5.39%，但commit增加0.54秒，
**不能根据局部GPU时间下降宣称commit墙钟改善**。
短提交前缀的重打包和启动开销抵消了恢复kernel的收益，因此继续用
`a10bfa1` 消除恢复入口的三个 `torch.stack`，直接读取已有记录。
两个版本数学顺序相同；第二版的完整4K结果单独记录。

初版正式A/B的4×4096输出全部与既有target-only一致，接受数11202、
接受率30.919981%、接受长度直方图和所有verifier执行计数完全不变。
紧凑更新峰值170164224字节；逐token完整state快照为0。
每个版本目前各一次正式运行，以上数据不构成稳定加速证明。

## 最终版本 a10bfa1 的正式结果

在基线、初版完成后运行直接读取版本。除指定优化开关外，manifest参数一致；
初版到最终版只有 `src/myvllm/models/qwen35_gdn.py` 的源码哈希发生变化。
该文件本地与远端原始SHA256均为
`acac4353cac91b72fcd623564a4a1b6afb4ed24156a8d7c75974a81412d0cdec`。

| 指标 | 基线 | 最终版本 | 相对变化 |
|---|---:|---:|---:|
| 输出吞吐 tok/s | 33.2559 | 34.6059 | +4.06% |
| 最后请求完成墙钟 秒 | 492.6641 | 473.4451 | -19.2190秒 |
| verifier 秒 | 357.5997 | 340.8583 | -4.68% |
| commit 秒 | 47.0416 | 43.7398 | -7.02% |
| proposal 秒 | 27.4425 | 28.2848 | +3.07% |
| TTFT均值 秒 | 60.2111 | 60.1958 | 基本不变 |
| TPOT均值 ms | 102.9953 | 98.4325 | -4.43% |
| TPOT p99 ms | 105.5186 | 100.8330 | -4.44% |
| 峰值分配 GiB | 59.4788 | 59.4846 | +0.0059 GiB |
| 峰值保留 GiB | 60.2148 | 60.0078 | -0.2070 GiB |

最终版本降低了commit墙钟，但本次吞吐与初版相近：34.6059 vs 34.6218。
proposal未改，却出现约3%的耗时变化，说明单轮仍有环境／运行波动。
不能将初版与最终版之间0.046%的吞吐差认定为回归，也不能宣称第二版有稳定吞吐收益。

最终版接受11202 / 提议36229 = 30.919981%；接受长度0–7直方图仍为
`[1319,1230,875,587,362,228,166,411]`。verifier统计也与基线完全一致：
1345次验证调用、25372个实际执行的请求token位置、避免执行16035个位置。
因此收益没有来自改变输出、接受策略或少计验证工作。

三次正式运行共12请求、49152输出token，外加两次算子诊断8请求、32768输出token，
合计 **20请求、81920 token** 与同一个已审计target-only序列完全一致。
每请求输入仍分别187、194、190、201 tokens；输出4096，包含推理token，ignore_eos。
显存控制保持：紧凑更新峰值170164224字节，逐token完整state快照为0。

## 使用与后续门槛

继续使用原来的 K=7 / 4-token 并行验证 / vectorized target / fused GDN 配置，
额外加上三个开关即可启用本阶段最终版本：

```text
--fused-prefix-commit --reuse-verification-samples --compact-verification-copies
```

目前保留开关默认关闭，便于同代码做回退和配对。服务器已安装最终源码；
`candidate.bundle` 和 `refined_candidate.bundle` 保存完整Git历史，验证可恢复，
没有重置远端原有HEAD或未提交改动。所有GPU进程在本阶段完成后退出。

剩余成本仍以verifier为主，约占最终版计时解码阶段的82.6%。
下一轮应针对full-attention、投影与CPU调度做更集中的优化，再做至少三轮
target-only／DFlash2配对确认稳定收益。当前只证明了这组负载的一轮成本下降；
未完成新的C8矩阵或稳定端到端加速验收。

大型 `.trace.json` 原件留在服务器，归档清单记录其哈希；
紧凑本地证据归档包含配置、源码、diff、全部请求JSONL、summary、算子表、测试与分析脚本。
