# Qwen3.8 DFlash2 高并发基准报告 - 2026-09-07

## 当前状态

截至 2026-09-12，Mini-vLLM Qwen3.8-27B target-only 与 Mini-vLLM
DFlash2 的高并发验证已完成主要可采信矩阵的回收和聚合。Git 历史已在本地与
远端统一到提交 `70239ac3ad2d37593704ba05de12306f709f5c46`，远端原 main
已备份为 `backup/remote-main-8ca09-20260910`。

- 本地完整回归：`153 passed, 2 skipped`。
- 远端完整回归：`152 passed, 2 skipped`。
- 远端运行环境：NVIDIA A800-SXM4-80GB、Python 3.11.16、BF16。
- 已归档远端结果包：`D:\Downloads\qwen38_dflash2_results_20260912_103718.tgz`。
- 聚合产物：
  - `reports/qwen38_dflash2_c1_c32_aggregate_2026-09-12.json`
  - `reports/qwen38_dflash2_c1_c32_paired_2026-09-12.csv`

纳入最终聚合的所有 target-only/DFlash2 fairness 文件均为
`exact_token_match=true`，且 mismatch 数为 0。因此，在这些 Mini-vLLM
target-only 对比样本中，当前 DFlash2 路径满足严格 token ID 一致性。

仍需保留两个边界结论：

- 官方 vLLM 0.28 V2 Oracle 对齐仍为 BLOCKED。此前 Oracle runner 仍表现出
  非无损语义，因此本报告不能被解读为已经完成与官方 DFlash2 的全面对齐。
- C8 mixed soak 尚无可采信正式结果。旧脚本中的 `--soak-minutes` 只被解析，
  未作为 wall-clock 截止条件；`qwen38_dflash2_soak_mixed_c8_500_20260910`
  因而不能代表 10 分钟 soak。随后 `qwen38_dflash2_soak_mixed_c8_64_timeout20m_20260911`
  以 `KeyboardInterrupt` 结束且只有 target-only summary，也不能作为配对
  soak 性能结论。

## 关键修复

本轮工作修复并验证了以下高并发相关问题：

- 目标模型贪心 tie-break 统一为稳定策略，并在 Qwen target-only/verifier
  路径中对 BF16 精确并列使用 FP32 `lm_head` 复判，残余并列回退到固定 token
  ID 策略。
- 资源清理和运行时统计重置已覆盖 proposer、model runner、NCCL/process group
  与 DFlash2 profile 计数器，远端完整回归不再复现跨测试残留导致的 OOM。
- `scripts/benchmark_qwen38_dflash2_concurrency.py` 已补上真实 soak wall-clock
  截止逻辑：到点后停止提交新请求，取消仍在运行的请求，并在 summary 的
  `failures.soak_time_limit` 中记录触发状态、剩余 pending 和取消数量。

## 基准工具

基准脚本为 `scripts/benchmark_qwen38_dflash2_concurrency.py`。

主要特性：

- target-only 与 DFlash2 均通过 `OnlineEngine` 执行，稳态请求延迟不包含模型
  加载和启动时间。
- 两种路径使用相同 tokenizer、prompt token ID、负载定义、准入路径、数据类型、
  缓存设置、`temperature=0.0` 和随机种子。
- 支持闭环并发、开环泊松到达、固定间隔突发到达、混合变长负载、正式测量前
  预热，以及 soak 取消边界。
- 输出逐请求 JSONL、监控 JSONL、单次 summary JSON、聚合 JSON 和 CSV。
- 记录吞吐量、TTFT、TPOT、ITL、E2E、goodput SLO、队列和运行中请求采样、
  DFlash2 proposal profile、接受率、接受长度直方图、batch 大小分桶、CUDA
  显存快照和失败记录。

默认 goodput SLO：

- TTFT <= 5.0s
- TPOT <= 0.25s

## 最终聚合结果

下表为可采信结果的配对聚合。多轮单元使用输出 tok/s 中位数；单轮容量单元使用
该次测量值。`D/T` 为 DFlash2 输出 tok/s 除以 target-only 输出 tok/s。

| 负载 | C | Target tok/s | DFlash2 tok/s | D/T | 接受率 | Target 峰值 GiB | DFlash2 峰值 GiB | 重复数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 4.469 | 3.550 | 0.794x | 25.51% | 55.18 | 58.07 | 4 |
| short | 2 | 5.397 | 4.456 | 0.826x | 24.87% | 55.18 | 58.07 | 1 |
| short | 4 | 5.860 | 5.164 | 0.881x | 23.94% | 55.18 | 58.07 | 1 |
| short | 8 | 6.454 | 5.736 | 0.889x | 26.46% | 55.18 | 58.91 | 4 |
| short | 16 | 6.345 | 5.807 | 0.915x | 23.94% | 55.18 | 63.98 | 4 |
| short | 32 | 6.466 | 6.067 | 0.938x | 23.33% | 55.46 | 74.08 | 1 |
| medium | 1 | 2.702 | 2.253 | 0.834x | 17.87% | 55.05 | 58.01 | 3 |
| medium | 8 | 3.291 | 3.110 | 0.945x | 16.78% | 55.05 | 60.17 | 3 |
| medium | 16 | 3.259 | 3.122 | 0.958x | 18.28% | 55.05 | 66.52 | 4 |
| mixed | 1 | 2.785 | 2.368 | 0.850x | 24.16% | 55.05 | 57.95 | 3 |
| mixed | 8 | 3.293 | 3.054 | 0.927x | 25.05% | 55.05 | 60.17 | 5 |
| mixed | 16 | 3.062 | 2.949 | 0.963x | 24.33% | 54.99 | 66.27 | 4 |
| mixed | 32 | 3.215 | 3.041 | 0.946x | 22.25% | 56.58 | 76.68 | 1 |
| long | 16 | 1.561 | 1.435 | 0.920x | 14.82% | 54.98 | 67.54 | 1 |

medium C32 被拆分执行：target-only 来自
`qwen38_dflash2_capacity_medium_c32_20260909`，DFlash2 来自
`qwen38_dflash2_capacity_medium_c32_dflash2_20260910`，并通过迁移 target
输出进行 fairness 校验：

| 负载 | C | Target tok/s | DFlash2 tok/s | D/T | 接受率 | Target 峰值 GiB | DFlash2 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 32 | 2.706 | 2.498 | 0.923x | 17.55% | 57.78 | 75.48 |

## 代表性三轮

short/medium/mixed 的代表性三轮重复已覆盖 C1、C8、C16：

- C1/C8：`qwen38_dflash2_representative_smm_c1c8_r3_20260910`
- C16：`qwen38_dflash2_representative_smm_c16_r3_20260910`

所有纳入的 fairness 文件均严格通过。代表性重复显示 DFlash2 在高并发下相对吞吐
更接近 target-only，但仍未超过 target-only。

## Open-Loop 与 Burst

| 场景 | 负载 | C | 请求数 | Target tok/s | DFlash2 tok/s | D/T | 接受率 | 严格一致 |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| open-loop, rate=0.5 | mixed | 8 | 128 | 2.997 | 2.836 | 0.946x | 20.53% | 是 |
| burst | mixed | 8 | 128 | 3.034 | 2.884 | 0.950x | 21.13% | 是 |

这两项均完成 target-only 与 DFlash2 配对 summary，且 fairness 为
`exact_token_match=true`。

## Soak 状态

本次结果包中没有可采信的 C8 mixed soak 配对结果。

- `qwen38_dflash2_soak_mixed_c8_500_20260910`：旧脚本只生成 manifest/workload
  和 target-only request log，没有 summary；并且旧版 `--soak-minutes` 未被执行
  为时间截止，不能作为 10 分钟 soak。
- `qwen38_dflash2_soak_mixed_c8_64_timeout20m_20260911`：以 `KeyboardInterrupt`
  结束，只有 target-only summary，没有 DFlash2 配对结果。

因此 soak 结论应标记为 BLOCKED/INVALID，而不是 PASS。后续若需要补齐正式 soak，
应使用已修复脚本，并建议在命令外层继续保留 Linux `timeout` 作为进程级保险丝。

## 排除结果

以下结果仅作为诊断或失败证据保留，不进入最终性能聚合：

- 早期 OOM / process-group 失败：`qwen38_dflash2_pilot_c2_20260908`、
  `qwen38_dflash2_short_matrix_k1_20260908`。
- tie-break 修复前的非无损矩阵/诊断：
  `qwen38_dflash2_short_matrix_k1_c2c4c8_20260908`、
  `qwen38_dflash2_fixedbatch_c4_gate_20260908`、
  `qwen38_dflash2_target_decision_diag_c4_20260908`。
- 中间验证或单边运行：
  `qwen38_dflash2_isolate_c4_req0_20260908`、
  `qwen38_engine_cleanup_validation_20260908`、
  `qwen38_target_repro_c4_r2_20260908`、
  `qwen38_dflash2_representative_smm_c1c8c16c32_r3_20260910`。
- 无效 soak：
  `qwen38_dflash2_soak_mixed_c8_500_20260910`、
  `qwen38_dflash2_soak_mixed_c8_64_timeout20m_20260911`。

## 最终结论

在 Mini-vLLM target-only 对照下，当前 DFlash2 实现已经通过纳入聚合的严格
token ID 一致性检查；此前 C4 BF16 tie 分歧已通过统一贪心 tie-break 和 Qwen
目标决策复判策略解决。

性能方面，本轮 A800 结果没有观察到 DFlash2 加速。DFlash2 在 short/medium/mixed
及 long 已测单元中均慢于 target-only；相对吞吐随并发度升高有所改善，典型从
C1 的约 0.79x-0.85x 提升到 C16/C32 的约 0.92x-0.96x，但仍低于 1.0x。DFlash2
额外显存开销随并发度上升明显，C32 mixed 峰值约 76.68 GiB，C32 medium 峰值约
75.48 GiB。

因此，最终判断为：

- Mini-vLLM 内部 target-lossless：PASS，限于已采样 fairness 覆盖范围。
- C1-C32 高并发性能矩阵：PARTIAL PASS，结果完整但未显示加速。
- open-loop/burst：PASS，严格一致但仍慢于 target-only。
- C8 mixed soak：BLOCKED/INVALID，需要用修复后的脚本重跑。
- 官方 vLLM Oracle 对齐：BLOCKED，不能宣称完成官方 DFlash2 语义对齐。
