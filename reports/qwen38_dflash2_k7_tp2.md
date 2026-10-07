# Qwen3.8 + DFlash2 K=7 / TP=2 阶段报告

初始日期: 2026-09-20；最新更新: 2026-09-30

起点: `c911bc4415cd3a63f673860b887e3ae8e8c881a7`

最新状态（2026-09-30）: 共同 target 数值路径与分块验证已接入；本轮实现
GDN 融合递推和精确前缀恢复（checkpoint `dfc47c9`），C4/C8 各完成三轮配对，
144/144 请求、9216/9216 输出 token 严格一致，最终测试 248 passed、2 skipped。
C4 三轮吞吐比为 1.0028/0.9915/0.9741，C8 为 1.0061/1.0919/0.9556，
两者均未证明稳定端到端加速。DFlash2 显存峰值分别约 55.613/57.065 GiB。
GDN 耗时已明显下降；当前主要瓶颈为线性投影、缓存调度及占总耗时约
86%–88% 的逐 token prefill。新数值路径改变部分历史输出，实验开关默认关闭。
当前服务器仅有一张 A800，TP=2 与 C16/C32 双卡验证未完成。
完整证据、方法、限制和下一步见 [2026-09-30 性能报告](qwen38_dflash2_gdn_profile_20260930.md)。

历史状态（2026-09-20）: RoPE 初始化修复、全接受缓存复用及部分接受状态恢复均已通过 C2/C4 验证。当时 C2 为 8/8、C4 为 16/16 输出一致；DFlash2 吞吐分别为 3.7986 和 4.3530 token/s，仍低于成对 target-only。2026-09-20 在端口 57195 读取并独立复核完整 C4 结果，当时未启动 C8/TP2。下文按日期保留历史结果。

## 阶段边界

本轮严格按阶段推进，没有跳过阶段。

1. 阶段一: 修复 Qwen3.8 + DFlash2 的显存复杂度风险，并补内存 trace。已完成 after C1/C2/C4/C8；before baseline 有 C1/C2/C4，C8 被用户要求暂停后中断，不纳入有效结果。
2. 阶段二: 单卡 K=7 的 C1/C2/C4/C8 正确性与稳定性。旧实现已通过；RoPE 修复后需重新验证。
3. 阶段三: 接受率和性能方向分析。进行中，已确认 RoPE 加载缺陷并修复。
4. 阶段四: TP=2 下 C16/C32 验证。未开始。

阶段一 memory 证据保留；当前先完成 RoPE 修复后的成对正确性验证，再决定性能优化方向。

## 已读历史证据

- `reports/qwen38_dflash2_handoff_2026-09-04.md`: 之前已完成 Qwen3.8 text-only target、DFlash2 draft 权重加载、隐状态 tap、DFlash2 proposal/verify/commit 的基础集成；官方 Oracle 仍受阻。
- `reports/qwen38_dflash2_high_concurrency_2026-09-07.md`: 历史 A800 80GB 结果显示 K=1 的 C1-C32 target/DFlash2 输出逐 token 一致，但 DFlash2 性能慢于 target-only，显存峰值曾到约 75-77 GiB。该证据只覆盖旧实现和 K=1，不能替代本阶段 K=7 的 before/after 内存证据。

## 阶段一实现

主要修改目标是把会随 `batch * K * state` 放大的显存路径降下来，并让 benchmark 能导出阶段级内存快照。

- Qwen3.8 不再分配未被模型使用的 generic paged KV pool。之前 block budget 使用 full-attention 层数估算，但实际 tensor 形状按 `num_layers` 分配；Qwen3.8 attention 模块也没有 `k_cache/v_cache` 消费该池。现在 Qwen3.5/Qwen3.8 分支完全跳过 generic pool，按真实动态缓存估算 `max_cached_blocks`。
- 新增 Qwen3.5/Qwen3.8 动态缓存 byte model，区分 full-attention token KV、linear recurrent state、linear conv state、DFlash2 hidden/context/projected KV，并写入 `config["qwen35_dynamic_cache_capacity"]`。
- 新增 `FullAttentionLayerState`，full-attention KV 在已有 capacity 内 append，容量不足时翻倍扩容，避免每 token `torch.cat + clone` 复制全历史。
- Hybrid cache transaction 从每 token clone 完整 request state 改成轻量 `record_step()`。commit 时先 restore 原始状态，再只 replay accepted prefix，并用 `return_logits=False` 跳过 logits 分配。
- verification pending entry 不再保存完整 target output/logits，只保留 DFlash2 需要的 hidden taps、target token ids、start position 和 query tokens。
- Qwen3.8 cached decode / verification / accepted-prefix replay 默认固定 padding 到 `max_model_length` 和最多 4 个请求，三条路径使用相同 CUDA kernel shape；超过 4 个请求时拆成固定 micro-batch 4。可通过显式设置 `qwen35_cached_attention_pad_to_length` / `qwen35_cached_batch_pad_to_size` 覆盖。
- 新增 `MemoryTrace`，benchmark 可通过 `--memory-trace` 导出 load、warmup、cache allocation、prefill、target decode、proposal、verify、commit、rollback、release 等阶段的 CUDA allocated/reserved/max/free/total 以及估算字段。
- `DFlash2DraftCache.estimate_bytes()` 会报告 hidden tap、context position、projected KV 和 active proposal snapshot 数量。

## 本地验证

本地环境只有 RTX 4060 8GB，且当前 `.venv` 的 PyTorch 是 CPU-only:

```text
nvidia-smi: NVIDIA GeForce RTX 4060, 8188 MiB total
torch.cuda.is_available(): False
torch.version.cuda: None
```

因此本机不能运行 Qwen3.8-27B CUDA 验证，也不能采集 A800 before/after 显存数据。

已完成的本地验证:

```text
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pytest -q tests\test_hybrid_cache_transactions.py
4 passed
.\.venv\Scripts\python.exe -m pytest -q tests\test_qwen35_config_and_model.py
22 passed
.\.venv\Scripts\python.exe -m pytest -q tests\test_qwen38_dflash2_concurrency_benchmark.py
12 passed
.\.venv\Scripts\python.exe -m pytest -q tests\test_dflash2_contract.py tests\test_spec_coordinator.py
38 passed
.\.venv\Scripts\python.exe -m pytest -q tests\test_engine_spec_integration.py tests\test_engine_requests.py
7 passed
.\.venv\Scripts\python.exe -m pytest -q tests\test_verification_input.py tests\test_kv_transactions.py
10 passed
.\.venv\Scripts\python.exe -m pytest -q
156 passed, 2 skipped
git diff --check
exit 0, only CRLF warnings on Windows
```

## A800 阶段一采集命令

以下为阶段一 after 证据的采集命令模板；本轮已在 A800 80GB、CUDA PyTorch、Qwen3.8-27B target 和 DFlash2 draft 权重环境上完成采集，实际结果见后文。

```bash
cd /root/autodl-tmp/Mini-vLLM
export MINIVLLM_QWEN38_PATH=/path/to/Qwen3.8-27B
export MINIVLLM_DFLASH2_PATH=/path/to/DFlash2

python -m pytest -q \
  tests/test_hybrid_cache_transactions.py \
  tests/test_qwen35_config_and_model.py \
  tests/test_qwen38_dflash2_concurrency_benchmark.py \
  tests/test_dflash2_contract.py \
  tests/test_spec_coordinator.py \
  tests/test_engine_spec_integration.py \
  tests/test_engine_requests.py \
  tests/test_verification_input.py \
  tests/test_kv_transactions.py

python scripts/benchmark_qwen38_dflash2_concurrency.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --dflash2-safe-open-device cuda \
  --num-speculative-tokens 7 \
  --concurrency 1 2 4 8 \
  --workload short \
  --variants target_only dflash2 \
  --repeats 1 \
  --requests-per-run 32 \
  --warmup-requests 1 \
  --profile-dflash2-stages \
  --memory-trace \
  --run-id qwen38_dflash2_phase1_k7_mem_after_$(date +%Y%m%d_%H%M%S)
```

阶段一 before/after 对比需要旧代码基线。旧代码没有 `--memory-trace`，所以有两种可接受方式:

1. 在旧起点 `c911bc4415cd3a63f673860b887e3ae8e8c881a7` 上跑同等 benchmark，使用进程外 GPU sampler 的 peak 显存和 OOM/成功状态作为 before。
2. 把纯 instrumentation patch 移植到旧起点，但不要带阶段一修复，再跑相同 `--memory-trace` 命令作为 before。

本轮 before 采用方式 1；用户要求跑到 C4 后暂停，因此 before 有效结果只覆盖 C1/C2/C4。

## 远端暂停点

用户提供 SSH 后，已连接远端 `connect.bjb1.seetacloud.com:41649`，确认 GPU 为 `NVIDIA A800-SXM4-80GB`，空卡，远端 `.venv` 为 PyTorch `2.9.1+cu128` 且 CUDA 可用。

远端仓库 `/root/autodl-tmp/Mini-vLLM` 当前 `main` 为 `70239ac3ad2d37593704ba05de12306f709f5c46`，本轮在其上创建工作分支 `codex/qwen38-dflash2-k7-memory` 并应用阶段一补丁。远端全量测试通过:

```text
/root/autodl-tmp/Mini-vLLM/.venv/bin/python -m pytest -q /root/autodl-tmp/Mini-vLLM/tests
155 passed, 2 skipped
```

按用户要求，阶段一 before baseline 跑到 C4 后暂停。baseline worktree 为 `/root/autodl-tmp/Mini-vLLM-baseline`，commit 为 `70239ac3ad2d37593704ba05de12306f709f5c46`。有效结果只统计 C1/C2/C4；C8 已启动但被中断，不纳入有效结果。

结果包已保存:

```text
/root/autodl-tmp/qwen38_dflash2_phase1_k7_mem_before_c1_c4_20260912_2030.tgz
```

baseline before summary:

| C | requests | output TPS | TTFT p50 s | TPOT p50 s | acceptance | peak allocated GiB | error |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 1.6618 | 9.5575 | 0.4618 | 0.0459 | 58.0677 | none |
| 2 | 8 | 2.5703 | 9.9636 | 0.6156 | 0.0431 | 58.0747 | none |
| 4 | 8 | 3.3934 | 29.2118 | 0.7264 | 0.0439 | 60.1096 | none |

暂停时远端 GPU 已释放:

```text
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
0 MiB, 0 %, 35
```

## 阶段一 after 结果

2026-09-13 在新服务器 `connect.bjb1.seetacloud.com:26664` 继续。确认 GPU 为 `NVIDIA A800-SXM4-80GB`，空卡，远端 `.venv` 为 PyTorch `2.9.1+cu128` 且 CUDA 可用。

当前补丁分支 `/root/autodl-tmp/Mini-vLLM` 的阶段一 after 命令已完成:

```text
run_id: qwen38_dflash2_phase1_k7_mem_after_20260913_1030
results: /root/autodl-tmp/Mini-vLLM/results/qwen38_dflash2_perf/qwen38_dflash2_phase1_k7_mem_after_20260913_1030
logs: /root/autodl-tmp/Mini-vLLM/logs/qwen38_dflash2_perf/qwen38_dflash2_phase1_k7_mem_after_20260913_1030
package: /root/autodl-tmp/qwen38_dflash2_phase1_k7_mem_after_c1_c8_20260913_1030.tgz
```

after summary:

| C | requests | output TPS | TTFT p50 s | TPOT p50 s | acceptance | peak allocated GiB | error |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 1.3695 | 9.6789 | 0.5941 | 0.0474 | 55.6515 | none |
| 2 | 8 | 1.9659 | 10.3611 | 0.8553 | 0.0451 | 55.9798 | none |
| 4 | 8 | 2.4914 | 34.3701 | 1.1533 | 0.0424 | 56.6451 | none |
| 8 | 8 | 2.9569 | 75.9496 | 1.5149 | 0.0467 | 57.9783 | none |

after memory trace 关键检查:

| C | trace points | max snapshot bytes | max recorded steps | max generic pool bytes | max generic layers | max CUDA allocated GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3511 | 0 | 8 | 0 | 0 | 55.6515 |
| 2 | 2031 | 0 | 16 | 0 | 0 | 55.9798 |
| 4 | 1263 | 0 | 32 | 0 | 0 | 56.6451 |
| 8 | 855 | 0 | 64 | 0 | 0 | 57.9783 |

阶段一 memory 结论: 当前实现已在 K=7 C1/C2/C4/C8 下证明不再保留 `batch * K` 份完整 hybrid cache snapshot，Qwen3.8 generic KV pool 全程为 0，C8 peak allocated 为 57.98 GiB，远离 A800 80GB 上限。

## 阶段二 C1-C4 暂停点

按用户要求，本轮在 C4 DFlash2 完成后暂停。进程当时已经进入 C8 target-only，已立即中断；C8 不纳入有效结果。

```text
run_id: qwen38_dflash2_k7_c1_c8_correctness_tie_policy_20260913_1150
results: /root/autodl-tmp/Mini-vLLM/results/qwen38_dflash2_perf/qwen38_dflash2_k7_c1_c8_correctness_tie_policy_20260913_1150
logs: /root/autodl-tmp/Mini-vLLM/logs/qwen38_dflash2_perf/qwen38_dflash2_k7_c1_c8_correctness_tie_policy_20260913_1150
package: /root/autodl-tmp/qwen38_dflash2_k7_correctness_tie_policy_c1_c4_20260913_1150.tgz
```

性能与显存:

| variant | C | requests | output TPS | TTFT p50 s | TPOT p50 s | acceptance | peak allocated GiB | error |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| target_only | 1 | 8 | 4.3177 | 9.3994 | 0.0849 | 0.0000 | 50.4555 | none |
| dflash2 | 1 | 8 | 1.3835 | 9.5095 | 0.5852 | 0.0469 | 55.6474 | none |
| target_only | 2 | 8 | 5.4032 | 18.2156 | 0.0845 | 0.0000 | 50.6042 | none |
| dflash2 | 2 | 8 | 2.1179 | 18.2986 | 0.6626 | 0.0442 | 57.0095 | none |
| target_only | 4 | 8 | 5.7181 | 37.9993 | 0.0922 | 0.0000 | 50.9313 | none |
| dflash2 | 4 | 8 | 2.9459 | 34.1256 | 0.8327 | 0.0420 | 59.2141 | none |

Token equality:

| C | exact token match | mismatches | sampled requests | mismatch details |
|---:|---|---:|---:|---|
| 1 | true | 0 | 8 | none |
| 2 | false | 3 | 8 | `short-000004` idx 25 target 68 vs candidate 9997; `short-000005` idx 60 target 26023 vs candidate 52789; `short-000007` idx 19 target 198 vs candidate 271 |
| 4 | false | 1 | 8 | `short-000002` idx 59 target 1156 vs candidate 9640 |

阶段二 memory trace 关键检查:

| C | trace points | max snapshot bytes | max original bytes | max recorded steps | max generic pool bytes | max CUDA allocated GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3533 | 0 | 166789120 | 8 | 0 | 55.6474 |
| 2 | 2055 | 0 | 332726272 | 16 | 0 | 57.0095 |
| 4 | 1261 | 0 | 665452544 | 32 | 0 | 59.2141 |

该组旧结果的结论: memory 修复仍然成立，但移除默认 cached padding 后，K=7 correctness 在 sampled=8 的 C2/C4 下未通过。之前 `qwen38_dflash2_k7_c2_debug_tie_policy_20260913_1145` 的 C2 小样本 debug 只 sampled=2 且通过，不能覆盖本轮 C2 sampled=8 暴露出的三处 mismatch。

暂停时远端 GPU 已释放:

```text
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
0 MiB, 0 %, 37
```

## 阶段二根因修复与复验

两处 C2 mismatch 的共同特征是 target-only decode 的 top-1 比 top-2 高 `0.125`，而 speculative target verifier 在同一位置把两者量化为 BF16 精确并列。进一步审计发现，阶段一为了降低临时显存，把 cached decode / verifier 的默认固定 padding 移除了；accepted-prefix replay 也使用变化的活跃 batch。不同 batch/context shape 会选择不同 CUDA GEMM reduction 形状，导致 GDN/attention cache 状态产生微小 BF16 舍入差异，经过多轮 replay 后改变边界 token 的 greedy 决策。

修复内容:

- `ModelRunner` 对 Qwen3.8 默认设置 `qwen35_cached_batch_pad_to_size=min(4, max_num_sequences)` 和 `qwen35_cached_attention_pad_to_length=max_model_length`。
- `forward_with_cache_batch()` 对超过固定 padding 大小的请求自动分块；C8 的 decode、verify、replay 均拆成两个 batch-4 kernel，尾块不足 4 时补齐。
- benchmark 显式写入并记录这两个值，确保 target-only、verify、replay 的形状一致。
- tie policy 保持文档约定的 highest-token-id；不再用 tie policy 掩盖上游 logits 漂移。
- fairness 比较改为检查全部请求，`sample_limit` 只限制保存的 mismatch 明细数量。

修复后结果:

| variant | C | requests | output TPS | TTFT p50 s | TPOT p50 s | acceptance | peak allocated GiB | error |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| target_only | 2 | 8 | 4.9971 | 19.6448 | 0.0932 | 0.0000 | 50.6043 | none |
| dflash2 | 2 | 8 | 1.9668 | 11.1885 | 0.8389 | 0.0430 | 57.7100 | none |
| target_only | 4 | 16 | 5.7254 | 37.8838 | 0.0963 | 0.0000 | 50.9314 | none |
| dflash2 | 4 | 16 | 2.8470 | 15.0249 | 1.0285 | 0.0388 | 63.9155 | none |
| target_only | 8 | 32 | 5.8624 | 75.8354 | 0.1823 | 0.0000 | 51.5413 | none |
| dflash2 | 8 | 32 | 2.9243 | 19.0032 | 2.2184 | 0.0428 | 73.3831 | none |

Token equality:

| C | exact token match | compared requests | mismatches |
|---:|---|---:|---:|
| 2 | true | 8/8 | 0 |
| 4 | true | 16/16 | 0 |
| 8 | true | 32/32 | 0 |

C4 DFlash2 memory trace 共 2193 个点；`max_snapshot_bytes=0`、`max_generic_kv_bytes=0`、`max_generic_pool_bytes=0`、`max_recorded_steps=32`，事务原始状态峰值 `664469504` bytes。固定 shape 使 C4 DFlash2 峰值升至 63.92 GiB，但仍低于 A800 80GB，且没有恢复 `batch * K` snapshot 或未使用的 generic KV pool。

首次 C8 使用固定 batch 8，DFlash2 在第二波请求中达到 78.23 GiB，并在额外申请 20 MiB 时 OOM；16 个已完成请求与 target-only 全部一致，其余请求因引擎 unhealthy 未完成。改为固定 micro-batch 4 后，32 个请求全部成功，峰值降到 73.38 GiB。C8 memory trace 共 2183 个点；`max_snapshot_bytes=0`、`max_generic_kv_bytes=0`、`max_generic_pool_bytes=0`、`max_recorded_steps=64`，事务原始状态峰值 `1327038464` bytes，admission failure 为 0。

结果目录:

```text
/root/autodl-tmp/Mini-vLLM/results/qwen38_dflash2_perf/qwen38_dflash2_k7_c2_fixed_shape_20260913_1420
/root/autodl-tmp/Mini-vLLM/results/qwen38_dflash2_perf/qwen38_dflash2_k7_c4_fixed_shape_20260913_1435
/root/autodl-tmp/Mini-vLLM/results/qwen38_dflash2_perf/qwen38_dflash2_k7_c8_fixed_shape_20260913_1625
/root/autodl-tmp/Mini-vLLM/results/qwen38_dflash2_perf/qwen38_dflash2_k7_c8_microbatch4_20260913_1645
```

C4 原始 benchmark fairness 文件按旧逻辑只抽查 8 条；同目录额外保存 `fairness_short_c4_r0_full.json`，它由两份 request JSONL 的全部 16 条输出直接比较得到。

最新聚焦测试:

```text
tests/test_qwen38_dflash2_concurrency_benchmark.py + tests/test_qwen35_config_and_model.py + speculative transaction tests
59 passed (remote), 39 + 20 passed (local focused groups)
git diff --check: exit 0 (Windows CRLF warnings only)
```

## 阶段一 memory 验收口径

阶段一 memory 部分只有在以下条件同时满足时才算完成:

- memory trace 或外部 sampler 证明当前实现不再保留 `batch * K` 份完整 hybrid cache snapshot。
- `qwen35_capacity_generic_kv_pool_bytes` 为 0，`generic_kv_cache_layers` 为 0。
- cached decode / verifier / replay 使用相同的固定 batch/context shape；固定 padding 的临时显存不能恢复完整 transaction snapshot 或 generic KV pool。
- 无新增本地测试失败，`git diff --check` 通过。

## 提交状态

阶段一和阶段二已固化为 checkpoint `307b174` (`Stabilize Qwen3.8 DFlash2 memory and correctness`)。阶段三诊断工具与结果单独提交，不混入该 checkpoint。

## 阶段三 acceptance 根因定位

先复用阶段二 C2 request 0，未重跑并发矩阵。其 122-token prompt 可由 tokenizer 无损 round-trip；Mini target-only 与 DFlash2 已在阶段二对该请求保持 64/64 token 完全一致。

Oracle 运行时审计:

- vLLM `0.28.0`，实际类为 `vllm.v1.worker.gpu.spec_decode.dflash2.speculator.DFlash2Speculator`。
- 基类固定 `num_query_per_req = 1 + num_speculative_steps`。
- `sample_from_anchor=True` 被显式拒绝，运行时 `sample_from_anchor=False`。
- 缺失 `ninja` 可执行文件会使默认 FlashInfer sampler 在 warmup 失败；设置 `VLLM_USE_FLASHINFER_SAMPLER=0` 后使用 PyTorch sampler，不改变 greedy 语义。

同输入 target-only 对照:

```text
Mini target:   [1204, 63520, 45543, 72346, ...]
Oracle target: [1204, 63520, 45543,   271, ...]
```

两者前三个生成 token 相同，首个 target baseline 差异位于生成位置 3（零基）。因此后续 Oracle 全序列不能直接作为 Mini verifier 的逐 token 数值基准。

首次 proposal 对照仍可比较，因为它们共享相同 prompt、anchor 和 position:

```text
anchor_token_id=1204
anchor_position=122
target_layer_ids=[5, 19, 33, 47, 61]
Oracle proposal=[63520, 45543, 72346, 54102, 2468, 21106, 13]
Mini proposal=  [54102,   424, 54102,   424, 54102,   424, 54102]
```

Oracle proposal 与 Oracle target 前三步一致，fixture 通过正式 target-lossless gate；Mini proposal 在位置 0 即与 Mini target top-1 `63520` 不同。这解释了 K=7 仅约 4% 的总体 token acceptance：根因发生在 draft proposal 路径，早于 verifier rejection/commit。

旧 exporter 在 greedy DFlash2 路径读取 `_cached_candidate_ids`，但该缓冲区只在 probabilistic draft logits 路径写入，因此得到全零占位，不能判断候选 top-k 与 selector 谁先分叉。阶段三工具已改为直接挂钩 `compute_candidates()`，并补充真实 candidate IDs、unary scores、combined context/proposal hidden checksum、selector selected indices；Mini fixture 同时补充 hidden-tap checksum、target top1/top2 和 first rejection index。用户已授权上传并完成远端诊断。

结果目录:

```text
/root/autodl-tmp/Mini-vLLM/results/qwen38_target/oracle_alignment_stage3/oracle_target_c2_request0.txt
/root/autodl-tmp/Mini-vLLM/results/qwen38_target/oracle_alignment_stage3/oracle_dflash2_c2_request0.json
/root/autodl-tmp/Mini-vLLM/results/qwen38_target/oracle_alignment_stage3/minivllm_dflash2_c2_request0.json
```

阶段三当前结论是继续定位 draft proposal 的首个数值差异，暂不启动 TP2 或正式三重复性能测试。

## 2026-09-17 RoPE 加载缺陷修复

新服务器端口为 `22330`，GPU 为 A800 80GB。原始诊断文件和模型快照均保留。

已确认目标模型 `Qwen35RotaryEmbedding` 和草稿模型 `DFlash2Attention` 的
`inv_freq` 是非持久化 buffer：在 meta 设备构造后调用 `to_empty()` 会分配未初始化
存储，而 safetensors 加载仅恢复参数，不会恢复此 buffer。CPU 小模型已复现其值
不等于配置定义的 RoPE 频率。该问题同时存在于生产 runner 和 standalone smoke 的
加载流程，此前逐层误差在首个 full attention 后放大与此一致。

两处模块现在在 `_apply()` 后依据配置和目标设备重新生成 FP32 频率，覆盖
`to_empty()` 以及 dtype/device 转换。回归测试覆盖父模块递归 meta materialization、
BF16/FP16/FP32 转换和多个位置（含长位置）的 cos/sin，不依赖 checkpoint 保存 buffer。

原 C2 `short-000000` 单样本，未注入 Oracle hidden taps、未启用实验性 GDN 替换：

| 项目 | 修复前 | 修复后 |
|---|---|---|
| proposal | `[54102,424,54102,424,54102,424,54102]` | `[63520,45543,72346,54102,2468,21106,13]` |
| 与 Oracle proposal 一致 | 否 | 7/7 |
| 首次 rejection index（零基） | 0 | 2 |

相同 anchor 为 token `1204`、position `122`。使用 FP64 dot/norm 比较已有 Oracle
张量：normalized context cosine 为 `0.99304280`、relative L2 为 `0.11789589`；
proposal hidden cosine 为 `0.99923154`、relative L2 为 `0.03920015`。
候选 token 对齐不表示中间浮点数完全一致，也不表示七个候选全被 target 接受。

此前注入 Oracle target taps 仍产生错误交替 proposal，说明不能将草稿错误仅归于
目标模型数值漂移。RoPE 修复后候选恢复，确认这是本轮定位到的关键加载缺陷。
所有修复前的 correctness/performance 数据保留作历史记录，不能作为有效 RoPE 下的
正确性或性能验收依据，需要重新运行成对验证。

本地验证：`pytest -q` 为 **166 passed, 2 skipped**。

单样本结果（远端目录 `results/qwen38_target/oracle_alignment_stage3/`）：

- `minivllm_c2_rope_fixed.json`
- `minivllm_c2_rope_fixed.txt`
- `minivllm_c2_rope_fixed_tensors.pt`

C2 八请求、K7、单重复的 target-only / DFlash2 成对复验已完成，结果目录：
`results/qwen38_dflash2_perf/qwen38_dflash2_k7_c2_rope_fixed_20260917/`。

| variant | C | requests | output TPS | TTFT p50 s | TPOT p50 s | acceptance | peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 2 | 8 | 5.2752 | 18.6931 | 0.0863 | 0 | 50.6043 |
| dflash2 | 2 | 8 | 3.0683 | 10.7807 | 0.4628 | 0.471605 | 63.0080 |
| target_only | 4 | 16 | 5.7354 | 38.3540 | 0.0916 | 0 | 50.9322 |
| dflash2 | 4 | 16 | 3.6368 | 19.5539 | 0.6702 | 0.392473 | 72.4169 |

`fairness_short_c2_r0.json` 比较全部 8 条请求：exact token match 为 true，
mismatch count 为 0；两组均完成 512 个输出 token，未报告运行错误。
DFlash2 接受 382/810 个候选。接受率从历史 4.3% 提升到 47.2%，但本轮
吞吐仍仅为成对 target-only 的约 0.582 倍，不能宣称已有性能加速。

C2 speculative 阶段计时：verify 为 47.3840 s（51.72%），commit 为
42.7176 s（46.63%），proposal 为 1.5067 s（1.64%）。这些比例仅针对
speculative 阶段，不是总运行时间的比例。后续性能方向应优先审计 verify/commit。

服务器 CPU RoPE 回归测试为 2 passed。C4 使用 16 请求、K7、单重复，
目录为 `results/qwen38_dflash2_perf/qwen38_dflash2_k7_c4_rope_fixed_20260917/`。
`fairness_short_c4_r0.json` 全量比较 16 条请求，exact token match 为 true、
mismatch count 为 0；两组各完成 1024 个输出 token，无 admission failure 或 schema error。
本地再次从原始 request JSONL 独立复核，C2 和 C4 均为零差异。

C4 接受 730/1860 个候选，接受率 39.25%（修复前历史值约 3.88%）；
吞吐为成对 target-only 的约 0.634 倍，尚未获得吞吐加速。
C4 speculative 阶段耗时为 verify 58.4539 s、commit 63.3116 s、proposal
1.8381 s，仍主要消耗在 verify/commit。

| DFlash2 memory trace | C2 | C4 |
|---|---:|---:|
| trace points | 787 | 887 |
| max snapshot bytes | 0 | 0 |
| max original state bytes | 330891264 | 663355392 |
| max recorded steps | 16 | 32 |
| max generic KV pool bytes | 0 | 0 |

结果已保存在远端；本地工作区保存了 C2 全部指标、C4 汇总/一致性/manifest、
两组完整 request JSONL 以及单样本 Oracle 对照 JSON。
对应请求明细位于 `logs/qwen38_dflash2_perf/<run-id>/`。
全部为单次验证，不作为三重复性能统计结论。

暂停确认：C4 命令退出码为 0，`nvidia-smi` 为 **0 MiB, 0%**；未启动 C8/TP2。
后续优化方向是减少 target verification 和 accepted-prefix commit/replay 的成本，
同时保留当前逐 token 一致性与显存约束。本次不继续修改该路径。

## 2026-09-17 全接受缓存复用优化

用户随后授权继续性能优化。`commit_verified_cache()` 原先无论接受长度多少，
都会恢复事务起始状态并重新执行接受的前缀。现在，当接受长度等于完整验证
query 长度（包含 pending anchor）时，直接保留验证结束后的缓存；部分接受和
零接受仍走原来的恢复逻辑。最终提交仍检查缓存 logical length，异常仍可回滚。
配置 `qwen35_reuse_verified_cache=False` 可以关闭新路径，默认开启。

本地全量测试为 **173 passed, 2 skipped**；新增覆盖全部接受、部分接受、零接受、
混合 batch、优化关闭和提交异常后的状态恢复。`git diff --check` 通过。

C2 成对测试目录：
`results/qwen38_dflash2_perf/qwen38_dflash2_k7_c2_reuse_verified_20260917/`。

| C2 指标 | RoPE 修复后、优化前 | 全接受复用后 |
|---|---:|---:|
| target-only TPS | 5.2752 | 5.2459 |
| DFlash2 TPS | 3.0683 | 3.3764 |
| DFlash2 acceptance | 47.1605% | 47.1605% |
| commit 秒 | 42.7176 | 25.6196 |
| verify 秒 | 47.3840 | 47.9902 |
| DFlash2 peak allocated GiB | 63.0080 | 61.9437 |
| 全量逐 token 一致 | 8/8 | 8/8 |

吞吐单次观测提升约 10.0%，commit 下降约 40.0%，显存峰值下降约 1.06 GiB。
接受数保持 382/810，原始请求 JSONL 对比确认优化前后 8 条输出均相同。
优化后 memory trace 仍为 787 点、snapshot bytes 为 0、generic KV pool 为 0，
original state 峰值 330891264 bytes、recorded steps 峰值 16。
该优化尚未使 DFlash2 超过 target-only。

C4 首次运行 `qwen38_dflash2_k7_c4_reuse_verified_20260917` 因连接断开结束，
仅保存 12/16 条 target-only 请求，不作为有效结果。保留原目录，使用
`qwen38_dflash2_k7_c4_reuse_verified_20260917_retry` 独立进程重新验证。

### C4 完整结果与迁移后复核

2026-09-18 在新服务器端口 `57195` 找到完整 retry 结果。此次读取的是迁移保存的
已完成测试，不是在新服务器上重跑所得的数据。

| C4 指标 | RoPE 修复后、优化前 | 全接受复用后 |
|---|---:|---:|
| target-only TPS | 5.7354 | 5.5703 |
| DFlash2 TPS | 3.6368 | 3.8708 |
| DFlash2 acceptance | 39.2473% | 39.2473% |
| commit 秒 | 63.3116 | 42.7575 |
| verify 秒 | 58.4539 | 59.2265 |
| DFlash2 peak allocated GiB | 72.4169 | 70.2856 |
| 全量逐 token 一致 | 16/16 | 16/16 |

C4 单次观测吞吐提升约 6.4%，commit 下降约 32.5%，显存峰值下降约 2.13 GiB。
接受数保持 730/1860。fairness 全量通过，且本地从原始 request JSONL 独立复核：
优化前后输出无差异，本轮 target-only 与 DFlash2 的 1024 个 token 也全部一致。
没有 admission failure 或 schema error。

显存 trace 共 885 点，snapshot bytes 和 generic KV pool 峰值均为 0，
original state 峰值 663355392 bytes、recorded steps 峰值 32。
最新 DFlash2 吞吐约为成对 target-only 的 0.695 倍，不能宣称已经解决性能倒挂。
上述性能改变量来自单次运行比较；target-only 计时也存在波动，尚未做三重复统计。

完整远端目录：
`results/qwen38_dflash2_perf/qwen38_dflash2_k7_c4_reuse_verified_20260917_retry/`。
本地同名 results 目录保存 summary、fairness、manifest；同名 logs 目录保存两组
原始 request JSONL。原始完整指标和 trace 保留在远端。

缓存复用优化验收完成。下一步应研究部分接受时如何避免整模型重放，以及如何
降低逐 token verifier 的成本；分块验证需要单独检验 BF16 计算形状变化引起的
输出差异，不能仅凭理论并行度宣称加速。此次未启动新的性能实验或 C8/TP2。
迁移后检查 GPU 为 **0 MiB, 0%**。

## 2026-09-18 部分接受状态恢复

验证期间保存每个线性注意力层的 FP32 decay、key、delta，以及较小的卷积状态。
提交部分接受的前缀时，从事务原始 recurrent state 按验证时相同的两次运算
`state = state * decay; state = state + key * delta` 恢复状态，不再执行目标模型
的投影、注意力和 MLP；full-attention KV 保留已验证内容并截断逻辑长度。
零接受恢复原始状态，全接受继续直接复用验证后状态。没有足够更新记录时回退到
整模型重放；事务原始状态仍保留供异常回滚。

更新记录由 `transaction_linear_update_bytes` 单独计入显存估算与 total bytes，
不再仅检查旧 snapshot 指标。配置 `qwen35_replay_state_updates=False` 可关闭优化；
benchmark 支持 `--replay-state-updates` / `--no-replay-state-updates`，默认开启。

最终本地测试 **214 passed, 2 skipped**。新增精确状态及后续 logits 对照覆盖
FP32/BF16、空/已有缓存、每种接受长度、ragged batch、micro-batch 1/4；还覆盖
开关关闭、记录不足回退、异常恢复，以及提交后记录不再计入活跃事务显存。

新服务器仍为 A800-SXM4-80GB，端口 57195。C2 新路径目录：
`results/qwen38_dflash2_perf/qwen38_dflash2_k7_c2_state_updates_20260918/`。
此 run 启动时 benchmark 尚未加入 CLI 开关，实际使用 runner 默认开启的新路径；
后续对照使用显式 `--no-replay-state-updates`。

C2 新路径 8/8 请求全量逐 token 一致，DFlash2 TPS 为 3.7986，成对 target-only
为 5.0455，接受数保持 382/810（47.1605%）。commit 为 1.5017 s，verify 为
52.2540 s，proposal 为 1.5296 s。peak allocated 为 54.5984 GiB，新增状态更新
记录峰值为 85082112 bytes（约 81.14 MiB），完整 snapshot 和 generic KV pool 均为 0。

同服务器旧路径对照目录：
`results/qwen38_dflash2_perf/qwen38_dflash2_k7_c2_state_updates_off_20260918/`。
该组只跑 DFlash2，沿用相同输入和参数，用于避免把服务器迁移或环境波动当作优化收益。

同机 C2 A/B 已完成，工作负载 JSON 和全部 8 条输出均完全一致：

| 指标 | 更新记录关闭（整模型重放） | 更新记录开启 |
|---|---:|---:|
| DFlash2 output TPS | 3.3849 | 3.7986 |
| commit 秒 | 26.2413 | 1.5017 |
| verify 秒 | 47.6831 | 52.2540 |
| proposal 秒 | 1.5291 | 1.5296 |
| peak allocated GiB | 61.9425 | 54.5984 |
| accepted/proposed | 382/810 | 382/810 |

单次同机观测吞吐提升约 12.2%，commit 下降约 94.3%，显存峰值下降约 7.34 GiB。
更新记录使 verify 增加约 4.57 s，但少做的整模型重放仍带来净收益。
两组均保留上一轮全接受缓存复用；这里只切换部分接受的状态恢复方式。
新路径仍慢于本轮 target-only 的 5.0455 TPS，整体加速目标尚未达到。

C4 新路径已在相同服务器完成 16 请求成对验证，目录为
`results/qwen38_dflash2_perf/qwen38_dflash2_k7_c4_state_updates_20260918/`。

### C4 状态恢复验收

2026-09-20 恢复工作后读取到完整结果；上次仅测试文件上传受自动审批的用量限制
阻止，独立运行的 benchmark 已正常完成。测试文件现已补齐同步。
服务器聚焦回归（模型、并发基准配置及缓存事务）为 **91 passed**。

| C4 指标 | target-only | DFlash2 状态恢复 |
|---|---:|---:|
| requests | 16 | 16 |
| output tokens | 1024 | 1024 |
| output TPS | 5.5169 | 4.3530 |
| TTFT p50 s | 39.5428 | 20.3822 |
| TPOT p50 s | 0.0973 | 0.5221 |
| peak allocated GiB | 50.9314 | 55.4921 |

全量 fairness 为 true、mismatch count 为 0。下载两组原始 request JSONL 后再次
独立复核：与本轮 target-only 及上一轮全接受复用版本相比，16 条请求的所有
1024 个 token 均一致。接受数仍为 730/1860（39.2473%）。没有 admission failure
或 schema error。

speculative 阶段耗时：verify 66.1148 s，commit 2.8835 s，proposal 1.8283 s。
verify 已占这部分耗时约 93.3%，成为下一步的主要性能瓶颈。
状态更新记录峰值 170164224 bytes（约 162.28 MiB），完整 snapshot 和
generic KV pool 峰值均为 0。新增记录确实计入显存统计，没有恢复逐 token
完整 recurrent-state snapshot。

历史 C4 全接受复用版本为 3.8708 TPS、70.2856 GiB，本轮为 4.3530 TPS、
55.4921 GiB；该历史对比跨服务器迁移，仅作参考。本轮成对 DFlash2 吞吐仍为
target-only 的约 0.789 倍，不能宣称整体加速目标已完成。同机 C2 A/B 才是此次
部分接受优化的直接性能证据；所有数据仍为单重复验证。

结果与原始输出已保存在服务器和本地同名 results/logs 目录；GPU 已确认
**0 MiB, 0%**。本轮不启动 C8/TP2。后续应优先研究逐 token verifier 如何减少
无效后缀计算或实现分块验证，同时继续检查 BF16 计算形状变化对精确输出的影响。

## 2026-09-28 验证器提前停止与性能倒挂归因

本轮首先把 RoPE 修复和缓存恢复整理成可回滚 checkpoint `9f63861`；
随后在 `e8bf0c8` 实现每个请求在首次 greedy mismatch 后停止后缀验证。
拒绝位置的目标 token 仍被计算，全接受时仍计算 bonus token；hidden taps、
GDN 更新记录和缓存事务仅保留实际执行的前缀。固定 batch/context 计算形状
继续沿用，避免 BF16 greedy 漂移。远端完整回归 **227 passed, 2 skipped**；
本地 `git diff --check` 通过。服务器 GPU 为 A800-SXM4-80GB。

各结果目录均保存 manifest（目标/草稿模型 snapshot、GPU、依赖、配置、
源码文件哈希和工作树 diff 哈希）、原始 request JSONL、summary 及日志；
远端原件保留在 `/root/autodl-tmp/Mini-vLLM`，本地同名 `results/`、`logs/`
目录保存副本。远端 Git HEAD 仍为 `70239ac` 且工作树含阶段改动，
因此性能实验必须以 manifest 的源码哈希和 diff 为准，不能只看远端 HEAD。

| K7 单次观测 | C2 target-only | C2 DFlash 关闭提前停止 | C2 DFlash 开启 | C4 target-only | C4 DFlash 关闭 | C4 DFlash 开启 |
|---|---:|---:|---:|---:|---:|---:|
| output TPS | 5.4767 | 4.1601 | 4.7136 | 5.8365 | 4.7430 | 5.0616 |
| measured elapsed s | 93.50 | 123.08 | 108.63 | 175.46 | 215.90 | 202.31 |
| TTFT p50 s | 17.99 | 9.90 | 9.88 | 37.42 | 19.05 | 19.41 |
| TPOT p50 s | 0.0837 | 0.3203 | 0.2593 | 0.0907 | 0.4789 | 0.4533 |
| target verifier model steps | — | 584 | 376 | — | 670 | 502 |
| verifier request-token steps | — | 932 | 504 | — | 2138 | 1008 |
| verifier seconds | — | 47.67 | 31.91 | — | 60.72 | 45.82 |
| proposal + commit seconds | — | 2.82 | 2.37 | — | 4.53 | 3.33 |
| peak allocated GiB | 50.6043 | 54.5984 | 54.5980 | 50.9314 | 55.4921 | 55.6099 |

C2 的 8 条请求、512 个输出 token，C4 的 16 条请求、1024 个输出 token，
均从原始 JSONL 独立核对为 target-only、DFlash 开/关逐 token 完全一致。
C2 接受数保持 382/810，C4 保持 730/1860；拒绝位置直方图也不变。
提前停止避免了 C2 的 208/584、C4 的 168/670 个目标模型验证步，
单次 DFlash 吞吐分别提高约 13.3%、6.7%，但仍低于成对 target-only。
这些只是单轮观察值，尚未证明稳定收益。

### 为什么 DFlash 端到端吞吐仍慢

代码中 `forward_with_cache_batch()` 的验证循环仍按 token 位置调用
`forward_cached_batch_step()`，即 K7 提案没有让 target 在一次模型前向中
并行完成 7 个位置。C4 target-only 的 252 次 decode step，每次有 4 个
有效请求，恰好完成 1008 个非首 token；提前停止后的 DFlash verifier
也必须验证这 1008 个请求 token，却需要 502 次目标模型 step，平均
每步约 2.01 个有效请求。C2 对应为 target-only 252 次、DFlash 376 次，
后者平均每步约 1.34 个有效请求。为保持数值形状，验证器每步仍补足
到固定 micro-batch 4，因此后期的空行也消耗模型计算。C4 验证阶段
45.82 秒，约占 verify/proposal/commit 三项之和的 93.2%。

按本轮计时，C4 DFlash 开启提前停止后仍比 target-only 多 26.85 秒。
若其他阶段和调度完全不变，要追平需要把 verifier 从 45.82 秒降至
约 18.97 秒（再降约 59%）；C2 对应约 31.91 降至 16.78 秒
（再降约 47%）。这是基于单次端到端差额的近似 break-even 估算，
不能当成独立的模块性能测量。后续优化应先降低逐位置 target 前向的
次数或提高有效行利用率，并重新验证固定形状下的逐 token 一致性。

### 为什么 DFlash 的 TTFT 中位数反而较低

C4 原始 JSONL 中首批 4 条同时提交请求的 TTFT：target-only 均约
37.42 秒，DFlash 均约 37.19 秒。两者的首批 prefill/首 token 时间接近。
压测为闭环并发：请求完成后才补交下一条。target-only 的后续请求
大多 4 条同时完成、再 4 条同时入场；DFlash 请求完成时间更分散，
后续常以 1–2 条入场，TTFT 多为约 10–20 秒。因此 DFlash 的较低
总体 TTFT p50 主要反映两组不同的入场批量，而非首批计算更快。
该归因来自逐请求提交/首 token 时间与调度方式；若要隔离模型本身的
首 token 成本，还需相同入场时间的开放环路或固定单批测试。

下一步才扩到 RoPE 修复后的 C8，并对 C4/C8 做至少三轮成对统计；
继续记录接受长度、TTFT、TPOT、显存与波动。TP2 尚未实现或验证。

## 2026-09-28 C8 数值边界回归与 LM head 形状修复

首次开启提前停止的 C8 成对运行
`qwen38_k7_c8_earlystop_on_20260928` 完成了两组各 32 条请求，
但 fairness 为 false：`short-000022` 在输出索引 43 处 target-only
为 `694`、DFlash 为 `321`。这次运行不纳入有效性能结论。
关闭提前停止并开启决策诊断的 C8 对照
`qwen38_k7_c8_earlystop_off_debug_20260928` 与原 target-only 的
32/32 条、2048/2048 个 token 全部一致；开启诊断的运行
`qwen38_k7_c8_earlystop_on_debug_20260928` 重现相同分歧。

两侧在分歧位置之前的输出和接受过程相同；位置 43 是一次全接受后的
bonus token。关闭提前停止时，该位置 `321`、`694` 的 BF16 logit
均为 25.25，greedy 按较大 token id 选 `694`，诊断记录的活跃请求数为 7。
开启时活跃数仅 3，`321` 的 logit 升至 25.375，`694` 仍为 25.25。
审计发现目标模型主体仍固定填充到 micro-batch 4，但提前停止后
LM head 的矩阵乘法按剩余活跃行数执行，改变了 BF16 reduction 形状。

修复方式是在提前停止路径中，LM head 仍按当前位置原计划的请求行数
计算，再只保留实际活跃请求的 logits；目标模型主体、target-only 路径、
hidden taps 和缓存提交均未改变。异长 batch 的 tiny-model 测试新增
LM head 行数一致断言，聚焦测试 **11 passed**，远端全量测试
**227 passed, 2 skipped**。修复前远端源码与测试原件另存于开启诊断的
结果目录，所有失败日志和 JSONL 继续保留。

修复后 C8 成对运行目录为
`qwen38_k7_c8_lmhead_shape_fix_20260928`，原始 JSONL 独立复核
**32/32 请求、2048/2048 token 完全一致**，benchmark fairness 亦为 true。
单次观测如下：

| C8 K7 指标 | target-only | DFlash2 提前停止 + LM head 修复 |
|---|---:|---:|
| output TPS | 5.8257 | 5.0902 |
| elapsed s | 351.57 | 402.35 |
| TTFT p50 s | 76.12 | 19.87 |
| TPOT p50 s | 0.1776 | 0.8940 |
| peak allocated GiB | 51.5413 | 57.0286 |

DFlash 接受 1486/3455 个候选；验证实际执行 934 个目标模型步，
避免 230 个后缀步；验证 88.2061 秒，proposal 1.8193 秒，
commit 2.7421 秒。DFlash 吞吐为 target-only 的约 0.874 倍，
仍无端到端加速。这一轮和前述 C2/C4 的源码版本不同，
且只有单次运行；跨 batch size 的稳定结论须在同一修复版本上配对重复。

## 2026-09-28 同版 C2/C4/C8 与 C4/C8 三轮配对统计

在 LM head 修复后，同版 C2/C4 成对运行目录为
`qwen38_k7_c2_c4_lmhead_shape_fix_20260928`；C8 首轮为上节目录。
C4/C8 另两轮目录为
`qwen38_k7_c4_c8_lmhead_shape_fix_repeats_20260928`（repeat 0/1）。
所有有效运行的 `source_tree_sha256` 均为
`4c8b1a80c2f2b8f6b898718636159bd0be62f1ef0ca845e1b4ca105367cfc728`。
模型 snapshot、K7、固定 micro-batch 4、short 工作负载、温度 0、
ignore EOS 和状态恢复配置一致。所有原始 JSONL、summary、manifest、
memory trace 与公平性文件保留在远端，本地同名 logs/results 也已保存。

| 并发数 | 成对轮次 | target-only TPS 均值（范围） | DFlash2 TPS 均值（范围） | 配对 DFlash/target 均值（范围） | 每轮严格一致 |
|---|---:|---:|---:|---:|---|
| C2 | 1 | 5.3838 | 4.6759 | 0.8685 | 8/8，512 token |
| C4 | 3 | 5.8033（5.7776–5.8245） | 5.0512（5.0137–5.0870） | 0.8704（0.8608–0.8759） | 16/16，1024 token |
| C8 | 3 | 5.9393（5.8257–6.0302） | 5.1641（5.0902–5.2052） | 0.8695（0.8618–0.8738） | 32/32，2048 token |

独立读取每组原始 request JSONL，逐请求、逐 token 比较均为零差异；
C4 共 48 次请求执行、3072 个输出 token，C8 共 96 次、6144 个输出
token 全部匹配。三轮 C4 的接受数均为 730/1860，接受长度分布
`0..7 = [71,48,50,29,10,11,13,46]`；三轮 C8 均为 1486/3455，
分布 `[124,99,73,51,39,28,20,96]`。C2 为 382/810。
每轮无 admission failure、schema error 或 GPU OOM。

| 三轮均值 | C4 target-only | C4 DFlash2 | C8 target-only | C8 DFlash2 |
|---|---:|---:|---:|---:|
| TTFT p50 s | 37.74 | 19.01 | 74.70 | 19.32 |
| TPOT p50 s | 0.0910 | 0.4532 | 0.1744 | 0.8829 |
| peak allocated GiB | 50.932 | 55.611 | 51.542 | 57.029 |

TTFT 是闭环负载各请求的中位数；DFlash 请求完成时间错开，后续请求
通常以较小批量入场。首批同时提交时两种模式的 TTFT 接近，
因此此处较低的 DFlash TTFT 不能解释为 target 首 token 前向更快。
TPOT 则在 C4/C8 均明显较高，和较低的端到端吞吐一致。

C4 三轮 DFlash verifier 平均 **46.26 秒**（范围 46.04–46.56），
proposal 1.75 秒、commit 1.60 秒；C8 分别为 **87.01 秒**
（86.31–88.21）、1.82 秒和 2.77 秒。验证占三项之和的均值
分别约 93.3% 与 95.0%。目标 verifier 的实际模型步数三轮固定为
C4 **502**、C8 **934**；被提前停止避免的步数分别为 **168**、
**230**。必要的请求 token 验证步数为 C4 **1008**、C8 **2016**，
没有低于 target-only 的非首 token 数。C8 target-only 的 252 个
scheduler decode step 以两个 micro-batch 4 执行，约 504 次目标模型
前向；DFlash verifier 仍需 934 次。并发从 C4 扩至 C8 后，
配对吞吐比仍约 0.87，验证位置串行和有效行不足仍是主要瓶颈。

以每轮端到端时间差近似估算，如果其他开销与调度不变，C4 还需
从 verifier 平均 46.26 秒减少约 26.27 秒（约 56.8%），
C8 还需从 87.01 秒减少约 51.72 秒（约 59.5%）才能追平。
这是单卡、当前负载下的 break-even 估算，不是独立测得的
可实现加速。结论：提前停止和 LM head 修复稳定保证了严格一致，
显存保持在 57.03 GiB 以内，但 K7 在 C4/C8 **稳定没有端到端
吞吐收益**。下一步性能工作应减少目标模型逐位置前向次数或提高
每次前向的有效请求数，并在相同形状下重新做严格逐 token 验证。
TP2 尚未实现或验证，不能据此推断 C16/C32 表现。

## 2026-09-28 官方整块验证机制与并行原型诊断

核对官方 SGLang DFlash2 worker：候选 token 块在因果掩码下交给 target
一次 forward 验证。Mini-vLLM 此前的 `forward_with_cache_batch` 则逐候选
位置循环调用 target。该串行验证是本项目漏掉的关键实现，不是 A800
显存不足：先前有效 C8 峰值约 57.03/79.25 GiB；框架需要补上
Qwen3.8 hybrid KV/GDN 的多 token 缓存接口。

新增默认关闭的 `qwen35_parallel_verification` 实验路径：full attention
使用显式因果掩码，GDN 在整块输入中保持顺序递推并记录每个前缀的
状态更新，事务按接受前缀提交。tiny-model 覆盖接受长度 0..7、异长
请求、hidden taps 和提交后缓存；远端正式 tests 目录全量测试在
首版原型为 **229 passed, 2 skipped**，修复 LM head 形状后的聚焦测试
为 **10 passed**。这些测试无法替代真实 BF16 模型逐 token 对照。

首轮 C2 原型 `qwen38_k7_c2_parallel_target_probe_20260928`：target-only
5.3697 tok/s，并行 DFlash2 5.8264 tok/s；估计 target forward 次数
257 对 77，DFlash2 verify 12.4157 秒，峰值 54.5898 GiB。但原始
JSONL 为 **6/8 请求一致**，因此吞吐数据仅是失效原型的诊断观测。
修复并行 LM head 以与串行逐位置保持相同行数后，第二轮
`qwen38_k7_c2_parallel_headshape_probe_20260928` 为 5.3416 对
5.6677 tok/s，估计 forward 257 对 79，verify 13.8237 秒，峰值
54.5995 GiB，接受 382/810 与串行 C2 旧结果相同；独立读取
原始 request JSONL 后仍有 **1/8 请求不一致**：`short-000003`
输出索引 4 的 target-only 为 271，并行结果为 1622。这轮同样
**不能计为有效加速**。两个运行的 manifest、源码 patch、原始 JSONL、
fairness、memory trace 和结果均保留在远端同名 logs/results 目录。

真实模型逐层对照显示，多 token 块在第 0 层已经有约 1e-3 的
BF16 差异，后续层放大；`short-000003` 的输入 token 26 在末层
隐藏状态最大差异达数十，导致 greedy 选择改变。单 token 对照的
末层 hidden taps 完全相同，只有 LM head 的矩阵行数产生 0.0625
的 logit 差异，已按原串行行数修复。将线性层、卷积和注意力乘法
单独提高到 FP32 仍未稳定保持原 BF16 greedy；把整个验证段激活
改为 FP32 能缩小串并行 logit 差异，但串行结果本身也发生改变
（上述 token 位置从 271 变成 2574），不满足保持现有 target-only
输出的要求。原型仍默认关闭，未形成可发布 checkpoint。

下一步必须先解决跨计算形状的严格数值一致性，再做 C4/C8 的并行
性能统计；目前没有证据声称官方整块验证路径已在 Mini-vLLM 中
合格实现，也不能以单次、且不一致的 C2 吞吐作为加速结论。

## 2026-09-28 严格一致的整块验证与分块优化

继续追踪 BF16 数值差异后，确认归一化归约、线性层 GEMM、GDN
卷积与注意力乘法都可能因 query 维度变动改变舍入。整块验证现保留
一次 target 模型层遍历和因果缓存接口，但在敏感算子内部按单 token
形状计算；真实 Qwen3.8 固定前缀的全部 logits 与原串行路径位级
相同。C2 端到端 `qwen38_k7_c2_parallel_exactops_20260928` 的原始
JSONL 独立对比 **8/8 请求、512/512 token 一致**，接受 382/810，
峰值 54.5982 GiB。此路径解决了正确性，但许多权重 GEMM 仍逐
token 执行，单次吞吐低于 target-only。

在 A800 上逐形状比较 BF16 GEMM 后，增加默认关闭的 B=2 实验开关：
部分 MLP、full-attention 和 GDN 投影可整块计算且在探针中位级
一致；MLP down 的两 token 小块也位级一致。另增加 2/4 token
分块验证：首次拒绝后跳过后续块，已执行的多余缓存用记录的紧凑
GDN 更新恢复到接受前缀。tiny-model 覆盖接受长度 0..7、异长
请求和采样异常回滚，正式测试 **230 passed、2 skipped**。

下表均为单轮 C2 K7 成对观测。每轮原始 request JSONL 独立比较
都是 **8/8 请求、512/512 token 完全一致**，接受均为 382/810；
所有结果目录前缀为远端 `results/qwen38_dflash2_perf/`，原始 JSONL
在同名 `logs/qwen38_dflash2_perf/` 下，manifest 保存各轮源码 hash、
模型 snapshot 和配置。不同源码版本及单轮波动不能合并声称稳定收益。

| 实验目录（省略日期后缀） | target-only TPS | DFlash2 TPS | DFlash/target | verify 秒 | DFlash 峰值 GiB |
|---|---:|---:|---:|---:|---:|
| `qwen38_k7_c2_parallel_exactops` | 5.1066 | 4.1586 | 0.8144 | 41.85 | 54.598 |
| `qwen38_k7_c2_parallel_batchedmlp` | 5.1935 | 4.3404 | 0.8357 | 38.71 | 54.598 |
| `qwen38_k7_c2_parallel_safe_linears` | 5.2498 | 4.5350 | 0.8638 | 34.98 | 54.598 |
| `qwen38_k7_c2_parallel_chunk4_exact` | 5.1913 | 4.4670 | 0.8605 | 35.12 | 54.589 |
| `qwen38_k7_c2_parallel_chunk4_safe` | 5.1885 | 4.7563 | 0.9167 | 29.97 | 54.589 |
| `qwen38_k7_c2_parallel_chunk2_safe` | 5.2186 | 4.6380 | 0.8887 | 32.41 | 54.597 |

4-token 分块执行 677 个请求验证位置，避免 255 个后缀位置；
2-token 分块执行 554、避免 378，但额外块调用使单次耗时更长。
4-token 分块加 B=2 精确投影是目前有效 C2 中吞吐最高的版本，
仍比同轮 target-only 慢约 8.3%。Triton 批量 GEMM 的常规 tile
设置未复现旧版单 token cuBLAS 结果，因此不能直接用来替换
仍昂贵的 GDN qkv 和 MLP down 算子。

当前分块与精确批量投影均为默认关闭的实验配置。以下 C4/C8
结果更新了当时尚未完成的回归结论。

## 2026-09-28 C4/C8 并行验证回归

采用相同的 4-token 分块、micro-batch 4 和经过逐位探针筛选的
B=4 安全合批投影。C4、C8 两次运行的 manifest 记录了相同的
模型 snapshot、源码树 SHA256
`f71091e487c01bf144c132fedbc1b9f2c7cf86786e04573d73dbbca78037a418`
及源码 patch SHA256
`311311bd7bdbfcf1282b4d70d3cda406881ac60c41b74cc271440ed278307797`。
远端实验目录分别为
`qwen38_k7_c4_parallel_b4_safe_20260928` 和
`qwen38_k7_c8_parallel_b4_safe_20260928`，位于
`results/qwen38_dflash2_perf/`；原始 JSONL 位于对应的
`logs/qwen38_dflash2_perf/`，保留 manifest、fairness 和显存记录。
独立按 request ID 比较原始 target-only/DFlash2 JSONL：C4 为
**16/16 请求、1024/1024 token 一致**，C8 为
**32/32 请求、2048/2048 token 一致**。远端全量测试
**230 passed、2 skipped**，`git diff --check` 通过。

| 并发 | target-only tok/s | DFlash2 tok/s | 比值 | 接受数/提议数 | verify 秒 | DFlash2 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|
| C4 | 5.5974 | 4.9012 | 0.8756 | 730/1860 | 42.27 | 55.604 |
| C8 | 5.6483 | 5.3062 | 0.9394 | 1486/3455 | 67.76 | 57.054 |

C4 的接受率为 39.25%，C8 为 43.01%。C8 的首次拒绝位置
直方图为位置 0..6 分别 117、90、69、46、39、24、18，
全接受 127；执行了 2788 个请求验证位置，避免了 1197 个
原计划位置。分块 verifier 的模型步数为 252，不能把这个统计
等同于 252 次完整 target 前向，也不能只据调用次数推断速度。

C4 的配对总耗时为 182.94/208.93 秒，C8 为 362.58/385.96 秒
（前者 target-only，后者 DFlash2）。假设其他耗时不变，C4 的
verify 还需由 42.27 降至约 16.28 秒，C8 由 67.76 降至
约 44.38 秒才能追平；这分别要求约 61.5% 和 34.5% 的下降。
尽管整块接口解决了因果验证与 hybrid cache 提交的正确性，
旧 target-only BF16 greedy 对矩阵行数敏感，当前仍有昂贵的
逐 token 投影。C4/C8 都只有单轮，尚无稳定加速证据，也没有
完成 TP2。下一步应优先降低这些投影的实际耗时，并以相同源码
做至少三轮配对测试；不能用不一致的全合批原型充当收益。

### 形状稳定矩阵核的淘汰实验

另外尝试用同一 Triton BF16 矩阵核处理 target-only decode 与
并行 verifier；探针中同一个输入行在 1/2/4/8/16 行批量下对
GDN qkv、MLP down/gate 投影得到逐位相同的结果。实验代码默认
关闭，仅用于 `qwen38_k7_c2_shape_stable_triton_probe_20260928`。
该次 manifest 的源码树 SHA256 为
`3eb209d6e3aa5b02fa84bcdefdfda304588e7d9e02648d05d9b36eb29eba387e`，
原始 patch 保存在结果目录。C2 原始 JSONL 中两模式 8/8 请求、
512/512 token 一致，DFlash2 接受 376/864，峰值 54.589 GiB；
但 target-only 为 5.0183 tok/s、DFlash2 仅 4.5962 tok/s，
verify 30.44 秒。与旧 target-only C2 基线相比，仅 3/8 请求
输出保持相同，证明这会改变原有 greedy 数值路径，却没有形成
吞吐收益。因此实验核不纳入阶段三代码检查点；保留原始日志、
源码 patch 和结果供后续分析。

### C8 micro-batch 8 对照

`qwen38_k7_c8_parallel_b8_microbatch_probe_20260928` 使用与上方
C8 micro-batch 4 相同的源码树 SHA256，仅把固定 target
micro-batch 由 4 改为 8。单轮配对运行的原始 JSONL 独立比对
为 **32/32 请求、2048/2048 token 一致**。target-only 为
6.2827 tok/s、325.98 秒，DFlash2 为 5.5366 tok/s、
369.90 秒，吞吐比 **0.8812**；DFlash2 接受 1466/3573，
verify 60.40 秒，峰值 57.665 GiB。micro-batch 8 相对
micro-batch 4 提高了两种模式的绝对吞吐，但 target-only 提高
更多，投机解码的相对收益下降。两次 batch 形状不同，BF16
输出及接受数也不同；每次都必须与同轮的 target-only 做公平
逐 token 对照。单次结果不证明稳定性，也不能用扩批替代
对昂贵逐 token 投影的优化。

## 数值验收口径澄清

本阶段为排查 BF16 形状漂移，把旧版 target-only 的逐 token
输出额外当作兼容性基线，因此牺牲了若干算子的整块合批能力。
项目真正的严格正确性要求是：**同一模型、同一请求和同一明确
定义的 target 数值路径下，target-only 与 DFlash2 的 greedy
输出逐 token 一致**。不要求新版共同数值路径逐 token 复现
旧版 Mini-vLLM 的 BF16 kernel 输出。旧版输出继续记录为
行为变更与模型质量参考，不作为阻止并行 target 实现的门槛。

后续应统一两种模式的 target kernel/精度策略，优先实现真正
整块批量计算；用原始 JSONL 检查新路径内的严格一致性，
再用固定模型、配置和请求做多轮配对吞吐测量。任何新路径
都需要检查缓存状态、EOS、异长 batch、异常回滚、显存峰值，
并明确报告与旧版输出的差别。

### 统一 target 数值路径的首轮结果

实验开关 `--vectorized-target-numerics` 同时作用于 target-only
decode 与 DFlash2 verifier，预填充仍由相同的原路径处理。
BF16 投影使用按输出行独立的 Triton 核；RMSNorm 使用按行独立的
融合归约；full-attention 中已证实对 query 形状敏感的乘法与
softmax 继续逐位置处理。LM head 在两个模式中固定到相同的
micro-batch 行数。4-token 因果分块在首次拒绝后跳过后续块，
GDN 仍按位置更新并按接受前缀提交。由此允许旧版 target-only
输出变化，同时要求新 target-only 与 DFlash2 逐 token 一致。

首轮 C2/C4/C8 使用相同源码树 SHA256
`2f3e04bf7884c15d1cbf2c1613fc184b6b9a64b49a68b43e79f74a01df418d88`，
同一 A800、K=7、micro-batch 4。结果目录分别为
`qwen38_k7_c2_vectorized_fusednorm_fixedhead_20260928`、
`qwen38_k7_c4_vectorized_fusednorm_fixedhead_20260928` 和
`qwen38_k7_c8_vectorized_fusednorm_fixedhead_20260928`。
独立按 request ID 比较两模式的原始 JSONL：C2 **8/8**、
C4 **16/16**、C8 **32/32** 请求一致，分别为 512、1024、
2048 个输出 token。随后补充 3 个定向测试并重跑远端全量测试，
结果为 **233 passed、2 skipped**；生产源码与上述实验相同。

| 并发 | target-only tok/s | DFlash2 tok/s | DFlash/target | 接受数/提议数 | verify 秒 | DFlash2 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|
| C2 | 5.0994 | 5.3961 | 1.0582 | 382/812 | 16.11 | 54.590 |
| C4 | 5.5601 | 5.5678 | 1.0014 | 717/1943 | 22.35 | 55.600 |
| C8 | 5.8143 | 5.8011 | 0.9977 | 1482/3500 | 38.92 | 56.972 |

这些都是单轮观测。C2 显示约 5.8% 的有效配对收益，
C4/C8 接近持平，尚不能声称稳定加速。完整 8-token 块与
3-token 分块在 C2 都严格一致，但 verify 分别为 24.83 秒与
24.95 秒，慢于 4-token 分块的 23.28 秒（融合 RMSNorm 前
的同源码对照）。另一个融合 RMSNorm 但 LM head 未固定行数
的原型曾达到 5.30 对 5.18 tok/s，原始 JSONL 却有 2/8
请求不一致；这轮仅保留为失败诊断，不能计入收益。

### C2/C4/C8 三轮配对复测

在上述首轮之外，各并发量又用相同源码和模型各做两轮配对：
`qwen38_k7_c2_vectorized_fusednorm_fixedhead_repeat2_20260928`、
`qwen38_k7_c4_vectorized_fusednorm_fixedhead_repeat2_20260928`、
`qwen38_k7_c8_vectorized_fusednorm_fixedhead_repeat2_20260928`。
逐轮独立比较原始 request JSONL，C2 为每轮 8/8 请求与
512/512 token 一致，C4 为 16/16 与 1024/1024，C8 为
32/32 与 2048/2048。每轮 manifest 的源码树 SHA256 均为
`2f3e04bf7884c15d1cbf2c1613fc184b6b9a64b49a68b43e79f74a01df418d88`。

| 并发 | 三轮 target-only tok/s | 三轮 DFlash2 tok/s | 三轮配对吞吐比 | 比值均值 | verify 秒范围 | DFlash2 峰值 GiB 范围 |
|---|---|---|---|---:|---:|---:|
| C2 | 5.099, 5.259, 5.322 | 5.396, 5.492, 5.505 | 1.0582, 1.0442, 1.0344 | 1.0456 | 15.89–16.20 | 54.589–54.590 |
| C4 | 5.560, 5.415, 5.464 | 5.568, 5.344, 5.388 | 1.0014, 0.9868, 0.9860 | 0.9914 | 22.35–23.25 | 55.600 |
| C8 | 5.814, 5.695, 5.661 | 5.801, 5.690, 5.704 | 0.9977, 0.9991, 1.0076 | 1.0015 | 38.92–40.37 | 56.972–57.061 |

C2 三轮均为 382/812 接受，接受长度 0..7 的直方图为
27、21、14、10、7、11、4、28；C4 三轮均为 717/1943，
直方图为 76、59、43、37、13、13、6、44。C8 首轮
1482/3500，后两轮 1481/3507；直方图在 0..7 各位置
范围为 129–130、105、68、47–48、39–40、27、21、97。
少量 C8 接受数差异没有改变同轮严格输出一致性。

闭环负载下，各请求 TTFT 中位数三轮范围：C2 target-only
18.20–19.31 秒、DFlash2 9.93–14.12 秒；C4 分别
39.24–40.34 与 19.82–20.59 秒；C8 分别 76.00–77.50
与 19.78–20.51 秒。各请求 TPOT 中位数范围：C2
0.0903–0.0921 与 0.0764–0.1401 秒；C4 0.0978–0.1015
与 0.4183–0.4326 秒；C8 0.1890–0.1905 与
0.8218–0.8621 秒（顺序仍为 target-only、DFlash2）。
请求完成时间错开，TTFT/TPOT 不能单独解释整体吞吐比。

结论仅限当前单张 A800、短输出负载和 K=7：**C2 有三轮
均为正的约 3.4%–5.8% 端到端收益**；C4 与 C8 在零附近
波动，不能声称稳定收益。相同 C2 请求的新版 target-only
仅 5/8 请求与旧 BF16 基线逐 token 相同；这是有意更换
共同 target 数值路径的行为变更，不是新路径中两模式的
不一致。正式推广前还需扩展模型质量与输入覆盖面，并优先
优化 C4/C8 剩余的 full-attention 逐位置算子与验证开销。

## 2026-10-07 补充：自然长任务、每请求 4096 输出

用户要求将性能测试输出加至 4K，并选用自然需要长回答的代码／推理任务。
在融合 GDN 的共同 target 数值路径上，增加轻量实时统计，避免每个 decode
步骤序列化全部历史；完整统计在计时结束后由引擎工作线程导出。
实现 checkpoint 为 `b2081ac`，远端全量测试 **254 passed、2 skipped**。

同一 A800、C4、K=7、4-token 验证块、每轮 4 请求 × 4096 输出，三轮
带细分 profiling 的配对全部通过原始 JSONL 独立审计：**12/12 请求、
49,152/49,152 token 严格一致**，跨轮也一致，均无提前 EOS。
target-only 三轮吞吐为 37.534、37.567、37.788 tok/s；DFlash2 为
33.868、33.910、33.740 tok/s，比值平均 0.8993。每轮接受率30.92%，
平均每请求投机周期输出3.16 tokens，DFlash2峰值分配显存不超过59.501 GiB。

这组 4K 数据仍没有端到端收益；另做关闭细分 profiling 的单变量控制后，
target-only为37.722、DFlash2为33.730 tok/s，比值0.8942，仍未获得收益。
控制轮输出与接受分布不变；四组配对累计16请求、65,536输出token严格一致。
该控制只有一轮，不与三轮诊断结果混合统计。详细配置、延迟、波动、接受分布、耗时
break-even、证据路径和控制结果统一记录在
[4K 阶段报告](qwen38_dflash2_decode4096_20261007.md)。
本节属于单批 C4 长输出结果，不外推到持续在线负载或 TP=2；TP=2／C16／C32
仍未实现或验证。本节之前的短输出观测保留为历史，不能代替本轮长输出结论。

## 2026-10-07 补充：低接受率差异定位

官方 vLLM 0.28.0 的同任务 4 × 4096 输出诊断已完成：K=7 接受率30.3129%，
Mini为30.9200%。两引擎输出序列不同，此为接受水平对照，不是同周期因果消融。
已复现 Mini selector 使用 BF16 总分而官方使用 FP32 的差异；两次固定周期
影子对照中修正精度使接受数从11202变成11160，并未提高接受率。
长前缀缓存与整窗投影的差异进一步定位到 BF16 GEMM 形状；按原706段投影
重放，五层KV逐位完全一致。两个固定前缀的Mini draft整窗重放与官方均7/7一致。
详细证据、限制和下一步见[接受率诊断报告](qwen38_dflash2_acceptance_diagnosis_20261007.md)。
生产模型路径本次未修改；优先优化 verifier/commit 成本，不能承诺精度对齐会恢复高接受率。
