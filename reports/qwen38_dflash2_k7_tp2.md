# Qwen3.8 + DFlash2 K=7 / TP=2 阶段报告

日期: 2026-09-20

起点: `c911bc4415cd3a63f673860b887e3ae8e8c881a7`

当前状态: RoPE 初始化修复、全接受缓存复用及部分接受状态恢复均已通过 C2/C4 验证。最新 C2 为 8/8、C4 为 16/16 输出一致；DFlash2 吞吐分别为 3.7986 和 4.3530 token/s，仍低于成对 target-only。2026-09-20 已在端口 57195 读取并独立复核完整 C4 结果，GPU 已释放，未启动 C8/TP2。历史数据和最新优化结果在下文分开记录。

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
