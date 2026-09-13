# Qwen3.8 + DFlash2 K=7 / TP=2 阶段报告

日期: 2026-09-13

起点: `c911bc4415cd3a63f673860b887e3ae8e8c881a7`

当前状态: 阶段一显存复杂度修复已在 A800 80GB 上完成 before/after 验证。阶段二先修复 cached target 计算 shape 不稳定导致的 BF16 舍入漂移，再用固定 micro-batch 4 解决 C8 batch-8 OOM；修复后 C2 为 8/8、C4 为 16/16、C8 为 32/32 全量逐 token 一致。阶段二已完成，阶段三、阶段四未开始。

## 阶段边界

本轮严格按阶段推进，没有跳过阶段。

1. 阶段一: 修复 Qwen3.8 + DFlash2 的显存复杂度风险，并补内存 trace。已完成 after C1/C2/C4/C8；before baseline 有 C1/C2/C4，C8 被用户要求暂停后中断，不纳入有效结果。
2. 阶段二: 单卡 K=7 的 C1/C2/C4/C8 正确性与稳定性。已完成；C1/C2/C4/C8 correctness 均通过。
3. 阶段三: 接受率和性能方向分析。未开始。
4. 阶段四: TP=2 下 C16/C32 验证。未开始。

阶段一 memory 证据已满足进入阶段二的最低条件；阶段二已通过，可进入阶段三接受率和性能方向分析。

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

本轮尚未创建 git commit。原因是用户要求的最终提交应发生在阶段四完成后；当前阶段二已完成，阶段三和阶段四仍未完成。
