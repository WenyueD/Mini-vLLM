# Qwen3.8 DFlash2 High-Concurrency Benchmark - 2026-09-07

## Status

This phase produced the reproducible benchmark harness and CPU-verifiable tests,
but did not produce A800 performance numbers in this local run.

- Locally verified: `141 passed, 2 skipped` with
  `D:\LLM项目\MinivLLM\.venv\Scripts\python.exe -m pytest`.
- Locally verified: dry-run plan generation with
  `D:\LLM项目\MinivLLM\.venv\Scripts\python.exe scripts\benchmark_qwen38_dflash2_concurrency.py --dry-run --run-id local_dry_run`.
- Locally blocked: the local `.venv` has `torch 2.9.1+cpu`; no CUDA runtime is
  available here for real Qwen3.8-27B execution.
- Remote blocked: the prior AutoDL SSH endpoint
  `root@connect.bjb1.seetacloud.com:15723` refused the connection after an
  escalated SSH reachability check.

The benchmark remains Mini-vLLM target-only vs Mini-vLLM DFlash2. The official
vLLM 0.28 V2 Oracle semantic alignment limitation remains in force: the Oracle
runner is blocked by non-lossless runtime behavior, so this report must not be
read as full official DFlash2 alignment.

## Harness

Added `scripts/benchmark_qwen38_dflash2_concurrency.py`.

Key properties:

- Uses `OnlineEngine` for both variants and excludes model load/startup time
  from steady-state request latency.
- Runs target-only and DFlash2 sequentially with the same tokenizer, prompt
  token IDs, workload definitions, admission path, dtype, cache settings,
  temperature `0.0`, and seed.
- Supports closed-loop concurrency, optional open-loop Poisson arrivals,
  fixed-interval burst arrivals, mixed ragged workloads, warmup before measured
  iterations, and soak-mode cancellation edges.
- Writes raw per-request JSONL, monitor JSONL, per-run summary JSON, aggregate
  JSON, and CSV under `results/qwen38_dflash2_perf/<run_id>/`; raw logs go under
  `logs/qwen38_dflash2_perf/<run_id>/`.
- Records submit/admit/first-token/per-token/finish timestamps, throughput,
  TTFT, TPOT, ITL, E2E p50/p90/p95/p99, goodput SLO, queue/running samples,
  stage totals, DFlash2 proposal profile, acceptance by position, acceptance by
  decode step, accepted-length histogram, batch-size buckets, CUDA memory
  snapshots, and failure records.

The default goodput SLO is explicit and configurable:

- TTFT <= 5.0s
- TPOT <= 0.25s

Added `OnlineRequestSpec` and `OnlineEngine.submit_requests()` so mixed
workloads can atomically admit requests with per-request `SamplingParams`.
Existing `submit()` and `submit_batch()` behavior is preserved.

## Test Design

Local tests cover:

- percentile interpolation and latency summaries;
- throughput, TTFT, TPOT, ITL, E2E, and goodput aggregation;
- per-step/per-position acceptance aggregation and batch-size buckets;
- summary schema validation and mismatch diagnostics;
- `OnlineEngine.submit_requests()` with different per-request generation
  lengths while still reaching shared decode batching.

The harness records exact token-ID comparisons for sampled target-only/DFlash2
outputs. If a sampled mismatch appears, the current streaming API cannot attach
top-2 logit margins directly; the mismatch record is preserved and should be
followed by the existing cached-batch logit diagnostic to classify exact argmax
ties.

## Remote Commands

Pilot after the code is available on AutoDL:

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

Formal matrix after pilot logs pass fairness and memory checks:

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

Soak:

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

Do not silently lower concurrency after OOM. Keep failed summary JSON files and
use the recorded memory/failure fields as capacity evidence.

## Current Conclusion

No new A800 throughput or latency conclusion is available from this run. The
latest measured performance conclusion remains the prior log-verified B=2/K=1
short run: target-only around 6.6040s and DFlash2 around 7.2836s, with DFlash2
about 10.3% slower. Treat that as prior evidence, not as a result from the new
high-concurrency matrix.

The next useful step is to run the pilot command on a reachable A800 host, check
lossless sampled outputs, drafted/accepted counters, queue/active-batch traces,
and memory release, then expand to the formal matrix without changing the
algorithm between runs.
