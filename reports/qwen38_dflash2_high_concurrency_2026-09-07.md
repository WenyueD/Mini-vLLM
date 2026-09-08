# Qwen3.8 DFlash2 High-Concurrency Benchmark - 2026-09-07

## Status

The A800 host became reachable at the new endpoint on 2026-09-08. Pilot,
short-matrix, cleanup, repeatability, and target-decision diagnostic runs were
completed. The performance cells run successfully, but strict lossless status
is blocked by one reproducible C=4 BF16 near-tie mismatch.

- Local regression: `150 passed, 2 skipped`.
- Remote focused Qwen regression at commit `525f7cf`: `19 passed`.
- Remote runtime: NVIDIA A800-SXM4-80GB, Python 3.11.16, PyTorch 2.9.1+cu128,
  CUDA 12.8, BF16.
- Target-only reproducibility: two independent C=4 model reloads produced
  identical outputs for all 8 requests and all 512 generated tokens.
- Short matrix: C=1, C=2, and C=8 passed exact token equality; C=4 did not.
- Formal multi-workload matrix and soak remain gated because C=4 is not
  strictly target-lossless.

The benchmark remains Mini-vLLM target-only vs Mini-vLLM DFlash2. The official
vLLM 0.28 V2 Oracle semantic alignment limitation remains in force: the Oracle
runner is blocked by non-lossless runtime behavior, so this report must not be
read as full official DFlash2 alignment.

## Remote Fixes

The remote campaign exposed and fixed several independent implementation and
harness failures:

- `0531f1b`: run target inference under `torch.inference_mode()` and make
  constructor failures clean up distributed state.
- `62f7143`: preserve every runtime counter key when resetting after warmup.
- `46776ba`: release the speculative proposer and retained model runner during
  engine shutdown. This removed a 53.7 GiB cross-cell GPU memory leak.
- `de89110`: hold cached Qwen attention context width constant.
- `f3534e2`: hold cached Qwen batch size constant with dummy request states.
- `525f7cf`: capture bounded target-decision top-2 diagnostics for ordinary and
  speculative paths.

All remote cells after the cleanup fix could load target-only and DFlash2
sequentially in one process without OOM or NCCL shutdown failures.

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
outputs. With `--debug-metadata`, it also records the request, output index,
source path, query offset, active batch size, input token, actual argmax token,
top-two token IDs/logits, and top-two margin. Fairness remains a strict token-ID
comparison; diagnostics classify failures but never turn them into passes.

## Short Results

The C=1/C=2/C=8 rows are from the fixed-attention-width short matrix at
`de89110`. The C=4 row is the newer fixed-batch gate at `f3534e2`. Each row has
8 requests, 128 input tokens, 64 output tokens, K=1, and one measured repeat.

| C | Target output tok/s | DFlash2 output tok/s | DFlash/target | Acceptance | Target peak GiB | DFlash peak GiB | Exact |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 4.275 | 3.454 | 0.808x | 25.19% | 55.17 | 58.07 | yes |
| 2 | 5.381 | 4.436 | 0.824x | 24.87% | 55.18 | 58.07 | yes |
| 4 | 5.825 | 5.118 | 0.879x | 25.19% | 55.17 | 58.07 | no |
| 8 | 6.272 | 5.586 | 0.891x | 27.04% | 55.18 | 58.91 | yes |

DFlash2 is slower than target-only at every tested concurrency. Its relative
throughput improves as concurrency rises, but this pilot does not demonstrate a
speedup. C=4 DFlash2 has better median TTFT (34.07s versus 37.81s) but worse
median E2E latency (49.86s versus 43.94s) and worse median TPOT (0.251s versus
0.084s).

Artifacts:

- `results/qwen38_dflash2_perf/qwen38_dflash2_short_matrix_fixedpad_k1_20260908`
- `results/qwen38_dflash2_perf/qwen38_dflash2_fixedbatch_c4_gate_20260908`
- `logs/qwen38_dflash2_perf/qwen38_target_repro_c4_r2_20260908`

## Fairness Diagnosis

Fixed attention width reduced the short matrix to one C=4 mismatch. Fixed batch
size removed that mismatch, but another appeared for `short-000000` at output
index 61. The diagnostic rerun at `525f7cf` reproduced the same token pair after
an exact prefix through index 60:

| Path | Token 25 logit | Token 13 logit | Margin | Greedy result |
|---|---:|---:|---:|---:|
| target-only decode | 22.375 | 22.250 | 0.125 | 25 |
| DFlash2 verifier, query offset 0 | 22.250 | 22.250 | 0.000 | 13 |

The DFlash2 verifier produces an exact BF16 tie, and `torch.argmax` selects the
lower token ID, 13. This is consistent with accumulated low-precision state
sensitivity after accepted speculative tokens. It is not consistent with an
unstable target-only baseline, and no request failure, cache exception, OOM, or
shutdown error accompanied it.

Artifacts:

- `results/qwen38_dflash2_perf/qwen38_dflash2_target_decision_diag_c4_20260908`
- `logs/qwen38_dflash2_perf/qwen38_dflash2_target_decision_diag_c4_20260908`

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

The A800 execution and memory lifecycle are now healthy, and the concurrency
harness produces complete performance artifacts. K=1 DFlash2 remains slower
than target-only across C=1/2/4/8, with roughly 25-27% draft acceptance.

The implementation cannot yet claim strict target losslessness at C=4. The
formal matrix should remain blocked until the cached target state is numerically
invariant across ordinary and accepted-speculative scheduling, or a documented
higher-precision target decision policy is implemented for both paths. A logit
tolerance must not be used to relabel the current mismatch as an exact pass.
