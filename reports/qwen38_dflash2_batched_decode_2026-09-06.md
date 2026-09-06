# Qwen3.8 Batched Decode Validation - 2026-09-06

## Outcome

- Commit `f6798a8` tensor-batches ordinary Qwen3.8 cached decode, matching the
  request-dimension batch semantics already used by DFlash2 target verification.
- Local regression: `115 passed, 2 skipped`.
- AutoDL focused regression: `15 passed`.
- The previously failing long-prompt B=2/K=1 run is target-lossless and reports
  `dflash2_online_b2=PASS`.

## Root Cause

The apparent divergence was not cache contamination. A real Qwen3.8
teacher-forced comparison used identical prefixes for B=1 and B=2 and found one
argmax mismatch across 64 request positions. At that position, B=1 had an exact
top-two logit tie; B=2 BF16 GEMM rounding produced a 0.125 margin. The old smoke
compared a request-serial target-only path against tensor-batched verification,
so it compared different numerical batch semantics.

Diagnostic log:

`logs/qwen38_cached_b1_b2_diagnostic_20260906.log`

## GPU Acceptance

The fixed long-prompt run used two concurrent requests, K=1, 32 output tokens,
and target cache fraction 0.05:

- exact target-only/DFlash2 token IDs for both requests;
- 42 proposed / 19 accepted tokens, acceptance 45.2%;
- verifier max batch size 2;
- 40 tensor-batched token steps out of 45;
- target-only 6.32s, DFlash2 7.28s;
- no CUDA or shutdown error.

Log:

`logs/dflash2_corpus_pair1_k1_tensor_target_20260906_rerun.log`

## K Decision

Across three heterogeneous prompt pairs, every valid K=1/K=2 run remained exact
and reached a real shared B=2 speculative batch.

| K | Target total | DFlash2 total | Proposed | Accepted | Acceptance |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.01s | 20.94s | 128 | 54 | 42.2% |
| 2 | 18.15s | 25.43s | 230 | 66 | 28.7% |

K=1 won every pair. The B=2 smoke now defaults to K=1. Adaptive K is deferred:
the current evidence shows no workload where K=2 recovers its added verification
cost, and DFlash2 remains slower than the newly batched target-only baseline on
this broader corpus.

The translation/arithmetic pair required target cache fraction 0.08 to allocate
two DFlash cache blocks at `max_model_length=47`; fraction 0.05 was lossless but
request-serial and was rejected by the strict B=2 assertion.
