# Mini-vLLM: Qwen Hybrid Inference & DFlash2

[English](README.md) | [简体中文](README_zh.md)

A research inference engine for **Qwen3.5/Qwen3.8 hybrid models and DFlash2 speculative decoding**, built on Mini-vLLM. The project focuses on real-model execution, greedy-output validation, hybrid cache transactions, and draft/verify/commit performance across concurrency levels.

Current experiments use a Qwen3.8-27B target and a compatible DFlash2 draft on NVIDIA A800 80GB. This is an independent implementation; official vLLM is used as a separate comparison and diagnostic reference.

## Main features

- **Hybrid target:** Full Attention and Gated DeltaNet (GDN), checkpoint loading, RoPE, cached decoding, and target hidden-state taps.
- **DFlash2 proposals:** up to seven speculative tokens, target-conditioned draft context, and batched proposals.
- **Verification and commit:** greedy acceptance/correction, transactional KV and recurrent state, accepted-prefix restoration, and rollback.
- **Optimizations:** parallel block verification, fused GDN recurrence/prefix restoration, append-only KV reuse, batched verification sampling, and reduced intermediate copies.
- **Measurements:** paired target-only/DFlash2 runs, exact token comparison, acceptance, stage timing, memory tracking, and fixed-prefix official-vLLM diagnostics.
- **Online runtime:** concurrent admission, cancellation, streaming infrastructure, and request metrics. See [online serving](docs/online_serving.md) for its separate entry points and scope.

Original attention benchmarks and learning examples remain available but are no longer the main project entry point.

## Results and boundaries

Performance depends on workload, concurrency, numerical settings, and version. A faster LM head does not establish an equivalent end-to-end speedup.

| Experiment | Observation | Scope |
|---|---|---|
| C1 real-model pair | Target-only 8.80 tok/s; DFlash2 18.80 tok/s; acceptance 29.57% | Four requests × 3,000 output tokens; unmerged head; all 12,000 tokens matched; single observation |
| C24 LM-head A/B | Accumulated verify 651.38 → 634.48 s (−2.59%) | Both runs use DFlash2; 24 × 3,000 outputs matched; not a target-only comparison |
| B4 matched-input profile | LM-head GPU activity 11.89 → 1.48 ms; official 1.48 ms | Same prompts/candidates, K=7; one warmed verification; profiler data, not throughput |

Different versions and workloads must not be combined into one scaling curve. Detailed experiment artifacts are maintained outside the published repository.

**Numerical boundary:** BF16 kernel shape changes can change greedy decisions. `--merged-lm-head` remains experimental and off by default; validate exact output equality for your configuration before enabling it.

## Installation

Real-model execution requires Linux, an NVIDIA GPU, compatible CUDA drivers, and local target/draft weights. The tested large-model setup uses A800 80GB; capacity at a requested concurrency must be measured.

```bash
git clone https://github.com/WenyueD/Mini-vLLM.git
cd Mini-vLLM
uv sync --python 3.11 --group dev
```

Dependencies pin PyTorch 2.9.1, Transformers 4.57.3, and Linux x86-64 Triton 3.5.1. Official vLLM diagnostics use a **separate environment**, not a dependency of Mini-vLLM inference. The optional `oracle` dependency does not pin the historical reference build.

## Paired real-model benchmark

Weights are not included. Set checkpoint directories, then run a C1 smoke pair:

```bash
export MINIVLLM_QWEN38_PATH=/path/to/Qwen3.8-27B
export MINIVLLM_DFLASH2_PATH=/path/to/Qwen3.8-27B-DFlash2

uv run python scripts/benchmark_qwen38_dflash2_concurrency.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --variants target_only dflash2 \
  --workload short --concurrency 1 \
  --requests-per-run 4 --repeats 1 \
  --warmup-requests 1 --warmup-output-tokens 16 \
  --num-speculative-tokens 7 \
  --parallel-target-verification --parallel-verification-chunk-size 8 \
  --vectorized-target-numerics --fused-gdn-target \
  --cached-target-micro-batch-size 1 \
  --fused-prefix-commit --reuse-verification-samples \
  --compact-verification-copies --short-prefill-attention \
  --lightweight-live-stats --timeout 3600 \
  --run-id qwen_dflash2_c1_smoke
```

This short smoke run is **not** the 3,000-token experiment above. `--workload tasks4096` selects the four long-form tasks with a 4,096-token output budget. Start at C1 before raising concurrency, and set `--cached-target-micro-batch-size` deliberately: it changes execution shapes. Maximum admitted concurrency and actual verification batch size are different metrics.

Results: `results/qwen38_dflash2_perf/<run-id>/`; logs: `logs/qwen38_dflash2_perf/<run-id>/`. Check fairness/token matches, failures, actual batches, and warmup settings before interpreting throughput. Raw logs, traces, weights, and environments are not committed.

## Key options

| Option | Purpose / boundary |
|---|---|
| `--num-speculative-tokens 7` | Seven proposals; the complete verification query also contains an anchor |
| `--parallel-target-verification` | Verify a causal block rather than a separate target forward per position |
| `--vectorized-target-numerics` | Experimental shared BF16 vectorized target path |
| `--fused-gdn-target`, `--fused-prefix-commit` | Fuse recurrence and accepted-prefix restoration respectively |
| `--short-prefill-attention` | Direct-KV cached attention; requires vectorized target numerics |
| `--merged-lm-head` | One GEMM across verification positions; experimental, requires numerical validation |
| `--stop-verification-at-mismatch` | Optional early stopping, off by default; savings depend on mode/chunk size |

Use `--help` for all switches. Numerical-path changes require exact greedy-output regression. This project does not claim general stochastic rejection sampling, universal equality to official vLLM, or a speedup at every concurrency.

## Tests and diagnosis

```bash
uv run pytest -q
```

Tests cover tiny models, cache/rollback, and CUDA paths when available. Real-checkpoint tests require additional opt-in settings; skipped tests are not real-model validation. Use [profile_qwen35_verifier.py](scripts/profile_qwen35_verifier.py) for operator attribution, separately from throughput runs.

## Repository map

| Path | Contents |
|---|---|
| `src/myvllm/models/` | Qwen models, hybrid attention, GDN kernels |
| `src/myvllm/spec_decode/` | DFlash2, proposal/verification coordination, greedy acceptance |
| `src/myvllm/engine/` | Runner, sequences, scheduling, hybrid cache transactions |
| `src/myvllm/online/` | Concurrent online runtime |
| `scripts/` | Benchmarks, oracle fixtures, diagnostic tools |
| `tests/` | Functional, numerical, rollback, and GPU tests |

Further reading: [target notes](docs/qwen35_target.md), [speculative design](docs/speculative_decoding_design.md), [server commands](docs/autodl_qwen38_validation.md).

## Origins and license

This work extends the original [MiniVLLM](https://github.com/Wenyueh/MinivLLM), based on Nano-vLLM with standalone attention benchmarks. Original learning material remains in [HowToApproachvLLM.md](HowToApproachvLLM.md); the previous overview is preserved as [historical documentation](docs/legacy_minivllm_readme.md). Historical capability statements are not current project status.

The repository retains its [Apache-2.0 license](LICENSE). Model weights and external reference implementations have their own licenses.
