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

# DFlash2 Evaluation Record
> Base Configuration: Qwen3.8-27B target, paired DFlash2 draft, NVIDIA A800 80GB
> Note: Performance and acceptance rate are affected by task, sampling parameters, concurrency, output length and execution configuration.

## GSM8K Acceptance Rate Evaluation
Test Condition: Same set of 128 GSM8K problems, concurrency = 8, maximum output length = 4096 tokens, at most 7 draft tokens proposed per round. Both implementations share identical input and model weights; warmup results are excluded.

| Implementation | Sampling Parameters | Overall Draft Acceptance Rate | Per-request Average Accepted Length | Total Output Tokens / Verification Rounds |
| ---- | -------- | ---------------- | ------------------ | ------------------------ |
| Mini-vLLM + DFlash2 | greedy | 54.65% | 5.6117 | 4.8253 |
| vLLM + DFlash2 | greedy | 55.19% | 5.6519 | 4.8637 |
| vLLM + DFlash2 | temperature=1, top_p=0.95, top_k=20 | 41.47% | 4.8372 | 3.9037 |

**Metric Definition**
- Overall draft acceptance rate: Total number of accepted draft tokens across all requests divided by total proposed draft tokens. Target-supplemented tokens are excluded.
- Per-request average accepted length: For each request, divide output token count by verification rounds, then take the arithmetic mean over all requests.
- Aggregate advancement length: Total output tokens divided by total verification rounds. Requests with more verification rounds carry higher weight, so this value generally differs from the per-request average accepted length.

**Remarks**
The overall draft acceptance rate of Mini-vLLM and vLLM under greedy mode differs by approximately 0.54 percentage points. A close acceptance rate does not guarantee identical token-wise outputs; this table does not serve as strict proof of output consistency.

The official model card reports a per-request average accepted length of 5.46 on GSM8K, using SGLang, H200, FlashAttention 3 and official recommended stochastic sampling parameters. Our vLLM stochastic sampling result of 4.8372 uses the same metric definition, but runs on different hardware and environment, so it does not constitute a full reproduction.
Reference: https://huggingface.co/incoai/Qwen3.8-27B-DFlash2#evaluation

## GSM8K Throughput Benchmark
Timing Scope: Generation calls only. Model loading, warmup and counter reading are excluded. vLLM runs in eager mode with CUDA Graph and asynchronous scheduling disabled. These are diagnostic measurements and do not represent the optimal performance of each engine.

| Implementation | Request Count | Concurrency | Sampling | Output Limit | Output Throughput |
| ---- | ------ | ---- | ---- | -------- | -------- |
| Mini-vLLM + DFlash2 | 128 | 8 | greedy | 4096 token | 172.8 tok/s |
| vLLM + DFlash2 | 128 | 8 | greedy | 4096 token | 226.7 tok/s |
| vLLM + DFlash2 | 128 | 8 | Official recommended stochastic sampling | 4096 token | 204.3 tok/s |
| Mini-vLLM + DFlash2 | 32 | 8 | greedy | No artificial output cap | 165.8 tok/s |

## Historical Pairing & Operator Optimization Experiments
> Results below come from different workloads and configurations, retained only for validation logging.

| Experiment | Observation | Scope |
| ---- | -------- | -------- |
| C1 Real Model Pairing | Target-only: 12.860 tok/s; DFlash2: 39.86 tok/s; ~3.1× speedup; acceptance rate 54.9% | 4 requests, each generating 4096 tokens; head merging disabled by default; 16384 output tokens fully consistent; single observation |
| C24 LM head A/B | Cumulative verify time reduced from 277.38 s to 249.8 s, a 10.22% drop | Both sides use DFlash2; 24 requests each generating 4096 tokens, consistent outputs; not a target-only baseline |
| B4 Same Input Diagnostic | LM head GPU cumulative activity reduced from 11.89 ms to 1.48 ms; official vLLM baseline = 1.48 ms | Identical prompt and candidates, K=7; first round after warmup; instrumented profiling, not a throughput metric |

## Global Supplementary Notes
Numerical boundary: Kernel shape changes under BF16 may alter greedy sampling decisions. `--merged-lm-head` remains an experimental option disabled by default. Strict output consistency should be validated for your specific setup before enabling it.


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
