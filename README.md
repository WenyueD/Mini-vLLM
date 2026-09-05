<p align="center">
  <img src="./assets/minivllm.png" alt="MinivLLM" width="50%" height="50%">
</p>

<p align="center">
| <a href="./README.md"><b>English</b></a> 
| <a href="./README_zh.md"><b>简体中文</b></a> |
</p>

# miniVLLM

A custom implementation of vLLM inference engine with attention mechanism benchmarks, based on Nano-vLLM but with self-contained paged attention and flash attention implementation. 

Benchmarking on flash attention in prefilling time and paged attention in decoding time are provided.

**New to vLLM?** Check out [HowToApproachvLLM.md](HowToApproachvLLM.md) for a step-by-step implementation guide covering layers, models, paged attention, CUDA graphs, and scheduling.

## Quickstart

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run the main inference engine
uv run python main.py

# Run prefilling benchmark
uv run python benchmark_prefilling.py

# Run decoding benchmark
uv run python benchmark_decoding.py
```

To run multi-GPU setting, simply change world_size to n > 1 in config in main.py

## What Each Script Does

```bash
uv run python main.py
```

This is the main inference engine demo

Demonstrates the complete LLM inference pipeline using a custom engine implementation:
- Create a small version of Qwen3 with random initialization
- Creates 60 chat prompts (2 base prompts repeated 30 times each)
- Processes them through the custom LLM engine with batch processing
- Uses paged attention and KV cache management for efficient inference
- Generates up to 256 tokens per prompt with temperature sampling

This showcases how the custom vLLM implementation handles batched text generation with memory-efficient attention.

```bash
uv run python benchmark_prefilling.py
```

This is the prefilling phase comparison

Compares three attention implementations during the **prefilling phase** (processing input prompts):

1. **PyTorch Standard (O(N²) memory)**: Traditional attention that materializes full attention matrix
2. **Naive Triton (O(N²) memory)**: GPU kernel that also uses O(N²) memory, limited by shared memory constraints (≤128 tokens)
3. **Flash Attention (O(N) memory)**: Memory-efficient online softmax algorithm that processes attention in blocks

```bash
uv run python benchmark_decoding.py
```

This is the decoding phase comparison

Compares three implementations during the **decoding phase** (generating output tokens one at a time):

1. **Naive PyTorch**: Simple loop-based implementation using paged KV cache
2. **Optimized PyTorch**: Vectorized implementation with batch gathering and masking
3. **Triton Kernel**: Custom GPU kernel optimized for paged attention decode


## Project Structure

```
myvllm/
├── src/
│   └── myvllm/           # Core vLLM implementation
│       ├── models/       # Model implementations
│       ├── engine/       # LLM engine logic, including sequence definition for input prompts, block management for KV cache management for GPU, scheduler for iteration-based scheduling of sequences, runner for actual implementation of running prefilling and decoding, and engine for generation API interface
│       ├── layers/       # Model layer components (activation, attention, embeddings, etc.)
│       ├── utils/        # Utility helpers and inference context management
│       └── sampling_parameters.py
├── main.py              # Full inference demo
├── benchmark_prefilling.py   # Prefilling attention comparison
└── benchmark_decoding.py     # Decoding attention comparison
```

## Requirements

- Python ≥3.11, < 3.12
- CUDA-capable GPU
- Dependencies: `transformers`, `torch`, `xxhash` (managed by uv)

## Speculative decoding MVP

MiniVLLM now has an opt-in, proposer-independent speculative decode path:

```python
config["speculative_config"] = {
    "enabled": True,
    "proposer": "ngram",
    "num_speculative_tokens": 4,
    "ngram_min_size": 2,
    "ngram_max_size": 4,
}
sampling_params = SamplingParams(temperature=0, max_tokens=128)
```

`temperature=0` is exact deterministic argmax. Target-only and speculative
decoding use the same logits-to-token decision function. The MVP supports prompt
n-gram lookup, a deterministic mock proposer, heterogeneous proposal/acceptance
lengths, K+1 eager verification (including the bonus token), EOS/length limits,
and transactional paged-KV reserve/commit/rollback. When prompt lookup finds no
match or draft capacity is unavailable, that request performs ordinary one-token
greedy decode.

Runtime statistics are available through `engine.get_speculative_stats()`. They
include proposed/accepted counts, acceptance lengths and per-position rates, plus
proposal/verify/sample/commit wall times. A CPU-only correctness and overhead smoke
benchmark is available without model weights:

```bash
uv run python benchmark_speculative.py
uv run pytest -q
```

`benchmark_speculative.py` uses a deterministic fake target/verifier whose output
is the repeating cycle `0,1,2,3,4`, with a matching repeated prompt. Its K=4 / high
acceptance result validates the lossless coordinator, transaction pipeline, and
statistics plumbing only. It is **not** a real-model throughput measurement, an
n-gram quality claim, or evidence of GPU speedup. Real performance and acceptance
numbers must come from the Qwen3 GPU command below.

The implementation does **not** claim probability rejection sampling, non-greedy
speculation, a neural draft model, EAGLE/DFlash2, verification CUDA graphs, or
dynamic-K kernels. Positive-temperature requests continue through the unchanged
ordinary sampler. Speculative verification is forced eager. Prefix-cache hash
metadata is only published on transaction commit; speculative reservations never
publish provisional prefixes.

The state/data-flow rationale and failure invariants are documented in
[`docs/speculative_decoding_design.md`](docs/speculative_decoding_design.md).

Qwen3.5/Qwen3.8 hybrid text target scaffolding is documented in
[`docs/qwen35_target.md`](docs/qwen35_target.md). It covers nested `text_config`
parsing, zero-based hidden taps `[5, 19, 33, 47, 61]`, request-isolated eager
hybrid cache transactions, and the current text-only support boundary for the
later DFlash2 proposer integration.

### Real Qwen3-0.6B GPU validation

The repository includes an opt-in target-only versus speculative greedy equality
test. Its primary test replays target-only baseline tokens as proposals and asserts
positive proposed/accepted counts plus an acceptance length greater than one. Thus
it necessarily exercises real Qwen3 K+1 verification and cannot silently pass via
K=0 fallback. A separate n-gram equality smoke test is also included. The test
never downloads weights implicitly; point it at a local checkpoint:

```bash
MINIVLLM_QWEN3_PATH=/absolute/path/to/Qwen3-0.6B \
  uv run pytest -q -s tests/gpu/test_qwen3_speculative_equivalence.py
```

On Windows PowerShell:

```powershell
$env:MINIVLLM_QWEN3_PATH = "D:\models\Qwen3-0.6B"
uv run pytest -q -s tests/gpu/test_qwen3_speculative_equivalence.py
```

This environment had no CUDA device, so the real-model test is provided but was
not reported as passed. CPU tests validate coordinator, scheduler, acceptance,
verification metadata, and KV transaction behavior with deterministic targets.

## Online request processing

The online layer has three explicit boundaries:

- `LLMEngine` owns request IDs, incremental committed-token events, cancellation,
  scheduler state, and step-level metrics.
- `OnlineEngine` is the single background writer to `LLMEngine` and GPU state.
  HTTP threads only enqueue add/cancel commands and consume output queues. When no
  requests exist, the loop blocks on the command queue and does not busy-spin.
- The standard-library HTTP server handles JSON/SSE only. It does not call the
  model runner or speculative coordinator directly, and adds no new dependency.

Edit the model path in `examples/online_qwen3_config.json`, then start the server:

```bash
uv run python -m myvllm.online.server \
  --config examples/online_qwen3_config.json --host 0.0.0.0 --port 8000
```

Non-streaming completion:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"red blue red blue","temperature":0,"max_tokens":32}'
```

SSE streaming (`-N` disables curl buffering):

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"red blue red blue","temperature":0,"max_tokens":32,"stream":true,"request_id":"demo-stream"}'
```

Cancel explicitly with
`POST /v1/requests/demo-stream/cancel`. A disconnected streaming client also
causes the handler to enqueue cancellation. Health and step-level statistics are
available at `/health` and `/stats`.

This is a minimal OpenAI-style completions surface, not full OpenAI API
compatibility. SSE detokenization retains an eight-token mutable suffix so common
BPE boundary rewrites do not corrupt concatenated chunks; a rewrite extending
beyond that window is reported as an explicit stream error. After a terminal event
is consumed, request/channel state is released and `/stats` retains only a bounded
terminal status history while keeping lifetime submitted/released counters.

The speculative online endpoint rejects positive temperatures with HTTP 400 rather
than silently changing sampling semantics. To serve positive-temperature requests,
start a target-only configuration with `speculative_config.enabled=false`.

Send concurrent requests from a shell:

```bash
seq 1 8 | xargs -P 8 -I{} curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain paged KV cache","temperature":0,"max_tokens":64,"request_id":"curl-{}"}'
curl -s http://127.0.0.1:8000/stats
```

Or run the repeatable HTTP benchmark:

```bash
uv run python benchmark_online.py --requests 32 --concurrency 8 --max-tokens 64
```

The benchmark reports `max_observed_decode_batch_size` and
`observed_continuous_batching`. A single client issuing one request after another
is only serial inference wrapped in HTTP. Dynamic continuous batching is actually
demonstrated only when overlapping requests produce a recorded decode batch size
greater than one.

For a real Qwen3-0.6B server smoke test, run the server command on a CUDA host, then
run both the concurrent benchmark and the `/stats` query above. This local
environment has no CUDA and did not permit reading an SSH configuration, so no
remote GPU/online result is claimed here.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Wenyueh/MinivLLM&type=date&legend=top-left)](https://www.star-history.com/?utm_source=chatgpt.com#Wenyueh/MinivLLM&type=date&legend=top-left)
