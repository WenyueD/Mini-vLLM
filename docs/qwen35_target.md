# Qwen3.5/Qwen3.8 Text Target Support

This repository now has a first-stage Qwen3.5/Qwen3.8 text-only target path for
hybrid models whose outer architecture is `Qwen3_5ForConditionalGeneration` and
whose language model lives under nested `text_config`.

## Current Boundary

- Supported: text-only target execution, greedy correctness scaffolding,
  request-isolated eager cached target steps, Qwen3.5/Qwen3.8 config parsing,
  weight mapping validation, hidden taps, hybrid cache transaction semantics,
  and DFlash2 draft execution behind the speculative verifier. Cached target
  verification tensor-batches active requests at each K+1 token position while
  preserving request-isolated state and ragged query lengths.
- Not claimed: multimodal inputs, tensor-batched ordinary target decode or
  DFlash2 proposal, production dynamic batching, CUDA Graph optimization for
  verification, Oracle proposal-ID alignment, or performance numbers.
- Multimodal tensors such as `pixel_values`, `image_grid_thw`, and
  `video_grid_thw` are rejected explicitly.
- Target cache sizing can be constrained with `target_cache_memory_fraction` or
  by reserving future draft space with `draft_memory_reservation_fraction`.

## Model Facts Captured

- Hidden tap IDs are zero-based layer-completion outputs:
  `[5, 19, 33, 47, 61]`.
- A log line showing `(6, 20, 34, 48, 62)` is a one-based display convention,
  not the Python/API key.
- The fixture used by tests mirrors the public Qwen/Qwen3.8-27B text config:
  64 text layers, `hidden_size=5120`, `intermediate_size=17408`,
  `num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`,
  `vocab_size=248320`, and every fourth layer as `full_attention`; other layers
  are `linear_attention`.
- Full-attention layers and linear/GDN layers own distinct cache state.
- The DFlash2 draft-side contract is separate from the target text config:
  `block_size=8`, K=7, `target_layer_ids=[5,19,33,47,61]`,
  `selector_top_k=16`, `selector_rank=256`, `conv_kernel_size=2`,
  `conv_group_size=16`, 5 sliding-attention layers, and
  `sliding_window=2048`.

## Cache Contract

`HybridCacheManager` owns request-scoped state:

- `full_attention_layers[layer_id]`: appended K/V tensors for full-attention
  layers in each request's eager path.
- `linear_layers[layer_id]`: GDN recurrent state and causal convolution state.
- `logical_length`: committed target prefix length for that request.

Transaction API:

```python
tx = cache.begin_transaction(request_ids, reserve_tokens)
model.forward_with_cache(proposal_tokens, cache, request_id, transaction=tx)
cache.commit(tx, {request_id: accepted_tokens})
cache.rollback(tx)
```

`accepted_tokens` is the number of executed provisional tokens to retain. For a
K+1 verifier, Mini-vLLM maps `accepted_proposal_tokens` to retained target cache
tokens as `1 + accepted_proposal_tokens`, because the verifier first computes the
already-committed pending token and then up to K draft tokens. The bonus target
token emitted after all K drafts are accepted is not yet in cache. The manager
supports heterogeneous batch commits such as `{a: 0, b: 3, c: 7}` and restores
full KV, recurrent state, conv state, and logical length on rollback.

`SpecDecodeCoordinator.execute()` accepts optional `commit_verified_cache` and
`rollback_verified_cache` hooks. The Qwen3.5/Qwen3.8 `ModelRunner.verify()` path
uses these hooks to begin a `HybridCacheManager` transaction before verification,
record one snapshot per verifier token, commit the accepted prefix after
rejection sampling, and roll back on validation/commit errors.

## Target Verification Shape Contract

For the next DFlash2 stage, target verification should call the target with:

- token IDs: the current mandatory target token plus up to K proposal tokens;
- K currently expected up to 7 proposals;
- `capture_hidden_taps=True` when DFlash2 hidden features are needed;
- `hidden_taps[layer_id]` shaped `[1, query_tokens, hidden_size]` per request;
- logits shaped `[1, query_tokens, vocab_size]`.

The HTTP/online layer should continue to depend only on target logits and emitted
token IDs; DFlash2-specific metadata belongs below the runner/coordinator boundary.

## DFlash2 Boundary

`myvllm.spec_decode.dflash2` now contains the draft-side interface contract:

- `DFlash2Config` validates the known Qwen3.8-DFlash2 constants and rejects
  one-based target tap IDs. It reads the real nested `dflash_config` object from
  the checkpoint config.
- `select_topk_candidates()` covers the selector top-k tensor path without
  treating it as a complete DFlash2 neural block.
- `ProposalBatch` carries `[batch, K]` token IDs plus per-request valid lengths.
- `DFlash2DraftCache` tracks sliding-window request lengths.
- `DFlash2DraftModel` is a loadable module shell whose parameter names match the
  observed 81-tensor checkpoint: candidate selector, `fc`, `hidden_norm`, 5
  sliding-attention layers, dynamic conv weights, and final `norm`.
- `audit_dflash2_checkpoint()` validates safetensors headers without loading
  large tensor payloads.
- `forward_block()` is the standalone B=1/Batched draft path: rolling target
  hidden-tap history is fused through `fc`, the target-generated bonus token plus
  mask tokens are embedded with the target embedding table, the DFlash2 draft
  backbone runs once over the whole block, and the target `lm_head` supplies
  unary logits for lattice-based candidate path selection on mask positions.
- `DFlash2StandaloneProposer` only accepts a draft model advertising
  `supports_block_parallel=True` and calls `propose_block()` once per batch.
- `SpeculativeConfig(proposer="dflash2")` wires a runner-bound proposer into
  `LLMEngine`; `ModelRunner` owns draft loading, target hidden-context capture,
  standalone proposal, and verifier-commit context refresh.

This path is still not a production DFlash2 integration. The draft checkpoint
can be audited and loaded, standalone block proposal can be run, and multiple
requests can share an engine scheduler batch. Cached target verification now
forms a real request-dimension tensor batch at each active query-token position;
ordinary target decode and DFlash2 proposal remain request-serial. Oracle
proposal-ID alignment remains required before broader claims.

## GPU Validation

AutoDL A800 validation on 2026-09-05 passed a heterogeneous B=2 online run with
the real Qwen/Qwen3.8-27B target. Both requests matched target-only token IDs for
32 generated tokens. Verification reported `max_batch_size=2` and 108
`tensor_batched_token_steps` out of 166 request-token steps. The clean run is in
`logs/dflash2_online_b2_tensor_batched_clean_20260905.log`.

Compared with the pre-optimization heterogeneous run, cached verification fell
from 19.43s to 12.95s (33.3%) and total DFlash2 generation fell from 20.89s to
14.43s (30.9%). The latest target-only run took 7.19s. Peak allocated memory was
71.99 GiB. The validation records:

- startup arguments and environment manifest;
- GPU name, available memory, and peak allocated memory;
- generated token IDs, with tie-aware argmax diagnostics;
- `236 proposed / 24 accepted`, acceptance `0.1016949152542373`;
- per-request accepted counts of 8 and 16;
- shared target and speculative batch size 2;
- no shutdown exception or residual GPU process.

A same-input K sweep then showed that proposal length, rather than request
batching, is the dominant remaining control for this workload. K=1 completed in
5.31s versus its 7.06s target-only baseline (1.33x faster), while K=2 took 6.52s,
K=3 took 8.22s, and K=7 took 14.29s. All variants remained target-lossless.
K=1 is therefore the current workload-specific recommendation; it is not yet a
global default or an adaptive-K claim.

Oracle proposal alignment remains `PENDING` because the available Oracle fixture
is not target-lossless.
