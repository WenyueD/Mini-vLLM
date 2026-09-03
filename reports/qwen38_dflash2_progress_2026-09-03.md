# Qwen3.8 + DFlash2 Progress - 2026-09-03

## User-Reported Remote Result

The AutoDL A800 target baseline commands all passed:

- `smoke_qwen38_load_only.py`
- `smoke_qwen38_single_token.py`
- `smoke_qwen38_cache_vs_recompute.py`
- `smoke_qwen38_engine_target_only.py`

This moves P1 target real baseline to PASS, assuming the server logs are retained
under `results/qwen38_target/`.

The user also reported:

- `dflash2_weight_name_audit=PASS`
- `dflash2_load_only=PASS`
- `dflash2_standalone_proposal=PASS`
- `qwen38_dflash2_engine_b1=PASS` under old prompt-final/single-context semantics; STALE after anchor/context/selector fixes
- `dflash2_online_b1=PASS` under old prompt-final/single-context semantics; STALE after anchor/context/selector fixes
- `dflash2_oracle_alignment=PENDING`; Oracle/vLLM method is `dflash`, and the current blocker is fixture/semantic parity, not Oracle capability
- `qwen38_target_oracle_anchor=FAIL` for prompt `"Hello"`: Oracle
  target-only first completion token is `11`, while Mini-vLLM currently predicts
  `846`
- After the split RMSNorm/GatedRMSNorm fix, `qwen38_target_oracle_anchor=PASS`
  for prompt `"Hello"` with `next_token_id=11`.
- The first post-fix cache-vs-recompute rerun hit a strict all-vocab bf16
  tolerance failure at `mismatch_step=26` (`max_abs_diff=0.5`), but
  `cached_token` and `recompute_token` both remained `353` and the top-5 token
  IDs matched.
- The follow-up token-decision cache-vs-recompute run passed with
  `qwen38_cache_vs_recompute=PASS`, and the target-only engine smoke passed with
  `qwen38_engine_target_only=PASS`.
- The fresh post-fix DFlash2 online B=1 run passed target-lossless validation:
  `dflash2_online_b1=PASS`, with `dflash2_proposed_tokens=189`,
  `dflash2_accepted_tokens=0`, and `dflash2_acceptance_rate=0.0`.

## DFlash2 Checkpoint Audit Input

The real DFlash2 checkpoint audit shows:

- `architectures=["DFlash2DraftModel"]`
- nested `dflash_config`
- K=7 via `block_size=8`
- `target_layer_ids=[5,19,33,47,61]`
- `selector_top_k=16`, `selector_rank=256`
- `conv_kernel_size=2`, `conv_group_size=16`
- 5 `sliding_attention` layers
- `sliding_window=2048`
- `hidden_size=5120`, `vocab_size=248320`
- one `model.safetensors` with 81 BF16 tensors

## New Local Changes

- `DFlash2Config.from_mapping()` now reads the real nested `dflash_config`.
- Added `DFlash2DraftModel`, a loadable shell with state-dict names matching all
  81 observed checkpoint tensors.
- Added expected shape generation for candidate selector, `fc`, norms,
  dynamic-conv parameters, 5 attention layers, and MLPs.
- Added `audit_dflash2_checkpoint()` for header-only safetensors validation.
- Added `load_dflash2_weights_from_safetensors()` for load-only validation.
- Added standalone draft math for `combine_hidden_taps`, grouped dynamic conv,
  dense sliding attention, MLP, target-lm-head unary logits, and lattice-based
  candidate path selection.
- DFlash2 target context now keeps rolling target hidden-tap history up to the
  sliding window instead of a single decision token.
- Added `DFlash2TargetContext` provider support to `DFlash2StandaloneProposer`,
  so one target-context fetch can drive one block proposal.
- Added engine-level DFlash2 proposer wiring for B=1:
  - `SpeculativeConfig(proposer="dflash2")`
  - runner-bound proposer in `LLMEngine`
  - DFlash2 draft loading inside `ModelRunner`
  - target hidden context capture and verifier-commit context refresh
- Split Qwen3.5 normalization semantics to match the official implementation:
  ordinary `Qwen35RMSNorm` keeps delta checkpoint semantics (`1 + weight`), while
  the GDN `Qwen35RMSNormGated` uses checkpoint weights directly.  This affects
  both Qwen3.8 target linear-attention layers and DFlash2 draft layers, and must
  be revalidated on AutoDL.
- Added scripts:
  - `scripts/audit_dflash2_weights.py`
  - `scripts/smoke_dflash2_load_only.py`
  - `scripts/smoke_dflash2_standalone_proposal.py`
  - `scripts/smoke_qwen38_dflash2_engine_b1.py`
  - `scripts/smoke_qwen38_dflash2_online_b1.py`
  - `scripts/compare_dflash2_oracle_fixture.py`
  - `scripts/export_oracle_dflash2_fixture.py`
- Added tests for nested config, expected shapes, header-only audit, selector
  top-k, and guarded block-parallel proposer behavior.

## Local Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp tests\test_dflash2_contract.py
```

Result: `9 passed`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp
```

Result: `84 passed, 2 skipped`.

Latest local regression after standalone proposal math:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp
```

Result: `87 passed, 2 skipped`.

Latest local regression after the split RMSNorm/GatedRMSNorm checkpoint-semantics
fix:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp
```

Result: `91 passed, 2 skipped`.

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Result: PASS.

## P0-P7 Status

| Priority | Status | Evidence |
| --- | --- | --- |
| P0 regression | PASS local | `84 passed, 2 skipped`; compileall passed. |
| P1 target real baseline | PASS remote/user-reported | Target Oracle anchor PASS after gated norm fix; token-decision cache-vs-recompute PASS; target-only engine smoke PASS. |
| P2 cache transaction | PASS local | Hybrid transaction/coordinator/Qwen cached verifier tests. |
| P3 hidden taps | PASS local | Zero-based taps and logits-invariance tests. |
| P4 DFlash2 standalone | PENDING | Real config + 81-key checkpoint audit/load-only PASS; draft model executes. Proposal token semantic parity with Oracle remains pending. |
| P5 B=1 end-to-end | PASS/PARTIAL | Fresh online B=1 is target-lossless and reports `dflash2_online_b1=PASS`, but accepted tokens are still zero, so proposal quality/alignment is not complete. |
| P6 batch | PENDING | Real DFlash2 batch not implemented. |
| P7 performance | PENDING | Oracle target parity, Oracle alignment, fresh B=1 validation, and batch validation are not complete. |

## Next Remote Commands

Run after pushing/pulling this change:

```bash
python scripts/audit_dflash2_weights.py \
  | tee results/qwen38_target/dflash2_weight_audit.txt

python scripts/smoke_dflash2_load_only.py \
  --model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  | tee results/qwen38_target/dflash2_load_only.txt
```

Then run one real standalone block proposal:

```bash
python scripts/smoke_dflash2_standalone_proposal.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --prompt "Hello" \
  | tee results/qwen38_target/dflash2_standalone_proposal_hello.txt
```

After standalone proposal passes, run the offline B=1 DFlash2 engine check:

```bash
python scripts/smoke_qwen38_dflash2_engine_b1.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --dflash2-safe-open-device cuda \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/dflash2_engine_b1_hello_32.txt
```

After offline engine B=1 passes, run the online B=1 streaming check:

```bash
python scripts/smoke_qwen38_dflash2_online_b1.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --dflash2-safe-open-device cuda \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/dflash2_online_b1_hello_32.txt
```

If CUDA staging is tight, retry with CPU staging:

```bash
python scripts/smoke_qwen38_dflash2_engine_b1.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cpu \
  --dflash2-safe-open-device cpu \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/dflash2_engine_b1_hello_32_cpu_staging.txt
```

If CUDA staging is tight for online B=1, retry with CPU staging:

```bash
python scripts/smoke_qwen38_dflash2_online_b1.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cpu \
  --dflash2-safe-open-device cpu \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/dflash2_online_b1_hello_32_cpu_staging.txt
```

If CUDA staging is tight, retry with CPU staging:

```bash
python scripts/smoke_dflash2_standalone_proposal.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cpu \
  --prompt "Hello" \
  | tee results/qwen38_target/dflash2_standalone_proposal_hello_cpu_staging.txt
```

If CUDA staging is tight, retry only load-only with CPU staging:

```bash
python scripts/smoke_dflash2_load_only.py \
  --model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cpu \
  | tee results/qwen38_target/dflash2_load_only_cpu_staging.txt
```

After online B=1 passes, export a Mini-vLLM proposal fixture and compare it with
an Oracle fixture containing the same prompt, anchor, tap IDs, and proposal
token IDs:

```bash
mkdir -p results/qwen38_target/oracle_alignment

python scripts/smoke_dflash2_standalone_proposal.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --prompt "Hello" \
  --json-output results/qwen38_target/oracle_alignment/minivllm_dflash2_hello.json \
  | tee results/qwen38_target/oracle_alignment/minivllm_dflash2_hello.txt

python scripts/compare_dflash2_oracle_fixture.py \
  --minivllm results/qwen38_target/oracle_alignment/minivllm_dflash2_hello.json \
  --oracle results/qwen38_target/oracle_alignment/oracle_dflash2_hello.json \
  --checksum-atol 0.5 \
  | tee results/qwen38_target/oracle_alignment/dflash2_oracle_alignment_hello.txt
```

Oracle fixture generation command:
The exporter skips prefill captures with `num_sampled=0` by default and selects
the first post-target-sample DFlash proposal block.  The current Oracle/vLLM
build accepts `--method dflash`; `--method dflash2` is rejected by its
SpeculativeConfig validation.

```bash
source /path/to/oracle/.venv/bin/activate
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=fork

python scripts/export_oracle_dflash2_fixture.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --output results/qwen38_target/oracle_alignment/oracle_dflash2_hello.json \
  --prompt "Hello" \
  --max-tokens 16 \
  --num-speculative-tokens 7 \
  --method dflash \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 256 \
  --anchor-token-mode minivllm \
  --anchor-position-mode minivllm \
  --trust-remote-code \
  --enforce-eager \
  | tee results/qwen38_target/oracle_alignment/oracle_dflash2_hello.txt
```
