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
- `qwen38_dflash2_engine_b1=PASS`
- `dflash2_online_b1=PENDING`
- `dflash2_oracle_alignment=PENDING`

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
  dense sliding attention, MLP, target-lm-head unary logits, and candidate path
  selection.
- Added `DFlash2TargetContext` provider support to `DFlash2StandaloneProposer`,
  so one target-context fetch can drive one block proposal.
- Added engine-level DFlash2 proposer wiring for B=1:
  - `SpeculativeConfig(proposer="dflash2")`
  - runner-bound proposer in `LLMEngine`
  - DFlash2 draft loading inside `ModelRunner`
  - target hidden context capture and verifier-commit context refresh
- Added scripts:
  - `scripts/audit_dflash2_weights.py`
  - `scripts/smoke_dflash2_load_only.py`
  - `scripts/smoke_dflash2_standalone_proposal.py`
  - `scripts/smoke_qwen38_dflash2_engine_b1.py`
  - `scripts/smoke_qwen38_dflash2_online_b1.py`
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

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Result: PASS.

## P0-P7 Status

| Priority | Status | Evidence |
| --- | --- | --- |
| P0 regression | PASS local | `84 passed, 2 skipped`; compileall passed. |
| P1 target real baseline | PASS remote/user-reported | All four Qwen3.8 target smoke scripts reported PASS on AutoDL A800. |
| P2 cache transaction | PASS local | Hybrid transaction/coordinator/Qwen cached verifier tests. |
| P3 hidden taps | PASS local | Zero-based taps and logits-invariance tests. |
| P4 DFlash2 standalone | PASS/PENDING align | Real config + 81-key checkpoint audit/load-only PASS; remote standalone proposal PASS; Oracle proposal-ID alignment still PENDING. |
| P5 B=1 end-to-end | PARTIAL | Offline engine B=1 PASS remote/user-reported; online B=1 smoke script added locally and remains PENDING. |
| P6 batch | PENDING | Real DFlash2 batch not implemented. |
| P7 performance | PENDING | Correctness stage not complete. |

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
