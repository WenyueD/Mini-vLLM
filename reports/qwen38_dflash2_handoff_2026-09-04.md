# Qwen3.8 + DFlash2 Handoff - 2026-09-04

## Scope

This handoff starts from clean HEAD `584391f` (`Dump Oracle DFlash speculator runtime debug`) in `D:\LLM项目\MinivLLM`.

This local turn did not run AutoDL/GPU validation. It audited the current Mini-vLLM DFlash2 code path, pinned deterministic selector/cache semantics, and fixed draft-cache proposal transaction handling so rejected or failed proposals cannot leave provisional length state behind.

## Evidence Matrix

| Area | Status | Evidence |
| --- | --- | --- |
| Starting worktree | PASS | `git status --short` was empty before edits. |
| Official config name | PASS | Prior user/server evidence says Oracle accepts `method=dflash` with the DFlash2 checkpoint. This is not contradicted by local code. |
| Official selector shape | PASS local/source-audited | Official DFlash2 runtime builds per-step top-k candidates, scores predecessor x successor edge tensors, then walks from the anchor candidate row by argmax. Mini now has a deterministic reference test for this edge-walk behavior. Sources inspected: `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/spec_decode/dflash.py` and `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_dflash.py`. |
| Mini selector all-pairs edge scoring | PASS local | `test_dflash2_candidate_selector_matches_official_edge_walk_reference` and `test_dflash2_candidate_selector_scores_all_previous_candidate_edges` pass. The selector is not a single-predecessor scoring shortcut. |
| Mini DFlash2 target-hidden fallback cache | PASS local | `DFlash2DraftCache.update_target_context()` now records the actual clipped target-hidden context length and stores sliding-window hidden taps. This remains a correctness fallback, not the final high-performance official per-layer draft KV cache. |
| Proposal rollback/commit semantics | PASS local | `DFlash2DraftCache.begin_proposal()`, `commit_proposal()`, and `rollback_proposal()` now snapshot provisional proposal length. Standalone proposer failures roll back, runner proposal exceptions roll back, and verified commit reconciles to accepted target context length. |
| Local DFlash2 contract tests | PASS local | `24 passed` after this change. |
| Local regression | PASS local | `101 passed, 2 skipped`; `compileall` passed. |
| Qwen3.8 target baseline | PASS remote/user-reported | Prior report says target anchor, cache-vs-recompute token decision, and target-only engine smoke passed on AutoDL after the gated norm fix. I did not rerun these locally. |
| DFlash2 weight/load-only | PASS remote/user-reported | Prior report says checkpoint audit and load-only passed. I did not rerun them locally. |
| Fresh DFlash2 standalone proposal | PASS remote/user-reported | User reported `dflash2_standalone_proposal=PASS` after pulling commit `89fd865`. I did not rerun this locally. |
| Old DFlash2 B=1 offline/online | STALE | Old B=1 PASS claims must be rerun after anchor/path/cache changes. The reported online B=1 run had `dflash2_proposed_tokens=189` and `dflash2_accepted_tokens=0`, so it proves target losslessness under rejection, not useful DFlash2 acceptance. |
| Oracle fixtures with prompt-repeat/raw-output anomalies | TOOLING_BUG | Treat these as fixture/export interpretation failures, not evidence that official DFlash is non-lossless. The user reported Oracle exporter failures where DFlash output repeated prompt token `9419` and capture anchors were all `0`; after warmup capture was disabled, capture row 0 became single-prompt and started with `[11, 353, ...]`, while vLLM `CompletionOutput` itself still contained repeated `9419`. This now points mainly at Oracle LLM entrypoint/output behavior. Do not force formal alignment PASS from this fixture. |
| Mini vs Oracle one-step proposal alignment | PENDING | Formal alignment still requires Oracle target-lossless output. A diagnostic-only proposal compare is now supported for the current non-lossless Oracle fixture and prints `dflash2_oracle_proposal_diagnostic=PASS` instead of formal alignment PASS. |
| Real GPU B=1 offline drafted > 0 and accepted > 0 | PASS remote/user-reported | User reported exact target-only/DFlash2 token ID equality for 32 generated tokens with `dflash2_proposed_tokens=143`, `dflash2_accepted_tokens=8`, `dflash2_acceptance_rate=0.055944055944055944`, and `qwen38_dflash2_engine_b1=PASS`. I did not rerun this locally. |
| Real GPU B=1 online drafted > 0 and accepted > 0 | PASS remote/user-reported | User reported `dflash2_online_b1=PASS` with `dflash2_proposed_tokens=143`, `dflash2_accepted_tokens=8`, `dflash2_acceptance_rate=0.055944055944055944`, and multi-token streaming chunks. I did not rerun this locally. |
| B>1 continuous batching and performance | PENDING | Do not start until Oracle proposal alignment is resolved and B=1 traces are archived. |

## Local Changes

- Added deterministic selector reference coverage in `tests/test_dflash2_contract.py`.
- Added tests proving all predecessor-candidate rows are scored, rather than only the currently selected predecessor.
- Added DFlash2 draft cache proposal snapshots:
  - `begin_proposal(request_id)` saves the pre-proposal length, including the case where no length existed.
  - `rollback_proposal(request_id)` restores that exact state and removes new-request pollution.
  - `commit_proposal(request_id)` clears the provisional snapshot after accepted target context has been recorded.
- Made `update_target_context()` require non-empty, equal-length hidden taps and set the cache length to the real clipped context length.
- Wrapped `DFlash2StandaloneProposer` and `ModelRunner.propose_dflash2()` so proposal failures roll back draft-cache state.
- Updated `ModelRunner.commit_verified_cache()` to clear the DFlash2 proposal snapshot after accepted target-hidden history is stored.
- Updated `ModelRunner.rollback_verified_cache()` to roll back active DFlash2 proposal snapshots while staying compatible with `ModelRunner.__new__` tests that do not initialize DFlash2 fields.
- Hardened `scripts/export_oracle_dflash2_fixture.py` after an Oracle diagnostic failure:
  - binds `DFlashSpeculator.propose()` arguments by inspected signature before falling back to positional indices;
  - captures active request-state indices from the vLLM input batch when available;
  - uses the single-prompt row for this exporter instead of trusting an anomalous inferred `num_reqs=7`;
  - writes input-batch mapping diagnostics into the capture.
- Hardened the exporter again after a CUDA `IndexKernel.cu` assert:
  - proposal capture is disabled during `LLM` construction/warmup and enabled only around the real `generate()` call;
  - vLLM CUDA tensors are no longer advanced-indexed by active request-state indices inside the hook;
  - active indices are recorded only as diagnostics.
- Added DFlash output-object summaries and `proposal_first_token_id` diagnostics so future non-lossless Oracle reports show exactly which vLLM output field yielded repeated prompt tokens.
- Added diagnostic-only proposal comparison:
  - `export_oracle_dflash2_fixture.py` can set the diagnostic Mini-compatible anchor from `--expected-target-first-token-id` when `--allow-non-lossless-fixture` is used and Oracle output is known bad;
  - `compare_dflash2_oracle_fixture.py --allow-oracle-non-lossless` compares prompt/anchor/taps/proposal fields but prints `dflash2_oracle_proposal_diagnostic=PASS`, never formal `dflash2_oracle_alignment=PASS`.

## Commands Run Locally

```powershell
git status --short
git log --oneline -12
rg --files
rg -n "DFlash2|dflash|draft|proposal|hidden_tap|spec" src tests scripts reports docs
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp tests\test_dflash2_contract.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp tests\test_dflash2_contract.py tests\test_spec_coordinator.py tests\test_engine_spec_integration.py tests\test_hybrid_cache_transactions.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m compileall -q scripts\export_oracle_dflash2_fixture.py
.\.venv\Scripts\python.exe -m compileall -q scripts\export_oracle_dflash2_fixture.py scripts\compare_dflash2_oracle_fixture.py
.\.venv\Scripts\python.exe scripts\compare_dflash2_oracle_fixture.py --minivllm .\.tmp-oracle-check\mini.json --oracle .\.tmp-oracle-check\oracle.json --allow-oracle-non-lossless
```

Results:

- Focused DFlash2 before edits: `19 passed`.
- Focused DFlash2 after edits: `24 passed`.
- Related spec/cache/engine tests: `37 passed`.
- Full local regression: `101 passed, 2 skipped`.
- Compileall: PASS.
- Exporter compile check after Oracle diagnostic hardening: PASS.
- Diagnostic-only comparator smoke: `dflash2_oracle_alignment=SKIP ... proposal_diagnostic_only=true`, then `dflash2_oracle_proposal_diagnostic=PASS`.

## Remote User-Reported Results

Reported by the user after pulling commit `89fd865` on AutoDL:

```text
dflash2_standalone_proposal=PASS
dflash2_after_generate: allocated=53.69 GiB reserved=55.43 GiB max_allocated=70.10 GiB free=23.31 GiB total=79.25 GiB
target_only_token_ids=[11, 353, 2688, 264, 220, 17, 15, 4514, 6086, 8330, 11, 321, 353, 2908, 978, 23330, 264, 24207, 38288, 364, 279, 3162, 220, 17, 5381, 13, 353, 2908, 978, 4483, 888, 10288]
dflash2_token_ids=[11, 353, 2688, 264, 220, 17, 15, 4514, 6086, 8330, 11, 321, 353, 2908, 978, 23330, 264, 24207, 38288, 364, 279, 3162, 220, 17, 5381, 13, 353, 2908, 978, 4483, 888, 10288]
dflash2_proposed_tokens=143
dflash2_accepted_tokens=8
dflash2_acceptance_rate=0.055944055944055944
qwen38_dflash2_engine_b1=PASS
dflash2_online_b1=PENDING
dflash2_oracle_alignment=PENDING
```

Interpretation:

- This is the first useful B=1 offline DFlash2 evidence after the cache transaction fix because it has both `proposed_tokens > 0` and `accepted_tokens > 0`.
- The reported target-only and DFlash2 token IDs are exactly equal for the displayed 32-token output.
- Online B=1 and Oracle proposal alignment remained pending at this checkpoint.

Reported by the user after running online B=1:

```text
dflash2_text=", I'm a 20-year-old male, and I've been experiencing a persistent cough for the past 2 weeks. I've been taking over-the"
target_only_chunk_sizes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
dflash2_chunk_sizes=[1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1]
dflash2_proposed_tokens=143
dflash2_accepted_tokens=8
dflash2_acceptance_rate=0.055944055944055944
dflash2_online_b1=PASS
dflash2_oracle_alignment=PENDING
```

Interpretation:

- Online B=1 now has real acceptance and emits multi-token chunks (`2` and `3`) where target-only streamed one token per chunk.
- Offline and online B=1 both report the same proposed/accepted counts for this 32-token run.
- Oracle proposal alignment remains pending.

Reported by the user after the first Oracle exporter attempt:

```text
dflash2_standalone_proposal=PASS
dflash2_oracle_alignment=PENDING
Oracle DFlash generation is not target-lossless for this prompt:
target_first=11 dflash_first=9419 target_tokens=[11]
dflash_tokens=[9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419]
capture anchor_token_id=0 for all displayed captures
first displayed capture inferred multiple rows from a single-prompt run
```

Interpretation:

- `9419` is the `"Hello"` prompt token in this setup, so the Oracle fixture output path is still producing or reading prompt-token repetition.
- `anchor_token_id=0` across captures is not credible target-bonus-token state and points to capture mapping trouble.
- The exporter has been hardened after this report; rerun it before drawing any Mini/Oracle proposal conclusion.

Reported by the user after the first hardened exporter rerun:

```text
/__w/pytorch/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111:
Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
torch.AcceleratorError: CUDA error: device-side assert triggered
stack reached scripts/export_oracle_dflash2_fixture.py tensor_rows_to_list()
during vLLM compile_or_warm_up_model() -> sample_tokens() -> speculator.propose()
```

Interpretation:

- The exporter was still observing vLLM warmup-time speculator calls.
- The attempted active-index remapping was too invasive for internal CUDA tensors and could trigger device-side asserts.
- The exporter has been changed to avoid both behaviors.

Reported by the user after disabling warmup capture and unsafe indexing:

```text
Processed prompts: 1/1
Oracle DFlash generation is not target-lossless for this prompt:
target_first=11 dflash_first=9419
dflash_tokens=[9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419]
captures[0].num_sampled=[1]
captures[0].anchor_token_id=0
captures[0].proposal_rows=[[11, 353, 1076, 198, 9419, 11, 24]]
```

Interpretation:

- Capture filtering improved: it now records a single prompt row and no longer crashes during warmup.
- `captures[0].proposal_rows[0][0] == 11` matches the target first token, which is useful evidence that the speculator return may be closer to the desired point than the exported `output_token_ids` indicate.
- `anchor_token_id=0` remains suspicious and needs `last_sampled`/output-object diagnostics from the next run.

Reported by the user after output diagnostics:

```text
selected_capture_index=0
capture_num_sampled=[1]
raw_output_token_ids=[9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419]
output_token_ids=[9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419, 9419]
dflash_output_summary CompletionOutput text="HelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHelloHello"
anchor_token_id=9419
oracle_capture_anchor_token_id=0
anchor_position=1
proposal_token_ids=[11, 353, 1076, 198, 9419, 11, 24]
oracle_target_lossless=False
capture.proposal_first_token_id=11
capture.active_request_indices=[1022]
capture.input_batch_summary.idx_mapping=[1022]
capture.last_sampled=[0]
capture.next_prefill_tokens length=1024
```

Interpretation:

- The repeated `9419` is present inside vLLM's `CompletionOutput`, so the exporter is no longer merely slicing prompt tokens incorrectly.
- The captured speculator proposal starts with the expected target token `11`, but formal Oracle alignment remains blocked because the Oracle generation output is not target-lossless through this `LLM.generate` path.
- Use diagnostic-only comparison to see whether Mini's proposal IDs match the captured Oracle speculator return, while keeping formal alignment PENDING.

## Proposal Semantics Spec

Current Mini-vLLM implementation is expected to match this minimal DFlash2 proposal contract:

1. Target prefill/decode samples one bonus anchor token from target logits. That token is the first DFlash query input, not a prompt-final token.
2. Draft query inputs are `[anchor_token, mask, mask, mask, mask, mask, mask, mask]` for block size 8.
3. The seven mask positions produce the seven speculative token proposals.
4. Target hidden taps are zero-based layers `(5, 19, 33, 47, 61)`.
5. Mini currently recomputes draft-layer context key/value from the stored rolling target-hidden tap history for correctness. This is acceptable as a first-stage fallback, but not the final efficient official cache implementation.
6. Candidate selection takes top-k unary candidates for each speculative position, constructs predecessor x successor edge scores for each adjacent step, starts from anchor row index 0, and follows argmax successor indices step by step.
7. Rejection or failure must not advance the stored target-hidden fallback context beyond the accepted target prefix.

## Next AutoDL Validation

Standalone proposal, offline B=1, and online B=1 have now been reported as passed by the user. Run Oracle proposal alignment next:

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
```

Then run a diagnostic Oracle exporter pass in the official vLLM/Oracle environment. Use `--allow-non-lossless-fixture` for this pass so the JSON is saved:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
export CUDA_LAUNCH_BLOCKING=1

python scripts/export_oracle_dflash2_fixture.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --output results/qwen38_target/oracle_alignment/oracle_dflash2_hello_debug_nonlossless.json \
  --prompt "Hello" \
  --max-tokens 16 \
  --num-speculative-tokens 7 \
  --method dflash \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 256 \
  --expected-target-first-token-id 11 \
  --skip-target-preflight \
  --anchor-token-mode minivllm \
  --anchor-position-mode minivllm \
  --trust-remote-code \
  --enforce-eager \
  --dump-speculator-debug \
  --allow-non-lossless-fixture \
  | tee results/qwen38_target/oracle_alignment/oracle_dflash2_hello_debug_nonlossless.txt
```

Do not compare this diagnostic fixture unless it prints `oracle_dflash2_fixture=PASS`. If it still prints `DEBUG_NON_LOSSLESS`, inspect and report:

- `dflash_output_summary`
- `capture.last_sampled`
- `capture.next_prefill_tokens`
- `capture.proposal_first_token_id`
- `capture.input_batch_summary`
- `raw_output_token_ids`
- `output_token_ids`

For proposal-only diagnostics against the currently non-lossless Oracle fixture, run:

```bash
python scripts/compare_dflash2_oracle_fixture.py \
  --minivllm results/qwen38_target/oracle_alignment/minivllm_dflash2_hello.json \
  --oracle results/qwen38_target/oracle_alignment/oracle_dflash2_hello_debug_nonlossless.json \
  --checksum-atol 0.5 \
  --allow-oracle-non-lossless \
  | tee results/qwen38_target/oracle_alignment/dflash2_oracle_proposal_diagnostic_hello.txt
```

This diagnostic compare may print `dflash2_oracle_proposal_diagnostic=PASS`; that is useful, but it does not satisfy formal `dflash2_oracle_alignment=PASS`.

Compare fixtures:

```bash
python scripts/compare_dflash2_oracle_fixture.py \
  --minivllm results/qwen38_target/oracle_alignment/minivllm_dflash2_hello.json \
  --oracle results/qwen38_target/oracle_alignment/oracle_dflash2_hello.json \
  --checksum-atol 0.5 \
  | tee results/qwen38_target/oracle_alignment/dflash2_oracle_alignment_hello.txt
```

Required evidence for Oracle alignment:

- Oracle fixture has `fixture_schema_version=2` and `anchor_semantics=target_bonus_token`.
- Oracle generation remains target-lossless for the compared prompt.
- Mini and Oracle agree on prompt token IDs, anchor token ID, anchor position, target tap IDs, and proposal token IDs, or any mismatch is explained with concrete candidate/selector diagnostics.

If CUDA staging is tight for the Mini fixture, rerun it with `--safe-open-device cpu`. For the Oracle exporter, prefer reducing `--gpu-memory-utilization` or using CPU offload via supported vLLM kwargs rather than changing dtype or quantizing, because either can change the proposal surface.

## Remaining Work

- Regenerate Oracle fixture from the official DFlashSpeculator return point and verify the output-token extraction path. Do not compare against old prompt-repeat/anchor-0 fixtures.
- Align one Mini proposal block with Oracle under identical anchor token, anchor position, hidden taps, and checkpoint.
- If proposals still have zero acceptance after this cache fix, inspect target hidden tap values and selector candidate lattice before changing anchor fields.
- Implement official-style per-layer draft context/KV cache for performance after correctness fallback alignment is proven.
- Extend to B>1 only after B=1 has drafted and accepted real tokens.
