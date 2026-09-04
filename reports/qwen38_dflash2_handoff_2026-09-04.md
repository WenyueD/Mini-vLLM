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
| Oracle fixtures with prompt-repeat/raw-output anomalies | TOOLING_BUG | Treat these as fixture/export interpretation failures, not evidence that official DFlash is non-lossless. Do not force top-level anchor fields to make the comparator pass. |
| Mini vs Oracle one-step proposal alignment | PENDING | Needs same prompt/tokenized input, same target bonus anchor token and position, same hidden taps, same checkpoint, and tie-aware proposal comparison. |
| Real GPU B=1 offline drafted > 0 and accepted > 0 | PASS remote/user-reported | User reported exact target-only/DFlash2 token ID equality for 32 generated tokens with `dflash2_proposed_tokens=143`, `dflash2_accepted_tokens=8`, `dflash2_acceptance_rate=0.055944055944055944`, and `qwen38_dflash2_engine_b1=PASS`. I did not rerun this locally. |
| Real GPU B=1 online drafted > 0 and accepted > 0 | PENDING | User reported `dflash2_online_b1=PENDING`; run online smoke next. |
| B>1 continuous batching and performance | PENDING | Do not start until selector/cache/oracle alignment and B=1 real acceptance are proven. |

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
```

Results:

- Focused DFlash2 before edits: `19 passed`.
- Focused DFlash2 after edits: `24 passed`.
- Related spec/cache/engine tests: `37 passed`.
- Full local regression: `101 passed, 2 skipped`.
- Compileall: PASS.

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
- Online B=1 and Oracle proposal alignment remain pending.

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

Standalone proposal and offline B=1 have now been reported as passed by the user. Run online B=1 next:

```bash
mkdir -p results/qwen38_target/dflash2_2026_09_04

python scripts/smoke_qwen38_dflash2_online_b1.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --dflash2-safe-open-device cuda \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/dflash2_2026_09_04/dflash2_online_b1_hello_32.txt
```

Required evidence for B=1 to count as real DFlash2 progress:

- `drafted_tokens > 0` or equivalent proposed-token counter.
- `accepted_tokens > 0`.
- Per-step proposal token IDs.
- Target verifier greedy IDs/probabilities or logits around the proposal IDs.
- Accepted prefix length per step.
- Cache context length and rollback/commit trace.
- Final generated IDs equal target-only IDs, with explicit tie-aware treatment for exact argmax ties.

If CUDA staging is tight, rerun the same commands with `--safe-open-device cpu` and `--dflash2-safe-open-device cpu`; do not change dtype or quantize for alignment unless the goal is explicitly diagnostic.

## Remaining Work

- Regenerate Oracle fixture from the official DFlashSpeculator return point and verify the output-token extraction path. Do not compare against old prompt-repeat/anchor-0 fixtures.
- Align one Mini proposal block with Oracle under identical anchor token, anchor position, hidden taps, and checkpoint.
- If proposals still have zero acceptance after this cache fix, inspect target hidden tap values and selector candidate lattice before changing anchor fields.
- Implement official-style per-layer draft context/KV cache for performance after correctness fallback alignment is proven.
- Extend to B>1 only after B=1 has drafted and accepted real tokens.
