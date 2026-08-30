# Speculative decoding MVP design

## Scope and repository audit

MiniVLLM currently has a two-phase engine loop. `Scheduler.schedule()` allocates
prompt blocks for prefill and, during decode, calls `BlockManager.append()` before
`ModelRunner.run()`. The runner then evaluates exactly one pending token per
sequence and `Scheduler.postprocess()` appends exactly one sampled output token.
Consequently, after every successful non-terminal step the logical sequence has
one committed token whose KV has not yet been computed.

The existing decode attention kernel already reads paged KV through a block table,
but the runner only prepares one query per request. Prefill is not a safe shortcut
for verification: its attention kernel only attends to K/V from that invocation,
and its RoPE positions restart at zero. Speculative verification therefore gets a
dedicated eager decode input path.

The initial repository test baseline on 2026-08-29 was 8 failures: every existing
scheduler test instantiated `Sequence` without the required `block_size`. The
working tree also contained user-owned edits to `pyproject.toml`, `uv.lock`, and an
untracked `.idea/`; this implementation does not alter or remove them.

## Supported MVP

- deterministic greedy decoding (`temperature=0` means exact argmax);
- a proposer-independent runtime with mock and prompt n-gram proposers;
- variable proposal lengths in one batch, multi-token eager target verification,
  first/middle rejection, all accepted, all rejected, and a target bonus token;
- per-sequence KV reservation transactions without copying KV tensors;
- output-limit, EOS, model-length, capacity fallback, and ordinary decode fallback;
- counters, per-position acceptance, acceptance-length observations, and timings
  for proposal, verification, sampling, and commit.

Probability-based rejection sampling is **not implemented**. Temperature greater
than zero continues to use the ordinary sampler and disables speculative decoding.
Neural draft models, EAGLE, DFlash2, CUDA graphs for verification, and dynamic-K
kernels are future work.

## Data flow

```text
Scheduler selects running requests (without mutating their lengths)
  -> Proposer.propose(sequence, max_tokens)
  -> Sequence begins a provisional transaction
  -> BlockManager reserves only newly required physical blocks
  -> TargetVerifier evaluates [pending committed token, draft tokens...]
  -> RejectionSampler compares target argmax tokens with draft tokens
  -> commit accepted drafts + mismatch/bonus token
  -> rollback unused slots/blocks and finish or requeue the request
```

`Proposal`, `SpecDecodeBatch`, and `VerificationResult` are deliberately small
transport objects. A proposer only sees committed token IDs and a token budget. A
target verifier consumes the batch and returns one target decision per query
position (proposal length plus one). This interface permits a future parallel
proposer to attach metadata without changing transaction ownership.

`Proposer.propose_batch(sequences, per_request_budgets)` is the neural/parallel
extension point. The base implementation loops over `propose`; draft-model,
EAGLE, or DFlash2 implementations can override the batch method and set
`supports_parallel=True` while returning the same `Proposal` transport objects.

## Sequence state and invariants

The terms have exact meanings:

- **committed**: tokens visible to the user and future proposers (`token_ids`);
- **computed**: leading committed tokens whose target KV is valid;
- **provisional**: draft tokens visible only inside the active transaction;
- **reserved**: logical capacity backed by the current block table; it covers all
  committed and provisional query positions during verification.

At rest:

```text
0 <= num_computed_tokens <= num_tokens
num_provisional_tokens == 0
no active Sequence or BlockManager transaction
num_tokens - num_computed_tokens is 0 before prefill, then 1 during decode
len(block_table) == ceil(num_computed_tokens / block_size) during active decode
```

The last emitted token is committed but pending: it has no valid KV until the next
ordinary scheduler append or speculative reservation/verification. If that pending
token is inside the retained partial block, its physical block already exists but
its slot is not valid KV. If it starts a new block, `block_table` deliberately does
not contain that block yet. Consequently `Sequence.num_blocks` (derived from all
committed tokens) may be one greater than `len(block_table)` between decode steps.
Finished sequences are deallocated and have an empty table; waiting/prefill states
follow their allocation phase rather than this active-decode equality.

During verification:

```text
reserved_tokens == num_tokens + num_provisional_tokens
len(block_table) == ceil(reserved_tokens / block_size)
query tokens == committed[num_computed_tokens:] + provisional
```

After accepting `a` proposal tokens, target KV is retained for the old pending
token plus those `a` tokens. The mismatch token, or the bonus token when all draft
tokens match, is committed but remains the next pending token. A terminal output
may leave no pending token because its blocks are immediately deallocated.

Transactions are single-use and per sequence. Nested reserve, duplicate commit,
duplicate rollback, and a transaction/sequence mismatch are errors. Rollback in a
partially used block only changes logical state; stale KV is unreachable and will
be overwritten. Rollback across a boundary decrements and frees only the newly
allocated tail blocks. No KV tensor is cloned.

## Failure handling

- **No proposal / no n-gram match:** verify one pending token exactly like ordinary
  greedy decode.
- **Insufficient KV for K drafts:** retry reservation with zero drafts; if even the
  pending token cannot be backed, the scheduler preempts using its normal policy.
- **EOS / max_tokens / max_model_length:** truncate committed outputs at the first
  stop and deallocate the finished request.
- **Mixed batch lengths:** flattened verification rows carry a duplicated block
  table and a position-specific context length; results are regrouped by request.
- **Verifier exception:** roll back every active Sequence and block reservation,
  then re-raise; no request is partially committed.
- **Prefix cache:** cross-request prefix reuse is already effectively disabled by
  the current deallocation path. Speculative reservations do not publish hashes;
  tests assert that rollback cannot introduce cache entries. A future functional
  prefix cache must add transaction-aware hash publication.

## Verification input contract

For each request with computed prefix length `c`, committed pending token `x`, and
draft `d1..dk`, the eager runner evaluates flattened rows `x,d1,..,dk`. Row `j`
uses position `c+j`, context length `c+j+1`, the request's reserved block table,
and the physical slot for logical token `c+j`. Because every layer stores all row
K/V before paged attention reads them, later rows can attend to earlier draft rows.
The returned argmax tokens are decisions for `d1..dk` and one bonus/replacement.

This path intentionally bypasses CUDA graph capture in the MVP. Its metadata shape
allows variable q lengths today and can later be bucketed for fixed/dynamic K.

## Validation plan

CPU tests cover proposer behavior, exact greedy decisions, rejection boundaries,
bonus tokens, mixed batches, scheduler integration with a fake target, within- and
cross-block rollback, capacity fallback, duplicate-operation guards, and randomized
state transitions/ref-count conservation. A fake deterministic causal target also
checks batched verification decisions against token-by-token execution.

The CPU benchmark is intentionally a pipeline smoke workload: a deterministic fake
verifier emits a repeating five-token cycle that prompt lookup predicts perfectly.
Its timing and acceptance rate only validate framework overhead/counters and exact
output equality. They are not real-model n-gram effectiveness or acceleration data.

GPU validation requires CUDA plus local Qwen3-0.6B weights. The exact command and
observed result are documented in `README.md`; absence of either is reported as an
unverified boundary, never as a pass.

The real-GPU acceptance test first produces target-only baseline tokens, then uses
a test-only replay proposer. On the first proposal call, prefill has already
committed baseline completion token 0, so `sequence.num_completion_tokens` directly
indexes the next baseline token. Hard assertions require positive proposed and
accepted counts and an acceptance length greater than one; output equality cannot
silently pass through a K=0 fallback. A separate n-gram equality test remains a
smoke check and does not claim that its workload necessarily finds a prompt match.
