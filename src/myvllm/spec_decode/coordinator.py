from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from myvllm.engine.block_manager import BlockManager, KVReservation
from myvllm.engine.sequence import Sequence, SequenceStatus, SequenceTransaction
from myvllm.spec_decode.config import SpeculativeConfig
from myvllm.spec_decode.metrics import SpecDecodeMetrics
from myvllm.spec_decode.proposer import Proposer
from myvllm.spec_decode.rejection_sampler import RejectionSampler
from myvllm.spec_decode.types import (
    Proposal,
    SampleResult,
    SpecDecodeBatch,
    SpecDecodeStepResult,
    VerificationResult,
)

VerifyFn = Callable[[SpecDecodeBatch], list[VerificationResult]]


class SpecDecodeCoordinator:
    """Owns the proposer -> verifier -> sampler -> KV transaction chain."""

    def __init__(
        self,
        config: SpeculativeConfig,
        proposer: Proposer,
        rejection_sampler: RejectionSampler | None = None,
        metrics: SpecDecodeMetrics | None = None,
    ):
        self.config = config
        self.proposer = proposer
        self.rejection_sampler = rejection_sampler or RejectionSampler()
        self.metrics = metrics or SpecDecodeMetrics()

    def execute(
        self,
        sequences: list[Sequence],
        block_manager: BlockManager,
        verify: VerifyFn,
        eos_token_id: int,
        max_verification_tokens: int | None = None,
    ) -> list[SpecDecodeStepResult]:
        if any(not seq.is_greedy for seq in sequences):
            raise ValueError("speculative decoding MVP only supports temperature=0")

        started = perf_counter()
        remaining_batch_tokens = (
            None
            if max_verification_tokens is None
            else max(0, max_verification_tokens - len(sequences))
        )
        proposal_budgets = []
        for seq in sequences:
            budget = self._proposal_budget(seq, remaining_batch_tokens)
            proposal_budgets.append(budget)
            if remaining_batch_tokens is not None:
                remaining_batch_tokens -= budget
        proposals = self.proposer.propose_batch(sequences, proposal_budgets)
        if len(proposals) != len(sequences):
            raise ValueError("proposer returned the wrong batch size")
        for seq, budget, proposal in zip(sequences, proposal_budgets, proposals):
            if proposal.sequence_id != seq.seq_id:
                raise ValueError("proposer returned a proposal for another sequence")
            if len(proposal.token_ids) > budget:
                raise ValueError("proposer exceeded its token budget")
        self.metrics.add_stage_time("proposal", perf_counter() - started)

        reservations = block_manager.reserve_batch([(seq, 0) for seq in sequences])
        chosen: list[Proposal] = []
        seq_transactions: list[SequenceTransaction] = []
        try:
            for seq, proposal, reservation in zip(sequences, proposals, reservations):
                try:
                    block_manager.extend_reservation(
                        seq, reservation, len(proposal.token_ids)
                    )
                    selected = proposal
                except MemoryError:
                    selected = Proposal(
                        seq.seq_id, (), {"kind": "decode-fallback", "capacity": True}
                    )
                chosen.append(selected)
                seq_transactions.append(seq.begin_speculation(selected.token_ids))

            batch = SpecDecodeBatch(sequences, chosen)
            started = perf_counter()
            verifications = verify(batch)
            self.metrics.add_stage_time("verify", perf_counter() - started)
            self._validate_verifications(batch, verifications)

            started = perf_counter()
            samples = [
                self.rejection_sampler.sample(proposal, verification)
                for proposal, verification in zip(chosen, verifications)
            ]
            self.metrics.add_stage_time("sample", perf_counter() - started)

            started = perf_counter()
            results = self._commit_batch(
                sequences,
                chosen,
                samples,
                seq_transactions,
                reservations,
                block_manager,
                eos_token_id,
            )
            self.metrics.add_stage_time("commit", perf_counter() - started)
            return results
        except Exception:
            for seq, transaction in zip(sequences, seq_transactions):
                if seq.active_transaction is transaction:
                    seq.rollback(transaction)
            for seq, reservation in zip(sequences, reservations):
                if reservation.state.name == "ACTIVE":
                    block_manager.rollback(seq, reservation)
            raise

    def _proposal_budget(
        self, seq: Sequence, remaining_batch_tokens: int | None = None
    ) -> int:
        remaining = seq.remaining_completion_tokens
        if seq.remaining_model_tokens is not None:
            remaining = min(remaining, seq.remaining_model_tokens)
        # Keep room for the mismatch/bonus target token.
        budget = min(self.config.num_speculative_tokens, max(0, remaining - 1))
        if remaining_batch_tokens is not None:
            budget = min(budget, remaining_batch_tokens)
        return budget

    def _validate_verifications(
        self,
        batch: SpecDecodeBatch,
        results: list[VerificationResult],
    ) -> None:
        if len(results) != len(batch.sequences):
            raise ValueError("verifier returned the wrong batch size")
        for proposal, result in zip(batch.proposals, results):
            if result.sequence_id != proposal.sequence_id:
                raise ValueError("verifier result sequence order/id mismatch")
            if len(result.target_token_ids) != len(proposal.token_ids) + 1:
                raise ValueError("verifier returned the wrong query length")

    def _commit_batch(
        self,
        sequences: list[Sequence],
        proposals: list[Proposal],
        samples: list[SampleResult],
        transactions: list[SequenceTransaction],
        reservations: list[KVReservation],
        block_manager: BlockManager,
        eos_token_id: int,
    ) -> list[SpecDecodeStepResult]:
        plans = []
        for seq, proposal, sample, transaction, reservation in zip(
            sequences, proposals, samples, transactions, reservations
        ):
            emitted, finish_reason = self._truncate(
                seq, sample.emitted_token_ids, eos_token_id
            )
            accepted_kept = min(sample.accepted_proposal_tokens, len(emitted))
            seq.validate_commit(transaction, emitted, accepted_kept)
            future_tokens = seq.token_ids + list(emitted)
            future_computed = min(
                seq.num_computed_tokens + 1 + accepted_kept,
                len(future_tokens),
            )
            block_manager.validate_commit(
                seq, reservation, future_computed, future_tokens
            )
            committed_sample = SampleResult(
                sequence_id=sample.sequence_id,
                emitted_token_ids=tuple(emitted),
                accepted_proposal_tokens=accepted_kept,
                per_position_acceptance=tuple(
                    position < accepted_kept
                    for position in range(len(proposal.token_ids))
                ),
                used_bonus_token=(
                    sample.used_bonus_token
                    and len(emitted) > sample.accepted_proposal_tokens
                ),
            )
            plans.append(
                (
                    seq,
                    proposal,
                    transaction,
                    reservation,
                    emitted,
                    finish_reason,
                    committed_sample,
                )
            )

        # Every Sequence and BlockManager mutation has now been validated. The
        # second pass contains only deterministic state transitions, preventing a
        # later request's validation error from leaving an earlier request committed.
        results = []
        for (
            seq,
            proposal,
            transaction,
            reservation,
            emitted,
            finish_reason,
            committed_sample,
        ) in plans:
            accepted_kept = committed_sample.accepted_proposal_tokens
            seq.commit(transaction, emitted, accepted_kept)
            block_manager.commit(seq, reservation, seq.num_computed_tokens)
            if finish_reason is not None:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = finish_reason
            self.metrics.record(proposal, committed_sample)
            results.append(
                SpecDecodeStepResult(
                    sequence_id=seq.seq_id,
                    emitted_token_ids=tuple(emitted),
                    proposed_tokens=len(proposal.token_ids),
                    accepted_proposal_tokens=accepted_kept,
                    per_position_acceptance=(
                        committed_sample.per_position_acceptance
                    ),
                    used_bonus_token=committed_sample.used_bonus_token,
                    fell_back_to_decode=not proposal.token_ids,
                )
            )
        return results

    def _truncate(
        self, seq: Sequence, emitted: tuple[int, ...], eos_token_id: int
    ) -> tuple[list[int], str | None]:
        limit = seq.remaining_completion_tokens
        if seq.remaining_model_tokens is not None:
            limit = min(limit, seq.remaining_model_tokens)
        kept: list[int] = []
        finish_reason = "length" if limit == 0 else None
        for token_id in emitted[:limit]:
            kept.append(token_id)
            if not seq.ignore_eos and token_id == eos_token_id:
                finish_reason = "eos"
                break
        if len(kept) >= limit:
            finish_reason = finish_reason or "length"
        return kept, finish_reason
