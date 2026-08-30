from myvllm.spec_decode.types import (
    Proposal,
    SampleResult,
    VerificationResult,
)


class RejectionSampler:
    """Lossless greedy acceptance. Probability rejection is not implemented."""

    def sample(
        self, proposal: Proposal, verification: VerificationResult
    ) -> SampleResult:
        if proposal.sequence_id != verification.sequence_id:
            raise ValueError("proposal/verification sequence mismatch")
        if len(verification.target_token_ids) != len(proposal.token_ids) + 1:
            raise ValueError("verification must contain K target tokens plus one bonus")

        accepted = 0
        acceptance = [False] * len(proposal.token_ids)
        for position, draft_token in enumerate(proposal.token_ids):
            if draft_token != verification.target_token_ids[position]:
                break
            acceptance[position] = True
            accepted += 1

        emitted = proposal.token_ids[:accepted] + (
            verification.target_token_ids[accepted],
        )
        return SampleResult(
            sequence_id=proposal.sequence_id,
            emitted_token_ids=emitted,
            accepted_proposal_tokens=accepted,
            per_position_acceptance=tuple(acceptance),
            used_bonus_token=accepted == len(proposal.token_ids),
        )
