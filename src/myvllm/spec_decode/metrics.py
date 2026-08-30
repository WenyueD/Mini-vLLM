from __future__ import annotations

from dataclasses import asdict, dataclass, field

from myvllm.spec_decode.types import Proposal, SampleResult


@dataclass
class SpecDecodeMetrics:
    steps: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    acceptance_lengths: list[int] = field(default_factory=list)
    per_position_accepted: list[int] = field(default_factory=list)
    per_position_proposed: list[int] = field(default_factory=list)
    stage_seconds: dict[str, float] = field(
        default_factory=lambda: {
            "proposal": 0.0,
            "verify": 0.0,
            "sample": 0.0,
            "commit": 0.0,
        }
    )

    def add_stage_time(self, stage: str, seconds: float) -> None:
        self.stage_seconds[stage] += seconds

    def record(self, proposal: Proposal, sample: SampleResult) -> None:
        self.steps += 1
        self.proposed_tokens += len(proposal.token_ids)
        self.accepted_tokens += sample.accepted_proposal_tokens
        self.acceptance_lengths.append(sample.accepted_proposal_tokens)
        while len(self.per_position_accepted) < len(proposal.token_ids):
            self.per_position_accepted.append(0)
            self.per_position_proposed.append(0)
        for position, accepted in enumerate(sample.per_position_acceptance):
            self.per_position_proposed[position] += 1
            self.per_position_accepted[position] += int(accepted)

    def snapshot(self) -> dict:
        result = asdict(self)
        result["acceptance_rate"] = (
            self.accepted_tokens / self.proposed_tokens if self.proposed_tokens else 0.0
        )
        result["per_position_acceptance_rate"] = [
            accepted / proposed if proposed else 0.0
            for accepted, proposed in zip(
                self.per_position_accepted, self.per_position_proposed
            )
        ]
        return result
