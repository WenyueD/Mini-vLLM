from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from myvllm.engine.sequence import Sequence


@dataclass(frozen=True)
class Proposal:
    sequence_id: int
    token_ids: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class SpecDecodeBatch:
    sequences: list[Sequence]
    proposals: list[Proposal]

    def __post_init__(self) -> None:
        if len(self.sequences) != len(self.proposals):
            raise ValueError("sequences and proposals must have equal length")
        for seq, proposal in zip(self.sequences, self.proposals):
            if seq.seq_id != proposal.sequence_id:
                raise ValueError("proposal sequence order/id mismatch")

    @property
    def query_lengths(self) -> tuple[int, ...]:
        return tuple(len(proposal.token_ids) + 1 for proposal in self.proposals)


@dataclass(frozen=True)
class VerificationResult:
    sequence_id: int
    target_token_ids: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class SampleResult:
    sequence_id: int
    emitted_token_ids: tuple[int, ...]
    accepted_proposal_tokens: int
    per_position_acceptance: tuple[bool, ...]
    used_bonus_token: bool


@dataclass(frozen=True)
class SpecDecodeStepResult:
    sequence_id: int
    emitted_token_ids: tuple[int, ...]
    proposed_tokens: int
    accepted_proposal_tokens: int
    per_position_acceptance: tuple[bool, ...]
    used_bonus_token: bool
    fell_back_to_decode: bool
