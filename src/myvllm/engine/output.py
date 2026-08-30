from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestOutput:
    """Tokens committed for one request by exactly one engine step."""

    request_id: str
    token_ids: tuple[int, ...]
    finished: bool = False
    finish_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SpeculativeBatchObservation:
    """One atomic observation for one decode model batch."""

    engine_step: int
    active_batch_size: int
    speculative_batch_size: int
    proposed_tokens: int
    accepted_tokens: int
    batch_acceptance_rate: float
    per_request_accepted_length: dict[str, int] = field(default_factory=dict)
    per_position_proposed: tuple[int, ...] = ()
    per_position_accepted: tuple[int, ...] = ()


@dataclass(frozen=True)
class EngineStepResult:
    engine_step: int
    outputs: tuple[RequestOutput, ...]
    num_processed_tokens: int
    is_prefill: bool
    speculative: SpeculativeBatchObservation | None = None
