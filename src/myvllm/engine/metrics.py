from __future__ import annotations

from collections import deque
from dataclasses import asdict

from myvllm.engine.output import SpeculativeBatchObservation


class EngineMetrics:
    """Engine-step metrics; coordinator metrics remain algorithm-stage totals."""

    def __init__(self, max_step_history: int = 256):
        self.engine_steps = 0
        self.decode_steps = 0
        self.proposed_tokens = 0
        self.accepted_tokens = 0
        self.per_position_proposed: list[int] = []
        self.per_position_accepted: list[int] = []
        self.step_history: deque[SpeculativeBatchObservation] = deque(
            maxlen=max_step_history
        )

    def next_engine_step(self) -> int:
        self.engine_steps += 1
        return self.engine_steps

    def record_decode(self, observation: SpeculativeBatchObservation) -> None:
        self.decode_steps += 1
        self.proposed_tokens += observation.proposed_tokens
        self.accepted_tokens += observation.accepted_tokens
        while len(self.per_position_proposed) < len(observation.per_position_proposed):
            self.per_position_proposed.append(0)
            self.per_position_accepted.append(0)
        for position, proposed in enumerate(observation.per_position_proposed):
            self.per_position_proposed[position] += proposed
            self.per_position_accepted[position] += observation.per_position_accepted[
                position
            ]
        self.step_history.append(observation)

    def snapshot(self, *, include_history: bool = True) -> dict:
        return {
            "engine_steps": self.engine_steps,
            "decode_steps": self.decode_steps,
            "proposed_tokens": self.proposed_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": (
                self.accepted_tokens / self.proposed_tokens
                if self.proposed_tokens
                else 0.0
            ),
            "per_position_proposed": list(self.per_position_proposed),
            "per_position_accepted": list(self.per_position_accepted),
            "recent_decode_steps": [
                asdict(observation) for observation in self.step_history
            ] if include_history else [],
        }
