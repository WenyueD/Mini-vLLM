from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from myvllm.engine.sequence import Sequence
from myvllm.spec_decode.types import Proposal


class Proposer(ABC):
    """Stable algorithm boundary for serial or future parallel proposers."""

    supports_parallel: bool = False

    @abstractmethod
    def propose(self, sequence: Sequence, max_tokens: int) -> Proposal:
        raise NotImplementedError

    def propose_batch(
        self, sequences: list[Sequence], max_tokens: list[int]
    ) -> list[Proposal]:
        """Batch extension point for neural/EAGLE/DFlash-style proposers.

        CPU proposers use the default loop. A parallel proposer overrides this
        method and may execute one batched draft forward without runtime changes.
        """
        if len(sequences) != len(max_tokens):
            raise ValueError("sequence and proposer budget counts must match")
        return [
            self.propose(sequence, budget)
            for sequence, budget in zip(sequences, max_tokens)
        ]


class MockProposer(Proposer):
    def __init__(
        self,
        proposals: Mapping[int, list[int] | tuple[int, ...]] | None = None,
        default: list[int] | tuple[int, ...] = (),
    ):
        self.proposals = dict(proposals or {})
        self.default = tuple(default)

    def propose(self, sequence: Sequence, max_tokens: int) -> Proposal:
        tokens = tuple(self.proposals.get(sequence.seq_id, self.default))[:max_tokens]
        return Proposal(sequence.seq_id, tokens, {"kind": "mock"})


class NGramProposer(Proposer):
    """Prompt lookup proposer using the longest, most recent matching suffix."""

    def __init__(self, min_ngram_size: int = 2, max_ngram_size: int = 4):
        if min_ngram_size <= 0 or max_ngram_size < min_ngram_size:
            raise ValueError("invalid n-gram size range")
        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size

    def propose(self, sequence: Sequence, max_tokens: int) -> Proposal:
        history = sequence.token_ids
        if max_tokens <= 0:
            return Proposal(sequence.seq_id, (), {"kind": "ngram", "match": 0})
        largest = min(self.max_ngram_size, len(history) - 1)
        for n in range(largest, self.min_ngram_size - 1, -1):
            suffix = history[-n:]
            for start in range(len(history) - n - 1, -1, -1):
                if history[start : start + n] == suffix:
                    continuation = tuple(history[start + n : start + n + max_tokens])
                    if continuation:
                        return Proposal(
                            sequence.seq_id,
                            continuation,
                            {"kind": "ngram", "match": n, "start": start},
                        )
        return Proposal(sequence.seq_id, (), {"kind": "ngram", "match": 0})
