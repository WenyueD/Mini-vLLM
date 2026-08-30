from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count

from myvllm.sampling_parameters import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class TransactionState(Enum):
    ACTIVE = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()


@dataclass
class SequenceTransaction:
    transaction_id: int
    sequence_id: int
    original_num_tokens: int
    original_num_computed_tokens: int
    provisional_token_ids: tuple[int, ...]
    state: TransactionState = TransactionState.ACTIVE


class Sequence:
    """Logical request state, including speculative transaction boundaries.

    ``token_ids`` contains committed tokens only. Target KV is valid for the
    leading ``num_computed_tokens`` tokens, while ``provisional_token_ids`` is
    populated only during one active speculative transaction.
    """

    counter = count()
    transaction_counter = count()

    def __init__(
        self,
        token_ids: list[int],
        block_size: int = 4,
        sampling_params: SamplingParams | None = None,
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        sampling_params = sampling_params or SamplingParams()
        self.block_size = block_size
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = self.token_ids[-1] if self.token_ids else None
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.num_computed_tokens = 0
        self.provisional_token_ids: list[int] = []
        self._active_transaction: SequenceTransaction | None = None
        self._compact_worker_state = False
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        return self.blocks_for(self.num_tokens)

    @property
    def num_reserved_tokens(self) -> int:
        return self.num_tokens + len(self.provisional_token_ids)

    @property
    def num_reserved_blocks(self) -> int:
        return self.blocks_for(self.num_reserved_tokens)

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - max(self.num_blocks - 1, 0) * self.block_size

    @property
    def active_transaction(self) -> SequenceTransaction | None:
        return self._active_transaction

    @property
    def remaining_completion_tokens(self) -> int:
        return max(0, self.max_tokens - self.num_completion_tokens)

    @property
    def remaining_model_tokens(self) -> int | None:
        if self.max_model_length is None:
            return None
        return max(0, self.max_model_length - self.num_tokens)

    def blocks_for(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size) if num_tokens else 0

    def block(self, i):
        assert 0 <= i < self.num_blocks, (
            f"Block index {i} out of range [0, {self.num_blocks})"
        )
        start_idx = i * self.block_size
        return self.token_ids[start_idx : start_idx + self.block_size]

    def append_token(self, token_id: int) -> None:
        self.append_tokens([token_id])

    def append_tokens(self, token_ids: list[int] | tuple[int, ...]) -> None:
        if self._active_transaction is not None:
            raise RuntimeError("cannot append committed tokens during speculation")
        self.token_ids.extend(token_ids)
        self.num_tokens += len(token_ids)
        self.last_token = self.token_ids[-1] if self.token_ids else None
        self._check_invariants()

    def mark_computed(self, num_tokens: int) -> None:
        """Mark an absolute committed prefix length as having valid target KV."""
        if not self.num_computed_tokens <= num_tokens <= self.num_tokens:
            raise ValueError(
                f"computed prefix must advance within committed tokens: "
                f"{self.num_computed_tokens} <= {num_tokens} <= {self.num_tokens}"
            )
        self.num_computed_tokens = num_tokens
        self._check_invariants()

    def begin_speculation(
        self, provisional_token_ids: list[int] | tuple[int, ...]
    ) -> SequenceTransaction:
        if self._active_transaction is not None:
            raise RuntimeError("sequence already has an active transaction")
        if self.num_tokens - self.num_computed_tokens != 1:
            raise RuntimeError(
                "speculative decode requires exactly one committed pending token"
            )
        transaction = SequenceTransaction(
            transaction_id=next(Sequence.transaction_counter),
            sequence_id=self.seq_id,
            original_num_tokens=self.num_tokens,
            original_num_computed_tokens=self.num_computed_tokens,
            provisional_token_ids=tuple(provisional_token_ids),
        )
        self.provisional_token_ids = list(provisional_token_ids)
        self._active_transaction = transaction
        self._check_invariants()
        return transaction

    def validate_commit(
        self,
        transaction: SequenceTransaction,
        emitted_token_ids: list[int] | tuple[int, ...],
        accepted_proposal_tokens: int,
    ) -> None:
        self._require_active(transaction)
        emitted = tuple(emitted_token_ids)
        if not 0 <= accepted_proposal_tokens <= len(transaction.provisional_token_ids):
            raise ValueError("accepted proposal length is out of range")
        expected = transaction.provisional_token_ids[:accepted_proposal_tokens]
        if emitted[:accepted_proposal_tokens] != expected:
            raise ValueError("committed accepted tokens differ from the proposal")
        if len(emitted) < accepted_proposal_tokens:
            raise ValueError("emitted tokens cannot omit accepted proposal tokens")

    def commit(
        self,
        transaction: SequenceTransaction,
        emitted_token_ids: list[int] | tuple[int, ...],
        accepted_proposal_tokens: int,
    ) -> None:
        self.validate_commit(transaction, emitted_token_ids, accepted_proposal_tokens)
        emitted = list(emitted_token_ids)
        computed_increment = 1 + accepted_proposal_tokens
        self.token_ids.extend(emitted)
        self.num_tokens += len(emitted)
        self.num_computed_tokens += computed_increment
        self.num_computed_tokens = min(self.num_computed_tokens, self.num_tokens)
        self.last_token = self.token_ids[-1] if self.token_ids else None
        self.provisional_token_ids = []
        self._active_transaction = None
        transaction.state = TransactionState.COMMITTED
        self._check_invariants()

    def rollback(self, transaction: SequenceTransaction) -> None:
        self._require_active(transaction)
        self.provisional_token_ids = []
        self._active_transaction = None
        transaction.state = TransactionState.ROLLED_BACK
        self._check_invariants()

    def _require_active(self, transaction: SequenceTransaction) -> None:
        if transaction.sequence_id != self.seq_id:
            raise ValueError("transaction belongs to another sequence")
        if transaction.state is not TransactionState.ACTIVE:
            raise RuntimeError("transaction has already been finalized")
        if self._active_transaction is not transaction:
            raise RuntimeError("transaction is not active on this sequence")

    def _check_invariants(self) -> None:
        if self._compact_worker_state:
            return
        if self.num_tokens != len(self.token_ids):
            raise AssertionError("num_tokens disagrees with committed token_ids")
        if not 0 <= self.num_computed_tokens <= self.num_tokens:
            raise AssertionError("computed tokens must be a committed prefix")
        if self._active_transaction is None and self.provisional_token_ids:
            raise AssertionError("provisional tokens require an active transaction")

    def __getstate__(self):
        is_prefill = self.num_completion_tokens == 0
        # Workers need full IDs only for prefill. Decode and verification use the
        # single pending token plus separately transported provisional tokens.
        token_payload = self.token_ids if is_prefill else [self.last_token]
        return {
            "block_size": self.block_size,
            "seq_id": self.seq_id,
            "status": self.status,
            "token_ids": token_payload,
            "last_token": self.last_token,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "num_computed_tokens": self.num_computed_tokens,
            "provisional_token_ids": self.provisional_token_ids,
            "block_table": self.block_table,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
            "max_model_length": self.max_model_length,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._active_transaction = None
        self._compact_worker_state = len(self.token_ids) != self.num_tokens
