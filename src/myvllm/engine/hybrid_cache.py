from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Hashable

import torch


RequestId = Hashable


class HybridTransactionState(Enum):
    ACTIVE = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()


@dataclass
class LinearLayerState:
    recurrent_state: torch.Tensor
    conv_state: torch.Tensor

    def clone(self) -> "LinearLayerState":
        return LinearLayerState(
            recurrent_state=self.recurrent_state.clone(),
            conv_state=self.conv_state.clone(),
        )


@dataclass
class HybridRequestState:
    request_id: RequestId
    logical_length: int = 0
    full_attention_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    linear_layers: dict[int, LinearLayerState] = field(default_factory=dict)

    def clone(self) -> "HybridRequestState":
        return HybridRequestState(
            request_id=self.request_id,
            logical_length=self.logical_length,
            full_attention_layers={
                layer_id: (key.clone(), value.clone())
                for layer_id, (key, value) in self.full_attention_layers.items()
            },
            linear_layers={
                layer_id: state.clone()
                for layer_id, state in self.linear_layers.items()
            },
        )


@dataclass
class HybridCacheTransaction:
    transaction_id: int
    request_ids: tuple[RequestId, ...]
    reserve_tokens: dict[RequestId, int]
    original_states: dict[RequestId, HybridRequestState]
    snapshots: dict[RequestId, dict[int, HybridRequestState]] = field(
        default_factory=dict
    )
    state: HybridTransactionState = HybridTransactionState.ACTIVE


class HybridCacheManager:
    """Owns recurrent/conv cache state for Qwen3.5 hybrid text layers.

    The existing BlockManager already handles paged full-attention KV slots.  This
    manager covers the part that cannot be repaired by trimming sequence length:
    linear-attention state after each proposed token.  During verification callers
    record a snapshot after every provisional token, then commit a different
    accepted length for each request or roll the whole batch back.
    """

    def __init__(self) -> None:
        self._states: dict[RequestId, HybridRequestState] = {}
        self._active: dict[int, HybridCacheTransaction] = {}
        self._next_transaction_id = 0

    def ensure_request(self, request_id: RequestId) -> HybridRequestState:
        if request_id not in self._states:
            self._states[request_id] = HybridRequestState(request_id=request_id)
        return self._states[request_id]

    def get_request_state(self, request_id: RequestId) -> HybridRequestState:
        return self.ensure_request(request_id)

    def set_linear_layer_state(
        self,
        request_id: RequestId,
        layer_id: int,
        recurrent_state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> None:
        request_state = self.ensure_request(request_id)
        request_state.linear_layers[layer_id] = LinearLayerState(
            recurrent_state=recurrent_state.clone(),
            conv_state=conv_state.clone(),
        )

    def begin_transaction(
        self,
        request_ids: list[RequestId] | tuple[RequestId, ...],
        reserve_tokens: int | dict[RequestId, int],
    ) -> HybridCacheTransaction:
        request_ids = tuple(request_ids)
        if not request_ids:
            raise ValueError("begin_transaction requires at least one request")
        if isinstance(reserve_tokens, int):
            if reserve_tokens < 0:
                raise ValueError("reserve_tokens must be non-negative")
            reserve_map = {request_id: reserve_tokens for request_id in request_ids}
        else:
            reserve_map = dict(reserve_tokens)
            missing = [rid for rid in request_ids if rid not in reserve_map]
            if missing:
                raise ValueError(f"missing reserve_tokens for requests: {missing}")
            if any(value < 0 for value in reserve_map.values()):
                raise ValueError("reserve_tokens must be non-negative")

        transaction_id = self._next_transaction_id
        self._next_transaction_id += 1
        transaction = HybridCacheTransaction(
            transaction_id=transaction_id,
            request_ids=request_ids,
            reserve_tokens=reserve_map,
            original_states={
                request_id: self.ensure_request(request_id).clone()
                for request_id in request_ids
            },
            snapshots={request_id: {} for request_id in request_ids},
        )
        self._active[transaction_id] = transaction
        return transaction

    def record_snapshot(
        self,
        transaction: HybridCacheTransaction,
        request_id: RequestId,
        tokens_executed: int,
        state: HybridRequestState | None = None,
    ) -> None:
        self._require_active(transaction)
        if request_id not in transaction.request_ids:
            raise ValueError("request does not belong to this transaction")
        if not 1 <= tokens_executed <= transaction.reserve_tokens[request_id]:
            raise ValueError("snapshot token index is outside the reservation")
        snapshot = (state or self.ensure_request(request_id)).clone()
        snapshot.logical_length = (
            transaction.original_states[request_id].logical_length + tokens_executed
        )
        transaction.snapshots[request_id][tokens_executed] = snapshot

    def commit(
        self,
        transaction: HybridCacheTransaction,
        accepted_lengths_per_request: dict[RequestId, int],
    ) -> None:
        self._require_active(transaction)
        for request_id in transaction.request_ids:
            if request_id not in accepted_lengths_per_request:
                raise ValueError(f"missing accepted length for request {request_id}")
            accepted = accepted_lengths_per_request[request_id]
            if not 0 <= accepted <= transaction.reserve_tokens[request_id]:
                raise ValueError("accepted length is outside the reservation")
            if accepted == 0:
                self._states[request_id] = transaction.original_states[
                    request_id
                ].clone()
                continue
            try:
                self._states[request_id] = transaction.snapshots[request_id][
                    accepted
                ].clone()
            except KeyError as exc:
                raise RuntimeError(
                    "cannot commit a linear-attention state that was not recorded"
                ) from exc
        transaction.state = HybridTransactionState.COMMITTED
        del self._active[transaction.transaction_id]

    def rollback(self, transaction: HybridCacheTransaction) -> None:
        self._require_active(transaction)
        for request_id, state in transaction.original_states.items():
            self._states[request_id] = state.clone()
        transaction.state = HybridTransactionState.ROLLED_BACK
        del self._active[transaction.transaction_id]

    def release_request(self, request_id: RequestId) -> None:
        self._states.pop(request_id, None)

    def _require_active(self, transaction: HybridCacheTransaction) -> None:
        if transaction.state is not HybridTransactionState.ACTIVE:
            raise RuntimeError("hybrid cache transaction has already been finalized")
        if self._active.get(transaction.transaction_id) is not transaction:
            raise RuntimeError("hybrid cache transaction is not active")
