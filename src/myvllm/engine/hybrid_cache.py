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


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


@dataclass
class FullAttentionLayerState:
    key: torch.Tensor
    value: torch.Tensor
    logical_length: int

    @classmethod
    def from_tensors(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        capacity: int | None = None,
    ) -> "FullAttentionLayerState":
        if key.shape != value.shape or key.dim() != 4 or key.shape[0] != 1:
            raise ValueError(
                "full-attention K/V must be equal 4D [1, length, heads, dim]"
            )
        length = int(key.shape[1])
        capacity = max(length, length if capacity is None else int(capacity))
        stored_key = key.new_empty((key.shape[0], capacity, *key.shape[2:]))
        stored_value = value.new_empty((value.shape[0], capacity, *value.shape[2:]))
        if length:
            stored_key[:, :length].copy_(key.detach())
            stored_value[:, :length].copy_(value.detach())
        return cls(stored_key, stored_value, length)

    def clone(self) -> "FullAttentionLayerState":
        return FullAttentionLayerState.from_tensors(*self.tensors())

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.key[:, : self.logical_length],
            self.value[:, : self.logical_length],
        )

    def append(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key.shape != value.shape or key.dim() != 4 or key.shape[0] != 1:
            raise ValueError(
                "full-attention append K/V must be equal 4D [1, length, heads, dim]"
            )
        if key.shape[2:] != self.key.shape[2:]:
            raise ValueError("full-attention append K/V head shape changed")
        append_length = int(key.shape[1])
        new_length = self.logical_length + append_length
        self._ensure_capacity(new_length)
        if append_length:
            self.key[:, self.logical_length : new_length].copy_(key.detach())
            self.value[:, self.logical_length : new_length].copy_(value.detach())
        self.logical_length = new_length
        return self.tensors()

    def _ensure_capacity(self, needed: int) -> None:
        current = int(self.key.shape[1])
        if current >= needed:
            return
        new_capacity = max(needed, max(1, current * 2))
        new_key = self.key.new_empty(
            (self.key.shape[0], new_capacity, *self.key.shape[2:])
        )
        new_value = self.value.new_empty(
            (self.value.shape[0], new_capacity, *self.value.shape[2:])
        )
        if self.logical_length:
            new_key[:, : self.logical_length].copy_(self.key[:, : self.logical_length])
            new_value[:, : self.logical_length].copy_(
                self.value[:, : self.logical_length]
            )
        self.key = new_key
        self.value = new_value

    def nbytes(self, *, allocated_capacity: bool = True) -> int:
        if allocated_capacity:
            return tensor_nbytes(self.key) + tensor_nbytes(self.value)
        key, value = self.tensors()
        return tensor_nbytes(key) + tensor_nbytes(value)

    def __iter__(self):
        return iter(self.tensors())


@dataclass
class LinearLayerState:
    recurrent_state: torch.Tensor
    conv_state: torch.Tensor

    def clone(self) -> "LinearLayerState":
        return LinearLayerState(
            recurrent_state=self.recurrent_state.clone(),
            conv_state=self.conv_state.clone(),
        )

    def nbytes(self) -> int:
        return tensor_nbytes(self.recurrent_state) + tensor_nbytes(self.conv_state)


@dataclass
class HybridRequestState:
    request_id: RequestId
    logical_length: int = 0
    full_attention_layers: dict[int, FullAttentionLayerState] = field(
        default_factory=dict
    )
    linear_layers: dict[int, LinearLayerState] = field(default_factory=dict)

    def clone(self) -> "HybridRequestState":
        return HybridRequestState(
            request_id=self.request_id,
            logical_length=self.logical_length,
            full_attention_layers={
                layer_id: _clone_full_attention_state(state)
                for layer_id, state in self.full_attention_layers.items()
            },
            linear_layers={
                layer_id: state.clone()
                for layer_id, state in self.linear_layers.items()
            },
        )

    def nbytes(self, *, allocated_capacity: bool = True) -> dict[str, int]:
        full_attention = sum(
            _full_attention_state_nbytes(state, allocated_capacity=allocated_capacity)
            for state in self.full_attention_layers.values()
        )
        linear_recurrent = sum(
            tensor_nbytes(state.recurrent_state)
            for state in self.linear_layers.values()
        )
        linear_conv = sum(
            tensor_nbytes(state.conv_state) for state in self.linear_layers.values()
        )
        return {
            "full_attention_kv_bytes": full_attention,
            "linear_recurrent_bytes": linear_recurrent,
            "linear_conv_bytes": linear_conv,
            "total_bytes": full_attention + linear_recurrent + linear_conv,
        }


def _clone_full_attention_state(
    state: FullAttentionLayerState | tuple[torch.Tensor, torch.Tensor],
) -> FullAttentionLayerState:
    if isinstance(state, FullAttentionLayerState):
        return state.clone()
    key, value = state
    return FullAttentionLayerState.from_tensors(key, value)


def _full_attention_state_nbytes(
    state: FullAttentionLayerState | tuple[torch.Tensor, torch.Tensor],
    *,
    allocated_capacity: bool,
) -> int:
    if isinstance(state, FullAttentionLayerState):
        return state.nbytes(allocated_capacity=allocated_capacity)
    key, value = state
    return tensor_nbytes(key) + tensor_nbytes(value)


@dataclass
class LinearStateUpdate:
    decay: torch.Tensor
    key: torch.Tensor
    delta: torch.Tensor
    conv_state: torch.Tensor

    def nbytes(self) -> int:
        return sum(tensor_nbytes(value) for value in (
            self.decay, self.key, self.delta, self.conv_state,
        ))


@dataclass
class HybridCacheTransaction:
    transaction_id: int
    request_ids: tuple[RequestId, ...]
    reserve_tokens: dict[RequestId, int]
    original_states: dict[RequestId, HybridRequestState]
    snapshots: dict[RequestId, dict[int, HybridRequestState]] = field(
        default_factory=dict
    )
    recorded_steps: dict[RequestId, set[int]] = field(default_factory=dict)
    state: HybridTransactionState = HybridTransactionState.ACTIVE
    linear_updates: dict[RequestId, dict[int, list[LinearStateUpdate]]] = field(
        default_factory=dict
    )


class HybridCacheManager:
    """Owns recurrent/conv cache state for Qwen3.5 hybrid text layers.

    Transactions keep the original state for rollback. Verification can record
    compact linear-state updates to restore an accepted prefix while truncating
    full-attention KV. Snapshot-based commit remains available to legacy callers.
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
            recorded_steps={request_id: set() for request_id in request_ids},
            linear_updates={request_id: {} for request_id in request_ids},
        )
        self._active[transaction_id] = transaction
        return transaction

    def record_step(
        self,
        transaction: HybridCacheTransaction,
        request_id: RequestId,
        tokens_executed: int,
    ) -> None:
        self._require_active(transaction)
        if request_id not in transaction.request_ids:
            raise ValueError("request does not belong to this transaction")
        if not 1 <= tokens_executed <= transaction.reserve_tokens[request_id]:
            raise ValueError("snapshot token index is outside the reservation")
        transaction.recorded_steps[request_id].add(tokens_executed)

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
        transaction.recorded_steps[request_id].add(tokens_executed)
        snapshot = (state or self.ensure_request(request_id)).clone()
        snapshot.logical_length = (
            transaction.original_states[request_id].logical_length + tokens_executed
        )
        transaction.snapshots[request_id][tokens_executed] = snapshot

    def restore_original(
        self,
        transaction: HybridCacheTransaction,
        request_ids: list[RequestId] | tuple[RequestId, ...] | None = None,
    ) -> None:
        self._require_active(transaction)
        selected = (
            transaction.request_ids if request_ids is None else tuple(request_ids)
        )
        for request_id in selected:
            if request_id not in transaction.original_states:
                raise ValueError("request does not belong to this transaction")
            self._states[request_id] = transaction.original_states[request_id].clone()

    def restore_verified_prefix(
        self,
        transaction: HybridCacheTransaction,
        request_id: RequestId,
        accepted: int,
    ) -> bool:
        """Restore a prefix using recorded rank-one updates, without model replay."""
        self._require_active(transaction)
        if request_id not in transaction.request_ids:
            raise ValueError("request does not belong to this transaction")
        if not 0 <= accepted <= transaction.reserve_tokens[request_id]:
            raise ValueError("accepted length is outside the reservation")
        if accepted == 0:
            self.restore_original(transaction, [request_id])
            return True
        state = self.ensure_request(request_id)
        original = transaction.original_states[request_id]
        updates = transaction.linear_updates.get(request_id, {})
        if any(len(updates.get(layer_id, ())) < accepted for layer_id in state.linear_layers):
            return False
        if accepted not in transaction.recorded_steps[request_id]:
            raise RuntimeError("cannot restore an unverified cache prefix")
        length = original.logical_length + accepted
        if state.logical_length < length:
            raise RuntimeError("verified cache is shorter than the accepted prefix")
        restored_linear = {}
        for layer_id, current in state.linear_layers.items():
            previous = original.linear_layers.get(layer_id)
            recurrent = (
                torch.zeros_like(current.recurrent_state)
                if previous is None else previous.recurrent_state
            )
            for update in updates[layer_id][:accepted]:
                # Preserve the two FP32 operations used during verification.
                recurrent = recurrent * update.decay
                recurrent = recurrent + update.key.unsqueeze(-1) * update.delta.unsqueeze(-2)
            restored_linear[layer_id] = LinearLayerState(
                recurrent, updates[layer_id][accepted - 1].conv_state.clone(),
            )
        for layer_id, kv in state.full_attention_layers.items():
            if not isinstance(kv, FullAttentionLayerState):
                kv = FullAttentionLayerState.from_tensors(*kv)
                state.full_attention_layers[layer_id] = kv
            if kv.logical_length < length:
                raise RuntimeError("verified KV cache is shorter than the accepted prefix")
            kv.logical_length = length
        state.linear_layers = restored_linear
        state.logical_length = length
        return True

    def finalize_replayed_commit(
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
            expected_length = (
                transaction.original_states[request_id].logical_length + accepted
            )
            actual_length = self.ensure_request(request_id).logical_length
            if actual_length != expected_length:
                raise RuntimeError(
                    "replayed hybrid cache commit has the wrong logical length"
                )
        transaction.state = HybridTransactionState.COMMITTED
        del self._active[transaction.transaction_id]

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
                snapshot = transaction.snapshots[request_id][accepted]
            except KeyError as exc:
                raise RuntimeError(
                    "cannot commit a linear-attention state that was not recorded"
                ) from exc
            self._states[request_id] = snapshot.clone()
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

    def estimate_bytes(self) -> dict[str, int]:
        state_totals = {
            "full_attention_kv_bytes": 0,
            "linear_recurrent_bytes": 0,
            "linear_conv_bytes": 0,
            "total_bytes": 0,
        }
        for state in self._states.values():
            breakdown = state.nbytes()
            for key, value in breakdown.items():
                state_totals[key] += value

        original_bytes = 0
        snapshot_bytes = 0
        update_bytes = 0
        recorded_steps = 0
        for transaction in self._active.values():
            update_bytes += sum(
                update.nbytes()
                for layers in transaction.linear_updates.values()
                for updates in layers.values()
                for update in updates
            )
            original_bytes += sum(
                state.nbytes()["total_bytes"]
                for state in transaction.original_states.values()
            )
            snapshot_bytes += sum(
                snapshot.nbytes()["total_bytes"]
                for snapshots in transaction.snapshots.values()
                for snapshot in snapshots.values()
            )
            recorded_steps += sum(
                len(steps) for steps in transaction.recorded_steps.values()
            )
        return {
            "active_requests": len(self._states),
            "logical_tokens": sum(
                state.logical_length for state in self._states.values()
            ),
            "state_full_attention_kv_bytes": state_totals["full_attention_kv_bytes"],
            "state_linear_recurrent_bytes": state_totals["linear_recurrent_bytes"],
            "state_linear_conv_bytes": state_totals["linear_conv_bytes"],
            "state_total_bytes": state_totals["total_bytes"],
            "active_transactions": len(self._active),
            "transaction_original_state_bytes": original_bytes,
            "transaction_snapshot_state_bytes": snapshot_bytes,
            "transaction_linear_update_bytes": update_bytes,
            "transaction_recorded_steps": recorded_steps,
            "total_bytes": state_totals["total_bytes"]
            + original_bytes
            + snapshot_bytes
            + update_bytes,
        }

    def _require_active(self, transaction: HybridCacheTransaction) -> None:
        if transaction.state is not HybridTransactionState.ACTIVE:
            raise RuntimeError("hybrid cache transaction has already been finalized")
        if self._active.get(transaction.transaction_id) is not transaction:
            raise RuntimeError("hybrid cache transaction is not active")
