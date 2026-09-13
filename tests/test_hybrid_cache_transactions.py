import pytest
import torch

from myvllm.engine.hybrid_cache import (
    HybridCacheManager,
    HybridTransactionState,
)


def seed_state(manager, request_id, value, logical_length=10):
    state = manager.ensure_request(request_id)
    state.logical_length = logical_length
    manager.set_linear_layer_state(
        request_id,
        layer_id=0,
        recurrent_state=torch.full((1, 2, 3, 4), float(value)),
        conv_state=torch.full((1, 8, 3), float(value + 100)),
    )
    return manager.get_request_state(request_id).clone()


def record_all_snapshots(manager, tx, request_id, reserve_tokens):
    for step in range(1, reserve_tokens + 1):
        state = manager.get_request_state(request_id).clone()
        state.logical_length = tx.original_states[request_id].logical_length + step
        state.linear_layers[0].recurrent_state.fill_(step)
        state.linear_layers[0].conv_state.fill_(step + 100)
        manager.record_snapshot(tx, request_id, step, state)


def test_hybrid_cache_commits_different_accepted_lengths_per_request():
    manager = HybridCacheManager()
    seed_state(manager, "a", 10)
    seed_state(manager, "b", 20)
    seed_state(manager, "c", 30)
    tx = manager.begin_transaction(["a", "b", "c"], {"a": 7, "b": 7, "c": 7})
    for request_id in ["a", "b", "c"]:
        record_all_snapshots(manager, tx, request_id, 7)

    manager.commit(tx, {"a": 0, "b": 3, "c": 7})

    assert tx.state is HybridTransactionState.COMMITTED
    assert manager.get_request_state("a").logical_length == 10
    assert manager.get_request_state("b").logical_length == 13
    assert manager.get_request_state("c").logical_length == 17
    assert (
        manager.get_request_state("a").linear_layers[0].recurrent_state.unique().item()
        == 10
    )
    assert (
        manager.get_request_state("b").linear_layers[0].recurrent_state.unique().item()
        == 3
    )
    assert (
        manager.get_request_state("c").linear_layers[0].conv_state.unique().item()
        == 107
    )


def test_hybrid_cache_rollback_restores_recurrent_and_conv_state():
    manager = HybridCacheManager()
    before = seed_state(manager, "r0", 5, logical_length=4)
    tx = manager.begin_transaction(["r0"], reserve_tokens=7)
    record_all_snapshots(manager, tx, "r0", 7)

    manager.rollback(tx)

    after = manager.get_request_state("r0")
    assert tx.state is HybridTransactionState.ROLLED_BACK
    assert after.logical_length == before.logical_length
    assert torch.equal(
        after.linear_layers[0].recurrent_state,
        before.linear_layers[0].recurrent_state,
    )
    assert torch.equal(
        after.linear_layers[0].conv_state, before.linear_layers[0].conv_state
    )


def test_hybrid_cache_commit_requires_recorded_snapshot_for_accepted_tail():
    manager = HybridCacheManager()
    seed_state(manager, "r0", 1)
    tx = manager.begin_transaction(["r0"], reserve_tokens=7)
    with pytest.raises(RuntimeError, match="not recorded"):
        manager.commit(tx, {"r0": 7})


def test_hybrid_cache_replayed_commit_keeps_per_step_records_lightweight():
    manager = HybridCacheManager()
    before = seed_state(manager, "r0", 5, logical_length=4)
    tx = manager.begin_transaction(["r0"], reserve_tokens=7)

    for step in range(1, 8):
        state = manager.get_request_state("r0")
        state.logical_length = before.logical_length + step
        state.linear_layers[0].recurrent_state.fill_(step)
        manager.record_step(tx, "r0", step)

    estimate = manager.estimate_bytes()
    assert tx.snapshots["r0"] == {}
    assert estimate["transaction_recorded_steps"] == 7
    assert estimate["transaction_snapshot_state_bytes"] == 0

    manager.restore_original(tx)
    replayed = manager.get_request_state("r0")
    replayed.logical_length = before.logical_length + 3
    replayed.linear_layers[0].recurrent_state.fill_(3)
    manager.finalize_replayed_commit(tx, {"r0": 3})

    after = manager.get_request_state("r0")
    assert tx.state is HybridTransactionState.COMMITTED
    assert after.logical_length == 7
    assert after.linear_layers[0].recurrent_state.unique().item() == 3
