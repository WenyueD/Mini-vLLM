import random
import pickle

import pytest

from myvllm.engine.block_manager import BlockManager, ReservationState
from myvllm.engine.sequence import Sequence, TransactionState
from myvllm.sampling_parameters import SamplingParams


def running_sequence(tokens, block_size=4, num_blocks=16):
    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    seq = Sequence(tokens, block_size, SamplingParams(temperature=0))
    manager.allocate(seq)
    seq.mark_computed(len(seq))
    seq.append_token(99)
    return seq, manager


def test_within_block_rollback_does_not_free_or_copy_blocks():
    seq, manager = running_sequence([1, 2], block_size=4)
    original_table = list(seq.block_table)
    reservation = manager.reserve(seq, 1)
    transaction = seq.begin_speculation([7])
    assert seq.block_table == original_table

    seq.rollback(transaction)
    manager.rollback(seq, reservation)
    assert seq.block_table == original_table
    assert transaction.state is TransactionState.ROLLED_BACK
    assert reservation.state is ReservationState.ROLLED_BACK
    manager.assert_consistent()


def test_cross_block_rollback_releases_only_tail_blocks():
    seq, manager = running_sequence([1, 2, 3], block_size=4)
    original_table = list(seq.block_table)
    reservation = manager.reserve(seq, 5)
    transaction = seq.begin_speculation([5, 6, 7, 8, 9])
    added = set(seq.block_table) - set(original_table)
    assert len(added) == 2

    seq.rollback(transaction)
    manager.rollback(seq, reservation)
    assert seq.block_table == original_table
    assert added <= set(manager.free_block_ids)
    manager.assert_consistent()


def test_partial_commit_keeps_accepted_kv_and_frees_rejected_pages():
    seq, manager = running_sequence([1, 2, 3], block_size=4)
    reservation = manager.reserve(seq, 5)
    transaction = seq.begin_speculation([5, 6, 7, 8, 9])
    seq.commit(transaction, [5, 42], accepted_proposal_tokens=1)
    manager.commit(seq, reservation, seq.num_computed_tokens)

    assert seq.token_ids[-2:] == [5, 42]
    assert seq.num_tokens - seq.num_computed_tokens == 1
    assert len(seq.block_table) == 2
    manager.assert_consistent()


def test_spec_commit_finalizes_hash_for_following_ordinary_append():
    seq, manager = running_sequence([1, 2], block_size=4)
    reservation = manager.reserve(seq, 2)
    transaction = seq.begin_speculation([4, 5])
    seq.validate_commit(transaction, [4, 42], 1)
    future_tokens = seq.token_ids + [4, 42]
    manager.validate_commit(seq, reservation, 4, future_tokens)
    seq.commit(transaction, [4, 42], 1)
    manager.commit(seq, reservation, seq.num_computed_tokens)

    assert manager.blocks[seq.block_table[0]].hash != -1
    assert manager.blocks[seq.block_table[0]].token_ids == [1, 2, 99, 4]

    # The pending token starts a new block. Ordinary append relies on the full
    # predecessor having published hash metadata.
    assert manager.can_append(seq)
    manager.append(seq)
    seq.mark_computed(seq.num_computed_tokens + 1)
    seq.append_token(7)

    reservation2 = manager.reserve(seq, 3)
    transaction2 = seq.begin_speculation([8, 9, 10])
    seq.rollback(transaction2)
    manager.rollback(seq, reservation2)
    manager.deallocate(seq)
    manager.assert_consistent()
    assert len(manager.free_block_ids) == len(manager.blocks)


def test_decode_pickle_contract_is_compact_and_preserves_runner_fields():
    short = Sequence([1], 4, SamplingParams(temperature=0, max_tokens=20_000))
    short.mark_computed(1)
    short.append_token(2)
    small_payload = pickle.dumps(short)

    long = Sequence([1], 4, SamplingParams(temperature=0, max_tokens=20_000))
    long.mark_computed(1)
    long.append_tokens(list(range(10_000)))
    long.num_computed_tokens = long.num_tokens - 1
    long.provisional_token_ids = [7, 8, 9]
    long_payload = pickle.dumps(long)
    worker = pickle.loads(long_payload)

    assert len(long_payload) < len(small_payload) + 256
    assert worker.num_tokens == long.num_tokens
    assert worker.num_computed_tokens == long.num_computed_tokens
    assert worker.last_token == long.last_token
    assert worker.token_ids == [long.last_token]
    assert worker.provisional_token_ids == [7, 8, 9]


def test_capacity_and_duplicate_transaction_guards():
    seq, manager = running_sequence([1, 2], block_size=4, num_blocks=1)
    with pytest.raises(MemoryError):
        manager.reserve(seq, 4)

    reservation = manager.reserve(seq, 0)
    transaction = seq.begin_speculation([])
    with pytest.raises(RuntimeError):
        seq.begin_speculation([])
    seq.rollback(transaction)
    manager.rollback(seq, reservation)
    with pytest.raises(RuntimeError):
        seq.rollback(transaction)
    with pytest.raises(RuntimeError):
        manager.rollback(seq, reservation)


def test_reserve_batch_is_atomic_on_capacity_failure():
    manager = BlockManager(num_blocks=2, block_size=4)
    a = Sequence([1, 2], 4, SamplingParams(temperature=0))
    b = Sequence([3, 4], 4, SamplingParams(temperature=0))
    manager.allocate(a)
    manager.allocate(b)
    before = (list(a.block_table), list(b.block_table), list(manager.free_block_ids))
    with pytest.raises(MemoryError):
        manager.reserve_batch([(a, 4), (b, 4)])
    assert (a.block_table, b.block_table, list(manager.free_block_ids)) == before
    manager.assert_consistent()


def test_randomized_reserve_commit_rollback_has_no_block_leaks():
    rng = random.Random(12345)
    manager = BlockManager(num_blocks=256, block_size=4)
    seq = Sequence([1, 2, 3], 4, SamplingParams(temperature=0, max_tokens=500))
    manager.allocate(seq)
    seq.mark_computed(3)
    seq.append_token(4)

    for _ in range(300):
        k = rng.randint(0, 10)
        try:
            reservation = manager.reserve(seq, k)
        except MemoryError:
            reservation = manager.reserve(seq, 0)
            k = 0
        draft = [rng.randrange(100) for _ in range(k)]
        transaction = seq.begin_speculation(draft)
        if rng.random() < 0.25:
            seq.rollback(transaction)
            manager.rollback(seq, reservation)
        else:
            accepted = rng.randint(0, k)
            emitted = draft[:accepted] + [rng.randrange(100)]
            seq.commit(transaction, emitted, accepted)
            manager.commit(seq, reservation, seq.num_computed_tokens)
        manager.assert_consistent()
        assert seq.num_tokens - seq.num_computed_tokens == 1

    manager.deallocate(seq)
    manager.assert_consistent()
    assert len(manager.free_block_ids) == len(manager.blocks)
