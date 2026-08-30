import pytest

from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.scheduler import Scheduler
from myvllm.sampling_parameters import SamplingParams


def test_spec_schedule_does_not_advance_logical_or_kv_state():
    """Selecting work must not create the historical one-slot offset itself."""
    scheduler = Scheduler(4, 32, 8, 4, eos=0)
    seq = Sequence(
        [1, 2, 3, 4], block_size=4, sampling_params=SamplingParams(temperature=0)
    )
    scheduler.add_sequence(seq)
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill and scheduled == [seq]
    seq.mark_computed(len(seq))
    seq.append_token(5)

    before = (len(seq), seq.num_computed_tokens, tuple(seq.block_table))
    scheduled, is_prefill = scheduler.schedule(speculative=True)

    assert not is_prefill and scheduled == [seq]
    assert (len(seq), seq.num_computed_tokens, tuple(seq.block_table)) == before
    assert len(seq) - seq.num_computed_tokens == 1


def test_reservation_maps_pending_token_to_exact_logical_slot():
    manager = BlockManager(num_blocks=4, block_size=4)
    seq = Sequence([10, 11, 12, 13], block_size=4)
    manager.allocate(seq)
    seq.mark_computed(4)
    seq.append_token(14)

    reservation = manager.reserve(seq, num_provisional_tokens=2)
    assert manager.slot_for(seq, seq.num_computed_tokens) == seq.block_table[1] * 4
    assert reservation.reserved_num_tokens == 7
    manager.rollback(seq, reservation)


def test_spec_schedule_preempts_when_even_pending_slot_cannot_be_reserved():
    scheduler = Scheduler(2, 8, 1, 4, eos=0)
    seq = Sequence(
        [1, 2, 3, 4], block_size=4, sampling_params=SamplingParams(temperature=0)
    )
    scheduler.block_manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.mark_computed(4)
    seq.append_token(5)
    scheduler.running.append(seq)

    scheduled, is_prefill = scheduler.schedule(speculative=True)
    assert not is_prefill and scheduled == []
    assert seq in scheduler.waiting
    assert not scheduler.block_manager.used_block_ids
