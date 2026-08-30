from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode.types import Proposal, SpecDecodeBatch
from myvllm.spec_decode.verification import prepare_verification_input
import pickle


def make_reserved(tokens, draft, manager):
    seq = Sequence(tokens, 4, SamplingParams(temperature=0))
    manager.allocate(seq)
    seq.mark_computed(len(seq))
    seq.append_token(90 + seq.seq_id)
    reservation = manager.reserve(seq, len(draft))
    transaction = seq.begin_speculation(draft)
    return seq, reservation, transaction


def test_variable_q_verification_metadata_uses_prefix_positions_and_slots():
    manager = BlockManager(16, 4)
    a, a_res, a_tx = make_reserved([1, 2, 3, 4], [5, 6], manager)
    b, b_res, b_tx = make_reserved([7, 8], [9], manager)
    batch = SpecDecodeBatch(
        [a, b],
        [Proposal(a.seq_id, (5, 6)), Proposal(b.seq_id, (9,))],
    )

    plan = prepare_verification_input(batch, block_size=4)
    assert plan.query_lengths == (3, 2)
    assert plan.input_ids == (a.last_token, 5, 6, b.last_token, 9)
    assert plan.context_lens == (5, 6, 7, 3, 4)
    assert plan.slot_mappings[:3] == (
        a.block_table[1] * 4,
        a.block_table[1] * 4 + 1,
        a.block_table[1] * 4 + 2,
    )
    assert plan.slot_mappings[3:] == (
        b.block_table[0] * 4 + 2,
        b.block_table[0] * 4 + 3,
    )
    assert all(len(table) == 2 for table in plan.block_tables)

    a.rollback(a_tx)
    b.rollback(b_tx)
    manager.rollback(a, a_res)
    manager.rollback(b, b_res)


def test_compact_pickled_batch_has_every_worker_planner_field():
    manager = BlockManager(8, 4)
    seq, reservation, transaction = make_reserved([1, 2, 3, 4], [5, 6], manager)
    batch = SpecDecodeBatch([seq], [Proposal(seq.seq_id, (5, 6))])

    worker_batch = pickle.loads(pickle.dumps(batch))
    plan = prepare_verification_input(worker_batch, block_size=4)
    assert plan.input_ids == (seq.last_token, 5, 6)
    assert plan.context_lens == (5, 6, 7)
    assert plan.query_lengths == (3,)

    seq.rollback(transaction)
    manager.rollback(seq, reservation)
