from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode import (
    MockProposer,
    SpecDecodeCoordinator,
    SpeculativeConfig,
    VerificationResult,
)


def next_token(history):
    # Deterministic causal target with enough variation to create rejections.
    return (sum(history[-3:]) * 7 + len(history)) % 23


def fake_verify(batch):
    results = []
    for seq, proposal in zip(batch.sequences, batch.proposals):
        history = list(seq.token_ids)
        decisions = []
        for position in range(len(proposal.token_ids) + 1):
            decisions.append(next_token(history))
            if position < len(proposal.token_ids):
                history.append(proposal.token_ids[position])
        results.append(VerificationResult(seq.seq_id, tuple(decisions)))
    return results


def setup_seq(prompt, block_size=4, max_tokens=20, max_model_length=None):
    params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        max_model_length=max_model_length,
    )
    seq = Sequence(list(prompt), block_size, params)
    manager = BlockManager(32, block_size)
    manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.mark_computed(len(seq))
    seq.append_token(next_token(seq.token_ids))
    return seq, manager


def test_coordinator_variable_lengths_rejection_bonus_and_metrics():
    params = SamplingParams(temperature=0, max_tokens=20)
    a = Sequence([1, 2], 4, params)
    b = Sequence([4, 5, 6], 4, params)
    manager = BlockManager(32, 4)
    for seq in (a, b):
        manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        seq.mark_computed(len(seq))
        seq.append_token(next_token(seq.token_ids))

    # a: construct two correct drafts (all accepted + bonus); b: reject first.
    a_history = list(a.token_ids)
    a_draft = []
    for _ in range(2):
        token = next_token(a_history)
        a_draft.append(token)
        a_history.append(token)
    proposer = MockProposer({a.seq_id: a_draft, b.seq_id: [999]})
    coordinator = SpecDecodeCoordinator(
        SpeculativeConfig(enabled=True, proposer="mock", num_speculative_tokens=4),
        proposer,
    )
    results = coordinator.execute([a, b], manager, fake_verify, eos_token_id=-1)

    assert [result.proposed_tokens for result in results] == [2, 1]
    assert [result.accepted_proposal_tokens for result in results] == [2, 0]
    assert results[0].used_bonus_token
    assert not results[1].used_bonus_token
    snapshot = coordinator.metrics.snapshot()
    assert snapshot["sequence_records"] == 2
    assert "steps" not in snapshot
    assert snapshot["proposed_tokens"] == 3
    assert snapshot["accepted_tokens"] == 2
    assert snapshot["acceptance_lengths"] == [2, 0]
    assert set(snapshot["stage_seconds"]) == {"proposal", "verify", "sample", "commit"}
    manager.assert_consistent()


def test_capacity_shortage_falls_back_to_ordinary_greedy_decode():
    seq, manager = setup_seq([1, 2], block_size=4)
    # Shrink to the one already allocated block: pending fits, K does not.
    manager.blocks = manager.blocks[:1]
    manager.free_block_ids.clear()
    manager.used_block_ids = {0}
    proposer = MockProposer({seq.seq_id: [3, 4, 5, 6]})
    coordinator = SpecDecodeCoordinator(
        SpeculativeConfig(enabled=True, proposer="mock", num_speculative_tokens=4),
        proposer,
    )
    expected = next_token(seq.token_ids)
    result = coordinator.execute([seq], manager, fake_verify, eos_token_id=-1)[0]
    assert result.fell_back_to_decode
    assert result.emitted_token_ids == (expected,)
    manager.assert_consistent()


def test_eos_and_length_limits_truncate_multi_token_commit():
    seq, manager = setup_seq([1, 2], max_tokens=3)
    first = next_token(seq.token_ids)
    second = next_token(seq.token_ids + [first])
    coordinator = SpecDecodeCoordinator(
        SpeculativeConfig(enabled=True, proposer="mock", num_speculative_tokens=3),
        MockProposer({seq.seq_id: [first, second]}),
    )
    result = coordinator.execute([seq], manager, fake_verify, eos_token_id=first)[0]
    assert result.emitted_token_ids == (first,)
    assert seq.is_finished


def run_target_only(prompt, total_completion):
    history = list(prompt)
    outputs = []
    for _ in range(total_completion):
        token = next_token(history)
        outputs.append(token)
        history.append(token)
    return outputs


def test_greedy_speculative_is_tokenwise_equal_to_target_only():
    prompt = [3, 1, 4]
    completion_count = 40
    target_only = run_target_only(prompt, completion_count)
    params = SamplingParams(temperature=0, max_tokens=completion_count)
    seq = Sequence(prompt, 4, params)
    manager = BlockManager(64, 4)
    manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.mark_computed(len(seq))
    seq.append_token(next_token(seq.token_ids))

    coordinator = SpecDecodeCoordinator(
        SpeculativeConfig(enabled=True, proposer="mock", num_speculative_tokens=5),
        MockProposer(default=[0, 1, 2, 3, 4]),
    )
    while not seq.is_finished:
        coordinator.execute([seq], manager, fake_verify, eos_token_id=-1)

    assert seq.completion_token_ids == target_only


def test_batched_fake_verification_matches_sequential_target_execution():
    params = SamplingParams(temperature=0, max_tokens=10)
    sequences = [Sequence([1, 2, 3], 4, params), Sequence([7, 8], 4, params)]
    proposals = []
    from myvllm.spec_decode.types import Proposal, SpecDecodeBatch

    for seq, draft in zip(sequences, ([4, 5, 6], [9])):
        seq.mark_computed(len(seq) - 1)
        proposals.append(Proposal(seq.seq_id, tuple(draft)))
    batch = SpecDecodeBatch(sequences, proposals)
    batched = fake_verify(batch)
    for seq, proposal, result in zip(sequences, proposals, batched):
        history = list(seq.token_ids)
        sequential = []
        for draft in proposal.token_ids:
            sequential.append(next_token(history))
            history.append(draft)
        sequential.append(next_token(history))
        assert result.target_token_ids == tuple(sequential)


def test_validation_failure_rolls_back_both_sequence_and_kv(monkeypatch):
    import pytest

    seq, manager = setup_seq([1, 2, 3])
    original_tokens = list(seq.token_ids)
    original_table = list(seq.block_table)
    coordinator = SpecDecodeCoordinator(
        SpeculativeConfig(enabled=True, proposer="mock", num_speculative_tokens=2),
        MockProposer({seq.seq_id: [4, 5]}),
    )
    original_validate = manager.validate_commit

    def injected_failure(*args, **kwargs):
        raise ValueError("injected validation failure")

    monkeypatch.setattr(manager, "validate_commit", injected_failure)
    with pytest.raises(ValueError, match="injected"):
        coordinator.execute([seq], manager, fake_verify, eos_token_id=-1)

    assert seq.token_ids == original_tokens
    assert seq.active_transaction is None
    assert seq.block_table == original_table
    assert not manager._active_reservations
    manager.assert_consistent()
    monkeypatch.setattr(manager, "validate_commit", original_validate)


def test_late_batch_validation_failure_does_not_partially_commit(monkeypatch):
    import pytest

    params = SamplingParams(temperature=0, max_tokens=10)
    manager = BlockManager(16, 4)
    sequences = [Sequence([1, 2], 4, params), Sequence([3, 4], 4, params)]
    for seq in sequences:
        manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        seq.mark_computed(len(seq))
        seq.append_token(next_token(seq.token_ids))
    original_tokens = [list(seq.token_ids) for seq in sequences]
    original_tables = [list(seq.block_table) for seq in sequences]
    coordinator = SpecDecodeCoordinator(
        SpeculativeConfig(enabled=True, proposer="mock", num_speculative_tokens=1),
        MockProposer({seq.seq_id: [8] for seq in sequences}),
    )
    original_validate = manager.validate_commit
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("second request validation failed")
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(manager, "validate_commit", fail_second)
    with pytest.raises(ValueError, match="second request"):
        coordinator.execute(sequences, manager, fake_verify, eos_token_id=-1)

    assert [seq.token_ids for seq in sequences] == original_tokens
    assert [seq.block_table for seq in sequences] == original_tables
    assert all(seq.active_transaction is None for seq in sequences)
    assert not manager._active_reservations
    manager.assert_consistent()
