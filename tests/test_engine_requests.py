import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.metrics import EngineMetrics
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode import Proposer, SpecDecodeCoordinator, SpeculativeConfig
from myvllm.spec_decode.types import Proposal, VerificationResult


def target(history):
    return (sum(history[-3:]) * 5 + len(history)) % 29


class NumberTokenizer:
    def encode(self, prompt):
        return [int(value) for value in prompt.split(",")]

    def decode(self, token_ids):
        return ",".join(str(token) for token in token_ids)


class FakeRunner:
    def call(self, method, *args):
        if method == "run":
            sequences, _ = args
            return torch.tensor([target(seq.token_ids) for seq in sequences])
        if method == "verify":
            batch = args[0]
            results = []
            for seq, proposal in zip(batch.sequences, batch.proposals):
                history = list(seq.token_ids)
                decisions = []
                for position in range(len(proposal.token_ids) + 1):
                    decisions.append(target(history))
                    if position < len(proposal.token_ids):
                        history.append(proposal.token_ids[position])
                results.append(VerificationResult(seq.seq_id, tuple(decisions)))
            return results
        raise AssertionError(method)


class FakeQwenRunner(FakeRunner):
    is_qwen35_target = True

    def call(self, method, *args):
        if method == "get_qwen35_verification_stats":
            return {"max_batch_size": 2, "tensor_batched_token_steps": 7}
        return super().call(method, *args)


class OracleProposer(Proposer):
    def propose(self, sequence, max_tokens):
        history = list(sequence.token_ids)
        tokens = []
        for _ in range(max_tokens):
            token = target(history)
            tokens.append(token)
            history.append(token)
        return Proposal(sequence.seq_id, tuple(tokens), {"kind": "test-oracle"})


def make_engine(speculative=False, max_num_sequences=8):
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = {"block_size": 4, "max_metrics_history": 64}
    engine.scheduler = Scheduler(max_num_sequences, 64, 64, 4, eos=-1)
    engine.tokenizer = NumberTokenizer()
    engine.model_runner = FakeRunner()
    engine.requests = {}
    engine.metrics = EngineMetrics(64)
    if speculative:
        config = SpeculativeConfig(
            enabled=True, proposer="mock", num_speculative_tokens=3
        )
        engine.speculative_config = config
        engine.spec_decode_coordinator = SpecDecodeCoordinator(config, OracleProposer())
    else:
        engine.speculative_config = SpeculativeConfig(enabled=False)
        engine.spec_decode_coordinator = None
    return engine


def collect(engine, request_id):
    tokens = []
    event_sizes = []
    while not engine.get_request_status(request_id)["finished"]:
        result = engine.step()
        for output in result.outputs:
            if output.request_id == request_id:
                tokens.extend(output.token_ids)
                event_sizes.append(len(output.token_ids))
    return tokens, event_sizes


def test_stable_request_id_status_and_duplicate_rejection():
    import pytest

    engine = make_engine()
    request_id = engine.add_request(
        "1,2", SamplingParams(temperature=0, max_tokens=2), "client-id"
    )
    assert request_id == "client-id"
    assert engine.get_request_status(request_id)["status"] == "waiting"
    with pytest.raises(ValueError, match="duplicate"):
        engine.add_prompt("3,4", SamplingParams(temperature=0), request_id="client-id")


def test_abort_waiting_and_running_active_transaction_releases_all_blocks():
    scheduler = Scheduler(2, 32, 8, 4, eos=-1)
    waiting = Sequence([1, 2], 4, request_id="waiting")
    scheduler.add_sequence(waiting)
    assert scheduler.abort(waiting)
    assert waiting.status is SequenceStatus.ABORTED
    assert not scheduler.waiting

    running = Sequence(
        [3, 4, 5],
        4,
        SamplingParams(temperature=0),
        request_id="running",
    )
    scheduler.block_manager.allocate(running)
    running.status = SequenceStatus.RUNNING
    scheduler.running.append(running)
    running.mark_computed(3)
    running.append_token(6)
    reservation = scheduler.block_manager.reserve(running, 5)
    running.begin_speculation([7, 8, 9, 10, 11])

    assert scheduler.abort(running)
    assert reservation.state.name == "ROLLED_BACK"
    assert running.active_transaction is None
    assert not running.block_table
    assert not scheduler.running
    assert not scheduler.block_manager._active_reservations
    scheduler.block_manager.assert_consistent()
    assert len(scheduler.block_manager.free_block_ids) == len(
        scheduler.block_manager.blocks
    )


def test_incremental_ordinary_and_speculative_match_generate():
    params = SamplingParams(temperature=0, max_tokens=9, ignore_eos=True)
    for speculative in (False, True):
        incremental_engine = make_engine(speculative)
        request_id = incremental_engine.add_request("1,2,3", params)
        incremental, event_sizes = collect(incremental_engine, request_id)

        offline_engine = make_engine(speculative)
        offline = offline_engine.generate(["1,2,3"], params)["token_ids"][0]
        assert incremental == offline
        assert all(size >= 1 for size in event_sizes)
        if speculative:
            assert any(size > 1 for size in event_sizes)
        else:
            assert all(size == 1 for size in event_sizes)


def test_batch_step_metrics_are_atomic_and_weighted():
    engine = make_engine(speculative=True)
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    request_ids = [
        engine.add_request("1,2", params, "a"),
        engine.add_request("3,4", params, "b"),
    ]
    prefill = engine.step()
    assert prefill.is_prefill
    decode = engine.step()
    observation = decode.speculative
    assert observation is not None
    assert observation.active_batch_size == 2
    assert observation.speculative_batch_size == 2
    assert observation.proposed_tokens == 6
    assert observation.accepted_tokens == 6
    assert observation.per_request_accepted_length == {
        request_ids[0]: 3,
        request_ids[1]: 3,
    }
    assert observation.per_position_proposed == (2, 2, 2)
    assert observation.per_position_accepted == (2, 2, 2)

    second_decode = engine.step()
    second_observation = second_decode.speculative
    assert second_observation is not None
    assert second_observation.engine_step != observation.engine_step
    assert second_observation.proposed_tokens == 4
    assert second_observation.accepted_tokens == 4

    stats = engine.get_stats()
    assert stats["decode_steps"] == 2
    assert stats["proposed_tokens"] == sum(
        step["proposed_tokens"] for step in stats["recent_decode_steps"]
    )
    assert stats["accepted_tokens"] == sum(
        step["accepted_tokens"] for step in stats["recent_decode_steps"]
    )
    assert stats["acceptance_rate"] == 1.0


def test_qwen_verification_batch_stats_are_exposed_by_engine():
    engine = make_engine()
    engine.model_runner = FakeQwenRunner()

    stats = engine.get_stats()

    assert stats["qwen35_verification"] == {
        "max_batch_size": 2,
        "tensor_batched_token_steps": 7,
    }

    del engine.model_runner
    closed_stats = engine.get_stats()
    assert "qwen35_verification" not in closed_stats
