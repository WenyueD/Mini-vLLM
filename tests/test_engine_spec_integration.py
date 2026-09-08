import gc
import weakref
from types import SimpleNamespace

import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode import MockProposer, SpecDecodeCoordinator, SpeculativeConfig
from myvllm.spec_decode.types import VerificationResult


def target(history):
    return (sum(history[-2:]) + len(history)) % 31


def verify(batch):
    results = []
    for seq, proposal in zip(batch.sequences, batch.proposals):
        history = list(seq.token_ids)
        tokens = []
        for position in range(len(proposal.token_ids) + 1):
            tokens.append(target(history))
            if position < len(proposal.token_ids):
                history.append(proposal.token_ids[position])
        results.append(VerificationResult(seq.seq_id, tuple(tokens)))
    return results


class FakeRunner:
    def call(self, method, *args):
        if method == "verify":
            return verify(args[0])
        if method == "run":
            seqs, is_prefill = args
            assert not is_prefill
            for seq in seqs:
                logical_block = (len(seq) - 1) // seq.block_size
                assert logical_block < len(
                    seq.block_table
                ), "ordinary pending token was not allocated before ModelRunner"
            return torch.tensor([target(seq.token_ids) for seq in seqs])
        raise AssertionError(method)


def test_engine_exit_releases_dflash_runner_reference():
    calls = []

    class ExitRunner:
        def call(self, method, *args):
            calls.append((method, args))

    runner = ExitRunner()
    runner_ref = weakref.ref(runner)
    engine = LLMEngine.__new__(LLMEngine)
    engine._closed = False
    engine.processes = []
    engine.model_runner = runner
    engine.spec_decode_coordinator = SimpleNamespace(
        proposer=SimpleNamespace(model_runner=runner)
    )
    del runner

    engine.exit()
    gc.collect()

    assert calls == [("exit", ())]
    assert engine.spec_decode_coordinator is None
    assert not hasattr(engine, "model_runner")
    assert runner_ref() is None


def add_running(scheduler, tokens, temperature):
    seq = Sequence(
        tokens,
        4,
        SamplingParams(temperature=temperature, max_tokens=10),
    )
    scheduler.block_manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.mark_computed(len(seq))
    seq.append_token(target(seq.token_ids))
    scheduler.running.append(seq)
    return seq


def test_engine_step_mixes_multi_commit_greedy_and_ordinary_decode():
    scheduler = Scheduler(4, 16, 32, 4, eos=-1)
    greedy = add_running(scheduler, [1, 2], temperature=0)
    # Appending the first pending token to this full prompt crosses a block
    # boundary. The ordinary path must allocate block 2 before FakeRunner sees it.
    ordinary = add_running(scheduler, [3, 4, 5, 6], temperature=1.0)

    history = list(greedy.token_ids)
    correct_draft = []
    for _ in range(2):
        token = target(history)
        correct_draft.append(token)
        history.append(token)

    config = SpeculativeConfig(enabled=True, proposer="mock", num_speculative_tokens=2)
    engine = LLMEngine.__new__(LLMEngine)
    engine.speculative_config = config
    engine.spec_decode_coordinator = SpecDecodeCoordinator(
        config, MockProposer({greedy.seq_id: correct_draft})
    )
    engine.scheduler = scheduler
    engine.model_runner = FakeRunner()

    step = engine.step()
    assert not step.is_prefill
    assert len(greedy.completion_token_ids) == 4  # initial + 2 drafts + bonus
    assert len(ordinary.completion_token_ids) == 2  # initial + ordinary one-token step
    assert step.num_processed_tokens == 4
    assert {output.request_id for output in step.outputs} == {
        greedy.request_id,
        ordinary.request_id,
    }
    assert step.speculative is not None
    assert step.speculative.active_batch_size == 2
    assert step.speculative.speculative_batch_size == 1
    assert step.speculative.proposed_tokens == 2
    assert step.speculative.accepted_tokens == 2
    scheduler.block_manager.assert_consistent()
    scheduler.block_manager.deallocate(greedy)
    scheduler.block_manager.deallocate(ordinary)
    scheduler.block_manager.assert_consistent()
    assert len(scheduler.block_manager.free_block_ids) == len(
        scheduler.block_manager.blocks
    )
