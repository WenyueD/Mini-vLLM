import pytest
import torch

from myvllm.models.qwen35_gdn import fused_cached_gdn, fused_replay_gdn_prefix
from myvllm.engine.hybrid_cache import HybridCacheManager, LinearStateUpdate
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams
from myvllm.sampling import greedy_token_ids
from myvllm.spec_decode.types import Proposal, SpecDecodeBatch
from myvllm.spec_decode.rejection_sampler import RejectionSampler
from myvllm.models.qwen3_8 import Qwen35ForCausalLM
from test_qwen35_config_and_model import tiny_config, init_small

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")


@pytest.mark.parametrize("dim", [4, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_gdn_reference_shape_invariance_and_every_prefix(dim, dtype):
    torch.manual_seed(42)
    shape = (4, 8, 2, dim)
    q = torch.randn(shape, device="cuda", dtype=dtype) / dim**0.5
    k = torch.randn_like(q) / dim**0.5
    v = torch.randn_like(q)
    beta = torch.rand(shape[:3], device="cuda", dtype=dtype)
    decay = torch.rand(shape[:3], device="cuda", dtype=torch.float32) * 0.1 + 0.9
    state = torch.randn(4, 2, dim, dim, device="cuda", dtype=torch.float32) * 0.01
    initial = state.clone()
    out, final, deltas = fused_cached_gdn(q, k, v, beta, decay, state, record_updates=True)
    torch.testing.assert_close(state, initial, rtol=0, atol=0)

    # Independent FP64 equation reference, not the fused implementation.
    reference = state.double()
    expected = []
    for pos in range(8):
        reference *= decay[:, pos, :, None, None].double()
        key = k[:, pos].double()
        memory = torch.einsum("bhkv,bhk->bhv", reference, key)
        delta = (v[:, pos].double() - memory) * beta[:, pos, :, None].double()
        reference += torch.einsum("bhk,bhv->bhkv", key, delta)
        expected.append(torch.einsum("bhkv,bhk->bhv", reference, q[:, pos].double()))
    torch.testing.assert_close(final.double(), reference, rtol=5e-5, atol=2e-6)
    torch.testing.assert_close(out.float(), torch.stack(expected, dim=1).float(),
                               rtol=0.005 if dtype == torch.bfloat16 else 5e-5,
                               atol=0.008 if dtype == torch.bfloat16 else 2e-6)
    replay = state.clone()
    serial = state.clone()
    updates = []
    torch.testing.assert_close(fused_replay_gdn_prefix(state[:1], []), state[:1], rtol=0, atol=0)
    for pos in range(8):
        step, serial, _ = fused_cached_gdn(
            q[:, pos:pos+1], k[:, pos:pos+1], v[:, pos:pos+1],
            beta[:, pos:pos+1], decay[:, pos:pos+1], serial,
        )
        torch.testing.assert_close(step[:, 0], out[:, pos], rtol=0, atol=0)
        replay = replay * decay[:, pos, :, None, None]
        replay = replay + k[:, pos].float().unsqueeze(-1) * deltas[:, pos].unsqueeze(-2)
        torch.testing.assert_close(replay, serial, rtol=0, atol=0)
        updates.append(LinearStateUpdate(
            decay[:1, pos, :, None, None], k[:1, pos].float(),
            deltas[:1, pos], torch.empty(0, device="cuda"),
        ))
        fused_prefix = fused_replay_gdn_prefix(state[:1], updates)
        torch.testing.assert_close(fused_prefix, serial[:1], rtol=0, atol=0)
        torch.testing.assert_close(state, initial, rtol=0, atol=0)
    torch.testing.assert_close(serial, final, rtol=0, atol=0)
    for width in [3, 4]:
        chunk_state = state.clone()
        chunks = []
        for start in range(0, 8, width):
            sl = slice(start, start + width)
            core, chunk_state, _ = fused_cached_gdn(q[:, sl], k[:, sl], v[:, sl], beta[:, sl], decay[:, sl], chunk_state)
            chunks.append(core)
        torch.testing.assert_close(torch.cat(chunks, dim=1), out, rtol=0, atol=0)
        torch.testing.assert_close(chunk_state, final, rtol=0, atol=0)


def assert_state_exact(left, right):
    assert left.logical_length == right.logical_length
    for layer_id, state in left.linear_layers.items():
        other = right.linear_layers[layer_id]
        torch.testing.assert_close(state.recurrent_state, other.recurrent_state, rtol=0, atol=0)
        torch.testing.assert_close(state.conv_state, other.conv_state, rtol=0, atol=0)
    for layer_id, state in left.full_attention_layers.items():
        for a, b in zip(state, right.full_attention_layers[layer_id]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def make_model():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    return model.cuda().bfloat16().eval()


def make_runner(model, cache):
    runner = ModelRunner.__new__(ModelRunner)
    runner.model, runner.hybrid_cache_manager = model, cache
    runner.is_qwen35_target, runner.rank = True, 0
    runner.config = {
        "qwen35_vectorized_target": True, "qwen35_fused_gdn": True,
        "capture_verification_hidden_taps": True,
        "qwen35_replay_state_updates": True,
        "qwen35_parallel_verification": True,
        "qwen35_parallel_verification_chunk_size": 4,
        "qwen35_cached_batch_pad_to_size": 4,
        "qwen35_cached_attention_pad_to_length": 32,
        "qwen35_fused_prefix_commit": True,
        "qwen35_reuse_verification_samples": True,
        "qwen35_compact_verification_copies": True,
    }
    runner.sampler = lambda logits, _: greedy_token_ids(logits)
    runner._pending_verified_cache_transactions = {}
    return runner


def prefill(model, cache, rid, ids):
    model.set_experimental_vectorized_target(False)
    model.forward_with_cache(torch.tensor(ids, device="cuda"), cache, rid)
    model.set_experimental_vectorized_target(True, fused_gdn=True)


def step(model, cache, rid, token):
    return model.forward_with_cache_batch(
        torch.tensor([[token]], device="cuda"), [1], cache, [rid],
        attention_pad_to_length=32, batch_pad_to_size=4,
    )[0]


@pytest.mark.parametrize("accepted", range(8))
@torch.inference_mode()
def test_fused_verifier_all_acceptance_lengths_ragged_and_commit(accepted):
    model = make_model()
    reference = HybridCacheManager()
    prefill(model, reference, "a", [1, 2, 3])
    token, draft, states = 4, [], []
    for _ in range(8):
        output = step(model, reference, "a", token)
        states.append(reference.get_request_state("a").clone())
        token = int(greedy_token_ids(output.logits.reshape(1, -1)).item())
        draft.append(token)
    proposal_tokens = draft[:7]
    if accepted < 7:
        proposal_tokens[accepted] = (proposal_tokens[accepted] + 1) % 32
    cache = HybridCacheManager()
    prefill(model, cache, "a", [1, 2, 3])
    prefill(model, cache, "b", [5, 6])
    before_b = cache.get_request_state("b").clone()
    runner = make_runner(model, cache)
    a = Sequence([1, 2, 3, 4], 4, SamplingParams(temperature=0, max_tokens=16), request_id="a")
    b = Sequence([5, 6, 7], 4, SamplingParams(temperature=0, max_tokens=16), request_id="b")
    a.mark_computed(3); b.mark_computed(2)
    a.begin_speculation(proposal_tokens); b.begin_speculation([0, 1])
    proposals = [Proposal(a.seq_id, tuple(proposal_tokens)), Proposal(b.seq_id, (0, 1))]
    result = runner.verify(SpecDecodeBatch([a, b], proposals))
    sample = RejectionSampler().sample(proposals[0], result[0])
    assert sample.accepted_proposal_tokens == accepted
    assert sample.emitted_token_ids == tuple(draft[:accepted+1])
    pending = runner._pending_verified_cache_transactions[a.seq_id]
    assert all(t.shape[1] == len(pending.query_tokens) for t in pending.hidden_taps.values())
    assert cache.estimate_bytes()["transaction_snapshot_state_bytes"] == 0
    assert cache.estimate_bytes()["transaction_linear_update_bytes"] > 0
    # Packed views must not retain padded request rows or hidden full states.
    storages = {}
    for transaction in cache._active.values():
        for layers in transaction.linear_updates.values():
            for updates in layers.values():
                for update in updates:
                    for value in (update.decay, update.key, update.delta, update.conv_state):
                        storage = value.untyped_storage()
                        storages[storage.data_ptr()] = storage.nbytes()
    assert sum(storages.values()) == cache.estimate_bytes()["transaction_linear_update_bytes"]
    runner.commit_verified_cache({a.seq_id: accepted + 1})
    runner.rollback_verified_cache([b.seq_id])
    assert_state_exact(cache.get_request_state("a"), states[accepted])
    assert_state_exact(cache.get_request_state("b"), before_b)
    assert cache.estimate_bytes()["active_transactions"] == 0


@torch.inference_mode()
def test_fused_verifier_exception_rolls_back():
    model, cache = make_model(), HybridCacheManager()
    prefill(model, cache, "a", [1, 2, 3])
    before = cache.get_request_state("a").clone()
    runner = make_runner(model, cache)
    seq = Sequence([1, 2, 3, 4], 4, SamplingParams(temperature=0, max_tokens=16), request_id="a")
    seq.mark_computed(3); seq.begin_speculation([0] * 7)
    def fail(*a, **kw):
        raise RuntimeError("injected after fused recurrence")
    runner._sample_qwen35_logits = fail
    with pytest.raises(RuntimeError, match="injected"):
        runner.verify(SpecDecodeBatch([seq], [Proposal(seq.seq_id, (0,) * 7)]))
    assert_state_exact(cache.get_request_state("a"), before)
    assert cache.estimate_bytes()["active_transactions"] == 0


@torch.inference_mode()
def test_fused_prefix_commit_exception_rolls_back(monkeypatch):
    import myvllm.models.qwen35_gdn as gdn

    model, cache = make_model(), HybridCacheManager()
    prefill(model, cache, "a", [1, 2, 3])
    before = cache.get_request_state("a").clone()
    runner = make_runner(model, cache)
    seq = Sequence([1, 2, 3, 4], 4, SamplingParams(temperature=0, max_tokens=16), request_id="a")
    seq.mark_computed(3); seq.begin_speculation([0] * 7)
    runner.verify(SpecDecodeBatch([seq], [Proposal(seq.seq_id, (0,) * 7)]))
    original = gdn.fused_replay_gdn_prefix
    calls = []
    def fail_after_one_layer(*args, **kwargs):
        if calls:
            raise RuntimeError("injected during prefix commit")
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(gdn, "fused_replay_gdn_prefix", fail_after_one_layer)
    with pytest.raises(RuntimeError, match="injected during prefix commit"):
        runner.commit_verified_cache({seq.seq_id: 1})
    assert_state_exact(cache.get_request_state("a"), before)
    assert cache.estimate_bytes()["active_transactions"] == 0


@torch.inference_mode()
def test_fused_verifier_eos_truncates_cache_commit():
    from myvllm.engine.block_manager import BlockManager
    from myvllm.engine.sequence import SequenceStatus
    from myvllm.spec_decode import MockProposer, SpecDecodeCoordinator, SpeculativeConfig

    model, reference, cache = make_model(), HybridCacheManager(), HybridCacheManager()
    prefill(model, reference, "a", [1, 2, 3])
    token, draft, states = 4, [], []
    for _ in range(8):
        output = step(model, reference, "a", token)
        states.append(reference.get_request_state("a").clone())
        token = int(greedy_token_ids(output.logits.reshape(1, -1)).item())
        draft.append(token)
    prefill(model, cache, "a", [1, 2, 3])
    runner = make_runner(model, cache)
    seq = Sequence([1, 2, 3], 4, SamplingParams(temperature=0, max_tokens=16, ignore_eos=False), request_id="a")
    manager = BlockManager(32, 4)
    manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.mark_computed(3)
    seq.append_token(4)
    coordinator = SpecDecodeCoordinator(
        SpeculativeConfig(enabled=True, proposer="mock", num_speculative_tokens=7),
        MockProposer({seq.seq_id: draft[:7]}),
    )
    result = coordinator.execute(
        [seq], manager, runner.verify, eos_token_id=draft[0],
        commit_verified_cache=runner.commit_verified_cache,
        rollback_verified_cache=runner.rollback_verified_cache,
    )[0]
    assert result.emitted_token_ids == (draft[0],)
    assert seq.is_finished and seq.finish_reason == "eos"
    # EOS itself was accepted from the proposal, so its input cache is valid too.
    assert_state_exact(cache.get_request_state("a"), states[result.accepted_proposal_tokens])
    assert cache.estimate_bytes()["active_transactions"] == 0
    manager.assert_consistent()
