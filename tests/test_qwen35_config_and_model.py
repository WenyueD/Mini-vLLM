from types import SimpleNamespace

import pytest
import torch

from myvllm.models.qwen3_8 import (
    Qwen35ForCausalLM,
    Qwen35GatedDeltaNet,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
)
from myvllm.engine.hybrid_cache import FullAttentionLayerState, HybridCacheManager
from myvllm.engine.memory import MemoryTrace
from myvllm.engine.model_runner import ModelRunner
from myvllm.models.qwen3_8_config import (
    QWEN35_DEFAULT_HIDDEN_TAPS,
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)
from myvllm.sampling_parameters import SamplingParams
from myvllm.sampling import greedy_token_ids
from myvllm.spec_decode.rejection_sampler import RejectionSampler
from myvllm.spec_decode.types import Proposal, SpecDecodeBatch
from myvllm.spec_decode.dflash2 import DFlash2DraftCache, ProposalBatch
from myvllm.engine.sequence import Sequence


def qwen38_fixture():
    layer_types = [
        "full_attention" if (idx + 1) % 4 == 0 else "linear_attention"
        for idx in range(64)
    ]
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "layer_types": layer_types,
            "full_attention_interval": 4,
            "max_position_embeddings": 262144,
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "tie_word_embeddings": False,
            "eos_token_id": 151645,
        },
    }


def tiny_config(num_layers=8):
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "layer_types": [
                "full_attention" if (idx + 1) % 4 == 0 else "linear_attention"
                for idx in range(num_layers)
            ],
            "full_attention_interval": 4,
            "max_position_embeddings": 128,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            },
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "linear_conv_kernel_dim": 3,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "tie_word_embeddings": False,
            "eos_token_id": 2,
        },
    }


def init_small(module):
    generator = torch.Generator().manual_seed(20260901)
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(mean=0.0, std=0.02, generator=generator)


def test_qwen35_rms_norm_uses_checkpoint_weight_as_delta():
    norm = Qwen35RMSNorm(2, eps=0.0)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, 3.0]))

    x = torch.tensor([[[3.0, 4.0]]])
    normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True))

    assert torch.allclose(norm(x), normalized * torch.tensor([3.0, 4.0]))


def test_qwen35_gated_rms_norm_uses_checkpoint_weight_directly():
    norm = Qwen35RMSNormGated(2, eps=0.0)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, 3.0]))

    x = torch.tensor([[[3.0, 4.0]]])
    gate = torch.tensor([[[1.0, -1.0]]])
    normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True))
    expected = normalized * torch.tensor([2.0, 3.0]) * torch.nn.functional.silu(gate)

    assert torch.allclose(norm(x, gate), expected)


def test_qwen38_nested_text_config_fixture_parses_layer_types_and_taps():
    cfg = normalize_qwen35_text_config(qwen38_fixture())
    assert cfg.hidden_size == 5120
    assert cfg.vocab_size == 248320
    assert cfg.intermediate_size == 17408
    assert cfg.num_attention_heads == 24
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 256
    assert cfg.num_hidden_layers == 64
    assert cfg.full_attention_layers == tuple(range(3, 64, 4))
    assert cfg.linear_attention_layers[:3] == (0, 1, 2)
    assert cfg.linear_attention_layers[-1] == 62
    assert cfg.linear_conv_kernel_dim == 4
    assert cfg.linear_num_key_heads == 16
    assert cfg.linear_num_value_heads == 48
    assert QWEN35_DEFAULT_HIDDEN_TAPS == (5, 19, 33, 47, 61)


def test_qwen38_config_json_loader_does_not_require_transformers_autoconfig(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        __import__("json").dumps(qwen38_fixture()),
        encoding="utf-8",
    )
    raw = load_qwen35_config_dict(str(tmp_path))
    cfg = normalize_qwen35_text_config(raw)
    assert cfg.num_hidden_layers == 64
    assert cfg.full_attention_layers == tuple(range(3, 64, 4))


def test_qwen35_text_only_rejects_multimodal_inputs():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=2))
    init_small(model)
    with pytest.raises(NotImplementedError, match="text-only"):
        model(torch.tensor([[1, 2]]), pixel_values=torch.zeros(1, 3, 2, 2))


def test_hidden_taps_are_zero_based_layer_completion_outputs_and_do_not_change_logits():
    cfg = tiny_config(num_layers=64)
    model = Qwen35ForCausalLM.from_config_dict(cfg)
    init_small(model)
    input_ids = torch.tensor([[1, 2, 3]])
    no_taps = model(input_ids, return_dict=True)
    with_taps = model(input_ids, return_dict=True, capture_hidden_taps=True)
    assert sorted(with_taps.hidden_taps) == list(QWEN35_DEFAULT_HIDDEN_TAPS)
    assert all(value.shape == (1, 3, 16) for value in with_taps.hidden_taps.values())
    assert torch.allclose(no_taps.logits, with_taps.logits, atol=0, rtol=0)


def test_gdn_full_prefill_matches_chunked_prefill_and_decode_state():
    cfg = normalize_qwen35_text_config(tiny_config(num_layers=1))
    layer = Qwen35GatedDeltaNet(cfg, layer_idx=0)
    init_small(layer)
    x = torch.randn(2, 7, cfg.hidden_size, generator=torch.Generator().manual_seed(7))

    full, full_recurrent, full_conv = layer(x, return_final_state=True)
    left, recurrent, conv = layer(x[:, :3], return_final_state=True)
    right, recurrent, conv = layer(
        x[:, 3:],
        recurrent_state=recurrent,
        conv_state=conv,
        return_final_state=True,
    )
    chunked = torch.cat([left, right], dim=1)
    assert torch.allclose(full, chunked, atol=1e-5, rtol=1e-5)
    assert torch.allclose(full_recurrent, recurrent, atol=1e-5, rtol=1e-5)
    assert torch.allclose(full_conv, conv, atol=1e-5, rtol=1e-5)


def assert_cache_state_equal(left, right):
    assert left.logical_length == right.logical_length
    assert set(left.full_attention_layers) == set(right.full_attention_layers)
    assert set(left.linear_layers) == set(right.linear_layers)
    for layer_id in left.full_attention_layers:
        left_k, left_v = left.full_attention_layers[layer_id]
        right_k, right_v = right.full_attention_layers[layer_id]
        assert torch.allclose(left_k, right_k, atol=1e-5, rtol=1e-5)
        assert torch.allclose(left_v, right_v, atol=1e-5, rtol=1e-5)
    for layer_id in left.linear_layers:
        assert torch.allclose(
            left.linear_layers[layer_id].recurrent_state,
            right.linear_layers[layer_id].recurrent_state,
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.allclose(
            left.linear_layers[layer_id].conv_state,
            right.linear_layers[layer_id].conv_state,
            atol=1e-5,
            rtol=1e-5,
        )


def test_full_attention_layer_state_appends_inside_existing_capacity():
    key = torch.tensor([[[[1.0]], [[2.0]]]])
    value = key + 10
    state = FullAttentionLayerState.from_tensors(key, value, capacity=4)
    key_ptr = state.key.untyped_storage().data_ptr()
    value_ptr = state.value.untyped_storage().data_ptr()

    state.append(torch.tensor([[[[3.0]]]]), torch.tensor([[[[13.0]]]]))
    state.append(torch.tensor([[[[4.0]]]]), torch.tensor([[[[14.0]]]]))

    current_key, current_value = state.tensors()
    assert state.logical_length == 4
    assert state.key.untyped_storage().data_ptr() == key_ptr
    assert state.value.untyped_storage().data_ptr() == value_ptr
    assert current_key.reshape(-1).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert current_value.reshape(-1).tolist() == [11.0, 12.0, 13.0, 14.0]


def test_qwen35_cached_target_decode_matches_no_cache_full_recompute():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    full = model(input_ids, return_dict=True).logits

    cache = HybridCacheManager()
    cached = model.forward_with_cache(input_ids[0], cache, "req0").logits

    assert torch.allclose(full, cached, atol=1e-5, rtol=1e-5)


def test_qwen35_cache_transaction_rollback_and_partial_commit_replay_equivalence():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    cache = HybridCacheManager()
    prefix = torch.tensor([1, 2, 3])
    proposals = torch.tensor([4, 5, 6, 7, 8, 9, 10])
    model.forward_with_cache(prefix, cache, "req0")
    before = cache.get_request_state("req0").clone()

    tx = cache.begin_transaction(["req0"], reserve_tokens=7)
    model.forward_with_cache(proposals, cache, "req0", transaction=tx)
    cache.rollback(tx)
    assert_cache_state_equal(cache.get_request_state("req0"), before)

    tx = cache.begin_transaction(["req0"], reserve_tokens=7)
    model.forward_with_cache(proposals, cache, "req0", transaction=tx)
    cache.restore_original(tx)
    model.forward_with_cache(
        proposals[:3],
        cache,
        "req0",
        return_logits=False,
    )
    cache.finalize_replayed_commit(tx, {"req0": 3})

    replay_cache = HybridCacheManager()
    model.forward_with_cache(torch.tensor([1, 2, 3, 4, 5, 6]), replay_cache, "req0")
    assert_cache_state_equal(
        cache.get_request_state("req0"),
        replay_cache.get_request_state("req0"),
    )


def test_qwen35_ragged_cached_batch_matches_independent_serial_forwards():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    serial_cache = HybridCacheManager()
    batch_cache = HybridCacheManager()
    prefixes = {
        "req0": [1, 2, 3],
        "req1": [10, 11, 12, 13, 14],
    }
    queries = {
        "req0": [4, 5, 6],
        "req1": [15, 16],
    }
    for request_id, prefix in prefixes.items():
        model.forward_with_cache(torch.tensor(prefix), serial_cache, request_id)
        model.forward_with_cache(torch.tensor(prefix), batch_cache, request_id)

    serial_outputs = [
        model.forward_with_cache(
            torch.tensor(queries[request_id]), serial_cache, request_id
        )
        for request_id in prefixes
    ]
    padded_queries = torch.tensor([[4, 5, 6], [15, 16, 0]])
    batch_sizes = []
    original_batch_step = model.model.forward_cached_batch_step

    def tracked_batch_step(
        input_ids,
        request_states,
        hidden_tap_layers=None,
        attention_pad_to_length=None,
        state_update_logs=None,
    ):
        batch_sizes.append(len(request_states))
        assert attention_pad_to_length == 16
        return original_batch_step(
            input_ids,
            request_states,
            hidden_tap_layers=hidden_tap_layers,
            attention_pad_to_length=attention_pad_to_length,
            state_update_logs=state_update_logs,
        )

    model.model.forward_cached_batch_step = tracked_batch_step
    batch_outputs = model.forward_with_cache_batch(
        padded_queries,
        [3, 2],
        batch_cache,
        list(prefixes),
        attention_pad_to_length=16,
        batch_pad_to_size=4,
    )

    assert batch_sizes == [4, 4, 4]
    for request_id, serial_output, batch_output in zip(
        prefixes, serial_outputs, batch_outputs
    ):
        assert torch.allclose(
            batch_output.logits,
            serial_output.logits,
            atol=1e-5,
            rtol=1e-5,
        )
        assert_cache_state_equal(
            batch_cache.get_request_state(request_id),
            serial_cache.get_request_state(request_id),
        )


def test_qwen35_cached_batch_chunks_above_fixed_padding_size():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    serial_cache = HybridCacheManager()
    batch_cache = HybridCacheManager()
    request_ids = [f"req{index}" for index in range(5)]
    prefixes = [[index + 1, index + 2] for index in range(5)]
    queries = [index + 10 for index in range(5)]
    for request_id, prefix in zip(request_ids, prefixes):
        tokens = torch.tensor(prefix)
        model.forward_with_cache(tokens, serial_cache, request_id)
        model.forward_with_cache(tokens, batch_cache, request_id)
    serial_outputs = [
        model.forward_with_cache(torch.tensor([token]), serial_cache, request_id)
        for request_id, token in zip(request_ids, queries)
    ]

    batch_sizes = []
    original_batch_step = model.model.forward_cached_batch_step

    def tracked_batch_step(*args, **kwargs):
        batch_sizes.append(len(args[1]))
        return original_batch_step(*args, **kwargs)

    model.model.forward_cached_batch_step = tracked_batch_step
    batch_outputs = model.forward_with_cache_batch(
        torch.tensor(queries).view(5, 1),
        [1] * 5,
        batch_cache,
        request_ids,
        attention_pad_to_length=16,
        batch_pad_to_size=4,
    )

    assert batch_sizes == [4, 4]
    for request_id, serial_output, batch_output in zip(
        request_ids, serial_outputs, batch_outputs
    ):
        assert torch.allclose(
            batch_output.logits,
            serial_output.logits,
            atol=1e-5,
            rtol=1e-5,
        )
        assert_cache_state_equal(
            batch_cache.get_request_state(request_id),
            serial_cache.get_request_state(request_id),
        )


def test_qwen35_spec_verifier_records_transaction_and_commits_accepted_prefix():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    cache = HybridCacheManager()
    prefix = torch.tensor([1, 2, 3])
    model.forward_with_cache(prefix, cache, "req0")
    before = cache.get_request_state("req0").clone()

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner.hybrid_cache_manager = cache
    runner.is_qwen35_target = True
    runner.rank = 0
    runner.config = {"capture_verification_hidden_taps": True}
    runner.sampler = lambda logits, _temperature: logits.argmax(dim=-1)
    runner._pending_verified_cache_transactions = {}

    seq = Sequence(
        [1, 2, 3, 4],
        4,
        SamplingParams(temperature=0, max_tokens=16),
        request_id="req0",
    )
    seq.mark_computed(3)
    tx = seq.begin_speculation([5, 6])
    batch = SpecDecodeBatch([seq], [Proposal(seq.seq_id, (5, 6))])

    result = runner.verify(batch)[0]
    full = model(torch.tensor([[1, 2, 3, 4, 5, 6]]), return_dict=True).logits
    expected = full[:, 3:6, :].argmax(dim=-1).reshape(-1).tolist()

    assert result.target_token_ids == tuple(expected)
    assert result.metadata["verified_tokens"] == 3
    assert result.metadata["hidden_tap_layers"] == ()
    assert cache.get_request_state("req0").logical_length == 6

    runner.rollback_verified_cache([seq.seq_id])
    assert_cache_state_equal(cache.get_request_state("req0"), before)

    result = runner.verify(batch)[0]
    runner.commit_verified_cache({seq.seq_id: 2})
    replay_cache = HybridCacheManager()
    model.forward_with_cache(torch.tensor([1, 2, 3, 4, 5]), replay_cache, "req0")
    assert result.metadata["hybrid_cache_request_id"] == "req0"
    assert_cache_state_equal(
        cache.get_request_state("req0"),
        replay_cache.get_request_state("req0"),
    )
    seq.rollback(tx)


@pytest.mark.parametrize("accepted_lengths", [(2, 1), (3, 2), (3, 1), (0, 1)])
@pytest.mark.parametrize("reuse_verified_cache", [True, False])
@pytest.mark.parametrize("replay_state_updates", [True, False])
def test_qwen35_spec_verifier_isolates_two_request_cache_transactions(
    accepted_lengths, reuse_verified_cache, replay_state_updates, monkeypatch,
):
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    cache = HybridCacheManager()
    prefixes = {
        "req0": [1, 2, 3],
        "req1": [10, 11, 12],
    }
    for request_id, prefix in prefixes.items():
        model.forward_with_cache(torch.tensor(prefix), cache, request_id)
    before = {
        request_id: cache.get_request_state(request_id).clone()
        for request_id in prefixes
    }

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner.hybrid_cache_manager = cache
    runner.is_qwen35_target = True
    runner.rank = 0
    runner.config = {
        "capture_verification_hidden_taps": True,
        "qwen35_reuse_verified_cache": reuse_verified_cache,
        "qwen35_replay_state_updates": replay_state_updates,
    }
    runner.sampler = lambda logits, _temperature: logits.argmax(dim=-1)
    runner._pending_verified_cache_transactions = {}

    seq0 = Sequence(
        [1, 2, 3, 4],
        4,
        SamplingParams(temperature=0, max_tokens=16),
        request_id="req0",
    )
    seq1 = Sequence(
        [10, 11, 12, 13],
        4,
        SamplingParams(temperature=0, max_tokens=16),
        request_id="req1",
    )
    for seq in (seq0, seq1):
        seq.mark_computed(3)
    tx0 = seq0.begin_speculation([5, 6])
    tx1 = seq1.begin_speculation([14])
    batch = SpecDecodeBatch(
        [seq0, seq1],
        [Proposal(seq0.seq_id, (5, 6)), Proposal(seq1.seq_id, (14,))],
    )

    results = runner.verify(batch)

    assert [result.sequence_id for result in results] == [seq0.seq_id, seq1.seq_id]
    assert [result.metadata["verified_tokens"] for result in results] == [3, 2]
    assert all(result.metadata["tensor_batched"] for result in results)
    assert all(result.metadata["verification_batch_size"] == 2 for result in results)
    stats = runner.get_qwen35_verification_stats()
    assert stats["calls"] == 1
    assert stats["max_batch_size"] == 2
    assert stats["tensor_batched_token_steps"] == 2
    assert stats["total_token_steps"] == 3
    assert stats["last_active_batch_sizes"] == (2, 2, 1)
    assert stats["executed_request_token_steps"] == 5
    assert stats["avoided_request_token_steps"] == 0
    assert stats["executed_model_steps"] == 3
    assert stats["avoided_model_steps"] == 0
    assert sum(stats["first_rejection_position_histogram"].values()) == 2
    assert set(runner._pending_verified_cache_transactions) == {
        seq0.seq_id,
        seq1.seq_id,
    }

    runner.rollback_verified_cache([seq0.seq_id, seq1.seq_id])
    for request_id in prefixes:
        assert_cache_state_equal(
            cache.get_request_state(request_id), before[request_id]
        )

    results = runner.verify(batch)
    assert runner.get_qwen35_verification_stats()["calls"] == 2
    with monkeypatch.context() as failure_patch:
        def fail_replay(*args):
            raise RuntimeError("injected commit failure")

        failure_patch.setattr(runner, "_replay_qwen35_accepted_prefixes", fail_replay)
        with pytest.raises(RuntimeError, match="injected commit failure"):
            runner.commit_verified_cache(
                {seq0.seq_id: accepted_lengths[0], seq1.seq_id: accepted_lengths[1]}
            )
    for request_id in prefixes:
        assert_cache_state_equal(cache.get_request_state(request_id), before[request_id])
    runner.verify(batch)
    replay_calls = []
    original_forward = model.forward_with_cache_batch

    def record_replay(*args, **kwargs):
        replay_calls.append(tuple(args[3]))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward_with_cache_batch", record_replay)
    runner.commit_verified_cache(
        {seq0.seq_id: accepted_lengths[0], seq1.seq_id: accepted_lengths[1]}
    )
    expected_replay = tuple(
        request_id
        for request_id, accepted, verified in zip(
            ("req0", "req1"), accepted_lengths, (3, 2)
        )
        if accepted > 0 and (accepted < verified or not reuse_verified_cache)
        and not replay_state_updates
    )
    assert replay_calls == ([expected_replay] if expected_replay else [])
    assert not runner._pending_verified_cache_transactions
    expected_prefixes = {
        "req0": [1, 2, 3, 4, 5, 6][:3 + accepted_lengths[0]],
        "req1": [10, 11, 12, 13, 14][:3 + accepted_lengths[1]],
    }
    for request_id, expected_prefix in expected_prefixes.items():
        replay_cache = HybridCacheManager()
        model.forward_with_cache(
            torch.tensor(expected_prefix), replay_cache, request_id
        )
        assert_cache_state_equal(
            cache.get_request_state(request_id),
            replay_cache.get_request_state(request_id),
        )

    seq0.rollback(tx0)
    seq1.rollback(tx1)


@pytest.mark.parametrize("accepted", range(8))
def test_qwen35_early_stop_matches_full_verification_and_committed_cache(accepted):
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    reference_cache = HybridCacheManager()
    model.forward_with_cache(torch.tensor([1, 2, 3]), reference_cache, "reference")
    next_input = 4
    draft = []
    for _ in range(8):
        output = model.forward_with_cache(
            torch.tensor([next_input]), reference_cache, "reference"
        )
        next_input = int(greedy_token_ids(output.logits.reshape(1, -1)).item())
        draft.append(next_input)
    draft = draft[:7]
    if accepted < 7:
        draft[accepted] = (draft[accepted] + 1) % 32

    runs = []
    for stop_early in (False, True):
        cache = HybridCacheManager()
        model.forward_with_cache(torch.tensor([1, 2, 3]), cache, "request")
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = model
        runner.hybrid_cache_manager = cache
        runner.is_qwen35_target = True
        runner.rank = 0
        runner.config = {
            "capture_verification_hidden_taps": True,
            "qwen35_replay_state_updates": True,
            "qwen35_stop_verification_at_mismatch": stop_early,
            "qwen35_cached_batch_pad_to_size": 4,
            "qwen35_cached_attention_pad_to_length": 32,
        }
        runner.sampler = lambda logits, _temperature: greedy_token_ids(logits)
        runner._pending_verified_cache_transactions = {}
        seq = Sequence(
            [1, 2, 3, 4], 4,
            SamplingParams(temperature=0, max_tokens=16),
            request_id="request",
        )
        seq.mark_computed(3)
        seq.begin_speculation(draft)
        proposal = Proposal(seq.seq_id, tuple(draft))
        result = runner.verify(SpecDecodeBatch([seq], [proposal]))[0]
        sample = RejectionSampler().sample(proposal, result)
        expected_steps = accepted + 1 if stop_early else 8
        assert result.metadata["verified_tokens"] == expected_steps
        plan = runner._pending_verified_cache_transactions[seq.seq_id]
        assert len(plan.query_tokens) == expected_steps
        assert all(tap.shape[1] == expected_steps for tap in plan.hidden_taps.values())
        assert all(
            len(updates) == expected_steps
            for updates in plan.transaction.linear_updates["request"].values()
        )
        # The pending input token is committed along with accepted drafts.
        runner.commit_verified_cache({seq.seq_id: accepted + 1})
        assert not runner._pending_verified_cache_transactions
        stats = runner.get_qwen35_verification_stats()
        assert stats["executed_request_token_steps"] == expected_steps
        assert stats["avoided_request_token_steps"] == 8 - expected_steps
        assert stats["executed_model_steps"] == expected_steps
        assert stats["avoided_model_steps"] == 8 - expected_steps
        histogram_key = "all_accepted" if accepted == 7 else str(accepted)
        assert stats["first_rejection_position_histogram"] == {histogram_key: 1}
        runs.append((cache, sample))

    assert runs[0][1].emitted_token_ids == runs[1][1].emitted_token_ids
    assert runs[0][1].accepted_proposal_tokens == runs[1][1].accepted_proposal_tokens
    assert runs[0][1].per_position_acceptance == runs[1][1].per_position_acceptance
    assert runs[0][1].used_bonus_token == runs[1][1].used_bonus_token
    assert_cache_state_equal(
        runs[0][0].get_request_state("request"),
        runs[1][0].get_request_state("request"),
    )
    with torch.inference_mode():
        next_outputs = [
            model.forward_with_cache_batch(
                torch.tensor([[9]]), [1], cache, ["request"],
                attention_pad_to_length=32, batch_pad_to_size=4,
            )[0].logits
            for cache, _sample in runs
        ]
    torch.testing.assert_close(next_outputs[0], next_outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen35_early_stop_ragged_batch_preserves_other_request(dtype):
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    model.to(dtype=dtype)
    reference = HybridCacheManager()
    model.forward_with_cache(torch.tensor([5, 6, 7]), reference, "reference")
    next_input = 8
    accepted_draft = []
    for _ in range(2):
        output = model.forward_with_cache(
            torch.tensor([next_input]), reference, "reference"
        )
        next_input = int(greedy_token_ids(output.logits.reshape(1, -1)).item())
        accepted_draft.append(next_input)
    first_a = HybridCacheManager()
    model.forward_with_cache(torch.tensor([1, 2, 3]), first_a, "a")
    first_output = model.forward_with_cache(torch.tensor([4]), first_a, "a")
    wrong_token = (
        int(greedy_token_ids(first_output.logits.reshape(1, -1)).item()) + 1
    ) % 32
    drafts = ((wrong_token, 0, 0, 0, 0, 0, 0), tuple(accepted_draft))

    caches = []
    samples = []
    logit_row_counts = []
    for stop_early in (False, True):
        cache = HybridCacheManager()
        for request_id, prefix in (("a", [1, 2, 3]), ("b", [5, 6, 7])):
            model.forward_with_cache(torch.tensor(prefix), cache, request_id)
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = model
        runner.hybrid_cache_manager = cache
        runner.is_qwen35_target = True
        runner.rank = 0
        runner.config = {
            "capture_verification_hidden_taps": True,
            "qwen35_stop_verification_at_mismatch": stop_early,
            "qwen35_replay_state_updates": True,
            "qwen35_cached_batch_pad_to_size": 4,
            "qwen35_cached_attention_pad_to_length": 32,
        }
        runner.sampler = lambda logits, _temperature: greedy_token_ids(logits)
        runner._pending_verified_cache_transactions = {}
        sequences = [
            Sequence(tokens, 4, SamplingParams(temperature=0, max_tokens=16),
                     request_id=request_id)
            for tokens, request_id in (([1, 2, 3, 4], "a"), ([5, 6, 7, 8], "b"))
        ]
        for seq, draft in zip(sequences, drafts):
            seq.mark_computed(3)
            seq.begin_speculation(draft)
        proposals = [
            Proposal(seq.seq_id, draft)
            for seq, draft in zip(sequences, drafts)
        ]
        rows_this_run = []
        original_compute_logits = model.compute_logits
        def record_logits(hidden):
            rows_this_run.append(hidden.shape[0])
            return original_compute_logits(hidden)
        model.compute_logits = record_logits
        try:
            results = runner.verify(SpecDecodeBatch(sequences, proposals))
        finally:
            model.compute_logits = original_compute_logits
        logit_row_counts.append(rows_this_run)
        assert [result.metadata["verified_tokens"] for result in results] == (
            [1, 3] if stop_early else [8, 3]
        )
        sampled = [
            RejectionSampler().sample(proposal, result)
            for proposal, result in zip(proposals, results)
        ]
        assert [item.accepted_proposal_tokens for item in sampled] == [0, 2]
        runner.commit_verified_cache({sequences[0].seq_id: 1,
                                      sequences[1].seq_id: 3})
        caches.append(cache)
        samples.append(sampled)
    for baseline, early in zip(*samples):
        assert baseline.emitted_token_ids == early.emitted_token_ids
        assert baseline.per_position_acceptance == early.per_position_acceptance
    assert logit_row_counts[1] == logit_row_counts[0][:len(logit_row_counts[1])]
    for request_id in ("a", "b"):
        assert_cache_state_equal(
            caches[0].get_request_state(request_id),
            caches[1].get_request_state(request_id),
        )


def test_qwen35_early_stop_sampling_error_rolls_back_cache():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    cache = HybridCacheManager()
    model.forward_with_cache(torch.tensor([1, 2, 3]), cache, "request")
    before = cache.get_request_state("request").clone()
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner.hybrid_cache_manager = cache
    runner.is_qwen35_target = True
    runner.rank = 0
    runner.config = {
        "capture_verification_hidden_taps": True,
        "qwen35_stop_verification_at_mismatch": True,
        "qwen35_replay_state_updates": True,
        "qwen35_cached_batch_pad_to_size": 4,
        "qwen35_cached_attention_pad_to_length": 32,
    }
    runner._pending_verified_cache_transactions = {}
    calls = 0

    def fail_on_second_step(logits, temperatures, *, hidden_states=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected sampling failure")
        return greedy_token_ids(logits)

    runner._sample_qwen35_logits = fail_on_second_step
    probe = HybridCacheManager()
    model.forward_with_cache(torch.tensor([1, 2, 3]), probe, "probe")
    first_output = model.forward_with_cache(torch.tensor([4]), probe, "probe")
    first_draft = int(greedy_token_ids(first_output.logits.reshape(1, -1)).item())
    seq = Sequence(
        [1, 2, 3, 4], 4, SamplingParams(temperature=0, max_tokens=16),
        request_id="request",
    )
    seq.mark_computed(3)
    transaction = seq.begin_speculation([first_draft, 5])
    with pytest.raises(RuntimeError, match="injected sampling failure"):
        runner.verify(SpecDecodeBatch(
            [seq], [Proposal(seq.seq_id, (first_draft, 5))]
        ))
    assert calls == 2
    assert_cache_state_equal(cache.get_request_state("request"), before)
    assert cache.estimate_bytes()["active_transactions"] == 0
    assert not runner._pending_verified_cache_transactions
    seq.rollback(transaction)


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("micro_batch", [1, 4])
def test_rank_one_restore_matches_verified_prefix_exactly(
    accepted, prefill, dtype, micro_batch, monkeypatch,
):
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    init_small(model)
    model.to(dtype=dtype)
    cache = HybridCacheManager()
    ids = ["r0", "r1"]
    if prefill:
        for rid in ids:
            model.forward_with_cache(torch.tensor([1, 2, 3]), cache, rid)
    transactions = {
        rid: cache.begin_transaction([rid], {rid: length})
        for rid, length in zip(ids, [3, 2])
    }
    reference_states = {
        (rid, 0): cache.ensure_request(rid).clone() for rid in ids
    }
    record_step = cache.record_step

    def capture_step(tx, rid, tokens_executed):
        record_step(tx, rid, tokens_executed)
        reference_states[rid, tokens_executed] = cache.ensure_request(rid).clone()

    monkeypatch.setattr(cache, "record_step", capture_step)
    model.forward_with_cache_batch(
        torch.tensor([[4, 5, 6], [7, 8, 0]]), [3, 2], cache, ids,
        transactions=transactions, record_linear_updates=True,
        attention_pad_to_length=16, batch_pad_to_size=micro_batch,
    )
    stats = cache.estimate_bytes()
    assert stats["transaction_linear_update_bytes"] > 0
    assert stats["transaction_snapshot_state_bytes"] == 0
    if accepted:
        saved = transactions["r0"].linear_updates["r0"].pop(0)
        before_fallback = cache.ensure_request("r0").clone()
        assert not cache.restore_verified_prefix(transactions["r0"], "r0", accepted)
        assert_cache_state_equal(cache.ensure_request("r0"), before_fallback)
        transactions["r0"].linear_updates["r0"][0] = saved
    expected_cache = HybridCacheManager()
    for rid, length in zip(ids, [3, 2]):
        count = min(accepted, length)
        assert cache.restore_verified_prefix(transactions[rid], rid, count)
        actual = cache.ensure_request(rid)
        expected = reference_states[rid, count]
        assert actual.logical_length == expected.logical_length
        assert actual.linear_layers.keys() == expected.linear_layers.keys()
        for layer_id, state in actual.linear_layers.items():
            torch.testing.assert_close(
                state.recurrent_state, expected.linear_layers[layer_id].recurrent_state,
                rtol=0, atol=0,
            )
            torch.testing.assert_close(
                state.conv_state, expected.linear_layers[layer_id].conv_state,
                rtol=0, atol=0,
            )
        for layer_id, state in actual.full_attention_layers.items():
            for value, wanted in zip(state, expected.full_attention_layers[layer_id]):
                torch.testing.assert_close(value, wanted, rtol=0, atol=0)
        reference = expected_cache.ensure_request(rid)
        reference.logical_length = expected.logical_length
        reference.linear_layers = expected.linear_layers
        reference.full_attention_layers = expected.full_attention_layers
        cache.finalize_replayed_commit(transactions[rid], {rid: count})
    assert cache.estimate_bytes()["transaction_linear_update_bytes"] == 0
    actual_outputs = model.forward_with_cache_batch(
        torch.tensor([[9], [10]]), [1, 1], cache, ids,
        attention_pad_to_length=16, batch_pad_to_size=micro_batch,
    )
    expected_outputs = model.forward_with_cache_batch(
        torch.tensor([[9], [10]]), [1, 1], expected_cache, ids,
        attention_pad_to_length=16, batch_pad_to_size=micro_batch,
    )
    for actual, expected in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)


def test_qwen35_verified_commit_replays_accepted_prefixes_as_one_batch():
    class FakeReplayModel:
        def __init__(self):
            self.param = torch.empty(0)
            self.calls = []

        def parameters(self):
            return iter((self.param,))

        def forward_with_cache_batch(
            self,
            input_ids,
            query_lengths,
            cache_manager,
            request_ids,
            **kwargs,
        ):
            self.calls.append(
                {
                    "input_ids": input_ids.cpu().tolist(),
                    "query_lengths": tuple(query_lengths),
                    "request_ids": tuple(request_ids),
                    "return_logits": kwargs.get("return_logits"),
                    "capture_hidden_taps": kwargs.get("capture_hidden_taps"),
                }
            )
            for request_id, query_length in zip(request_ids, query_lengths):
                cache_manager.ensure_request(request_id).logical_length += query_length

    cache = HybridCacheManager()
    cache.ensure_request("req0").logical_length = 10
    cache.ensure_request("req1").logical_length = 20
    tx0 = cache.begin_transaction(["req0"], {"req0": 3})
    tx1 = cache.begin_transaction(["req1"], {"req1": 2})

    model = FakeReplayModel()
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner.hybrid_cache_manager = cache
    runner.config = {"qwen35_replay_state_updates": False}
    runner._pending_verified_cache_transactions = {
        101: (
            tx0,
            "req0",
            SimpleNamespace(hidden_taps={}),
            (31, 32, 33),
            10,
            (11, 12, 13),
        ),
        102: (
            tx1,
            "req1",
            SimpleNamespace(hidden_taps={}),
            (41, 42),
            20,
            (21, 22),
        ),
    }

    runner.commit_verified_cache({101: 2, 102: 1})

    assert model.calls == [
        {
            "input_ids": [[11, 12], [21, 0]],
            "query_lengths": (2, 1),
            "request_ids": ("req0", "req1"),
            "return_logits": False,
            "capture_hidden_taps": False,
        }
    ]
    assert cache.get_request_state("req0").logical_length == 12
    assert cache.get_request_state("req1").logical_length == 21
    assert runner._pending_verified_cache_transactions == {}


def test_dflash2_verified_commit_projects_only_new_accepted_context():
    class FakeDraftModel:
        config = SimpleNamespace(sliding_window=4)

        def __init__(self):
            self.projected_positions = []

        def combine_hidden_taps(self, hidden_taps):
            return next(iter(hidden_taps.values()))

        def project_combined_context_kv(self, context_states, positions):
            self.projected_positions.append(positions.clone())
            values = context_states.unsqueeze(1)
            return {0: (values, values + 1)}

    class FakeHybridCache:
        def __init__(self):
            self.commits = []

        def commit(self, transaction, accepted):
            self.commits.append((transaction, accepted))

    runner = ModelRunner.__new__(ModelRunner)
    runner.config = {"dflash2_profile_stages": False}
    runner.dflash2_draft_model = FakeDraftModel()
    runner.dflash2_draft_cache = DFlash2DraftCache(sliding_window=4)
    runner.hybrid_cache_manager = FakeHybridCache()
    old_hidden = {5: torch.zeros(1, 2, 2)}
    runner._dflash2_contexts = {
        17: {
            "request_id": "req",
            "anchor_token_id": 20,
            "anchor_position": 7,
            "hidden_taps": old_hidden,
        }
    }
    runner.dflash2_draft_cache.update_target_context(
        "req", old_hidden, anchor_token_id=20, anchor_position=7
    )
    runner.dflash2_draft_cache.install_projected_context(
        "req",
        torch.tensor([[5, 6]]),
        {0: (torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2))},
    )
    runner.dflash2_draft_cache.begin_proposal("req")
    transaction = object()
    output = SimpleNamespace(
        hidden_taps={5: torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])}
    )
    runner._pending_verified_cache_transactions = {
        17: (transaction, "req", output, [31, 32, 33], 7)
    }

    runner.commit_verified_cache({17: 2})

    assert runner.dflash2_draft_model.projected_positions[0].tolist() == [[7, 8]]
    assert runner.dflash2_draft_cache.context_positions_by_request["req"].tolist() == [
        [5, 6, 7, 8]
    ]
    assert runner.dflash2_draft_cache.context_kv_by_request["req"][0][0].shape[2] == 4
    assert runner._dflash2_contexts[17]["hidden_taps"][5].shape[1] == 4
    assert runner._dflash2_contexts[17]["anchor_token_id"] == 32
    assert runner._dflash2_contexts[17]["anchor_position"] == 9
    assert runner.hybrid_cache_manager.commits == [(transaction, {"req": 2})]
    assert runner._pending_verified_cache_transactions == {}
    assert runner.dflash2_draft_cache._proposal_context_snapshots_by_request == {}


def test_qwen35_cached_runner_prefills_requests_independently():
    calls = []

    class FakeModel:
        def forward_with_cache(
            self, input_ids, _cache_manager, request_id, capture_hidden_taps=False
        ):
            calls.append(
                (
                    request_id,
                    input_ids.tolist(),
                    capture_hidden_taps,
                    torch.is_inference_mode_enabled(),
                )
            )
            logits = torch.zeros(1, len(input_ids), 32)
            logits[:, -1, 7 if request_id == "req0" else 9] = 1
            return SimpleNamespace(logits=logits, hidden_taps={})

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = FakeModel()
    runner.hybrid_cache_manager = HybridCacheManager()
    runner.is_qwen35_target = True
    runner.dflash2_draft_model = None
    runner.rank = 0
    runner.sampler = lambda logits, _temperature: logits.argmax(dim=-1)
    runner.prepare_prefill = lambda seqs: torch.tensor(seqs[0].token_ids)
    runner.prepare_sample = lambda seqs: torch.zeros(len(seqs))

    seqs = [
        Sequence([1, 2], request_id="req0"),
        Sequence([3, 4], request_id="req1"),
    ]

    token_ids = runner.run(seqs, is_prefill=True)

    assert token_ids.tolist() == [7, 9]
    assert calls == [
        ("req0", [1, 2], False, True),
        ("req1", [3, 4], False, True),
    ]


def test_qwen35_cached_runner_tensor_batches_decode_in_request_order():
    calls = []

    class FakeModel:
        def forward_with_cache_batch(
            self,
            input_ids,
            query_lengths,
            _cache_manager,
            request_ids,
            capture_hidden_taps=False,
            attention_pad_to_length=None,
            batch_pad_to_size=None,
        ):
            calls.append(
                (
                    input_ids.tolist(),
                    list(query_lengths),
                    list(request_ids),
                    capture_hidden_taps,
                    attention_pad_to_length,
                    batch_pad_to_size,
                )
            )
            outputs = []
            for token_id in (7, 9):
                logits = torch.zeros(1, 1, 32)
                logits[:, :, token_id] = 1
                outputs.append(SimpleNamespace(logits=logits, hidden_taps={}))
            return outputs

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = FakeModel()
    runner.hybrid_cache_manager = HybridCacheManager()
    runner.is_qwen35_target = True
    runner.dflash2_draft_model = None
    runner.rank = 0
    runner.config = {"max_model_length": 64, "max_num_sequences": 8}
    runner.sampler = lambda logits, _temperature: logits.argmax(dim=-1)
    runner.prepare_decode = lambda seqs: torch.tensor([seq.last_token for seq in seqs])
    runner.prepare_sample = lambda seqs: torch.zeros(len(seqs))

    seqs = [
        Sequence([1, 2], request_id="req0"),
        Sequence([3, 4], request_id="req1"),
    ]

    token_ids = runner.run(seqs, is_prefill=False)

    assert token_ids.tolist() == [7, 9]
    assert calls == [
        (
            [[2], [4]],
            [1, 1],
            ["req0", "req1"],
            False,
            None,
            None,
        )
    ]


def test_qwen35_allocate_kv_cache_skips_unused_generic_pool(monkeypatch):
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = {
        "block_size": 4,
        "max_model_length": 16,
        "max_num_sequences": 2,
        "gpu_memory_utilization": 0.9,
        "target_cache_memory_fraction": 0.05,
    }
    runner.block_size = 4
    runner.rank = 0
    runner.world_size = 1
    runner.default_dtype = torch.float16
    runner.is_qwen35_target = True
    runner.qwen35_text_config = normalize_qwen35_text_config(tiny_config(num_layers=4))
    runner.model = SimpleNamespace(
        modules=lambda: iter(()), parameters=lambda: iter(())
    )
    runner.hybrid_cache_manager = HybridCacheManager()
    runner.dflash2_draft_model = None
    runner.dflash2_draft_cache = None
    runner._memory_trace = MemoryTrace(enabled=True)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 1024**3, 10 * 1024**3))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.peak": 2 * 1024**3,
            "allocated_bytes.all.current": 1 * 1024**3,
        },
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_args, **_kwargs: 0)

    def fail_zeros(*_args, **_kwargs):
        raise AssertionError("Qwen3.8 must not allocate the unused generic KV pool")

    monkeypatch.setattr(torch, "zeros", fail_zeros)

    runner.allocate_kv_cache()

    capacity = runner.config["qwen35_dynamic_cache_capacity"]
    assert runner.config["qwen35_uses_generic_kv_cache"] is False
    assert capacity["generic_kv_cache_layers"] == 0
    assert capacity["generic_kv_pool_bytes"] == 0
    assert runner.config["max_cached_blocks"] == capacity["configured_blocks"]


def test_runner_dflash2_b2_proposals_rollback_as_one_batch_on_failure():
    class FakeDraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                num_speculative_tokens=7,
                sliding_window=16,
            )
            self.calls = []

        def forward_block(self, **kwargs):
            self.calls.append(kwargs)
            return (
                SimpleNamespace(
                    token_ids=torch.tensor([[20, 21, 22], [30, 31, 32]]),
                    valid_lengths=torch.tensor([2]),
                ),
                {},
            )

    runner = ModelRunner.__new__(ModelRunner)
    runner.dflash2_draft_model = FakeDraft()
    runner.dflash2_draft_cache = DFlash2DraftCache(sliding_window=16)
    runner.config = {"dflash2_debug_metadata": False}
    runner.model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=object()),
        lm_head=object(),
    )
    seqs = [
        Sequence([1], request_id="req0"),
        Sequence([2], request_id="req1"),
    ]
    runner._dflash2_contexts = {
        seq.seq_id: {
            "request_id": seq.request_id,
            "anchor_token_id": seq.last_token,
            "anchor_position": 1,
            "hidden_taps": {5: torch.zeros(1, 1, 1)},
        }
        for seq in seqs
    }
    for seq in seqs:
        runner.dflash2_draft_cache.update_target_context(
            seq.request_id,
            {5: torch.zeros(1, 1, 1)},
            seq.last_token,
            1,
        )
        runner.dflash2_draft_cache.install_projected_context(
            seq.request_id,
            torch.tensor([[0]]),
            {0: (torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1))},
        )

    with pytest.raises(ValueError, match="valid lengths have the wrong batch size"):
        runner.propose_dflash2(seqs, [2, 2])

    assert len(runner.dflash2_draft_model.calls) == 1
    call = runner.dflash2_draft_model.calls[0]
    assert call["anchor_token_ids"].tolist() == [1, 2]
    assert call["context_valid_mask"].tolist() == [[True], [True]]
    assert runner.dflash2_draft_cache.lengths_by_request == {
        "req0": 1,
        "req1": 1,
    }
    assert runner.dflash2_draft_cache._proposal_length_snapshots_by_request == {}


def test_runner_target_decision_diagnostics_capture_top2_and_cap_history():
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = {
        "debug_target_decisions": True,
        "max_metrics_history": 2,
    }
    runner._target_decision_diagnostics = []
    seq = Sequence([10, 11], request_id="req0")

    runner._record_target_decision(
        seq,
        torch.tensor([0.5, 2.0, 1.75]),
        output_index=0,
        source="prefill",
        active_batch_size=1,
        input_token_id=11,
    )
    runner._record_target_decision(
        seq,
        torch.tensor([3.0, 2.5, 1.0]),
        output_index=1,
        source="decode",
        active_batch_size=4,
        input_token_id=2,
    )
    runner._record_target_decision(
        seq,
        torch.tensor([1.0, 4.0, 3.0]),
        output_index=2,
        source="verify",
        active_batch_size=3,
        input_token_id=7,
        query_offset=1,
    )

    diagnostics = runner.get_target_decision_diagnostics()
    assert [record["output_index"] for record in diagnostics] == [1, 2]
    assert diagnostics[0]["top1_token_id"] == 0
    assert diagnostics[0]["top2_token_id"] == 1
    assert diagnostics[0]["top2_margin"] == pytest.approx(0.5)
    assert diagnostics[1] == {
        "request_id": "req0",
        "sequence_id": seq.seq_id,
        "output_index": 2,
        "source": "verify",
        "active_batch_size": 3,
        "input_token_id": 7,
        "query_offset": 1,
        "argmax_token_id": 1,
        "exact_tie_break_policy": "highest_token_id",
        "max_tie_count": 1,
        "max_tie_min_token_id": 1,
        "max_tie_max_token_id": 1,
        "top1_token_id": 1,
        "top1_logit": 4.0,
        "top2_token_id": 2,
        "top2_logit": 3.0,
        "top2_margin": 1.0,
    }


def test_runner_target_decision_diagnostics_are_disabled_by_default():
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = {}
    runner._target_decision_diagnostics = []
    seq = Sequence([10], request_id="req0")

    runner._record_target_decision(
        seq,
        torch.tensor([0.0, 1.0]),
        output_index=0,
        source="prefill",
        active_batch_size=1,
    )

    assert runner.get_target_decision_diagnostics() == []


def test_runner_qwen35_resolves_bf16_greedy_ties_with_fp32_lm_head():
    config = tiny_config(num_layers=1)
    config["text_config"]["vocab_size"] = 4
    model = Qwen35ForCausalLM.from_config_dict(config)
    model.lm_head.weight.data.zero_()
    model.lm_head.weight.data[1, 0] = 1.0
    model.lm_head.weight.data[2, 0] = 0.5
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner.config = {"resolve_greedy_ties_with_fp32_lm_head": True}
    runner.sampler = lambda logits, _temperature: logits.argmax(dim=-1)

    logits = torch.tensor([[0.0, 2.0, 2.0, 1.0]])
    hidden_states = torch.zeros(1, model.config.hidden_size)
    hidden_states[0, 0] = 1.0

    token_ids = runner._sample_qwen35_logits(
        logits,
        torch.zeros(1),
        hidden_states=hidden_states,
    )

    assert token_ids.tolist() == [1]


def test_runner_qwen35_default_tie_policy_matches_sampling_documentation():
    config = tiny_config(num_layers=1)
    config["text_config"]["vocab_size"] = 4
    model = Qwen35ForCausalLM.from_config_dict(config)
    model.lm_head.weight.data.zero_()
    model.lm_head.weight.data[1, 0] = 1.0
    model.lm_head.weight.data[2, 0] = 0.5
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner.config = {}
    runner.sampler = lambda logits, _temperature: logits.argmax(dim=-1)

    logits = torch.tensor([[0.0, 2.0, 2.0, 1.0]])
    hidden_states = torch.zeros(1, model.config.hidden_size)
    hidden_states[0, 0] = 1.0

    token_ids = runner._sample_qwen35_logits(
        logits,
        torch.zeros(1),
        hidden_states=hidden_states,
    )

    assert token_ids.tolist() == [2]


def test_runner_dflash2_tensor_batches_unequal_contexts_and_independent_budgets():
    class FakeDraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(num_speculative_tokens=7, sliding_window=16)
            self.calls = []

        def forward_block(self, **kwargs):
            self.calls.append(kwargs)
            return (
                ProposalBatch(
                    token_ids=torch.tensor([[20, 21, 22, 23], [30, 31, 32, 33]]),
                    valid_lengths=torch.tensor([4, 3]),
                ),
                {},
            )

    runner = ModelRunner.__new__(ModelRunner)
    runner.dflash2_draft_model = FakeDraft()
    runner.dflash2_draft_cache = DFlash2DraftCache(sliding_window=16)
    runner.config = {
        "dflash2_debug_metadata": False,
        "dflash2_profile_stages": False,
    }
    runner.model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=object()),
        lm_head=object(),
    )
    seqs = [
        Sequence([1], request_id="req0"),
        Sequence([2], request_id="req1"),
        Sequence([3], request_id="req2"),
    ]
    runner._dflash2_contexts = {
        seq.seq_id: {
            "request_id": seq.request_id,
            "anchor_token_id": seq.last_token,
            "anchor_position": index + 1,
            "hidden_taps": {5: torch.zeros(1, index + 1, 1)},
        }
        for index, seq in enumerate(seqs)
    }
    for index, seq in enumerate(seqs):
        length = index + 1
        runner.dflash2_draft_cache.update_target_context(
            seq.request_id,
            {5: torch.zeros(1, length, 1)},
            seq.last_token,
            length,
        )
        runner.dflash2_draft_cache.install_projected_context(
            seq.request_id,
            torch.arange(length).view(1, -1),
            {
                0: (
                    torch.full((1, 1, length, 1), float(index)),
                    torch.full((1, 1, length, 1), float(index + 1)),
                )
            },
        )

    proposals = runner.propose_dflash2(seqs, [2, 0, 1])

    assert len(runner.dflash2_draft_model.calls) == 1
    call = runner.dflash2_draft_model.calls[0]
    assert call["anchor_token_ids"].tolist() == [1, 3]
    assert call["context_positions"].tolist() == [[0, 0, 0], [0, 1, 2]]
    assert call["context_valid_mask"].tolist() == [
        [True, False, False],
        [True, True, True],
    ]
    assert call["context_kv_by_layer"][0][0].shape == (2, 1, 3, 1)
    assert [proposal.token_ids for proposal in proposals] == [(20, 21), (), (30,)]
    assert proposals[0].metadata["proposal_batch_size"] == 2
    assert proposals[2].metadata["proposal_batch_size"] == 2
    assert runner.dflash2_draft_cache.lengths_by_request == {
        "req0": 3,
        "req1": 2,
        "req2": 4,
    }
    assert runner.get_dflash2_profile_stats()["proposal_calls"] == 1
    assert runner.get_dflash2_profile_stats()["proposal_requests"] == 2
    assert runner.get_dflash2_profile_stats()["max_proposal_batch_size"] == 2
    assert runner.get_dflash2_profile_stats()["tensor_batched_proposal_calls"] == 1
