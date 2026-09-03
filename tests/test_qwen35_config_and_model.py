import torch
import pytest

from myvllm.models.qwen3_8 import (
    Qwen35ForCausalLM,
    Qwen35GatedDeltaNet,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
)
from myvllm.engine.hybrid_cache import HybridCacheManager
from myvllm.engine.model_runner import ModelRunner
from myvllm.models.qwen3_8_config import (
    QWEN35_DEFAULT_HIDDEN_TAPS,
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode.types import Proposal, SpecDecodeBatch
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
    cache.commit(tx, {"req0": 3})

    replay_cache = HybridCacheManager()
    model.forward_with_cache(torch.tensor([1, 2, 3, 4, 5, 6]), replay_cache, "req0")
    assert_cache_state_equal(
        cache.get_request_state("req0"),
        replay_cache.get_request_state("req0"),
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
