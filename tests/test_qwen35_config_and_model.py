from types import SimpleNamespace

import pytest
import torch

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

    def tracked_batch_step(input_ids, request_states, hidden_tap_layers=None):
        batch_sizes.append(len(request_states))
        return original_batch_step(
            input_ids,
            request_states,
            hidden_tap_layers=hidden_tap_layers,
        )

    model.model.forward_cached_batch_step = tracked_batch_step
    batch_outputs = model.forward_with_cache_batch(
        padded_queries,
        [3, 2],
        batch_cache,
        list(prefixes),
    )

    assert batch_sizes == [2, 2, 1]
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


def test_qwen35_spec_verifier_isolates_two_request_cache_transactions():
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
    runner.config = {"capture_verification_hidden_taps": True}
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
    assert runner.get_qwen35_verification_stats() == {
        "calls": 1,
        "max_batch_size": 2,
        "tensor_batched_token_steps": 2,
        "total_token_steps": 3,
        "last_active_batch_sizes": (2, 2, 1),
    }
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
    runner.commit_verified_cache({seq0.seq_id: 2, seq1.seq_id: 1})
    expected_prefixes = {
        "req0": [1, 2, 3, 4, 5],
        "req1": [10, 11, 12, 13],
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
    assert runner.dflash2_draft_cache.context_positions_by_request[
        "req"
    ].tolist() == [[5, 6, 7, 8]]
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
        ):
            calls.append(
                (
                    input_ids.tolist(),
                    list(query_lengths),
                    list(request_ids),
                    capture_hidden_taps,
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
        )
    ]


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
                    token_ids=torch.tensor(
                        [[20, 21, 22, 23], [30, 31, 32, 33]]
                    ),
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
