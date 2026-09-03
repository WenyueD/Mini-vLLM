import pytest
import torch
import json
import struct

from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode.dflash2 import (
    DRAFT_TARGET_TAP_LAYERS,
    DFlash2Config,
    DFlash2DraftCache,
    DFlash2DraftModel,
    DFlash2StandaloneProposer,
    DFlash2TargetContext,
    ProposalBatch,
    audit_dflash2_checkpoint,
    expected_dflash2_weight_shapes,
    grouped_dynamic_conv,
    select_topk_candidates,
)


def test_dflash2_config_contract_uses_zero_based_target_taps():
    config = DFlash2Config.from_mapping(
        {
            "block_size": 8,
            "target_layer_ids": [5, 19, 33, 47, 61],
            "selector_top_k": 16,
            "selector_rank": 256,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "vocab_size": 248320,
            "num_sliding_attention_layers": 5,
            "sliding_window": 2048,
        }
    )

    config.validate_contract()
    assert config.num_speculative_tokens == 7
    assert config.target_layer_ids == DRAFT_TARGET_TAP_LAYERS


def test_dflash2_config_reads_real_nested_dflash_config():
    config = DFlash2Config.from_mapping(
        {
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 5,
            "sliding_window": 2048,
            "vocab_size": 248320,
            "rope_parameters": {"rope_theta": 10000000},
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 248070,
                "selector_rank": 256,
                "selector_top_k": 16,
                "target_layer_ids": [5, 19, 33, 47, 61],
            },
        }
    )

    config.validate_contract()
    assert config.mask_token_id == 248070
    assert config.rope_theta == 10000000
    assert config.num_sliding_attention_layers == 5


def test_dflash2_config_rejects_one_based_target_taps():
    config = DFlash2Config(target_layer_ids=(6, 20, 34, 48, 62))
    with pytest.raises(ValueError, match="zero-based"):
        config.validate_contract()


def test_proposal_batch_shape_and_budget_contract():
    batch = ProposalBatch(
        token_ids=torch.tensor([[10, 11, 12], [20, 0, 0]]),
        valid_lengths=torch.tensor([3, 1]),
    )
    seq = Sequence([1, 2], 4, SamplingParams(temperature=0), request_id="r0")

    proposal = batch.proposal_for(seq, row=0, include_metadata=False)

    assert proposal.token_ids == (10, 11, 12)
    assert proposal.metadata == {"kind": "dflash2", "block_parallel": True}


def test_expected_dflash2_weight_shapes_match_checkpoint_audit():
    shapes = expected_dflash2_weight_shapes(DFlash2Config())

    assert len(shapes) == 81
    assert shapes["candidate_selector.hidden_projection.weight"] == (256, 5120)
    assert shapes["candidate_selector.predecessor_codebook"] == (248320, 256)
    assert shapes["fc.weight"] == (5120, 25600)
    assert shapes["layers.0.attention_conv.base_kernel"] == (2, 2, 5120)
    assert shapes["layers.0.attention_conv.kernel_projection.weight"] == (1280, 5120)
    assert shapes["layers.4.self_attn.q_proj.weight"] == (4096, 5120)
    assert shapes["layers.4.self_attn.k_proj.weight"] == (1024, 5120)


def write_header_only_safetensors(path, shapes):
    header = {
        name: {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [0, 0],
        }
        for name, shape in shapes.items()
    }
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def test_audit_dflash2_checkpoint_passes_header_only_fixture(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "num_hidden_layers": 5,
                "sliding_window": 2048,
                "vocab_size": 248320,
                "dflash_config": {
                    "block_size": 8,
                    "conv_group_size": 16,
                    "conv_kernel_size": 2,
                    "mask_token_id": 248070,
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    "target_layer_ids": [5, 19, 33, 47, 61],
                },
            }
        ),
        encoding="utf-8",
    )
    write_header_only_safetensors(
        tmp_path / "model.safetensors",
        expected_dflash2_weight_shapes(DFlash2Config()),
    )

    audit = audit_dflash2_checkpoint(str(tmp_path))

    assert audit.passed
    assert audit.checkpoint_tensors == 81
    assert audit.expected_tensors == 81
    assert audit.dtype_counts == {"BF16": 81}


def test_dflash2_candidate_selector_returns_topk_path():
    scores = torch.tensor(
        [
            [
                [0.0, 4.0, 1.0, 3.0],
                [8.0, 2.0, 7.0, 1.0],
            ]
        ]
    )

    selected = select_topk_candidates(scores, top_k=2)

    assert selected.token_ids.tolist() == [[[1, 3], [0, 2]]]
    assert selected.scores.tolist() == [[[4.0, 3.0], [8.0, 7.0]]]


def test_grouped_dynamic_conv_resets_at_block_boundaries():
    hidden = torch.arange(1, 9, dtype=torch.float32).view(1, 4, 2)
    delta = torch.zeros(1, 4, 2, 1)
    base = torch.ones(2, 2)

    output = grouped_dynamic_conv(hidden, delta, base, block_size=2, group_size=2)

    assert output.tolist() == [[[1.0, 2.0], [4.0, 6.0], [5.0, 6.0], [12.0, 14.0]]]


def tiny_dflash2_config():
    return DFlash2Config(
        block_size=4,
        target_layer_ids=(0, 1),
        selector_top_k=3,
        selector_rank=3,
        conv_kernel_size=2,
        conv_group_size=2,
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_sliding_attention_layers=1,
        sliding_window=8,
        mask_token_id=15,
    )


def init_tiny_module(module):
    generator = torch.Generator().manual_seed(20260903)
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(0.0, 0.02, generator=generator)


def test_dflash2_draft_model_forward_block_proposes_in_one_parallel_call():
    config = tiny_dflash2_config()
    model = DFlash2DraftModel(config, enforce_contract=False)
    token_embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    init_tiny_module(model)
    init_tiny_module(token_embedding)
    init_tiny_module(lm_head)
    hidden_taps = {
        0: torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(1)),
        1: torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(2)),
    }

    proposals, debug = model.forward_block(
        anchor_token_ids=torch.tensor([2]),
        anchor_positions=torch.tensor([0]),
        hidden_taps=hidden_taps,
        token_embedding=token_embedding,
        lm_head=lm_head,
        return_debug=True,
    )

    assert proposals.token_ids.shape == (1, config.num_speculative_tokens)
    assert proposals.valid_lengths.tolist() == [config.num_speculative_tokens]
    assert debug["context_shape"] == (1, 1, config.hidden_size)
    assert debug["query_shape"] == (1, config.block_size, config.hidden_size)
    assert debug["unary_logits_shape"] == (
        1,
        config.num_speculative_tokens,
        config.vocab_size,
    )
    assert set(debug) == {
        "context_shape",
        "query_shape",
        "proposal_hidden_shape",
        "unary_logits_shape",
        "context_checksum",
        "proposal_hidden_checksum",
    }


def test_dflash2_proposer_requires_block_parallel_model():
    class SerialDraft:
        supports_block_parallel = False

    with pytest.raises(ValueError, match="block-parallel"):
        DFlash2StandaloneProposer(DFlash2Config(), SerialDraft())


def test_dflash2_proposer_calls_model_once_for_the_block_and_tracks_cache():
    calls = []

    class BlockDraft:
        supports_block_parallel = True

        def propose_block(self, sequences, max_tokens, draft_cache):
            calls.append((tuple(seq.request_id for seq in sequences), tuple(max_tokens)))
            assert isinstance(draft_cache, DFlash2DraftCache)
            return ProposalBatch(
                token_ids=torch.tensor([[101, 102, 0], [201, 0, 0]]),
                valid_lengths=torch.tensor([2, 1]),
                metadata={"selector_top_k": 16},
            )

    sequences = [
        Sequence([1, 2], 4, SamplingParams(temperature=0), request_id="a"),
        Sequence([3, 4], 4, SamplingParams(temperature=0), request_id="b"),
    ]
    cache = DFlash2DraftCache(sliding_window=2048)
    proposer = DFlash2StandaloneProposer(
        DFlash2Config(),
        BlockDraft(),
        draft_cache=cache,
        include_debug_metadata=True,
    )

    proposals = proposer.propose_batch(sequences, [3, 1])

    assert calls == [(("a", "b"), (3, 1))]
    assert [proposal.token_ids for proposal in proposals] == [(101, 102), (201,)]
    assert proposals[0].metadata["selector_top_k"] == 16
    assert cache.lengths_by_request == {"a": 2, "b": 1}


def test_dflash2_proposer_uses_target_context_provider_for_standalone_model():
    calls = []

    class ContextDraft:
        supports_block_parallel = True

        def forward_block(
            self,
            anchor_token_ids,
            anchor_positions,
            hidden_taps,
            token_embedding,
            lm_head,
            return_debug=False,
        ):
            calls.append(
                (
                    tuple(anchor_token_ids.tolist()),
                    tuple(anchor_positions.tolist()),
                    sorted(hidden_taps),
                    return_debug,
                )
            )
            return (
                ProposalBatch(
                    token_ids=torch.tensor([[10, 11, 12, 13, 14, 15, 16]]),
                    valid_lengths=torch.tensor([7]),
                    metadata={},
                ),
                {"query_shape": (1, 8, 5120)},
            )

        def propose_block(self, sequences, max_tokens, draft_cache):
            raise AssertionError("context provider path should call forward_block")

    seq = Sequence([1, 2], 4, SamplingParams(temperature=0), request_id="ctx")

    def provider(sequences):
        assert sequences == [seq]
        return DFlash2TargetContext(
            hidden_taps={5: torch.zeros(1, 1, 1)},
            anchor_token_ids=torch.tensor([2]),
            anchor_positions=torch.tensor([3]),
            token_embedding=lambda ids: ids,
            lm_head=lambda hidden: hidden,
        )

    proposer = DFlash2StandaloneProposer(
        DFlash2Config(),
        ContextDraft(),
        include_debug_metadata=True,
        target_context_provider=provider,
    )

    proposals = proposer.propose_batch([seq], [3])

    assert calls == [((2,), (3,), [5], True)]
    assert proposals[0].token_ids == (10, 11, 12)
    assert proposals[0].metadata["query_shape"] == (1, 8, 5120)
