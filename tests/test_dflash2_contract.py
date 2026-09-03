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
    DFlash2StandaloneProposer,
    ProposalBatch,
    audit_dflash2_checkpoint,
    expected_dflash2_weight_shapes,
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
