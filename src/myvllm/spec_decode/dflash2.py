from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
import torch.nn as nn

from myvllm.engine.sequence import Sequence
from myvllm.models.qwen3_8 import Qwen35RMSNorm
from myvllm.spec_decode.proposer import Proposer
from myvllm.spec_decode.types import Proposal


DRAFT_MODEL_ID = "incoai/Qwen3.8-27B-DFlash2"
DRAFT_TARGET_TAP_LAYERS = (5, 19, 33, 47, 61)


@dataclass(frozen=True)
class DFlash2Config:
    block_size: int = 8
    target_layer_ids: tuple[int, ...] = DRAFT_TARGET_TAP_LAYERS
    selector_top_k: int = 16
    selector_rank: int = 256
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_sliding_attention_layers: int = 5
    sliding_window: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    mask_token_id: int = 248070

    @property
    def num_speculative_tokens(self) -> int:
        return self.block_size - 1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DFlash2Config":
        dflash = raw.get("dflash_config", raw.get("draft_config", raw))
        if not isinstance(dflash, Mapping):
            raise ValueError("DFlash2 config must be a mapping")
        target_layer_ids = tuple(
            int(value)
            for value in dflash.get("target_layer_ids", cls.target_layer_ids)
        )
        rope_parameters = raw.get("rope_parameters") or {}
        return cls(
            block_size=int(dflash.get("block_size", cls.block_size)),
            target_layer_ids=target_layer_ids,
            selector_top_k=int(dflash.get("selector_top_k", cls.selector_top_k)),
            selector_rank=int(dflash.get("selector_rank", cls.selector_rank)),
            conv_kernel_size=int(
                dflash.get(
                    "conv_kernel_size",
                    dflash.get("conv_kernel", cls.conv_kernel_size),
                )
            ),
            conv_group_size=int(dflash.get("conv_group_size", cls.conv_group_size)),
            vocab_size=int(raw.get("vocab_size", dflash.get("vocab_size", cls.vocab_size))),
            hidden_size=int(raw.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(raw.get("intermediate_size", cls.intermediate_size)),
            num_attention_heads=int(raw.get("num_attention_heads", cls.num_attention_heads)),
            num_key_value_heads=int(raw.get("num_key_value_heads", cls.num_key_value_heads)),
            head_dim=int(raw.get("head_dim", cls.head_dim)),
            num_sliding_attention_layers=int(
                raw.get(
                    "num_hidden_layers",
                    dflash.get(
                    "num_sliding_attention_layers",
                        dflash.get(
                            "sliding_attention_layers",
                            cls.num_sliding_attention_layers,
                        ),
                    ),
                )
            ),
            sliding_window=int(raw.get("sliding_window", dflash.get("sliding_window", cls.sliding_window))),
            rms_norm_eps=float(raw.get("rms_norm_eps", cls.rms_norm_eps)),
            rope_theta=float(
                rope_parameters.get("rope_theta", raw.get("rope_theta", cls.rope_theta))
            ),
            mask_token_id=int(dflash.get("mask_token_id", cls.mask_token_id)),
        )

    @classmethod
    def from_pretrained_path(cls, model_name_or_path: str) -> "DFlash2Config":
        path = Path(model_name_or_path).expanduser()
        if path.is_dir():
            config_path = path / "config.json"
        else:
            from huggingface_hub import snapshot_download

            config_dir = Path(
                snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns=["config.json"],
                )
            )
            config_path = config_dir / "config.json"
        if not config_path.exists():
            raise ValueError(f"DFlash2 config.json not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def validate_contract(self) -> None:
        if self.block_size != 8:
            raise ValueError("DFlash2 Qwen3.8 contract requires block_size=8")
        if self.num_speculative_tokens != 7:
            raise ValueError("DFlash2 Qwen3.8 must propose exactly up to K=7")
        if self.target_layer_ids != DRAFT_TARGET_TAP_LAYERS:
            raise ValueError(
                "DFlash2 target taps must be zero-based layers "
                f"{DRAFT_TARGET_TAP_LAYERS}"
            )
        if self.selector_top_k <= 0 or self.selector_rank <= 0:
            raise ValueError("DFlash2 selector dimensions must be positive")
        if self.conv_kernel_size != 2:
            raise ValueError("DFlash2 Qwen3.8 contract requires conv_kernel_size=2")
        if self.vocab_size != 248320:
            raise ValueError("DFlash2 Qwen3.8 contract requires vocab_size=248320")
        if self.hidden_size != 5120:
            raise ValueError("DFlash2 Qwen3.8 contract requires hidden_size=5120")
        if self.num_sliding_attention_layers != 5:
            raise ValueError("DFlash2 Qwen3.8 contract requires 5 sliding layers")
        if self.num_attention_heads * self.head_dim != 4096:
            raise ValueError("DFlash2 attention projection width must be 4096")
        if self.num_key_value_heads * self.head_dim != 1024:
            raise ValueError("DFlash2 KV projection width must be 1024")


class DFlash2CandidateSelector(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.hidden_projection = nn.Linear(
            config.hidden_size,
            config.selector_rank,
            bias=False,
        )
        self.predecessor_codebook = nn.Parameter(
            torch.empty(config.vocab_size, config.selector_rank)
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(config.vocab_size, config.selector_rank)
        )


class DFlash2DynamicConv(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.base_kernel = nn.Parameter(
            torch.empty(config.conv_kernel_size, 2, config.hidden_size)
        )
        self.kernel_projection = nn.Linear(
            config.hidden_size,
            config.hidden_size // 4,
            bias=False,
        )


class DFlash2Attention(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = Qwen35RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(config.head_dim, eps=config.rms_norm_eps)


class DFlash2MLP(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )


class DFlash2DecoderLayer(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.attention_conv = DFlash2DynamicConv(config)
        self.input_layernorm = Qwen35RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp = DFlash2MLP(config)
        self.mlp_conv = DFlash2DynamicConv(config)
        self.post_attention_layernorm = Qwen35RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn = DFlash2Attention(config)


class DFlash2DraftModel(nn.Module):
    """Loadable DFlash2 checkpoint shell.

    This class deliberately does not advertise block-parallel proposal support
    until the standalone proposal math is implemented and aligned with Oracle.
    """

    supports_block_parallel = False

    def __init__(self, config: DFlash2Config):
        super().__init__()
        config.validate_contract()
        self.config = config
        self.candidate_selector = DFlash2CandidateSelector(config)
        self.fc = nn.Linear(
            config.hidden_size * len(config.target_layer_ids),
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            DFlash2DecoderLayer(config)
            for _ in range(config.num_sliding_attention_layers)
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def expected_dflash2_weight_shapes(config: DFlash2Config) -> dict[str, tuple[int, ...]]:
    config.validate_contract()
    shapes = {
        "candidate_selector.hidden_projection.weight": (
            config.selector_rank,
            config.hidden_size,
        ),
        "candidate_selector.predecessor_codebook": (
            config.vocab_size,
            config.selector_rank,
        ),
        "candidate_selector.successor_codebook": (
            config.vocab_size,
            config.selector_rank,
        ),
        "fc.weight": (
            config.hidden_size,
            config.hidden_size * len(config.target_layer_ids),
        ),
        "hidden_norm.weight": (config.hidden_size,),
        "norm.weight": (config.hidden_size,),
    }
    for layer_idx in range(config.num_sliding_attention_layers):
        prefix = f"layers.{layer_idx}"
        shapes.update(
            {
                f"{prefix}.attention_conv.base_kernel": (
                    config.conv_kernel_size,
                    2,
                    config.hidden_size,
                ),
                f"{prefix}.attention_conv.kernel_projection.weight": (
                    config.hidden_size // 4,
                    config.hidden_size,
                ),
                f"{prefix}.input_layernorm.weight": (config.hidden_size,),
                f"{prefix}.mlp.down_proj.weight": (
                    config.hidden_size,
                    config.intermediate_size,
                ),
                f"{prefix}.mlp.gate_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp_conv.base_kernel": (
                    config.conv_kernel_size,
                    2,
                    config.hidden_size,
                ),
                f"{prefix}.mlp_conv.kernel_projection.weight": (
                    config.hidden_size // 4,
                    config.hidden_size,
                ),
                f"{prefix}.post_attention_layernorm.weight": (config.hidden_size,),
                f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.k_proj.weight": (
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                ),
                f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.q_proj.weight": (
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                ),
                f"{prefix}.self_attn.v_proj.weight": (
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ),
            }
        )
    return shapes


@dataclass(frozen=True)
class DFlash2WeightAudit:
    checkpoint_tensors: int
    expected_tensors: int
    missing_weights: tuple[str, ...]
    unknown_weights: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
    dtype_counts: dict[str, int]

    @property
    def passed(self) -> bool:
        return not (
            self.missing_weights or self.unknown_weights or self.shape_mismatches
        )


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_len))


def dflash2_checkpoint_metadata(
    snapshot: Path,
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for tensor_file in sorted(snapshot.glob("*.safetensors")):
        header = read_safetensors_header(tensor_file)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shapes[name] = tuple(int(dim) for dim in meta["shape"])
            dtypes[name] = str(meta["dtype"])
    if not shapes:
        raise ValueError(f"No safetensors weights found under {snapshot}")
    return shapes, dtypes


def audit_dflash2_checkpoint(
    model_name_or_path: str,
    config: DFlash2Config | None = None,
) -> DFlash2WeightAudit:
    snapshot = Path(model_name_or_path).expanduser()
    if not snapshot.is_dir():
        raise ValueError(f"DFlash2 checkpoint path is not a directory: {snapshot}")
    config = config or DFlash2Config.from_pretrained_path(str(snapshot))
    expected = expected_dflash2_weight_shapes(config)
    actual, dtypes = dflash2_checkpoint_metadata(snapshot)
    missing = tuple(sorted(set(expected) - set(actual)))
    unknown = tuple(sorted(set(actual) - set(expected)))
    mismatches = tuple(
        sorted(
            (
                name,
                expected[name],
                actual[name],
            )
            for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        )
    )
    dtype_counts: dict[str, int] = {}
    for dtype in dtypes.values():
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    return DFlash2WeightAudit(
        checkpoint_tensors=len(actual),
        expected_tensors=len(expected),
        missing_weights=missing,
        unknown_weights=unknown,
        shape_mismatches=mismatches,
        dtype_counts=dtype_counts,
    )


def load_dflash2_weights_from_safetensors(
    model: DFlash2DraftModel,
    checkpoint_path: str,
    *,
    safe_open_device: str = "cpu",
) -> DFlash2WeightAudit:
    from safetensors import safe_open

    snapshot = Path(checkpoint_path).expanduser()
    audit = audit_dflash2_checkpoint(str(snapshot), model.config)
    if not audit.passed:
        raise RuntimeError("DFlash2 checkpoint audit failed; refusing to load weights")
    params = dict(model.named_parameters())
    loaded = set()
    with torch.no_grad():
        for tensor_file in sorted(snapshot.glob("*.safetensors")):
            with safe_open(
                os.fspath(tensor_file),
                framework="pt",
                device=safe_open_device,
            ) as handle:
                for name in handle.keys():
                    param = params[name]
                    tensor = handle.get_tensor(name)
                    if tensor.device != param.device:
                        tensor = tensor.to(device=param.device)
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(dtype=param.dtype)
                    param.copy_(tensor)
                    loaded.add(name)
    missing_load = sorted(set(params) - loaded)
    if missing_load:
        raise RuntimeError(
            "DFlash2 loader did not load expected parameters: "
            + ", ".join(missing_load[:10])
        )
    return audit


@dataclass(frozen=True)
class ProposalBatch:
    token_ids: torch.Tensor
    valid_lengths: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.token_ids.dim() != 2:
            raise ValueError("ProposalBatch token_ids must have shape [batch, K]")
        if self.valid_lengths.dim() != 1:
            raise ValueError("ProposalBatch valid_lengths must have shape [batch]")
        if self.valid_lengths.shape[0] != self.token_ids.shape[0]:
            raise ValueError("ProposalBatch valid_lengths batch size mismatch")
        max_k = self.token_ids.shape[1]
        if torch.any(self.valid_lengths < 0) or torch.any(self.valid_lengths > max_k):
            raise ValueError("ProposalBatch valid_lengths are outside [0, K]")

    def proposal_for(self, sequence: Sequence, row: int, include_metadata: bool) -> Proposal:
        valid_length = int(self.valid_lengths[row].item())
        tokens = tuple(int(v) for v in self.token_ids[row, :valid_length].tolist())
        metadata = dict(self.metadata) if include_metadata else {}
        metadata["kind"] = "dflash2"
        metadata["block_parallel"] = True
        return Proposal(sequence.seq_id, tokens, metadata)


@dataclass(frozen=True)
class CandidateSelection:
    token_ids: torch.Tensor
    scores: torch.Tensor

    def __post_init__(self) -> None:
        if self.token_ids.shape != self.scores.shape:
            raise ValueError("candidate token IDs and scores must have equal shape")
        if self.token_ids.dim() != 3:
            raise ValueError("candidate selection must have shape [batch, K, top_k]")


def select_topk_candidates(scores: torch.Tensor, top_k: int) -> CandidateSelection:
    if scores.dim() != 3:
        raise ValueError("DFlash2 selector scores must have shape [batch, K, vocab]")
    if not 0 < top_k <= scores.shape[-1]:
        raise ValueError("top_k must be within the vocabulary dimension")
    values, token_ids = torch.topk(scores, k=top_k, dim=-1)
    return CandidateSelection(token_ids=token_ids, scores=values)


@dataclass
class DFlash2DraftCache:
    sliding_window: int
    lengths_by_request: dict[str, int] = field(default_factory=dict)

    def advance(self, request_id: str, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("cannot advance draft cache by a negative length")
        current = self.lengths_by_request.get(request_id, 0)
        self.lengths_by_request[request_id] = min(
            self.sliding_window,
            current + tokens,
        )

    def rollback(self, request_id: str, length: int) -> None:
        if length < 0:
            raise ValueError("draft cache length cannot be negative")
        self.lengths_by_request[request_id] = min(length, self.sliding_window)

    def release(self, request_id: str) -> None:
        self.lengths_by_request.pop(request_id, None)


class BlockParallelDFlash2Model(Protocol):
    supports_block_parallel: bool

    def propose_block(
        self,
        sequences: list[Sequence],
        max_tokens: list[int],
        draft_cache: DFlash2DraftCache,
    ) -> ProposalBatch:
        ...


class DFlash2StandaloneProposer(Proposer):
    supports_parallel = True

    def __init__(
        self,
        config: DFlash2Config,
        draft_model: BlockParallelDFlash2Model,
        *,
        draft_cache: DFlash2DraftCache | None = None,
        include_debug_metadata: bool = False,
    ) -> None:
        config.validate_contract()
        if not getattr(draft_model, "supports_block_parallel", False):
            raise ValueError(
                "DFlash2 proposer requires a real block-parallel draft model; "
                "serial target/draft forwarding is not accepted"
            )
        self.config = config
        self.draft_model = draft_model
        self.draft_cache = draft_cache or DFlash2DraftCache(config.sliding_window)
        self.include_debug_metadata = include_debug_metadata

    def propose(self, sequence: Sequence, max_tokens: int) -> Proposal:
        return self.propose_batch([sequence], [max_tokens])[0]

    def propose_batch(
        self, sequences: list[Sequence], max_tokens: list[int]
    ) -> list[Proposal]:
        if len(sequences) != len(max_tokens):
            raise ValueError("sequence and proposer budget counts must match")
        bounded = [min(self.config.num_speculative_tokens, max(0, value)) for value in max_tokens]
        batch = self.draft_model.propose_block(sequences, bounded, self.draft_cache)
        if batch.token_ids.shape[0] != len(sequences):
            raise ValueError("DFlash2 draft model returned the wrong batch size")
        if batch.token_ids.shape[1] > self.config.num_speculative_tokens:
            raise ValueError("DFlash2 draft model exceeded K=7")
        proposals = []
        for row, (seq, budget) in enumerate(zip(sequences, bounded)):
            valid_length = int(batch.valid_lengths[row].item())
            if valid_length > budget:
                raise ValueError("DFlash2 draft model exceeded request budget")
            proposals.append(batch.proposal_for(seq, row, self.include_debug_metadata))
            self.draft_cache.advance(seq.request_id, valid_length)
        return proposals
