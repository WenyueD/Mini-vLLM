from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch

from myvllm.engine.sequence import Sequence
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
    num_sliding_attention_layers: int = 5
    sliding_window: int = 2048

    @property
    def num_speculative_tokens(self) -> int:
        return self.block_size - 1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DFlash2Config":
        source = raw.get("draft_config", raw)
        if not isinstance(source, Mapping):
            raise ValueError("DFlash2 config must be a mapping")
        target_layer_ids = tuple(
            int(value)
            for value in source.get("target_layer_ids", cls.target_layer_ids)
        )
        return cls(
            block_size=int(source.get("block_size", cls.block_size)),
            target_layer_ids=target_layer_ids,
            selector_top_k=int(source.get("selector_top_k", cls.selector_top_k)),
            selector_rank=int(source.get("selector_rank", cls.selector_rank)),
            conv_kernel_size=int(
                source.get(
                    "conv_kernel_size",
                    source.get("conv_kernel", cls.conv_kernel_size),
                )
            ),
            conv_group_size=int(source.get("conv_group_size", cls.conv_group_size)),
            vocab_size=int(source.get("vocab_size", cls.vocab_size)),
            num_sliding_attention_layers=int(
                source.get(
                    "num_sliding_attention_layers",
                    source.get("sliding_attention_layers", cls.num_sliding_attention_layers),
                )
            ),
            sliding_window=int(source.get("sliding_window", cls.sliding_window)),
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
