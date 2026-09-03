from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from myvllm.engine.sequence import Sequence
from myvllm.models.qwen3_8 import Qwen35RMSNorm, rotate_half
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
        self.top_k = config.selector_top_k
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

    def context(
        self, hidden_states: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        predecessor = self.predecessor_codebook[predecessor_ids.long()]
        projected_hidden = self.hidden_projection(hidden_states)
        return predecessor * projected_hidden.to(predecessor.dtype)

    def transition_scores(
        self,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context(hidden_states, predecessor_ids)
        successors = self.successor_codebook[candidate_ids.long()]
        return (context.unsqueeze(-2) * successors).sum(dim=-1)

    def score_candidates(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        unary_scores = unary_logits.gather(-1, candidate_ids)
        transition = self.transition_scores(
            hidden_states,
            predecessor_ids,
            candidate_ids,
        )
        return unary_scores + transition.to(unary_scores.dtype)

    def select(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        top_k: int | None = None,
    ) -> CandidateSelection:
        candidate_ids = unary_logits.topk(top_k or self.top_k, dim=-1).indices
        scores = self.score_candidates(
            unary_logits,
            hidden_states,
            predecessor_ids,
            candidate_ids,
        )
        return CandidateSelection(token_ids=candidate_ids, scores=scores)

    def select_path(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> ProposalBatch:
        if unary_logits.dim() != 3:
            raise ValueError("unary logits must have shape [batch, K, vocab]")
        batch_size, num_steps, _ = unary_logits.shape
        selected_tokens = []
        predecessor = anchor_token_ids.reshape(batch_size)
        for step in range(num_steps):
            selection = self.select(
                unary_logits[:, step : step + 1, :],
                hidden_states[:, step : step + 1, :],
                predecessor[:, None],
            )
            choice = selection.scores[:, 0, :].argmax(dim=-1)
            predecessor = selection.token_ids[:, 0, :].gather(
                -1,
                choice[:, None],
            ).squeeze(-1)
            selected_tokens.append(predecessor)
        if selected_tokens:
            token_ids = torch.stack(selected_tokens, dim=1)
        else:
            token_ids = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=unary_logits.device,
            )
        return ProposalBatch(
            token_ids=token_ids,
            valid_lengths=torch.full(
                (batch_size,),
                num_steps,
                dtype=torch.long,
                device=unary_logits.device,
            ),
        )


def grouped_dynamic_conv(
    hidden_states: torch.Tensor,
    delta_kernel: torch.Tensor,
    base_kernel: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    hidden_size = hidden_states.shape[-1]
    if hidden_size % group_size:
        raise ValueError("hidden_size must be divisible by group_size")
    kernel_size = base_kernel.shape[0]
    if kernel_size > block_size:
        raise ValueError("kernel_size cannot exceed block_size")
    num_groups = hidden_size // group_size
    if base_kernel.shape != (kernel_size, hidden_size):
        raise ValueError("base_kernel must have shape [kernel_size, hidden_size]")
    expected = (*hidden_states.shape[:-1], kernel_size, num_groups)
    if delta_kernel.shape != expected:
        raise ValueError(f"delta_kernel must have shape {expected}")

    original_shape = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, num_groups, group_size)
    flat_delta = delta_kernel.reshape(-1, kernel_size, num_groups)
    positions = torch.arange(
        flat_hidden.shape[0],
        device=hidden_states.device,
    ).remainder(block_size)
    output = torch.zeros_like(flat_hidden)
    for tap in range(kernel_size):
        if tap == 0:
            shifted = flat_hidden
        else:
            padding = flat_hidden.new_zeros(tap, num_groups, group_size)
            shifted = torch.cat([padding, flat_hidden[:-tap]], dim=0)
        coefficient = (
            base_kernel[tap].to(flat_hidden.dtype).view(1, num_groups, group_size)
            + flat_delta[:, tap, :, None]
        )
        output = output + shifted * coefficient * positions.ge(tap).view(-1, 1, 1)
    return output.reshape(original_shape)


class DFlash2DynamicConv(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.block_size = config.block_size
        self.group_size = config.conv_group_size
        self.kernel_size = config.conv_kernel_size
        self.num_groups = config.hidden_size // config.conv_group_size
        self.base_kernel = nn.Parameter(
            torch.empty(config.conv_kernel_size, 2, config.hidden_size)
        )
        self.kernel_projection = nn.Linear(
            config.hidden_size,
            config.conv_kernel_size * 2 * self.num_groups,
            bias=False,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kernels = self.kernel_projection(hidden_states).view(
            *hidden_states.shape[:-1],
            2,
            self.kernel_size,
            self.num_groups,
        )
        prepared = grouped_dynamic_conv(
            hidden_states,
            kernels[..., 0, :, :],
            self.base_kernel[0],
            block_size=self.block_size,
            group_size=self.group_size,
        )
        return prepared, kernels[..., 1, :, :]

    def finish(
        self,
        hidden_states: torch.Tensor,
        delta_kernel: torch.Tensor,
    ) -> torch.Tensor:
        return grouped_dynamic_conv(
            hidden_states,
            delta_kernel,
            self.base_kernel[1],
            block_size=self.block_size,
            group_size=self.group_size,
        )


class DFlash2Attention(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.sliding_window = config.sliding_window
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
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _position_embedding(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bt,d->btd", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _apply_rotary(
        self,
        states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = self._position_embedding(positions, states.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return states * cos + rotate_half(states) * sin

    def _project_kv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        key = self.k_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        return key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        context_states: torch.Tensor | None = None,
        context_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size,
            query_len,
            self.num_heads,
            self.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)
        query = self._apply_rotary(query, query_positions)

        query_key, query_value = self._project_kv(hidden_states)
        query_key = self._apply_rotary(query_key, query_positions)
        key_parts = [query_key]
        value_parts = [query_value]
        position_parts = [query_positions]
        if context_states is not None:
            if context_positions is None:
                raise ValueError("context_positions are required with context_states")
            context_key, context_value = self._project_kv(context_states)
            context_key = self._apply_rotary(context_key, context_positions)
            key_parts.insert(0, context_key)
            value_parts.insert(0, context_value)
            position_parts.insert(0, context_positions)

        key = torch.cat(key_parts, dim=2)
        value = torch.cat(value_parts, dim=2)
        key_positions = torch.cat(position_parts, dim=1)
        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        visible = (
            key_positions[:, None, None, :]
            >= query_positions[:, None, :, None] - self.sliding_window + 1
        )
        scores = scores.masked_fill(~visible, -torch.inf)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probs, value).transpose(1, 2).contiguous()
        return self.o_proj(output.reshape(batch_size, query_len, -1))


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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        context_states: torch.Tensor | None = None,
        context_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        conv_hidden, conv_tail = self.attention_conv.prepare(normed)
        attn_output = self.self_attn(
            conv_hidden,
            query_positions,
            context_states=context_states,
            context_positions=context_positions,
        )
        hidden_states = residual + self.attention_conv.finish(attn_output, conv_tail)
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        conv_hidden, conv_tail = self.mlp_conv.prepare(normed)
        mlp_output = self.mlp(conv_hidden)
        return residual + self.mlp_conv.finish(mlp_output, conv_tail)


class DFlash2DraftModel(nn.Module):
    """Loadable DFlash2 checkpoint shell.

    This class deliberately does not advertise block-parallel proposal support
    until the standalone proposal math is implemented and aligned with Oracle.
    """

    supports_block_parallel = True

    def __init__(self, config: DFlash2Config, *, enforce_contract: bool = True):
        super().__init__()
        if enforce_contract:
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

    def combine_hidden_taps(
        self,
        hidden_taps: Mapping[int, torch.Tensor],
    ) -> torch.Tensor:
        missing = [layer_id for layer_id in self.config.target_layer_ids if layer_id not in hidden_taps]
        if missing:
            raise ValueError(f"missing DFlash2 target hidden taps: {missing}")
        values = [hidden_taps[layer_id] for layer_id in self.config.target_layer_ids]
        reference_shape = values[0].shape[:-1]
        for value in values:
            if value.shape[:-1] != reference_shape:
                raise ValueError("DFlash2 hidden taps must share batch/sequence shape")
            if value.shape[-1] != self.config.hidden_size:
                raise ValueError("DFlash2 hidden tap hidden size mismatch")
        combined = torch.cat(values, dim=-1)
        return self.hidden_norm(self.fc(combined))

    def embed_block_inputs(
        self,
        anchor_token_ids: torch.Tensor,
        token_embedding,
    ) -> torch.Tensor:
        mask = torch.full(
            (
                anchor_token_ids.shape[0],
                self.config.num_speculative_tokens,
            ),
            self.config.mask_token_id,
            dtype=torch.long,
            device=anchor_token_ids.device,
        )
        input_ids = torch.cat([anchor_token_ids.reshape(-1, 1), mask], dim=1)
        return token_embedding(input_ids)

    def forward_block(
        self,
        anchor_token_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        hidden_taps: Mapping[int, torch.Tensor],
        token_embedding,
        lm_head,
        *,
        return_debug: bool = False,
    ) -> tuple[ProposalBatch, dict[str, Any]]:
        context_states = self.combine_hidden_taps(hidden_taps)
        context_len = context_states.shape[1]
        context_positions = (
            anchor_positions.reshape(-1, 1)
            - context_len
            + 1
            + torch.arange(
                context_len,
                device=anchor_positions.device,
                dtype=anchor_positions.dtype,
            ).view(1, -1)
        )
        hidden_states = self.embed_block_inputs(anchor_token_ids, token_embedding)
        query_positions = anchor_positions.reshape(-1, 1) + torch.arange(
            self.config.block_size,
            device=anchor_positions.device,
            dtype=anchor_positions.dtype,
        ).view(1, -1)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                query_positions,
                context_states=context_states,
                context_positions=context_positions,
            )
        hidden_states = self.norm(hidden_states)
        proposal_hidden = hidden_states[:, 1:, :]
        unary_logits = lm_head(proposal_hidden)
        proposal_batch = self.candidate_selector.select_path(
            unary_logits,
            proposal_hidden,
            anchor_token_ids,
        )
        debug = {}
        if return_debug:
            debug = {
                "context_shape": tuple(context_states.shape),
                "query_shape": tuple(hidden_states.shape),
                "proposal_hidden_shape": tuple(proposal_hidden.shape),
                "unary_logits_shape": tuple(unary_logits.shape),
                "context_checksum": float(context_states.float().sum().item()),
                "proposal_hidden_checksum": float(
                    proposal_hidden.float().sum().item()
                ),
            }
        return proposal_batch, debug


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
                    config.conv_kernel_size
                    * 2
                    * (config.hidden_size // config.conv_group_size),
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
                    config.conv_kernel_size
                    * 2
                    * (config.hidden_size // config.conv_group_size),
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


@dataclass(frozen=True)
class DFlash2TargetContext:
    hidden_taps: Mapping[int, torch.Tensor]
    anchor_token_ids: torch.Tensor
    anchor_positions: torch.Tensor
    token_embedding: Any
    lm_head: Any


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
        target_context_provider: (
            Callable[[list[Sequence]], DFlash2TargetContext] | None
        ) = None,
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
        self.target_context_provider = target_context_provider

    def propose(self, sequence: Sequence, max_tokens: int) -> Proposal:
        return self.propose_batch([sequence], [max_tokens])[0]

    def propose_batch(
        self, sequences: list[Sequence], max_tokens: list[int]
    ) -> list[Proposal]:
        if len(sequences) != len(max_tokens):
            raise ValueError("sequence and proposer budget counts must match")
        bounded = [
            min(self.config.num_speculative_tokens, max(0, value))
            for value in max_tokens
        ]
        if self.target_context_provider is None:
            batch = self.draft_model.propose_block(
                sequences,
                bounded,
                self.draft_cache,
            )
        else:
            context = self.target_context_provider(sequences)
            batch, debug = self.draft_model.forward_block(
                context.anchor_token_ids,
                context.anchor_positions,
                context.hidden_taps,
                context.token_embedding,
                context.lm_head,
                return_debug=self.include_debug_metadata,
            )
            if self.include_debug_metadata:
                batch.metadata.update(debug)
        if batch.token_ids.shape[0] != len(sequences):
            raise ValueError("DFlash2 draft model returned the wrong batch size")
        if batch.token_ids.shape[1] > self.config.num_speculative_tokens:
            raise ValueError("DFlash2 draft model exceeded K=7")
        proposals = []
        for row, (seq, budget) in enumerate(zip(sequences, bounded)):
            valid_length = min(int(batch.valid_lengths[row].item()), budget)
            tokens = tuple(
                int(v) for v in batch.token_ids[row, :valid_length].tolist()
            )
            metadata = dict(batch.metadata) if self.include_debug_metadata else {}
            metadata["kind"] = "dflash2"
            metadata["block_parallel"] = True
            proposals.append(Proposal(seq.seq_id, tokens, metadata))
            self.draft_cache.advance(seq.request_id, valid_length)
        return proposals
