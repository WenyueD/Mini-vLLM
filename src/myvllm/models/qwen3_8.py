from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from myvllm.engine.hybrid_cache import (
    HybridCacheManager,
    HybridRequestState,
    LinearLayerState,
)
from myvllm.models.qwen3_8_config import (
    QWEN35_DEFAULT_HIDDEN_TAPS,
    Qwen35TextConfig,
    normalize_qwen35_text_config,
    reject_qwen35_multimodal_inputs,
)


@dataclass(frozen=True)
class Qwen35ForwardOutput:
    logits: torch.Tensor | None
    hidden_states: torch.Tensor
    hidden_taps: dict[int, torch.Tensor]
    cache_transaction_id: int | None = None


class Qwen35RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)


class Qwen35RMSNormGated(Qwen35RMSNorm):
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * F.silu(gate).to(x.dtype)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q = torch.cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], dim=-1)
    k = torch.cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], dim=-1)
    return q, k


class Qwen35RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.rotary_dim = config.rotary_dim
        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError(
                "Qwen3.5 partial rotary dimension must be positive and even"
            )
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bt,d->btd", position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35Attention(nn.Module):
    def __init__(self, config: Qwen35TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen35RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def _project(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                bsz, seq_len, self.num_heads, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(bsz, seq_len, -1)
        key_states = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = apply_partial_rotary(
            query_states, key_states, position_embeddings[0], position_embeddings[1]
        )
        return query_states, key_states, value_states, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        query_states, key_states, value_states, gate = self._project(
            hidden_states, position_embeddings
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(
                self.num_key_value_groups, dim=1
            )
        scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        causal = torch.ones(
            seq_len, seq_len, device=scores.device, dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), -torch.inf)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(probs, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        request_state: HybridRequestState,
    ) -> torch.Tensor:
        if hidden_states.shape[:2] != (1, 1):
            raise NotImplementedError(
                "cached Qwen3.5 full attention currently supports B=1 step decode"
            )
        query_states, key_states, value_states, gate = self._project(
            hidden_states, position_embeddings
        )
        previous = request_state.full_attention_layers.get(self.layer_idx)
        if previous is None:
            cached_key = key_states
            cached_value = value_states
        else:
            cached_key = torch.cat([previous[0], key_states], dim=1)
            cached_value = torch.cat([previous[1], value_states], dim=1)
        request_state.full_attention_layers[self.layer_idx] = (
            cached_key.detach().clone(),
            cached_value.detach().clone(),
        )
        query_states = query_states.transpose(1, 2)
        cached_key = cached_key.transpose(1, 2)
        cached_value = cached_value.transpose(1, 2)
        if self.num_key_value_groups > 1:
            cached_key = cached_key.repeat_interleave(self.num_key_value_groups, dim=1)
            cached_value = cached_value.repeat_interleave(
                self.num_key_value_groups, dim=1
            )
        scores = torch.matmul(query_states, cached_key.transpose(2, 3)) * self.scaling
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(probs, cached_value).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, 1, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError("linear_num_value_heads must be a multiple of key heads")
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads).uniform_(0.01, 16).log_()
        )
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent = torch.zeros(
            batch_size,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            device=device,
            dtype=torch.float32,
        )
        conv = torch.zeros(
            batch_size,
            self.conv_dim,
            max(self.conv_kernel_size - 1, 0),
            device=device,
            dtype=dtype,
        )
        return recurrent, conv

    def _causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixed_qkv = mixed_qkv.transpose(1, 2)
        original_len = mixed_qkv.shape[-1]
        if conv_state is not None:
            mixed_qkv = torch.cat([conv_state.to(mixed_qkv.dtype), mixed_qkv], dim=-1)
        elif self.conv_kernel_size > 1:
            mixed_qkv = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))
        conv_out = F.conv1d(
            mixed_qkv,
            self.conv1d.weight,
            bias=self.conv1d.bias,
            groups=self.conv_dim,
        )
        conv_out = F.silu(conv_out[..., -original_len:])
        new_conv = mixed_qkv[..., -max(self.conv_kernel_size - 1, 0) :].detach().clone()
        return conv_out.transpose(1, 2), new_conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        if recurrent_state is None or conv_state is None:
            recurrent_state, conv_state = self.initial_state(
                batch_size, hidden_states.device, hidden_states.dtype
            )
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv, final_conv_state = self._causal_conv(mixed_qkv, conv_state)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        query = l2norm(query, dim=-1) / (query.shape[-1] ** 0.5)
        key = l2norm(key, dim=-1)
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )
        state = recurrent_state.to(value.device).float()
        outputs = []
        for idx in range(seq_len):
            decay_t = decay[:, idx].exp().view(batch_size, self.num_v_heads, 1, 1)
            state = state * decay_t
            k_t = key[:, idx].float()
            v_t = value[:, idx].float()
            beta_t = beta[:, idx].float().unsqueeze(-1)
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out_t = (state * query[:, idx].float().unsqueeze(-1)).sum(dim=-2)
            outputs.append(out_t)
        core = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        core = self.norm(
            core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        core = core.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(core)
        if return_final_state:
            return output, state.detach().clone(), final_conv_state
        return output


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = config.layer_types[layer_idx]
        self.input_layernorm = Qwen35RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen35RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen35Attention(config, layer_idx)
        self.mlp = Qwen35MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(
                hidden_states, position_embeddings, attention_mask=attention_mask
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen35TextModel(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen35DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        hidden_tap_layers: tuple[int, ...] | None = None,
        **multimodal_kwargs,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        reject_qwen35_multimodal_inputs(**multimodal_kwargs)
        squeeze = False
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze = True
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).view(1, -1)
            position_ids = position_ids.expand(batch_size, -1)
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(position_ids, hidden_states.dtype)
        taps = set(hidden_tap_layers or ())
        hidden_taps: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )
            if layer_idx in taps:
                hidden_taps[layer_idx] = hidden_states.detach()
        hidden_states = self.norm(hidden_states)
        if squeeze:
            hidden_states = hidden_states.squeeze(0)
            hidden_taps = {
                layer_id: value.squeeze(0) for layer_id, value in hidden_taps.items()
            }
        return hidden_states, hidden_taps

    def forward_cached_step(
        self,
        input_id: torch.Tensor,
        request_state: HybridRequestState,
        hidden_tap_layers: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        if input_id.numel() != 1:
            raise ValueError("cached Qwen3.5 target step expects exactly one token")
        input_ids = input_id.reshape(1, 1)
        position_ids = torch.tensor(
            [[request_state.logical_length]],
            dtype=torch.long,
            device=input_ids.device,
        )
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(position_ids, hidden_states.dtype)
        taps = set(hidden_tap_layers or ())
        hidden_taps: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            if layer.block_type == "linear_attention":
                previous = request_state.linear_layers.get(layer_idx)
                if previous is None:
                    attn_out, recurrent, conv = layer.linear_attn(
                        normed, return_final_state=True
                    )
                else:
                    attn_out, recurrent, conv = layer.linear_attn(
                        normed,
                        recurrent_state=previous.recurrent_state,
                        conv_state=previous.conv_state,
                        return_final_state=True,
                    )
                request_state.linear_layers[layer_idx] = LinearLayerState(
                    recurrent, conv
                )
            else:
                attn_out = layer.self_attn.forward_with_cache(
                    normed, position_embeddings, request_state
                )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)
            if layer_idx in taps:
                hidden_taps[layer_idx] = hidden_states.detach().clone()
        request_state.logical_length += 1
        return self.norm(hidden_states), hidden_taps


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config: Qwen35TextConfig | Mapping):
        super().__init__()
        if not isinstance(config, Qwen35TextConfig):
            config = normalize_qwen35_text_config(config)
        self.config = config
        self.model = Qwen35TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_config_dict(cls, config: Mapping) -> "Qwen35ForCausalLM":
        return cls(normalize_qwen35_text_config(config))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        capture_hidden_taps: bool = False,
        hidden_tap_layers: tuple[int, ...] = QWEN35_DEFAULT_HIDDEN_TAPS,
        return_dict: bool = False,
        cache_manager: HybridCacheManager | None = None,
        request_ids: tuple[str, ...] | None = None,
        **multimodal_kwargs,
    ) -> torch.Tensor | Qwen35ForwardOutput:
        if cache_manager is not None:
            for request_id in request_ids or ():
                cache_manager.ensure_request(request_id)
        tap_layers = hidden_tap_layers if capture_hidden_taps else ()
        hidden_states, hidden_taps = self.model(
            input_ids,
            position_ids=position_ids,
            hidden_tap_layers=tap_layers,
            **multimodal_kwargs,
        )
        if not return_dict:
            return hidden_states
        return Qwen35ForwardOutput(
            logits=self.compute_logits(hidden_states),
            hidden_states=hidden_states,
            hidden_taps=hidden_taps,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache_manager: HybridCacheManager,
        request_id: str,
        *,
        transaction=None,
        capture_hidden_taps: bool = False,
        hidden_tap_layers: tuple[int, ...] = QWEN35_DEFAULT_HIDDEN_TAPS,
    ) -> Qwen35ForwardOutput:
        """Run B=1 text target tokens through the hybrid cache contract.

        This is the eager correctness path used before DFlash2 integration:
        ordinary target-only decode and speculative verification can both update
        the same request-owned full-attention KV and GDN recurrent/conv state.
        """
        reject_qwen35_multimodal_inputs()
        if input_ids.dim() > 1:
            if input_ids.shape[0] != 1:
                raise NotImplementedError(
                    "Qwen3.5 cached target currently supports B=1"
                )
            input_ids = input_ids.reshape(-1)
        request_state = cache_manager.ensure_request(request_id)
        tap_layers = hidden_tap_layers if capture_hidden_taps else ()
        logits = []
        hidden_states = []
        merged_taps: dict[int, list[torch.Tensor]] = {
            layer_id: [] for layer_id in tap_layers
        }
        for index, token_id in enumerate(input_ids):
            hidden, taps = self.model.forward_cached_step(
                token_id,
                request_state,
                hidden_tap_layers=tap_layers,
            )
            hidden_states.append(hidden)
            logits.append(self.compute_logits(hidden))
            for layer_id, value in taps.items():
                merged_taps[layer_id].append(value)
            if transaction is not None:
                cache_manager.record_snapshot(
                    transaction,
                    request_id,
                    tokens_executed=index + 1,
                    state=request_state,
                )
        hidden_tensor = torch.cat(hidden_states, dim=1)
        logits_tensor = torch.cat(logits, dim=1)
        hidden_taps = {
            layer_id: torch.cat(values, dim=1)
            for layer_id, values in merged_taps.items()
            if values
        }
        return Qwen35ForwardOutput(
            logits=logits_tensor,
            hidden_states=hidden_tensor,
            hidden_taps=hidden_taps,
            cache_transaction_id=(
                None if transaction is None else transaction.transaction_id
            ),
        )
