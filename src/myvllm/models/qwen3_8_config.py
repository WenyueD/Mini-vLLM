from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


QWEN35_DEFAULT_HIDDEN_TAPS = (5, 19, 33, 47, 61)
QWEN35_SUPPORTED_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForCausalLM",
}


@dataclass(frozen=True)
class Qwen35TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: tuple[str, ...]
    full_attention_interval: int
    max_position_embeddings: int
    rope_theta: float
    partial_rotary_factor: float
    rms_norm_eps: float
    hidden_act: str
    attention_bias: bool
    attention_dropout: float
    attn_output_gate: bool
    tie_word_embeddings: bool
    eos_token_id: int | list[int] | None
    bos_token_id: int | None
    pad_token_id: int | None
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    output_gate_type: str
    dtype: str | None = None

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        )

    @property
    def linear_attention_layers(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type == "linear_attention"
        )

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)


def is_qwen35_config(config: Mapping[str, Any]) -> bool:
    model_type = config.get("model_type")
    architectures = set(config.get("architectures") or [])
    text_config = config.get("text_config")
    return (
        model_type in {"qwen3_5", "qwen3_5_text"}
        or bool(architectures & QWEN35_SUPPORTED_ARCHITECTURES)
        or (
            isinstance(text_config, Mapping)
            and text_config.get("model_type") == "qwen3_5_text"
        )
    )


def load_qwen35_config_dict(model_name_or_path: str) -> dict[str, Any]:
    """Load Qwen3.5/Qwen3.8 config.json without AutoConfig registration.

    Transformers 4.57.x can read the tokenizer files but does not register
    `model_type=qwen3_5` for AutoConfig.  Mini-vLLM only needs the JSON fields
    here, so reading config.json directly keeps the target runner independent of
    the Oracle/vLLM environment.
    """
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
        raise ValueError(f"Qwen3.5/Qwen3.8 config.json not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_qwen35_text_config(config: Mapping[str, Any]) -> Qwen35TextConfig:
    """Return the nested text config used by Qwen3.5/Qwen3.8 multimodal models."""
    raw_text_config = config.get("text_config", config)
    if not isinstance(raw_text_config, Mapping):
        raise ValueError("Qwen3.5/Qwen3.8 config must contain a text_config object")

    text = dict(raw_text_config)
    num_layers = int(text["num_hidden_layers"])
    interval = int(text.get("full_attention_interval", 4))
    layer_types = text.get("layer_types")
    if layer_types is None:
        layer_types = [
            "linear_attention" if (idx + 1) % interval else "full_attention"
            for idx in range(num_layers)
        ]
    layer_types = tuple(str(layer_type) for layer_type in layer_types)
    if len(layer_types) != num_layers:
        raise ValueError(
            "Qwen3.5/Qwen3.8 layer_types length must match num_hidden_layers"
        )
    invalid = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if invalid:
        raise ValueError(f"Unsupported Qwen3.5/Qwen3.8 layer types: {invalid}")

    rope_parameters = text.get("rope_parameters") or {}
    partial_rotary_factor = float(
        rope_parameters.get(
            "partial_rotary_factor", text.get("partial_rotary_factor", 0.25)
        )
    )
    head_dim = int(
        text.get("head_dim")
        or (int(text["hidden_size"]) // int(text["num_attention_heads"]))
    )

    return Qwen35TextConfig(
        vocab_size=int(text["vocab_size"]),
        hidden_size=int(text["hidden_size"]),
        intermediate_size=int(text["intermediate_size"]),
        num_hidden_layers=num_layers,
        num_attention_heads=int(text["num_attention_heads"]),
        num_key_value_heads=int(text["num_key_value_heads"]),
        head_dim=head_dim,
        layer_types=layer_types,
        full_attention_interval=interval,
        max_position_embeddings=int(text.get("max_position_embeddings", 32768)),
        rope_theta=float(
            rope_parameters.get("rope_theta", text.get("rope_theta", 10000))
        ),
        partial_rotary_factor=partial_rotary_factor,
        rms_norm_eps=float(
            text.get("rms_norm_eps", text.get("rms_norm_epsilon", 1e-6))
        ),
        hidden_act=str(text.get("hidden_act", "silu")),
        attention_bias=bool(text.get("attention_bias", False)),
        attention_dropout=float(text.get("attention_dropout", 0.0)),
        attn_output_gate=bool(text.get("attn_output_gate", True)),
        tie_word_embeddings=bool(
            text.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
        ),
        eos_token_id=text.get("eos_token_id"),
        bos_token_id=text.get("bos_token_id"),
        pad_token_id=text.get("pad_token_id"),
        linear_conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 4)),
        linear_key_head_dim=int(text.get("linear_key_head_dim", 128)),
        linear_value_head_dim=int(text.get("linear_value_head_dim", 128)),
        linear_num_key_heads=int(text.get("linear_num_key_heads", 16)),
        linear_num_value_heads=int(text.get("linear_num_value_heads", 32)),
        output_gate_type=str(text.get("output_gate_type", "swish")),
        dtype=text.get("dtype"),
    )


def reject_qwen35_multimodal_inputs(**kwargs: Any) -> None:
    multimodal_keys = {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "mm_token_type_ids",
        "image_embeds",
        "video_embeds",
    }
    present = sorted(key for key in multimodal_keys if kwargs.get(key) is not None)
    if present:
        joined = ", ".join(present)
        raise NotImplementedError(
            "Mini-vLLM currently supports Qwen3.5/Qwen3.8 text-only target "
            f"inference; multimodal inputs are not supported: {joined}"
        )
