import pytest
import torch

from myvllm.models.qwen3_8 import Qwen35ForCausalLM
from myvllm.utils.loader import load_weights_from_state_dict


def tiny_config(num_layers=2):
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


def synthetic_weights_for(model):
    generator = torch.Generator().manual_seed(123)
    return {
        name: torch.randn(param.shape, generator=generator, dtype=param.dtype)
        for name, param in model.named_parameters()
    }


def test_qwen35_loader_maps_complete_language_weights_and_skips_vision():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=2))
    weights = synthetic_weights_for(model)
    weights["visual.patch_embed.proj.weight"] = torch.randn(4, 4)
    weights["mtp.fc.weight"] = torch.randn(4, 4)

    summary = load_weights_from_state_dict(model, weights, strict_language=True)

    assert set(summary.loaded_model_params) == {
        name for name, _ in model.named_parameters()
    }
    assert set(summary.skipped_vision_weights) == {
        "visual.patch_embed.proj.weight",
        "mtp.fc.weight",
    }
    assert summary.unknown_language_weights == ()
    assert summary.unloaded_model_params == ()


def test_qwen35_loader_accepts_multimodal_language_model_prefix():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=4))
    weights = {
        f"model.language_model.{name.removeprefix('model.')}": value
        for name, value in synthetic_weights_for(model).items()
    }

    summary = load_weights_from_state_dict(model, weights, strict_language=True)

    assert set(summary.loaded_model_params) == {
        name for name, _ in model.named_parameters()
    }
    assert summary.unknown_language_weights == ()
    assert summary.unloaded_model_params == ()


def test_qwen35_loader_rejects_unknown_language_weight():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=1))
    weights = synthetic_weights_for(model)
    weights["model.layers.0.language_surprise.weight"] = torch.randn(1)

    with pytest.raises(RuntimeError, match="unknown language checkpoint weights"):
        load_weights_from_state_dict(model, weights, strict_language=True)


def test_qwen35_loader_reports_missing_language_weight():
    model = Qwen35ForCausalLM.from_config_dict(tiny_config(num_layers=1))
    weights = synthetic_weights_for(model)
    missing_name = next(iter(weights))
    del weights[missing_name]

    with pytest.raises(RuntimeError, match="unloaded model parameters"):
        load_weights_from_state_dict(model, weights, strict_language=True)
