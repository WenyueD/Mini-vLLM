import torch
from torch import nn
import os
from safetensors import safe_open
import re
from dataclasses import dataclass


def default_weight_loader(param, weight):
    """Default weight loader that copies weight data to parameter."""
    if param.shape != weight.shape:
        raise ValueError(
            f"Shape mismatch: param {param.shape} vs weight {weight.shape}"
        )
    param.data.copy_(weight)


@dataclass(frozen=True)
class WeightLoadSummary:
    loaded_model_params: frozenset[str]
    consumed_checkpoint_weights: frozenset[str]
    skipped_vision_weights: tuple[str, ...]
    skipped_merged_weights: tuple[str, ...]
    unknown_language_weights: tuple[str, ...]
    unloaded_model_params: tuple[str, ...]


def _is_vision_or_multimodal_weight(name: str) -> bool:
    return (
        name.startswith("visual.")
        or name.startswith("vision.")
        or name.startswith("model.visual.")
        or name.startswith("model.vision.")
        or name.startswith("multi_modal_projector.")
        or ".vision_tower." in name
    )


def _is_mtp_weight(name: str) -> bool:
    return name.startswith("mtp.") or name.startswith("model.mtp.")


def _is_skippable_non_target_weight(name: str) -> bool:
    return _is_vision_or_multimodal_weight(name) or _is_mtp_weight(name)


def canonical_language_weight_name(name: str) -> str:
    """Map Qwen3.5 multimodal wrapper names onto Mini-vLLM text target names."""
    if name.startswith("model.language_model.lm_head."):
        return name.removeprefix("model.language_model.")
    if name.startswith("language_model.lm_head."):
        return name.removeprefix("language_model.")
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name.startswith("language_model."):
        return "model." + name.removeprefix("language_model.")
    return name


def load_weights_from_state_dict(
    model: nn.Module,
    hf_weights: dict[str, torch.Tensor],
    *,
    strict_language: bool = True,
) -> WeightLoadSummary:
    # Now map and load weights into custom model
    loaded_model_params = set()
    consumed_checkpoint_weights = set()
    skipped_vision_weights = []
    skipped_merged_weights = []
    unknown_language_weights = []

    # Process each HF weight
    for hf_name, hf_weight in hf_weights.items():
        try:
            if _is_skippable_non_target_weight(hf_name):
                skipped_vision_weights.append(hf_name)
                consumed_checkpoint_weights.add(hf_name)
                continue
            model_name = canonical_language_weight_name(hf_name)

            # 0. Direct names first. Qwen3.5 text layers deliberately keep the
            # HF module names, while the legacy Qwen3 model still falls through
            # to packed qkv/gate_up handling below.
            try:
                param = model.get_parameter(model_name)
            except AttributeError:
                param = None
            if param is not None:
                default_weight_loader(param, hf_weight)
                loaded_model_params.add(model_name)
                consumed_checkpoint_weights.add(hf_name)
                continue

            # 1. Handle QKV merge (q_proj + k_proj + v_proj -> qkv_projection)
            if ".self_attn.q_proj.weight" in model_name:
                layer_match = re.search(r"layers\.(\d+)", model_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    k_name = hf_name.replace("q_proj", "k_proj")
                    v_name = hf_name.replace("q_proj", "v_proj")

                    if k_name in hf_weights and v_name in hf_weights:
                        q_weight = hf_weight
                        k_weight = hf_weights[k_name]
                        v_weight = hf_weights[v_name]

                        # Concatenate q, k, v along output dimension
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

                        custom_name = (
                            f"model.layers.{layer_idx}.self_attn.qkv_projection.weight"
                        )
                        try:
                            param = model.get_parameter(custom_name)
                            default_weight_loader(param, qkv_weight)
                            loaded_model_params.add(custom_name)
                            consumed_checkpoint_weights.update(
                                {hf_name, k_name, v_name}
                            )
                            continue
                        except AttributeError:
                            pass

            # 2. Handle gate_up merge (gate_proj + up_proj -> gate_up)
            elif ".mlp.gate_proj.weight" in model_name:
                layer_match = re.search(r"layers\.(\d+)", model_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    up_name = hf_name.replace("gate_proj", "up_proj")

                    if up_name in hf_weights:
                        gate_weight = hf_weight
                        up_weight = hf_weights[up_name]

                        # Concatenate gate and up along output dimension
                        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.weight"
                        try:
                            param = model.get_parameter(custom_name)
                            default_weight_loader(param, gate_up_weight)
                            loaded_model_params.add(custom_name)
                            consumed_checkpoint_weights.update({hf_name, up_name})
                            continue
                        except AttributeError:
                            pass

            # 3. Handle gate_up merge for bias (if present)
            elif ".mlp.gate_proj.bias" in model_name:
                layer_match = re.search(r"layers\.(\d+)", model_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    up_bias_name = hf_name.replace("gate_proj", "up_proj")

                    if up_bias_name in hf_weights:
                        gate_bias = hf_weight
                        up_bias = hf_weights[up_bias_name]
                        gate_up_bias = torch.cat([gate_bias, up_bias], dim=0)

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.bias"
                        try:
                            param = model.get_parameter(custom_name)
                            default_weight_loader(param, gate_up_bias)
                            loaded_model_params.add(custom_name)
                            consumed_checkpoint_weights.update({hf_name, up_bias_name})
                            continue
                        except AttributeError:
                            pass

            # 4. Skip k_proj, v_proj, up_proj only when their partner merge consumed them.
            if hf_name in consumed_checkpoint_weights and any(
                x in hf_name for x in [".k_proj.", ".v_proj.", ".up_proj."]
            ):
                skipped_merged_weights.append(hf_name)
            else:
                unknown_language_weights.append(hf_name)

        except Exception as e:
            unknown_language_weights.append(f"{hf_name}: {str(e)}")

    # Check for model parameters that weren't loaded. A tied lm_head may not be
    # yielded by named_parameters(), which is fine; get_parameter() was loaded via
    # the shared embedding weight when tie_word_embeddings=True.
    unloaded_model_params = []
    for name, param in model.named_parameters():
        if name not in loaded_model_params:
            unloaded_model_params.append(name)

    summary = WeightLoadSummary(
        loaded_model_params=frozenset(loaded_model_params),
        consumed_checkpoint_weights=frozenset(consumed_checkpoint_weights),
        skipped_vision_weights=tuple(skipped_vision_weights),
        skipped_merged_weights=tuple(skipped_merged_weights),
        unknown_language_weights=tuple(unknown_language_weights),
        unloaded_model_params=tuple(unloaded_model_params),
    )
    if strict_language and (
        summary.unknown_language_weights or summary.unloaded_model_params
    ):
        details = []
        if summary.unknown_language_weights:
            details.append(
                "unknown language checkpoint weights: "
                + ", ".join(summary.unknown_language_weights[:10])
            )
        if summary.unloaded_model_params:
            details.append(
                "unloaded model parameters: "
                + ", ".join(summary.unloaded_model_params[:10])
            )
        raise RuntimeError("Incomplete language weight load: " + "; ".join(details))
    return summary


def load_weights_from_safetensors_streaming(
    model: nn.Module,
    checkpoint_path: str,
    *,
    strict_language: bool = True,
    safe_open_device: str = "cpu",
) -> WeightLoadSummary:
    """Load direct-name safetensors weights one tensor at a time.

    This path is intended for Qwen3.5/Qwen3.8 target checkpoints whose Mini-vLLM
    parameter names already match the canonical language-model weights.  It avoids
    collecting the full 27B checkpoint in CPU memory before copying to the model.
    """
    safetensor_files = [
        f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
    ]
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    loaded_model_params = set()
    consumed_checkpoint_weights = set()
    skipped_vision_weights = []
    skipped_merged_weights = []
    unknown_language_weights = []

    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        with safe_open(file_path, framework="pt", device=safe_open_device) as handle:
            for hf_name in handle.keys():
                if _is_skippable_non_target_weight(hf_name):
                    skipped_vision_weights.append(hf_name)
                    consumed_checkpoint_weights.add(hf_name)
                    continue

                model_name = canonical_language_weight_name(hf_name)
                try:
                    param = model.get_parameter(model_name)
                except AttributeError:
                    param = None
                if param is None:
                    unknown_language_weights.append(hf_name)
                    continue

                weight = handle.get_tensor(hf_name)
                if weight.device != param.device:
                    weight = weight.to(device=param.device)
                try:
                    default_weight_loader(param, weight)
                except Exception as exc:
                    unknown_language_weights.append(f"{hf_name}: {exc}")
                    continue
                loaded_model_params.add(model_name)
                consumed_checkpoint_weights.add(hf_name)

    unloaded_model_params = []
    for name, _ in model.named_parameters():
        if name not in loaded_model_params:
            unloaded_model_params.append(name)

    summary = WeightLoadSummary(
        loaded_model_params=frozenset(loaded_model_params),
        consumed_checkpoint_weights=frozenset(consumed_checkpoint_weights),
        skipped_vision_weights=tuple(skipped_vision_weights),
        skipped_merged_weights=tuple(skipped_merged_weights),
        unknown_language_weights=tuple(unknown_language_weights),
        unloaded_model_params=tuple(unloaded_model_params),
    )
    if strict_language and (
        summary.unknown_language_weights or summary.unloaded_model_params
    ):
        details = []
        if summary.unknown_language_weights:
            details.append(
                "unknown language checkpoint weights: "
                + ", ".join(summary.unknown_language_weights[:10])
            )
        if summary.unloaded_model_params:
            details.append(
                "unloaded model parameters: "
                + ", ".join(summary.unloaded_model_params[:10])
            )
        raise RuntimeError("Incomplete language weight load: " + "; ".join(details))
    return summary


def load_weights_from_checkpoint(
    model: nn.Module,
    model_name_or_path: str,
    *,
    strict_language: bool = True,
    stream_safetensors: bool = False,
    safe_open_device: str = "cpu",
) -> WeightLoadSummary:
    """
    Load weights from a Hugging Face model checkpoint into the custom model.
    Handles QKV and gate_up weight merging for optimized layers.

    Args:
        model: The target model to load weights into
        model_name_or_path: Path to local checkpoint or Hugging Face model name
    """
    from huggingface_hub import snapshot_download

    # Try to resolve the path - could be local or from HF cache
    checkpoint_path = None

    # First, try local paths
    if model_name_or_path.startswith("~"):
        checkpoint_path = os.path.expanduser(model_name_or_path)
    elif os.path.isdir(model_name_or_path):
        checkpoint_path = model_name_or_path

    # If not a local path, try to download from HuggingFace
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=[
                    "*.msgpack",
                    "*.h5",
                    "*.bin",
                ],  # Skip non-safetensors weights
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path."
            )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    if stream_safetensors:
        summary = load_weights_from_safetensors_streaming(
            model,
            checkpoint_path,
            strict_language=strict_language,
            safe_open_device=safe_open_device,
        )
        print(f"\n{'='*80}")
        print("Weight Loading Summary:")
        print(f"{'='*80}")
        print(
            f"Successfully loaded: {len(summary.loaded_model_params)} "
            "model parameters"
        )
        print(f"Skipped vision/MTP weights: {len(summary.skipped_vision_weights)}")
        print(
            f"Consumed checkpoint weights: "
            f"{len(summary.consumed_checkpoint_weights)}"
        )
        if summary.unknown_language_weights:
            print(f"Unknown language weights: {len(summary.unknown_language_weights)}")
            for name in summary.unknown_language_weights[:15]:
                print(f"  - {name}")
            if len(summary.unknown_language_weights) > 15:
                print(f"  ... and {len(summary.unknown_language_weights) - 15} more")
        if summary.unloaded_model_params:
            print(
                f"\nWARNING: {len(summary.unloaded_model_params)} model "
                "parameters NOT loaded from checkpoint:"
            )
            for name in summary.unloaded_model_params[:15]:
                param = dict(model.named_parameters())[name]
                print(
                    f"  - {name} "
                    f"(shape: {param.shape}, mean: {param.data.mean():.6f})"
                )
            if len(summary.unloaded_model_params) > 15:
                print(f"  ... and {len(summary.unloaded_model_params) - 15} more")
        print(f"{'='*80}")
        return summary

    # Load all safetensors files in the checkpoint directory
    safetensor_files = [
        f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
    ]

    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # Collect all weights from HF model
    hf_weights = {}
    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                hf_weights[weight_name] = f.get_tensor(weight_name)

    summary = load_weights_from_state_dict(
        model, hf_weights, strict_language=strict_language
    )

    print(f"\n{'='*80}")
    print(f"Weight Loading Summary:")
    print(f"{'='*80}")
    print(f"Successfully loaded: {len(summary.loaded_model_params)} model parameters")
    print(f"Skipped vision/multimodal weights: {len(summary.skipped_vision_weights)}")
    print(f"Consumed checkpoint weights: {len(summary.consumed_checkpoint_weights)}")

    if summary.unloaded_model_params:
        print(
            f"\nWARNING: {len(summary.unloaded_model_params)} model parameters NOT loaded from checkpoint:"
        )
        for name in summary.unloaded_model_params[:15]:
            param = dict(model.named_parameters())[name]
            print(f"  - {name} (shape: {param.shape}, mean: {param.data.mean():.6f})")
        if len(summary.unloaded_model_params) > 15:
            print(f"  ... and {len(summary.unloaded_model_params) - 15} more")

    if summary.unknown_language_weights:
        print(f"Unknown language weights: {len(summary.unknown_language_weights)}")
        for name in summary.unknown_language_weights[:15]:
            print(f"  - {name}")
        if len(summary.unknown_language_weights) > 15:
            print(f"  ... and {len(summary.unknown_language_weights) - 15} more")

    print(f"{'='*80}")
    return summary
