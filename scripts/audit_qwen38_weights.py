from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import torch

from myvllm.models.qwen3_8 import Qwen35ForCausalLM
from myvllm.models.qwen3_8_config import (
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)
from myvllm.utils.loader import (
    _is_mtp_weight,
    _is_skippable_non_target_weight,
    _is_vision_or_multimodal_weight,
    canonical_language_weight_name,
)


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_len))


def checkpoint_shapes(snapshot: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for tensor_file in sorted(snapshot.glob("*.safetensors")):
        header = read_safetensors_header(tensor_file)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shapes[name] = tuple(meta["shape"])
    if not shapes:
        raise SystemExit(f"No safetensors weights found under {snapshot}")
    return shapes


def model_shapes(config_path: str) -> dict[str, tuple[int, ...]]:
    raw = load_qwen35_config_dict(config_path)
    config = normalize_qwen35_text_config(raw)
    with torch.device("meta"):
        model = Qwen35ForCausalLM(config)
    return {name: tuple(param.shape) for name, param in model.named_parameters()}


def main() -> None:
    path = os.environ.get("MINIVLLM_QWEN38_PATH")
    if not path:
        raise SystemExit("Set MINIVLLM_QWEN38_PATH to the local Qwen3.8 snapshot path")
    snapshot = Path(path).expanduser()
    ckpt = checkpoint_shapes(snapshot)
    model = model_shapes(str(snapshot))

    language_ckpt = {}
    duplicate_canonical_names = []
    for name, shape in ckpt.items():
        if _is_skippable_non_target_weight(name):
            continue
        canonical = canonical_language_weight_name(name)
        if canonical in language_ckpt:
            duplicate_canonical_names.append(canonical)
        language_ckpt[canonical] = shape
    vision = sorted(name for name in ckpt if _is_vision_or_multimodal_weight(name))
    mtp = sorted(name for name in ckpt if _is_mtp_weight(name))
    original_language_names = {
        name
        for name, shape in ckpt.items()
        if not _is_skippable_non_target_weight(name)
    }
    missing = sorted(set(model) - set(language_ckpt))
    unknown = sorted(set(language_ckpt) - set(model))
    shape_mismatch = sorted(
        (name, model[name], language_ckpt[name])
        for name in set(model) & set(language_ckpt)
        if model[name] != language_ckpt[name]
    )

    print(f"snapshot={snapshot}")
    print(f"checkpoint_tensors={len(ckpt)}")
    print(f"language_tensors={len(original_language_names)}")
    print(f"canonical_language_tensors={len(language_ckpt)}")
    print(f"duplicate_canonical_names={len(duplicate_canonical_names)}")
    print(f"vision_or_multimodal_skipped={len(vision)}")
    print(f"mtp_skipped={len(mtp)}")
    print(f"model_parameters={len(model)}")
    print(f"missing_model_parameters={len(missing)}")
    print(f"unknown_language_weights={len(unknown)}")
    print(f"shape_mismatches={len(shape_mismatch)}")

    for title, values in [
        ("missing", missing),
        ("unknown", unknown),
    ]:
        if values:
            print(f"\n{title} first 30:")
            for name in values[:30]:
                print(f"  {name}")

    if shape_mismatch:
        print("\nshape mismatches first 30:")
        for name, expected, actual in shape_mismatch[:30]:
            print(f"  {name}: model={expected} checkpoint={actual}")

    if duplicate_canonical_names:
        print("\nduplicate canonical names first 30:")
        for name in duplicate_canonical_names[:30]:
            print(f"  {name}")

    if missing or unknown or shape_mismatch or duplicate_canonical_names:
        raise SystemExit(1)
    print("weight_name_audit=PASS")


if __name__ == "__main__":
    main()
