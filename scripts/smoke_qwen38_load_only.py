from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

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


DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Qwen3.8-27B text-only target weights, print memory, exit."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MINIVLLM_QWEN38_PATH"),
        help="Local Qwen3.8 snapshot path. Defaults to MINIVLLM_QWEN38_PATH.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument(
        "--safe-open-device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Where safetensors should stage each tensor before copying.",
    )
    return parser.parse_args()


def bytes_to_gib(value: int) -> float:
    return value / 1024**3


def print_cuda_memory(label: str, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"{label}: allocated={bytes_to_gib(torch.cuda.memory_allocated(device)):.2f} GiB "
        f"reserved={bytes_to_gib(torch.cuda.memory_reserved(device)):.2f} GiB "
        f"max_allocated={bytes_to_gib(torch.cuda.max_memory_allocated(device)):.2f} GiB "
        f"free={bytes_to_gib(free):.2f} GiB total={bytes_to_gib(total):.2f} GiB",
        flush=True,
    )


def load_streaming(
    model: Qwen35ForCausalLM,
    snapshot: Path,
    device: torch.device,
    safe_open_device: str,
) -> dict[str, int]:
    params = dict(model.named_parameters())
    loaded = set()
    stats = {
        "checkpoint_tensors": 0,
        "loaded_language_tensors": 0,
        "vision_or_multimodal_skipped": 0,
        "mtp_skipped": 0,
        "unknown_language_weights": 0,
    }
    tensor_device = str(device) if safe_open_device == "cuda" else "cpu"
    with torch.no_grad():
        for tensor_file in sorted(snapshot.glob("*.safetensors")):
            file_loaded = 0
            started = time.time()
            with safe_open(tensor_file, framework="pt", device=tensor_device) as handle:
                for hf_name in handle.keys():
                    stats["checkpoint_tensors"] += 1
                    if _is_skippable_non_target_weight(hf_name):
                        if _is_mtp_weight(hf_name):
                            stats["mtp_skipped"] += 1
                        elif _is_vision_or_multimodal_weight(hf_name):
                            stats["vision_or_multimodal_skipped"] += 1
                        continue
                    model_name = canonical_language_weight_name(hf_name)
                    param = params.get(model_name)
                    if param is None:
                        stats["unknown_language_weights"] += 1
                        print(f"UNKNOWN {hf_name} -> {model_name}")
                        continue
                    tensor = handle.get_tensor(hf_name)
                    if tuple(param.shape) != tuple(tensor.shape):
                        raise RuntimeError(
                            f"shape mismatch for {hf_name} -> {model_name}: "
                            f"model={tuple(param.shape)} checkpoint={tuple(tensor.shape)}"
                        )
                    param.copy_(tensor.to(device=device, dtype=param.dtype))
                    loaded.add(model_name)
                    stats["loaded_language_tensors"] += 1
                    file_loaded += 1
                    del tensor
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"loaded {tensor_file.name}: language_tensors={file_loaded} "
                f"elapsed={time.time() - started:.2f}s",
                flush=True,
            )
            print_cuda_memory("after_file", device)

    missing = sorted(set(params) - loaded)
    if missing:
        print("missing first 30:")
        for name in missing[:30]:
            print(f"  {name}")
        raise RuntimeError(f"missing model parameters: {len(missing)}")
    if stats["unknown_language_weights"]:
        raise RuntimeError(
            f"unknown language weights: {stats['unknown_language_weights']}"
        )
    stats["model_parameters"] = len(params)
    return stats


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.model:
        raise SystemExit("Pass --model or set MINIVLLM_QWEN38_PATH")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen3.8-27B load-only smoke")

    snapshot = Path(args.model).expanduser()
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_default_dtype(dtype)

    print(f"snapshot={snapshot}", flush=True)
    print(f"device={device}", flush=True)
    print(f"dtype={dtype}", flush=True)
    print(f"safe_open_device={args.safe_open_device}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"device_name={torch.cuda.get_device_name(device)}", flush=True)
    print_cuda_memory("before_model", device)

    raw = load_qwen35_config_dict(str(snapshot))
    config = normalize_qwen35_text_config(raw)
    started = time.time()
    print("initializing model on meta device", flush=True)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = Qwen35ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    print("materializing empty parameters on target GPU", flush=True)
    model.to_empty(device=device)
    model.eval()
    print(f"model_initialized_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_model_init", device)

    started = time.time()
    stats = load_streaming(model, snapshot, device, args.safe_open_device)
    print(f"weights_loaded_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_weight_load", device)
    for key, value in stats.items():
        print(f"{key}={value}", flush=True)
    print("qwen38_load_only=PASS", flush=True)


if __name__ == "__main__":
    main()
