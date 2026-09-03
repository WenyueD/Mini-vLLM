from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from myvllm.spec_decode.dflash2 import (
    DFlash2Config,
    DFlash2DraftModel,
    load_dflash2_weights_from_safetensors,
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
        description="Load DFlash2 draft checkpoint weights, print memory, exit."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MINIVLLM_DFLASH2_PATH"),
        help="Local DFlash2 snapshot path. Defaults to MINIVLLM_DFLASH2_PATH.",
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


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.model:
        raise SystemExit("Pass --model or set MINIVLLM_DFLASH2_PATH")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DFlash2 load-only smoke")

    snapshot = Path(args.model).expanduser()
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(f"snapshot={snapshot}", flush=True)
    print(f"device={device}", flush=True)
    print(f"dtype={dtype}", flush=True)
    print(f"safe_open_device={args.safe_open_device}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"device_name={torch.cuda.get_device_name(device)}", flush=True)
    print_cuda_memory("before_model", device)

    config = DFlash2Config.from_pretrained_path(str(snapshot))
    config.validate_contract()
    started = time.time()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = DFlash2DraftModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    print("materializing empty DFlash2 parameters on target GPU", flush=True)
    model.to_empty(device=device)
    model.eval()
    print(f"model_initialized_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_model_init", device)

    started = time.time()
    safe_open_device = str(device) if args.safe_open_device == "cuda" else "cpu"
    audit = load_dflash2_weights_from_safetensors(
        model,
        str(snapshot),
        safe_open_device=safe_open_device,
    )
    print(f"weights_loaded_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_weight_load", device)
    print(f"checkpoint_tensors={audit.checkpoint_tensors}", flush=True)
    print(f"expected_tensors={audit.expected_tensors}", flush=True)
    print(f"dtype_counts={audit.dtype_counts}", flush=True)
    print("dflash2_load_only=PASS", flush=True)
    print("dflash2_standalone_proposal=PENDING", flush=True)


if __name__ == "__main__":
    main()
