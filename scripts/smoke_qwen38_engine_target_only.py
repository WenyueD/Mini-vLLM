


from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from myvllm.engine.llm_engine import LLMEngine
from myvllm.models.qwen3_8_config import (
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)
from myvllm.sampling_parameters import SamplingParams


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real LLMEngine target-only greedy smoke for Qwen3.8-27B."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MINIVLLM_QWEN38_PATH"),
        help="Local Qwen3.8 snapshot path. Defaults to MINIVLLM_QWEN38_PATH.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--safe-open-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--max-model-length",
        type=int,
        default=None,
        help=(
            "Prompt plus completion limit. Defaults to prompt tokens + "
            "max-new-tokens + 2."
        ),
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--target-cache-memory-fraction",
        type=float,
        default=0.05,
        help="Small KV-cache reservation for this short target-only smoke.",
    )
    parser.add_argument(
        "--cpu-staging",
        action="store_true",
        help="Alias for --safe-open-device cpu when direct CUDA staging is tight.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Keep generating until max-new-tokens even if EOS is emitted.",
    )
    return parser.parse_args()


def print_cuda_memory(label: str, device: torch.device) -> None:
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"{label}: allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB "
        f"max_allocated={max_allocated:.2f} GiB "
        f"free={free / 1024**3:.2f} GiB total={total / 1024**3:.2f} GiB",
        flush=True,
    )


def normalize_eos(eos_token_id) -> int:
    if isinstance(eos_token_id, int):
        return eos_token_id
    if isinstance(eos_token_id, list) and eos_token_id:
        return int(eos_token_id[0])
    return 248046


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.model:
        raise SystemExit("Pass --model or set MINIVLLM_QWEN38_PATH")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen3.8-27B engine smoke")

    device = torch.device(args.device)
    if device.type != "cuda" or (device.index not in (None, 0)):
        raise SystemExit(
            "This single-GPU engine smoke expects --device cuda:0. "
            "Use CUDA_VISIBLE_DEVICES to select a different physical GPU."
        )

    snapshot = Path(args.model).expanduser()
    dtype = DTYPES[args.dtype]
    safe_open_device = "cpu" if args.cpu_staging else args.safe_open_device
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_default_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    prompt_token_ids = tokenizer.encode(args.prompt)
    if not prompt_token_ids:
        raise RuntimeError("prompt encoded to zero tokens")

    raw_config = load_qwen35_config_dict(str(snapshot))
    text_config = normalize_qwen35_text_config(raw_config)
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = normalize_eos(text_config.eos_token_id)
    max_model_length = args.max_model_length or (
        len(prompt_token_ids) + args.max_new_tokens + 2
    )
    if max_model_length < len(prompt_token_ids) + args.max_new_tokens:
        raise SystemExit(
            "--max-model-length must fit prompt tokens plus max-new-tokens"
        )

    print(f"snapshot={snapshot}", flush=True)
    print(f"device={device}", flush=True)
    print(f"dtype={dtype}", flush=True)
    print(f"safe_open_device={safe_open_device}", flush=True)
    print(f"prompt={args.prompt!r}", flush=True)
    print(f"tokenizer_class={tokenizer.__class__.__name__}", flush=True)
    print(f"eos_token_id={eos}", flush=True)
    print(f"prompt_token_ids={prompt_token_ids}", flush=True)
    print(f"max_new_tokens={args.max_new_tokens}", flush=True)
    print(f"max_model_length={max_model_length}", flush=True)
    print_cuda_memory("before_engine", device)

    config = {
        "model_name_or_path": str(snapshot),
        "world_size": 1,
        "block_size": args.block_size,
        "max_num_sequences": 1,
        "max_num_seqs": 1,
        "max_num_batched_tokens": max_model_length,
        "max_num_batch_tokens": max_model_length,
        "max_model_length": max_model_length,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "target_cache_memory_fraction": args.target_cache_memory_fraction,
        "draft_memory_reservation_fraction": 0.0,
        "enforce_eager": True,
        "eos": int(eos),
        "dtype": args.dtype,
        "init_on_meta": True,
        "stream_weights": True,
        "safe_open_device": safe_open_device,
        "strict_language_weights": True,
    }

    engine = None
    started = time.time()
    try:
        print("initializing_engine", flush=True)
        engine = LLMEngine(config)
        print(f"engine_initialized_elapsed={time.time() - started:.2f}s", flush=True)
        print_cuda_memory("after_engine_init", device)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
            max_model_length=max_model_length,
        )
        generate_started = time.time()
        outputs = engine.generate([args.prompt], sampling_params)
        torch.cuda.synchronize(device)
        generated_token_ids = outputs["token_ids"][0]
        generated_text = outputs["text"][0]
        print(
            f"engine_generate_elapsed={time.time() - generate_started:.4f}s",
            flush=True,
        )
        print(f"generated_token_ids={generated_token_ids}", flush=True)
        print(f"generated_text={generated_text!r}", flush=True)
        print(f"generated_token_count={len(generated_token_ids)}", flush=True)
        print_cuda_memory("after_engine_generate", device)
        if not generated_token_ids:
            raise RuntimeError("engine generated zero tokens")
        print("qwen38_engine_target_only=PASS", flush=True)
    finally:
        if engine is not None:
            engine.exit()


if __name__ == "__main__":
    main()
