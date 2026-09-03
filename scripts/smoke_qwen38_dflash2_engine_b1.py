from __future__ import annotations

import argparse
import gc
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
        description="Compare Qwen3.8 target-only vs DFlash2 B=1 greedy engine output."
    )
    parser.add_argument("--target-model", default=os.environ.get("MINIVLLM_QWEN38_PATH"))
    parser.add_argument("--draft-model", default=os.environ.get("MINIVLLM_DFLASH2_PATH"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--safe-open-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dflash2-safe-open-device", default=None, choices=("cuda", "cpu"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--target-cache-memory-fraction", type=float, default=0.02)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--debug-metadata", action="store_true")
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


def normalize_eos(eos_token_id) -> int:
    if isinstance(eos_token_id, int):
        return eos_token_id
    if isinstance(eos_token_id, list) and eos_token_id:
        return int(eos_token_id[0])
    return 248046


def build_config(args: argparse.Namespace, speculative: bool, eos: int, max_model_length: int) -> dict:
    raw_config = load_qwen35_config_dict(args.target_model)
    config = dict(raw_config)
    safe_open_device = args.safe_open_device
    dflash_safe_open = args.dflash2_safe_open_device or safe_open_device
    config.update(
        {
            "model_name_or_path": args.target_model,
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
            "dflash2_safe_open_device": dflash_safe_open,
            "dflash2_debug_metadata": args.debug_metadata,
            "speculative_config": {
                "enabled": speculative,
                "proposer": "dflash2" if speculative else "ngram",
                "num_speculative_tokens": 7,
                "draft_model_name_or_path": args.draft_model,
            },
        }
    )
    return config


def run_engine(label: str, config: dict, prompt: str, args: argparse.Namespace, device: torch.device):
    engine = None
    try:
        print(f"{label}_initializing_engine", flush=True)
        started = time.time()
        engine = LLMEngine(config)
        print(f"{label}_engine_initialized_elapsed={time.time() - started:.2f}s", flush=True)
        print_cuda_memory(f"{label}_after_engine_init", device)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
            max_model_length=config["max_model_length"],
        )
        started = time.time()
        outputs = engine.generate([prompt], sampling_params)
        torch.cuda.synchronize(device)
        elapsed = time.time() - started
        stats = engine.get_stats()
        token_ids = outputs["token_ids"][0]
        print(f"{label}_generate_elapsed={elapsed:.4f}s", flush=True)
        print(f"{label}_token_ids={token_ids}", flush=True)
        print(f"{label}_text={outputs['text'][0]!r}", flush=True)
        print(f"{label}_stats={stats}", flush=True)
        print_cuda_memory(f"{label}_after_generate", device)
        return token_ids, stats
    finally:
        if engine is not None:
            engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.target_model:
        raise SystemExit("Pass --target-model or set MINIVLLM_QWEN38_PATH")
    if not args.draft_model:
        raise SystemExit("Pass --draft-model or set MINIVLLM_DFLASH2_PATH")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen3.8 DFlash2 engine smoke")

    args.target_model = os.fspath(Path(args.target_model).expanduser())
    args.draft_model = os.fspath(Path(args.draft_model).expanduser())
    device = torch.device(args.device)
    if device.type != "cuda" or (device.index not in (None, 0)):
        raise SystemExit("Use CUDA_VISIBLE_DEVICES to select a non-zero physical GPU.")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_default_dtype(DTYPES[args.dtype])

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    prompt_token_ids = tokenizer.encode(args.prompt)
    if not prompt_token_ids:
        raise RuntimeError("prompt encoded to zero tokens")
    raw_config = load_qwen35_config_dict(args.target_model)
    text_config = normalize_qwen35_text_config(raw_config)
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = normalize_eos(text_config.eos_token_id)
    max_model_length = len(prompt_token_ids) + args.max_new_tokens + 2

    print(f"target_model={args.target_model}", flush=True)
    print(f"draft_model={args.draft_model}", flush=True)
    print(f"prompt={args.prompt!r}", flush=True)
    print(f"prompt_token_ids={prompt_token_ids}", flush=True)
    print(f"max_new_tokens={args.max_new_tokens}", flush=True)
    print(f"max_model_length={max_model_length}", flush=True)
    print(f"eos_token_id={eos}", flush=True)
    print_cuda_memory("before_runs", device)

    target_config = build_config(args, False, eos, max_model_length)
    target_tokens, _target_stats = run_engine(
        "target_only",
        target_config,
        args.prompt,
        args,
        device,
    )

    spec_config = build_config(args, True, eos, max_model_length)
    spec_tokens, spec_stats = run_engine(
        "dflash2",
        spec_config,
        args.prompt,
        args,
        device,
    )

    print(f"target_only_token_ids={target_tokens}", flush=True)
    print(f"dflash2_token_ids={spec_tokens}", flush=True)
    if spec_tokens != target_tokens:
        for index, (left, right) in enumerate(zip(target_tokens, spec_tokens)):
            if left != right:
                print(f"first_divergence={index} target={left} dflash2={right}", flush=True)
                break
        if len(spec_tokens) != len(target_tokens):
            print(
                f"length_mismatch target={len(target_tokens)} dflash2={len(spec_tokens)}",
                flush=True,
            )
        raise SystemExit("DFlash2 engine output diverged from target-only")

    speculative = spec_stats["speculative"]
    print(f"dflash2_proposed_tokens={speculative['proposed_tokens']}", flush=True)
    print(f"dflash2_accepted_tokens={speculative['accepted_tokens']}", flush=True)
    print(f"dflash2_acceptance_rate={speculative['acceptance_rate']}", flush=True)
    print("qwen38_dflash2_engine_b1=PASS", flush=True)
    print("dflash2_online_b1=PENDING", flush=True)
    print("dflash2_oracle_alignment=PENDING", flush=True)


if __name__ == "__main__":
    main()
