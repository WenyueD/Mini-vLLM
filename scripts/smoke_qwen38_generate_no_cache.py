from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_qwen38_load_only import (  # noqa: E402
    DTYPES,
    load_streaming,
    print_cuda_memory,
)
from smoke_qwen38_single_token import build_empty_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Qwen3.8-27B target and run no-cache greedy generation."
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
    parser.add_argument(
        "--input-ids",
        default=None,
        help="Optional comma-separated token IDs. Skips tokenizer encoding if set.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def parse_input_ids(args: argparse.Namespace, tokenizer: AutoTokenizer) -> list[int]:
    if args.input_ids:
        return [int(part) for part in args.input_ids.split(",") if part.strip()]
    return tokenizer.encode(args.prompt)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.model:
        raise SystemExit("Pass --model or set MINIVLLM_QWEN38_PATH")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen3.8-27B no-cache generation")

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
    print(f"max_new_tokens={args.max_new_tokens}", flush=True)
    print_cuda_memory("before_model", device)

    started = time.time()
    print("initializing empty target model", flush=True)
    model = build_empty_model(snapshot, device, dtype)
    print(f"model_initialized_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_model_init", device)

    started = time.time()
    stats = load_streaming(model, snapshot, device, args.safe_open_device)
    print(f"weights_loaded_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_weight_load", device)
    for key, value in stats.items():
        print(f"{key}={value}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    token_ids = parse_input_ids(args, tokenizer)
    if not token_ids:
        raise RuntimeError("prompt encoded to zero tokens")
    print(f"prompt={args.prompt!r}", flush=True)
    print(f"tokenizer_class={tokenizer.__class__.__name__}", flush=True)
    print(f"eos_token_id={tokenizer.eos_token_id}", flush=True)
    print(f"prompt_token_ids={token_ids}", flush=True)

    generated: list[int] = []
    step_times = []
    started_total = time.time()
    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            started_step = time.time()
            output = model(input_ids, return_dict=True)
            logits = output.logits[:, -1, :]
            if not torch.isfinite(logits).all().item():
                raise RuntimeError(f"logits contain NaN or Inf at step {step}")
            next_token_id = int(logits.argmax(dim=-1).item())
            torch.cuda.synchronize(device)
            elapsed = time.time() - started_step
            step_times.append(elapsed)
            generated.append(next_token_id)
            token_ids.append(next_token_id)
            print(
                f"step={step + 1} next_token_id={next_token_id} "
                f"seq_len={len(token_ids)} elapsed={elapsed:.4f}s",
                flush=True,
            )
    total_elapsed = time.time() - started_total
    generated_text = tokenizer.decode(generated)
    full_text = tokenizer.decode(token_ids)

    print(f"generated_token_ids={generated}", flush=True)
    print(f"generated_text={generated_text!r}", flush=True)
    print(f"full_text={full_text!r}", flush=True)
    print(f"generation_total_elapsed={total_elapsed:.4f}s", flush=True)
    print(f"mean_step_elapsed={sum(step_times) / len(step_times):.4f}s", flush=True)
    print_cuda_memory("after_generation", device)
    print("qwen38_no_cache_greedy_32=PASS", flush=True)


if __name__ == "__main__":
    main()
