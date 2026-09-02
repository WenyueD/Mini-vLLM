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

from myvllm.engine.hybrid_cache import HybridCacheManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load Qwen3.8-27B target, run B=1 cached greedy decode, and compare "
            "cached logits against no-cache full recompute at each step."
        )
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
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    return parser.parse_args()


def parse_input_ids(args: argparse.Namespace, tokenizer: AutoTokenizer) -> list[int]:
    if args.input_ids:
        return [int(part) for part in args.input_ids.split(",") if part.strip()]
    return tokenizer.encode(args.prompt)


def check_close(
    cached_logits: torch.Tensor,
    recompute_logits: torch.Tensor,
    *,
    step: int,
    atol: float,
    rtol: float,
) -> tuple[float, int, int]:
    diff = (cached_logits.float() - recompute_logits.float()).abs()
    max_abs = float(diff.max().item())
    cached_token = int(cached_logits.argmax(dim=-1).item())
    recompute_token = int(recompute_logits.argmax(dim=-1).item())
    close = torch.allclose(
        cached_logits.float(), recompute_logits.float(), atol=atol, rtol=rtol
    )
    if cached_token != recompute_token or not close:
        top_cached = torch.topk(cached_logits.float(), k=5, dim=-1)
        top_recompute = torch.topk(recompute_logits.float(), k=5, dim=-1)
        print(f"mismatch_step={step}", flush=True)
        print(f"cached_token={cached_token}", flush=True)
        print(f"recompute_token={recompute_token}", flush=True)
        print(f"max_abs_diff={max_abs:.6f}", flush=True)
        print(f"cached_top5_ids={top_cached.indices[0].tolist()}", flush=True)
        print(f"cached_top5_values={top_cached.values[0].tolist()}", flush=True)
        print(f"recompute_top5_ids={top_recompute.indices[0].tolist()}", flush=True)
        print(f"recompute_top5_values={top_recompute.values[0].tolist()}", flush=True)
        raise RuntimeError("cached logits/token differ from full recompute")
    return max_abs, cached_token, recompute_token


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.model:
        raise SystemExit("Pass --model or set MINIVLLM_QWEN38_PATH")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen3.8-27B cache-vs-recompute")

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
    print(f"atol={args.atol} rtol={args.rtol}", flush=True)
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
    prefix_token_ids = parse_input_ids(args, tokenizer)
    if not prefix_token_ids:
        raise RuntimeError("prompt encoded to zero tokens")
    print(f"prompt={args.prompt!r}", flush=True)
    print(f"tokenizer_class={tokenizer.__class__.__name__}", flush=True)
    print(f"eos_token_id={tokenizer.eos_token_id}", flush=True)
    print(f"prompt_token_ids={prefix_token_ids}", flush=True)

    cache = HybridCacheManager()
    request_id = "qwen38-smoke"
    generated: list[int] = []
    token_ids = list(prefix_token_ids)
    max_abs_diffs: list[float] = []
    started_total = time.time()

    with torch.inference_mode():
        prompt_tensor = torch.tensor(prefix_token_ids, dtype=torch.long, device=device)
        cached_output = model.forward_with_cache(prompt_tensor, cache, request_id)
        recompute_input = torch.tensor([token_ids], dtype=torch.long, device=device)
        recompute_output = model(recompute_input, return_dict=True)
        cached_logits = cached_output.logits[:, -1, :]
        recompute_logits = recompute_output.logits[:, -1, :]
        if not torch.isfinite(cached_logits).all().item():
            raise RuntimeError("cached prompt logits contain NaN or Inf")
        if not torch.isfinite(recompute_logits).all().item():
            raise RuntimeError("recompute prompt logits contain NaN or Inf")
        max_abs, next_token_id, _ = check_close(
            cached_logits,
            recompute_logits,
            step=0,
            atol=args.atol,
            rtol=args.rtol,
        )
        max_abs_diffs.append(max_abs)

        for step in range(1, args.max_new_tokens + 1):
            generated.append(next_token_id)
            token_ids.append(next_token_id)
            new_token = torch.tensor([next_token_id], dtype=torch.long, device=device)
            cached_started = time.time()
            cached_output = model.forward_with_cache(new_token, cache, request_id)
            cached_logits = cached_output.logits[:, -1, :]
            torch.cuda.synchronize(device)
            cached_elapsed = time.time() - cached_started

            recompute_started = time.time()
            recompute_input = torch.tensor([token_ids], dtype=torch.long, device=device)
            recompute_output = model(recompute_input, return_dict=True)
            recompute_logits = recompute_output.logits[:, -1, :]
            torch.cuda.synchronize(device)
            recompute_elapsed = time.time() - recompute_started

            if not torch.isfinite(cached_logits).all().item():
                raise RuntimeError(f"cached logits contain NaN or Inf at step {step}")
            if not torch.isfinite(recompute_logits).all().item():
                raise RuntimeError(
                    f"recompute logits contain NaN or Inf at step {step}"
                )

            max_abs, next_token_id, _ = check_close(
                cached_logits,
                recompute_logits,
                step=step,
                atol=args.atol,
                rtol=args.rtol,
            )
            max_abs_diffs.append(max_abs)
            print(
                f"step={step} emitted_token_id={generated[-1]} "
                f"next_token_id={next_token_id} seq_len={len(token_ids)} "
                f"max_abs_diff={max_abs:.6f} "
                f"cached_elapsed={cached_elapsed:.4f}s "
                f"recompute_elapsed={recompute_elapsed:.4f}s",
                flush=True,
            )

    total_elapsed = time.time() - started_total
    print(f"generated_token_ids={generated}", flush=True)
    print(f"generated_text={tokenizer.decode(generated)!r}", flush=True)
    print(f"full_text={tokenizer.decode(token_ids)!r}", flush=True)
    print(f"max_seen_abs_diff={max(max_abs_diffs):.6f}", flush=True)
    print(f"decode_compare_total_elapsed={total_elapsed:.4f}s", flush=True)
    print_cuda_memory("after_cache_vs_recompute", device)
    print("qwen38_cache_vs_recompute=PASS", flush=True)


if __name__ == "__main__":
    main()
