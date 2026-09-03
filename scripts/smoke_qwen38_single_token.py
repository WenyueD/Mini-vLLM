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

from myvllm.models.qwen3_8 import Qwen35ForCausalLM  # noqa: E402
from myvllm.models.qwen3_8_config import (  # noqa: E402
    QWEN35_DEFAULT_HIDDEN_TAPS,
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Qwen3.8-27B target and run one short prefill forward."
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
        help="Optional comma-separated token IDs. Skips tokenizer if provided.",
    )
    parser.add_argument(
        "--expected-next-token-id",
        type=int,
        default=None,
        help="Optional expected greedy next token ID for Oracle target parity.",
    )
    return parser.parse_args()


def build_empty_model(snapshot: Path, device: torch.device, dtype: torch.dtype):
    raw = load_qwen35_config_dict(str(snapshot))
    config = normalize_qwen35_text_config(raw)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = Qwen35ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.to_empty(device=device)
    model.eval()
    return model


def parse_input_ids(args: argparse.Namespace, snapshot: Path) -> list[int]:
    if args.input_ids:
        return [int(part) for part in args.input_ids.split(",") if part.strip()]
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    token_ids = tokenizer.encode(args.prompt)
    print(f"prompt={args.prompt!r}", flush=True)
    print(f"tokenizer_class={tokenizer.__class__.__name__}", flush=True)
    print(f"eos_token_id={tokenizer.eos_token_id}", flush=True)
    return token_ids


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.model:
        raise SystemExit("Pass --model or set MINIVLLM_QWEN38_PATH")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen3.8-27B single-token smoke")

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

    token_ids = parse_input_ids(args, snapshot)
    if not token_ids:
        raise RuntimeError("prompt encoded to zero tokens")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    print(f"input_ids={token_ids}", flush=True)
    print(f"input_shape={tuple(input_ids.shape)}", flush=True)

    started = time.time()
    with torch.inference_mode():
        output = model(
            input_ids,
            return_dict=True,
            capture_hidden_taps=True,
            hidden_tap_layers=QWEN35_DEFAULT_HIDDEN_TAPS,
        )
        logits = output.logits
        last_logits = logits[:, -1, :]
        if not torch.isfinite(last_logits).all().item():
            raise RuntimeError("last-token logits contain NaN or Inf")
        next_token_id = int(last_logits.argmax(dim=-1).item())
        max_logit = float(last_logits.max().item())
        min_logit = float(last_logits.min().item())
    torch.cuda.synchronize(device)
    print(f"forward_elapsed={time.time() - started:.4f}s", flush=True)
    print(f"logits_shape={tuple(logits.shape)}", flush=True)
    print(f"hidden_tap_keys={sorted(output.hidden_taps)}", flush=True)
    print(f"next_token_id={next_token_id}", flush=True)
    if args.expected_next_token_id is not None:
        print(f"expected_next_token_id={args.expected_next_token_id}", flush=True)
        if next_token_id != args.expected_next_token_id:
            raise SystemExit(
                "qwen38_target_oracle_anchor=FAIL "
                f"expected={args.expected_next_token_id} actual={next_token_id}"
            )
        print("qwen38_target_oracle_anchor=PASS", flush=True)
    print(f"last_logit_min={min_logit:.6f}", flush=True)
    print(f"last_logit_max={max_logit:.6f}", flush=True)
    print_cuda_memory("after_forward", device)
    print("qwen38_single_token_forward=PASS", flush=True)


if __name__ == "__main__":
    main()
