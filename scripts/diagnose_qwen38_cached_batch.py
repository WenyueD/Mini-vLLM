from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

from myvllm.engine.hybrid_cache import HybridCacheManager
from myvllm.engine.llm_engine import LLMEngine
from myvllm.models.qwen3_8_config import load_qwen35_config_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare real Qwen3.8 cached B=1 and tensor-batched B=2 logits "
            "while teacher-forcing the same greedy token stream."
        )
    )
    parser.add_argument(
        "--target-model", default=os.environ.get("MINIVLLM_QWEN38_PATH")
    )
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--safe-open-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--target-cache-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def build_config(args: argparse.Namespace, max_model_length: int) -> dict:
    config = dict(load_qwen35_config_dict(args.target_model))
    config.update(
        {
            "model_name_or_path": args.target_model,
            "world_size": 1,
            "block_size": 256,
            "max_num_sequences": 2,
            "max_num_seqs": 2,
            "max_num_batched_tokens": max_model_length * 2,
            "max_num_batch_tokens": max_model_length * 2,
            "max_model_length": max_model_length,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "target_cache_memory_fraction": args.target_cache_memory_fraction,
            "enforce_eager": True,
            "dtype": args.dtype,
            "init_on_meta": True,
            "stream_weights": True,
            "safe_open_device": args.safe_open_device,
            "strict_language_weights": True,
            "speculative_config": {"enabled": False},
        }
    )
    return config


def margin(logits: torch.Tensor) -> float:
    top = torch.topk(logits.float(), k=2).values
    return float((top[0] - top[1]).item())


@torch.inference_mode()
def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.target_model:
        raise SystemExit("Pass --target-model or set MINIVLLM_QWEN38_PATH")
    prompts = args.prompts or [
        "Write a Python function that returns the first n Fibonacci numbers.",
        "Explain why the sky appears blue in two sentences.",
    ]
    if len(prompts) != 2:
        raise SystemExit("Pass exactly two --prompt values")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    prompt_ids = [tokenizer.encode(prompt) for prompt in prompts]
    max_model_length = max(map(len, prompt_ids)) + args.max_new_tokens + 1
    engine = None
    try:
        engine = LLMEngine(build_config(args, max_model_length))
        model = engine.model_runner.model
        device = next(model.parameters()).device
        serial_cache = HybridCacheManager()
        batch_cache = HybridCacheManager()
        request_ids = ["diagnostic-0", "diagnostic-1"]
        serial_tokens = []
        batch_tokens = []

        for request_id, token_ids in zip(request_ids, prompt_ids):
            input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            serial_output = model.forward_with_cache(
                input_ids, serial_cache, request_id
            )
            batch_output = model.forward_with_cache(input_ids, batch_cache, request_id)
            serial_tokens.append(int(serial_output.logits[0, -1].argmax().item()))
            batch_tokens.append(int(batch_output.logits[0, -1].argmax().item()))

        print(f"prompt_token_ids={prompt_ids}")
        print(f"position=0 serial_tokens={serial_tokens} batch_tokens={batch_tokens}")
        mismatch_count = int(serial_tokens != batch_tokens)
        token_streams = [[token] for token in serial_tokens]

        for position in range(1, args.max_new_tokens):
            serial_logits = []
            for row, request_id in enumerate(request_ids):
                output = model.forward_with_cache(
                    torch.tensor([serial_tokens[row]], dtype=torch.long, device=device),
                    serial_cache,
                    request_id,
                )
                serial_logits.append(output.logits[0, -1])

            batch_outputs = model.forward_with_cache_batch(
                torch.tensor(serial_tokens, dtype=torch.long, device=device).view(2, 1),
                [1, 1],
                batch_cache,
                request_ids,
            )
            batch_logits = [output.logits[0, -1] for output in batch_outputs]
            next_serial_tokens = [
                int(logits.argmax().item()) for logits in serial_logits
            ]
            next_batch_tokens = [int(logits.argmax().item()) for logits in batch_logits]

            row_diagnostics = []
            for row in range(2):
                max_abs = float(
                    (serial_logits[row].float() - batch_logits[row].float())
                    .abs()
                    .max()
                    .item()
                )
                row_diagnostics.append(
                    {
                        "row": row,
                        "serial": next_serial_tokens[row],
                        "batch": next_batch_tokens[row],
                        "serial_margin": margin(serial_logits[row]),
                        "batch_margin": margin(batch_logits[row]),
                        "max_abs_logit_diff": max_abs,
                    }
                )
                mismatch_count += int(next_serial_tokens[row] != next_batch_tokens[row])
                token_streams[row].append(next_serial_tokens[row])
            print(f"position={position} rows={row_diagnostics}")
            serial_tokens = next_serial_tokens

        print(f"serial_token_ids={token_streams}")
        print(f"cached_batch_argmax_mismatches={mismatch_count}")
        print(
            "qwen38_cached_batch_diagnostic="
            + ("PASS" if mismatch_count == 0 else "FAIL")
        )
        if mismatch_count:
            raise SystemExit(1)
    finally:
        if engine is not None:
            engine.exit()


if __name__ == "__main__":
    main()
