from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from myvllm.engine.llm_engine import LLMEngine
from myvllm.models.qwen3_8_config import (
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)
from myvllm.online import OnlineEngine
from myvllm.sampling_parameters import SamplingParams


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen3.8 target-only vs DFlash2 with two concurrent online "
            "requests and require a real shared speculative decode batch."
        )
    )
    parser.add_argument("--target-model", default=os.environ.get("MINIVLLM_QWEN38_PATH"))
    parser.add_argument("--draft-model", default=os.environ.get("MINIVLLM_DFLASH2_PATH"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--safe-open-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dflash2-safe-open-device", default=None, choices=("cuda", "cpu"))
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to submit; pass exactly twice. Defaults to two identical Hello prompts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        choices=range(1, 8),
        default=1,
        metavar="K",
        help="Maximum DFlash2 proposal length (1-7).",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--target-cache-memory-fraction", type=float, default=0.02)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--debug-metadata", action="store_true")
    parser.add_argument("--profile-dflash2-stages", action="store_true")
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


def normalize_eos(eos_token_id: Any) -> int:
    if isinstance(eos_token_id, int):
        return eos_token_id
    if isinstance(eos_token_id, list) and eos_token_id:
        return int(eos_token_id[0])
    return 248046


def build_config(
    args: argparse.Namespace,
    speculative: bool,
    eos: int,
    max_model_length: int,
) -> dict:
    config = dict(load_qwen35_config_dict(args.target_model))
    dflash_safe_open = args.dflash2_safe_open_device or args.safe_open_device
    max_batch_tokens = max_model_length * 2
    config.update(
        {
            "model_name_or_path": args.target_model,
            "world_size": 1,
            "block_size": args.block_size,
            "max_num_sequences": 2,
            "max_num_seqs": 2,
            "max_num_batched_tokens": max_batch_tokens,
            "max_num_batch_tokens": max_batch_tokens,
            "max_model_length": max_model_length,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "target_cache_memory_fraction": args.target_cache_memory_fraction,
            "draft_memory_reservation_fraction": 0.0,
            "enforce_eager": True,
            "eos": int(eos),
            "dtype": args.dtype,
            "init_on_meta": True,
            "stream_weights": True,
            "safe_open_device": args.safe_open_device,
            "strict_language_weights": True,
            "dflash2_safe_open_device": dflash_safe_open,
            "dflash2_debug_metadata": args.debug_metadata,
            "dflash2_profile_stages": args.profile_dflash2_stages,
            "speculative_config": {
                "enabled": speculative,
                "proposer": "dflash2" if speculative else "ngram",
                "num_speculative_tokens": args.num_speculative_tokens,
                "draft_model_name_or_path": args.draft_model,
            },
        }
    )
    return config


def batch_evidence(stats: dict, request_ids: list[str]) -> dict[str, Any]:
    steps = stats.get("recent_decode_steps", [])
    accepted_by_request = {request_id: 0 for request_id in request_ids}
    for step in steps:
        per_request = step.get("per_request_accepted_length", {})
        for request_id in request_ids:
            accepted_by_request[request_id] += int(per_request.get(request_id, 0))
    return {
        "max_active_batch_size": max(
            (int(step.get("active_batch_size", 0)) for step in steps), default=0
        ),
        "max_speculative_batch_size": max(
            (int(step.get("speculative_batch_size", 0)) for step in steps), default=0
        ),
        "max_position_zero_proposed": max(
            (
                int(step.get("per_position_proposed", (0,))[0])
                if step.get("per_position_proposed")
                else 0
                for step in steps
            ),
            default=0,
        ),
        "accepted_by_request": accepted_by_request,
    }


def consume_handle(label: str, index: int, handle, timeout: float) -> dict[str, Any]:
    chunks: list[list[int]] = []
    token_ids: list[int] = []
    finish_reason = None
    for output in handle.iter_outputs(timeout=timeout):
        if output.error:
            raise RuntimeError(output.error)
        chunk = list(output.token_ids)
        chunks.append(chunk)
        token_ids.extend(chunk)
        finish_reason = output.finish_reason
        print(
            f"{label}_request_{index}_chunk_{len(chunks) - 1}_token_ids={chunk}",
            flush=True,
        )
    return {
        "request_id": handle.request_id,
        "token_ids": token_ids,
        "chunks": chunks,
        "finish_reason": finish_reason,
    }


def run_online(
    label: str,
    config: dict,
    prompts: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict]:
    engine = None
    online = None
    try:
        print(f"{label}_initializing_engine", flush=True)
        started = time.time()
        engine = LLMEngine(config)
        print(f"{label}_engine_initialized_elapsed={time.time() - started:.2f}s", flush=True)
        print_cuda_memory(f"{label}_after_engine_init", device)

        online = OnlineEngine(engine)
        online.start()
        params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
            max_model_length=config["max_model_length"],
        )
        request_ids = [f"{label}-online-b2-{index}" for index in range(2)]
        handles = online.submit_batch(
            prompts,
            params,
            request_ids=request_ids,
            timeout=args.timeout,
        )

        started = time.time()
        results = [
            consume_handle(label, index, handle, args.timeout)
            for index, handle in enumerate(handles)
        ]
        torch.cuda.synchronize(device)
        stats = online.stats()
        evidence = batch_evidence(stats, request_ids)
        print(f"{label}_online_elapsed={time.time() - started:.4f}s", flush=True)
        for index, result in enumerate(results):
            print(f"{label}_request_{index}_token_ids={result['token_ids']}", flush=True)
            print(
                f"{label}_request_{index}_chunk_sizes="
                f"{[len(chunk) for chunk in result['chunks']]}",
                flush=True,
            )
            print(
                f"{label}_request_{index}_finish_reason={result['finish_reason']}",
                flush=True,
            )
        print(f"{label}_batch_evidence={evidence}", flush=True)
        print(f"{label}_stats={stats}", flush=True)
        print_cuda_memory(f"{label}_after_online_generate", device)
        return results, stats
    finally:
        if online is not None and online.is_running:
            online.close(timeout=args.timeout)
        elif engine is not None:
            engine.exit()
        del online
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
        raise SystemExit("CUDA is required for Qwen3.8 DFlash2 B=2 online smoke")

    prompts = args.prompts or ["Hello", "Hello"]
    if len(prompts) != 2:
        raise SystemExit("Pass --prompt exactly twice, or omit it for two Hello prompts")
    args.target_model = os.fspath(Path(args.target_model).expanduser())
    args.draft_model = os.fspath(Path(args.draft_model).expanduser())
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in (None, 0):
        raise SystemExit("Use CUDA_VISIBLE_DEVICES to select a non-zero physical GPU")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_default_dtype(DTYPES[args.dtype])

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
    if any(not token_ids for token_ids in prompt_token_ids):
        raise RuntimeError("A prompt encoded to zero tokens")
    text_config = normalize_qwen35_text_config(
        load_qwen35_config_dict(args.target_model)
    )
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = normalize_eos(text_config.eos_token_id)
    max_model_length = max(map(len, prompt_token_ids)) + args.max_new_tokens + 2

    print(f"prompts={prompts!r}", flush=True)
    print(f"prompt_token_ids={prompt_token_ids}", flush=True)
    print(f"max_new_tokens={args.max_new_tokens}", flush=True)
    print(f"num_speculative_tokens={args.num_speculative_tokens}", flush=True)
    print(f"max_model_length={max_model_length}", flush=True)
    print_cuda_memory("before_runs", device)

    target_results, target_stats = run_online(
        "target_only",
        build_config(args, False, eos, max_model_length),
        prompts,
        args,
        device,
    )
    dflash_results, dflash_stats = run_online(
        "dflash2",
        build_config(args, True, eos, max_model_length),
        prompts,
        args,
        device,
    )

    for index, (target, dflash) in enumerate(zip(target_results, dflash_results)):
        target_tokens = target["token_ids"]
        dflash_tokens = dflash["token_ids"]
        print(f"request_{index}_target_only_token_ids={target_tokens}", flush=True)
        print(f"request_{index}_dflash2_token_ids={dflash_tokens}", flush=True)
        if target_tokens != dflash_tokens:
            for position, (left, right) in enumerate(zip(target_tokens, dflash_tokens)):
                if left != right:
                    print(
                        f"request_{index}_first_divergence={position} "
                        f"target={left} dflash2={right}",
                        flush=True,
                    )
                    break
            raise SystemExit(f"DFlash2 B=2 request {index} diverged from target-only")

    target_ids = [result["request_id"] for result in target_results]
    dflash_ids = [result["request_id"] for result in dflash_results]
    target_evidence = batch_evidence(target_stats, target_ids)
    dflash_evidence = batch_evidence(dflash_stats, dflash_ids)
    speculative = dflash_stats["speculative"]
    verification = dflash_stats.get("qwen35_verification", {})
    if target_evidence["max_active_batch_size"] < 2:
        raise SystemExit("Target-only requests never shared a B=2 decode batch")
    if dflash_evidence["max_active_batch_size"] < 2:
        raise SystemExit("DFlash2 requests never shared a B=2 decode batch")
    if dflash_evidence["max_speculative_batch_size"] < 2:
        raise SystemExit("DFlash2 requests never shared a B=2 speculative batch")
    if dflash_evidence["max_position_zero_proposed"] < 2:
        raise SystemExit("A B=2 speculative step did not propose for both requests")
    if speculative["proposed_tokens"] <= 0:
        raise SystemExit("DFlash2 B=2 run produced no speculative proposals")
    if verification.get("max_batch_size", 0) < 2:
        raise SystemExit("Qwen3.8 verification never executed a tensor batch of 2")
    if verification.get("tensor_batched_token_steps", 0) <= 0:
        raise SystemExit("Qwen3.8 verification recorded no tensor-batched token step")
    if any(value <= 0 for value in dflash_evidence["accepted_by_request"].values()):
        raise SystemExit(
            "DFlash2 B=2 did not record an accepted draft token for each request: "
            f"{dflash_evidence['accepted_by_request']}"
        )

    print(f"target_only_b2_evidence={target_evidence}", flush=True)
    print(f"dflash2_b2_evidence={dflash_evidence}", flush=True)
    print(f"qwen35_verification_evidence={verification}", flush=True)
    print(f"dflash2_proposed_tokens={speculative['proposed_tokens']}", flush=True)
    print(f"dflash2_accepted_tokens={speculative['accepted_tokens']}", flush=True)
    print(f"dflash2_acceptance_rate={speculative['acceptance_rate']}", flush=True)
    print(
        f"dflash2_num_speculative_tokens={args.num_speculative_tokens}", flush=True
    )
    print("dflash2_online_b2=PASS", flush=True)
    print("dflash2_oracle_alignment=PENDING", flush=True)


if __name__ == "__main__":
    main()
