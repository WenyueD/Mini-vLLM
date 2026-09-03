from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from myvllm.engine.hybrid_cache import HybridCacheManager
from myvllm.models.qwen3_8 import Qwen35ForCausalLM
from myvllm.models.qwen3_8_config import (
    QWEN35_DEFAULT_HIDDEN_TAPS,
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)
from myvllm.spec_decode.dflash2 import (
    DFlash2Config,
    DFlash2DraftModel,
    load_dflash2_weights_from_safetensors,
)
from myvllm.utils.loader import load_weights_from_checkpoint


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
        description="Run one real DFlash2 standalone block proposal."
    )
    parser.add_argument(
        "--target-model",
        default=os.environ.get("MINIVLLM_QWEN38_PATH"),
        help="Local Qwen3.8 snapshot path. Defaults to MINIVLLM_QWEN38_PATH.",
    )
    parser.add_argument(
        "--draft-model",
        default=os.environ.get("MINIVLLM_DFLASH2_PATH"),
        help="Local DFlash2 snapshot path. Defaults to MINIVLLM_DFLASH2_PATH.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--safe-open-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write a proposal fixture for Oracle alignment.",
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


def materialize_target(snapshot: Path, device: torch.device, dtype: torch.dtype):
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


def materialize_draft(snapshot: Path, device: torch.device, dtype: torch.dtype):
    config = DFlash2Config.from_pretrained_path(str(snapshot))
    config.validate_contract()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = DFlash2DraftModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.to_empty(device=device)
    model.eval()
    return model


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.target_model:
        raise SystemExit("Pass --target-model or set MINIVLLM_QWEN38_PATH")
    if not args.draft_model:
        raise SystemExit("Pass --draft-model or set MINIVLLM_DFLASH2_PATH")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DFlash2 standalone proposal smoke")

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    target_snapshot = Path(args.target_model).expanduser()
    draft_snapshot = Path(args.draft_model).expanduser()
    safe_open_device = str(device) if args.safe_open_device == "cuda" else "cpu"

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"target_snapshot={target_snapshot}", flush=True)
    print(f"draft_snapshot={draft_snapshot}", flush=True)
    print(f"device={device}", flush=True)
    print(f"dtype={dtype}", flush=True)
    print(f"safe_open_device={safe_open_device}", flush=True)
    print(f"prompt={args.prompt!r}", flush=True)
    print_cuda_memory("before_models", device)

    started = time.time()
    target = materialize_target(target_snapshot, device, dtype)
    load_weights_from_checkpoint(
        target,
        str(target_snapshot),
        strict_language=True,
        stream_safetensors=True,
        safe_open_device=safe_open_device,
    )
    print(f"target_loaded_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_target_load", device)

    started = time.time()
    draft = materialize_draft(draft_snapshot, device, dtype)
    load_dflash2_weights_from_safetensors(
        draft,
        str(draft_snapshot),
        safe_open_device=safe_open_device,
    )
    print(f"draft_loaded_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_draft_load", device)

    tokenizer = AutoTokenizer.from_pretrained(str(target_snapshot))
    token_ids = tokenizer.encode(args.prompt)
    if not token_ids:
        raise RuntimeError("prompt encoded to no tokens")
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    cache = HybridCacheManager()
    with torch.inference_mode():
        target_output = target.forward_with_cache(
            input_ids,
            cache,
            "standalone",
            capture_hidden_taps=True,
            hidden_tap_layers=QWEN35_DEFAULT_HIDDEN_TAPS,
        )
        hidden_taps = {
            layer_id: value[:, -min(value.shape[1], draft.config.sliding_window) :, :]
            for layer_id, value in target_output.hidden_taps.items()
        }
        anchor_token_id = int(target_output.logits[:, -1, :].argmax(dim=-1).item())
        anchor_position = len(token_ids)
        proposal_batch, debug = draft.forward_block(
            anchor_token_ids=torch.tensor(
                [anchor_token_id],
                dtype=torch.long,
                device=device,
            ),
            anchor_positions=torch.tensor(
                [anchor_position],
                dtype=torch.long,
                device=device,
            ),
            hidden_taps=hidden_taps,
            token_embedding=target.model.embed_tokens,
            lm_head=target.lm_head,
            return_debug=True,
        )

    valid_length = int(proposal_batch.valid_lengths[0].item())
    proposed = proposal_batch.token_ids[0, :valid_length].tolist()
    debug_json = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in debug.items()
    }
    fixture = {
        "source": "minivllm",
        "prompt": args.prompt,
        "prompt_token_ids": token_ids,
        "anchor_token_id": anchor_token_id,
        "anchor_position": anchor_position,
        "target_layer_ids": list(draft.config.target_layer_ids),
        "proposal_valid_length": valid_length,
        "proposal_token_ids": proposed,
        **debug_json,
    }
    print(f"prompt_token_ids={token_ids}", flush=True)
    print(f"anchor_token_id={anchor_token_id}", flush=True)
    print(f"anchor_position={anchor_position}", flush=True)
    print(f"hidden_tap_keys={sorted(hidden_taps)}", flush=True)
    print(f"proposal_valid_lengths={proposal_batch.valid_lengths.tolist()}", flush=True)
    print(f"proposal_token_ids={proposed}", flush=True)
    for key, value in debug.items():
        print(f"{key}={value}", flush=True)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"json_output={output_path}", flush=True)
    print_cuda_memory("after_proposal", device)
    print("dflash2_standalone_proposal=PASS", flush=True)
    print("dflash2_oracle_alignment=PENDING", flush=True)


if __name__ == "__main__":
    main()
