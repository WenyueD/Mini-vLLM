from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any


CAPTURES: list[dict[str, Any]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Oracle/vLLM DFlash2 once and export the first captured draft "
            "proposal as a Mini-vLLM comparator fixture."
        )
    )
    parser.add_argument("--target-model", default=os.environ.get("MINIVLLM_QWEN38_PATH"))
    parser.add_argument("--draft-model", default=os.environ.get("MINIVLLM_DFLASH2_PATH"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--method", default="dflash")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional vLLM dtype, for example bfloat16, float16, or auto.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--skip-target-preflight",
        action="store_true",
        help=(
            "Skip the Oracle target-only comparison. By default the exporter "
            "first confirms that Oracle DFlash generation starts with the same "
            "completion token as Oracle target-only generation."
        ),
    )
    parser.add_argument(
        "--expected-target-first-token-id",
        type=int,
        default=None,
        help=(
            "Known Oracle target-only first completion token. When set together "
            "with --skip-target-preflight, the exporter still checks that Oracle "
            "DFlash generation starts with this token without loading a separate "
            "target-only LLM in the same process."
        ),
    )
    parser.add_argument(
        "--allow-prefill-anchor",
        action="store_true",
        help=(
            "Allow comparing a DFlash capture whose anchor came from "
            "next_prefill_tokens. By default the exporter selects the first "
            "capture with num_sampled > 0, matching target_bonus_token semantics."
        ),
    )
    parser.add_argument(
        "--anchor-position-mode",
        choices=("oracle", "minivllm", "prompt-final"),
        default="minivllm",
        help=(
            "Use 'oracle' or 'minivllm' for the first generated token position, "
            "or 'prompt-final' for the final prompt-token position."
        ),
    )
    parser.add_argument(
        "--anchor-token-mode",
        choices=("oracle", "minivllm", "prompt-final"),
        default="minivllm",
        help=(
            "Use 'oracle' for the token inferred from vLLM propose arguments, "
            "'minivllm' for the first generated target token used by the "
            "Mini-vLLM standalone fixture, or 'prompt-final' for the final "
            "prompt token."
        ),
    )
    parser.add_argument(
        "--extra-llm-kwargs",
        default="{}",
        help="JSON object merged into vLLM.LLM keyword arguments.",
    )
    return parser.parse_args()


def release_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def tensor_to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def as_int(value: Any) -> int:
    if isinstance(value, list):
        if not value:
            return 0
        return as_int(value[0])
    return int(value)


def install_dflash_capture_hook() -> None:
    try:
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )
    except Exception as exc:  # pragma: no cover - Oracle-only environment.
        raise RuntimeError(
            "Could not import vLLM DFlashSpeculator. Run this inside the "
            "Oracle/vLLM environment that can serve DFlash2."
        ) from exc

    original_propose = DFlashSpeculator.propose

    def wrapped_propose(self, *args, **kwargs):
        result = original_propose(self, *args, **kwargs)
        dummy_run = bool(kwargs.get("dummy_run", False))
        if len(args) >= 12:
            dummy_run = bool(args[11])
        if dummy_run:
            return result

        input_batch = args[0] if len(args) > 0 else kwargs.get("input_batch")
        num_sampled = args[5] if len(args) > 5 else kwargs.get("num_sampled")
        num_rejected = args[6] if len(args) > 6 else kwargs.get("num_rejected")
        last_sampled = args[7] if len(args) > 7 else kwargs.get("last_sampled")
        next_prefill_tokens = (
            args[8] if len(args) > 8 else kwargs.get("next_prefill_tokens")
        )

        num_reqs = int(getattr(input_batch, "num_reqs", 1) or 1)
        proposal_rows = tensor_to_list(result[:num_reqs])
        num_sampled_rows = tensor_to_list(num_sampled[:num_reqs])
        num_rejected_rows = tensor_to_list(num_rejected[:num_reqs])
        last_sampled_rows = tensor_to_list(last_sampled[:num_reqs])
        next_prefill_rows = tensor_to_list(next_prefill_tokens[:num_reqs])
        sample_pos = tensor_to_list(getattr(self, "sample_pos", [])[: self.num_speculative_steps])
        sample_indices = tensor_to_list(
            getattr(self, "sample_indices", [])[: self.num_speculative_steps]
        )

        sampled_count = as_int(num_sampled_rows)
        anchor_token_id = (
            as_int(last_sampled_rows)
            if sampled_count > 0
            else as_int(next_prefill_rows)
        )
        CAPTURES.append(
            {
                "proposal_rows": proposal_rows,
                "num_reqs": num_reqs,
                "num_sampled": num_sampled_rows,
                "num_rejected": num_rejected_rows,
                "last_sampled": last_sampled_rows,
                "next_prefill_tokens": next_prefill_rows,
                "anchor_token_id": anchor_token_id,
                "num_speculative_steps": int(self.num_speculative_steps),
                "num_query_per_req": int(getattr(self, "num_query_per_req", 0)),
                "sample_pos": sample_pos,
                "sample_indices": sample_indices,
                "sample_from_anchor": bool(getattr(self, "sample_from_anchor", False)),
            }
        )
        return result

    DFlashSpeculator.propose = wrapped_propose


def first_output_token_ids(output: Any) -> list[int]:
    for attr in ("outputs",):
        candidates = getattr(output, attr, None)
        if candidates:
            token_ids = getattr(candidates[0], "token_ids", None)
            if token_ids is not None:
                return [int(value) for value in token_ids]
    token_ids = getattr(output, "token_ids", None)
    if token_ids is not None:
        return [int(value) for value in token_ids]
    return []


def completion_token_ids(raw_token_ids: list[int], prompt_token_ids: list[int]) -> list[int]:
    if raw_token_ids[: len(prompt_token_ids)] == prompt_token_ids:
        return raw_token_ids[len(prompt_token_ids) :]
    return raw_token_ids


def build_llm_kwargs(args: argparse.Namespace, speculative: bool) -> dict[str, Any]:
    extra_kwargs = json.loads(args.extra_llm_kwargs)
    if not isinstance(extra_kwargs, dict):
        raise SystemExit("--extra-llm-kwargs must be a JSON object")

    llm_kwargs = {
        "model": args.target_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        **extra_kwargs,
    }
    if args.dtype:
        llm_kwargs["dtype"] = args.dtype
    if speculative:
        llm_kwargs["speculative_config"] = {
            "method": args.method,
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    return llm_kwargs


def main() -> None:
    args = parse_args()
    if not args.target_model:
        raise SystemExit("Pass --target-model or set MINIVLLM_QWEN38_PATH")
    if not args.draft_model:
        raise SystemExit("Pass --draft-model or set MINIVLLM_DFLASH2_PATH")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_token_ids = [int(value) for value in tokenizer.encode(args.prompt)]
    if not prompt_token_ids:
        raise SystemExit("prompt encoded to zero tokens")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    target_only_token_ids: list[int] = []
    if args.skip_target_preflight and args.expected_target_first_token_id is not None:
        target_only_token_ids = [int(args.expected_target_first_token_id)]
        print(
            f"oracle_target_only_token_ids={target_only_token_ids} "
            "source=expected-target-first-token-id",
            flush=True,
        )
    elif not args.skip_target_preflight:
        target_llm = LLM(**build_llm_kwargs(args, speculative=False))
        target_outputs = target_llm.generate([args.prompt], sampling_params)
        target_raw = first_output_token_ids(target_outputs[0]) if target_outputs else []
        target_only_token_ids = completion_token_ids(target_raw, prompt_token_ids)
        print(f"oracle_target_only_token_ids={target_only_token_ids}", flush=True)
        release_llm(target_llm)

    install_dflash_capture_hook()

    llm = LLM(**build_llm_kwargs(args, speculative=True))
    outputs = llm.generate([args.prompt], sampling_params)
    if not CAPTURES:
        raise SystemExit(
            "No non-dummy DFlash proposal was captured. Increase --max-tokens "
            "or check that vLLM is actually using DFlash/DFlash2. "
            "Debug: VLLM_ENABLE_V1_MULTIPROCESSING="
            f"{os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')!r}, "
            "VLLM_WORKER_MULTIPROC_METHOD="
            f"{os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')!r}."
        )

    selected_index = None
    for index, candidate in enumerate(CAPTURES):
        if args.allow_prefill_anchor or as_int(candidate["num_sampled"]) > 0:
            selected_index = index
            break
    if selected_index is None:
        summary = [
            {
                "index": index,
                "num_sampled": candidate.get("num_sampled"),
                "anchor_token_id": candidate.get("anchor_token_id"),
                "proposal_rows": candidate.get("proposal_rows"),
            }
            for index, candidate in enumerate(CAPTURES)
        ]
        raise SystemExit(
            "Only prefill-anchor DFlash captures were found; no capture with "
            f"num_sampled > 0 is available. Captures={summary}"
        )

    capture = CAPTURES[selected_index]
    proposal_rows = capture["proposal_rows"]
    proposal_token_ids = [int(value) for value in proposal_rows[0]]
    raw_output_token_ids = first_output_token_ids(outputs[0]) if outputs else []
    output_token_ids = completion_token_ids(raw_output_token_ids, prompt_token_ids)
    if not output_token_ids:
        raise SystemExit("Oracle generation produced no output tokens")
    if target_only_token_ids and output_token_ids[0] != target_only_token_ids[0]:
        raise SystemExit(
            "Oracle DFlash generation is not target-lossless for this prompt: "
            f"target_first={target_only_token_ids[0]} "
            f"dflash_first={output_token_ids[0]} "
            f"target_tokens={target_only_token_ids} "
            f"dflash_tokens={output_token_ids}. "
            "Retry with --method dflash2 if supported, or align a prompt/config "
            "where Oracle target-only and Oracle DFlash outputs match first."
        )
    sampled_count = as_int(capture["num_sampled"])
    oracle_anchor_position = len(prompt_token_ids) + max(0, sampled_count - 1)
    minivllm_anchor_position = len(prompt_token_ids)
    prompt_final_anchor_position = len(prompt_token_ids) - 1
    if args.anchor_position_mode == "oracle":
        anchor_position = oracle_anchor_position
    elif args.anchor_position_mode == "prompt-final":
        anchor_position = prompt_final_anchor_position
    else:
        anchor_position = minivllm_anchor_position
    oracle_anchor_token_id = int(capture["anchor_token_id"])
    output_anchor_index = max(0, sampled_count - 1)
    if output_anchor_index >= len(output_token_ids):
        output_anchor_index = 0
    minivllm_anchor_token_id = int(output_token_ids[output_anchor_index])
    prompt_final_anchor_token_id = int(prompt_token_ids[-1])
    if args.anchor_token_mode == "oracle":
        anchor_token_id = oracle_anchor_token_id
    elif args.anchor_token_mode == "prompt-final":
        anchor_token_id = prompt_final_anchor_token_id
    else:
        anchor_token_id = minivllm_anchor_token_id

    fixture = {
        "fixture_schema_version": 2,
        "anchor_semantics": "target_bonus_token",
        "source": "oracle-vllm",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_token_ids,
        "anchor_token_id": int(anchor_token_id),
        "anchor_token_mode": args.anchor_token_mode,
        "selected_capture_index": int(selected_index),
        "oracle_capture_anchor_token_id": int(oracle_anchor_token_id),
        "minivllm_anchor_token_id": int(minivllm_anchor_token_id),
        "prompt_final_anchor_token_id": int(prompt_final_anchor_token_id),
        "anchor_position": int(anchor_position),
        "anchor_position_mode": args.anchor_position_mode,
        "oracle_anchor_position": int(oracle_anchor_position),
        "minivllm_anchor_position": int(minivllm_anchor_position),
        "prompt_final_anchor_position": int(prompt_final_anchor_position),
        "target_layer_ids": [5, 19, 33, 47, 61],
        "proposal_valid_length": len(proposal_token_ids),
        "proposal_token_ids": proposal_token_ids,
        "oracle_target_only_token_ids": target_only_token_ids,
        "raw_output_token_ids": raw_output_token_ids,
        "output_token_ids": output_token_ids,
        "capture": capture,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prompt_token_ids={prompt_token_ids}", flush=True)
    print(f"oracle_target_only_token_ids={target_only_token_ids}", flush=True)
    print(f"selected_capture_index={selected_index}", flush=True)
    print(f"capture_num_sampled={capture['num_sampled']}", flush=True)
    print(f"raw_output_token_ids={raw_output_token_ids}", flush=True)
    print(f"output_token_ids={output_token_ids}", flush=True)
    print(f"anchor_token_id={fixture['anchor_token_id']}", flush=True)
    print(f"oracle_capture_anchor_token_id={oracle_anchor_token_id}", flush=True)
    print(f"anchor_position={fixture['anchor_position']}", flush=True)
    print(f"oracle_anchor_position={oracle_anchor_position}", flush=True)
    print(f"proposal_token_ids={proposal_token_ids}", flush=True)
    print(f"json_output={output_path}", flush=True)
    print("oracle_dflash2_fixture=PASS", flush=True)


if __name__ == "__main__":
    main()
