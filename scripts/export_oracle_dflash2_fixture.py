from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
from typing import Any


CAPTURES: list[dict[str, Any]] = []
SPECULATOR_DEBUG_PRINTED = False
CAPTURE_ENABLED = False


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
        "--allow-non-lossless-fixture",
        action="store_true",
        help=(
            "Write the diagnostic fixture even when Oracle speculative generation "
            "does not start with the expected target token. Such a fixture is for "
            "debug only and must not be treated as a passing Oracle alignment."
        ),
    )
    parser.add_argument(
        "--dump-speculator-debug",
        action="store_true",
        help=(
            "Print a compact DFlashSpeculator runtime attribute summary so "
            "version-specific Oracle capture fields can be mapped."
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


def tensor_attr_to_list(
    owner: Any,
    name: str,
    rows: int | None = None,
    *,
    indices: Any | None = None,
) -> Any:
    if not hasattr(owner, name):
        return None
    value = getattr(owner, name)
    if value is None:
        return None
    if rows is None:
        return tensor_to_list(value)
    return tensor_rows_to_list(value, rows, indices=indices)


def summarize_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        tensor = value.detach()
        summary: dict[str, Any] = {
            "type": type(value).__name__,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        if tensor.numel() <= 32:
            summary["value"] = tensor.cpu().tolist()
        return summary
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "items": [summarize_value(item) for item in list(value)[:4]],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "len": len(value),
            "keys": [str(key) for key in list(value)[:16]],
        }
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return {"type": type(value).__name__}


def summarize_speculator_attrs(owner: Any) -> dict[str, Any]:
    interesting = (
        "candidate",
        "draft",
        "hidden",
        "index",
        "indices",
        "logit",
        "proposal",
        "sample",
        "score",
        "token",
    )
    summary: dict[str, Any] = {}
    for name, value in sorted(vars(owner).items()):
        if any(part in name.lower() for part in interesting):
            summary[name] = summarize_value(value)
    return summary


def as_int(value: Any) -> int:
    if isinstance(value, list):
        if not value:
            return 0
        return as_int(value[0])
    return int(value)


def bound_propose_arguments(signature, owner, args, kwargs) -> dict[str, Any]:
    try:
        return dict(signature.bind_partial(owner, *args, **kwargs).arguments)
    except TypeError:
        return {}


def argument_value(
    bound: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    fallback_index: int,
) -> Any:
    if name in bound:
        return bound[name]
    if name in kwargs:
        return kwargs[name]
    if len(args) > fallback_index:
        return args[fallback_index]
    return None


def safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def tensor_rows_to_list(
    value: Any,
    rows: int,
    *,
    indices: Any | None = None,
) -> Any:
    if value is None:
        return None
    # Copy before selecting sparse request slots. Advanced-indexing vLLM's CUDA
    # buffers here can poison the process with a device-side assert when a slot is
    # stale or the source is a compact per-step tensor.
    copied = tensor_to_list(value)
    if indices is not None:
        copied_indices = tensor_to_list(indices)
        try:
            selected = [copied[as_int(index)] for index in copied_indices[:rows]]
        except (IndexError, TypeError, ValueError):
            selected = []
        if len(selected) == rows:
            return selected
    try:
        return copied[:rows]
    except Exception:
        return copied


def input_batch_active_indices(input_batch: Any, rows: int) -> Any | None:
    for name in (
        "req_indices",
        "request_indices",
        "req_state_indices",
        "request_state_indices",
        "idx_mapping",
    ):
        value = getattr(input_batch, name, None)
        if value is None:
            continue
        try:
            return value[:rows]
        except Exception:
            return value
    return None


def summarize_input_batch(input_batch: Any, rows: int) -> dict[str, Any]:
    if input_batch is None:
        return {}
    summary: dict[str, Any] = {}
    for name in (
        "num_reqs",
        "req_ids",
        "req_indices",
        "request_indices",
        "req_state_indices",
        "request_state_indices",
        "idx_mapping",
        "num_tokens",
    ):
        if hasattr(input_batch, name):
            value = getattr(input_batch, name)
            if hasattr(value, "detach") or hasattr(value, "__getitem__"):
                summary[name] = tensor_rows_to_list(value, rows)
            else:
                summary[name] = summarize_value(value)
    return summary


def install_dflash_capture_hook(
    *,
    dump_speculator_debug: bool = False,
    expected_num_prompts: int = 1,
) -> None:
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
    propose_signature = inspect.signature(original_propose)
    if dump_speculator_debug:
        print(
            f"oracle_speculator_class={DFlashSpeculator.__module__}."
            f"{DFlashSpeculator.__name__}",
            flush=True,
        )
        print(
            f"oracle_speculator_file={inspect.getsourcefile(DFlashSpeculator)}",
            flush=True,
        )
        print(
            f"oracle_speculator_propose_signature={propose_signature}",
            flush=True,
        )

    def wrapped_propose(self, *args, **kwargs):
        global SPECULATOR_DEBUG_PRINTED
        bound = bound_propose_arguments(propose_signature, self, args, kwargs)
        result = original_propose(self, *args, **kwargs)
        if not CAPTURE_ENABLED:
            return result
        dummy_run = bool(argument_value(bound, args, kwargs, "dummy_run", 11))
        if dummy_run:
            return result

        input_batch = argument_value(bound, args, kwargs, "input_batch", 0)
        num_sampled = argument_value(bound, args, kwargs, "num_sampled", 5)
        num_rejected = argument_value(bound, args, kwargs, "num_rejected", 6)
        last_sampled = argument_value(bound, args, kwargs, "last_sampled", 7)
        next_prefill_tokens = argument_value(
            bound,
            args,
            kwargs,
            "next_prefill_tokens",
            8,
        )

        inferred_num_reqs = int(getattr(input_batch, "num_reqs", 1) or 1)
        rows = min(expected_num_prompts, inferred_num_reqs)
        result_len = safe_len(result)
        if result_len is not None:
            rows = min(rows, result_len)
        rows = max(rows, 1)
        active_indices = input_batch_active_indices(input_batch, rows)
        proposal_rows = tensor_rows_to_list(result, rows)
        num_sampled_rows = tensor_rows_to_list(
            num_sampled, rows, indices=active_indices
        )
        num_rejected_rows = tensor_rows_to_list(
            num_rejected, rows, indices=active_indices
        )
        last_sampled_rows = tensor_rows_to_list(
            last_sampled, rows, indices=active_indices
        )
        next_prefill_rows = tensor_rows_to_list(
            next_prefill_tokens, rows, indices=active_indices
        )
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
        proposal_first_token_id = None
        try:
            proposal_first_token_id = int(proposal_rows[0][0])
        except Exception:
            pass
        CAPTURES.append(
            {
                "proposal_rows": proposal_rows,
                "proposal_first_token_id": proposal_first_token_id,
                "num_reqs": rows,
                "inferred_num_reqs": inferred_num_reqs,
                "active_request_indices": tensor_to_list(active_indices)
                if active_indices is not None
                else None,
                "input_batch_summary": summarize_input_batch(input_batch, rows),
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
                "candidate_token_ids": tensor_attr_to_list(
                    self,
                    "_cached_candidate_ids",
                    rows,
                ),
                "selector_selected_indices": tensor_attr_to_list(
                    self,
                    "_cached_sample_indices",
                    rows,
                ),
                "selector_realized_scores": tensor_attr_to_list(
                    self,
                    "_selector_scores",
                    rows,
                ),
                "draft_token_ids": tensor_attr_to_list(
                    self,
                    "draft_tokens",
                    rows,
                ),
            }
        )
        if dump_speculator_debug and not SPECULATOR_DEBUG_PRINTED:
            SPECULATOR_DEBUG_PRINTED = True
            attr_summary = summarize_speculator_attrs(self)
            CAPTURES[-1]["speculator_attr_summary"] = attr_summary
            print(
                "oracle_speculator_attr_summary="
                + json.dumps(attr_summary, ensure_ascii=False, sort_keys=True),
                flush=True,
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


def summarize_generation_output(output: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": type(output).__name__,
        "repr": repr(output)[:2000],
    }
    for name in (
        "prompt",
        "prompt_token_ids",
        "encoder_prompt",
        "encoder_prompt_token_ids",
        "request_id",
        "finished",
        "metrics",
    ):
        if hasattr(output, name):
            summary[name] = summarize_value(getattr(output, name))
    completions = getattr(output, "outputs", None)
    if completions is not None:
        summary["outputs_type"] = type(completions).__name__
        summary["outputs_len"] = safe_len(completions)
        items = []
        for completion in list(completions)[:2]:
            item: dict[str, Any] = {"type": type(completion).__name__}
            for name in (
                "index",
                "text",
                "token_ids",
                "cumulative_logprob",
                "logprobs",
                "finish_reason",
                "stop_reason",
            ):
                if hasattr(completion, name):
                    item[name] = summarize_value(getattr(completion, name))
            items.append(item)
        summary["outputs_items"] = items
    for name in ("token_ids", "text", "tokens"):
        if hasattr(output, name):
            summary[name] = summarize_value(getattr(output, name))
    return summary


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

    install_dflash_capture_hook(dump_speculator_debug=args.dump_speculator_debug)

    global CAPTURE_ENABLED
    llm = LLM(**build_llm_kwargs(args, speculative=True))
    CAPTURE_ENABLED = True
    try:
        outputs = llm.generate([args.prompt], sampling_params)
    finally:
        CAPTURE_ENABLED = False
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
    dflash_output_summary = (
        summarize_generation_output(outputs[0]) if outputs else {}
    )
    if not output_token_ids:
        raise SystemExit("Oracle generation produced no output tokens")
    oracle_target_lossless = not target_only_token_ids or (
        output_token_ids[0] == target_only_token_ids[0]
    )
    if target_only_token_ids and not oracle_target_lossless and not args.allow_non_lossless_fixture:
        summary = [
            {
                "index": index,
                "num_sampled": candidate.get("num_sampled"),
                "anchor_token_id": candidate.get("anchor_token_id"),
                "last_sampled": candidate.get("last_sampled"),
                "next_prefill_tokens": candidate.get("next_prefill_tokens"),
                "proposal_rows": candidate.get("proposal_rows"),
                "proposal_first_token_id": candidate.get("proposal_first_token_id"),
                "active_request_indices": candidate.get("active_request_indices"),
                "input_batch_summary": candidate.get("input_batch_summary"),
                "candidate_token_ids_present": candidate.get("candidate_token_ids")
                is not None,
                "selector_selected_indices_present": candidate.get(
                    "selector_selected_indices"
                )
                is not None,
            }
            for index, candidate in enumerate(CAPTURES)
        ]
        raise SystemExit(
            "Oracle DFlash generation is not target-lossless for this prompt: "
            f"target_first={target_only_token_ids[0]} "
            f"dflash_first={output_token_ids[0]} "
            f"target_tokens={target_only_token_ids} "
            f"dflash_tokens={output_token_ids} "
            f"dflash_output_summary={dflash_output_summary} "
            f"captures={summary}. "
            "Retry with --method dflash2 if supported, or align a prompt/config "
            "where Oracle target-only and Oracle DFlash outputs match first. "
            "For diagnostics only, rerun with --allow-non-lossless-fixture."
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
    minivllm_anchor_token_source = "output_token_ids"
    if (
        not oracle_target_lossless
        and args.allow_non_lossless_fixture
        and args.expected_target_first_token_id is not None
        and output_anchor_index == 0
    ):
        minivllm_anchor_token_id = int(args.expected_target_first_token_id)
        minivllm_anchor_token_source = "expected_target_first_token_id"
    else:
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
        "oracle_effective_anchor_matches_fixture": (
            int(oracle_anchor_token_id) == int(anchor_token_id)
        ),
        "minivllm_anchor_token_id": int(minivllm_anchor_token_id),
        "minivllm_anchor_token_source": minivllm_anchor_token_source,
        "prompt_final_anchor_token_id": int(prompt_final_anchor_token_id),
        "anchor_position": int(anchor_position),
        "anchor_position_mode": args.anchor_position_mode,
        "oracle_anchor_position": int(oracle_anchor_position),
        "minivllm_anchor_position": int(minivllm_anchor_position),
        "prompt_final_anchor_position": int(prompt_final_anchor_position),
        "target_layer_ids": [5, 19, 33, 47, 61],
        "proposal_valid_length": len(proposal_token_ids),
        "proposal_token_ids": proposal_token_ids,
        "oracle_target_lossless": oracle_target_lossless,
        "oracle_target_only_token_ids": target_only_token_ids,
        "raw_output_token_ids": raw_output_token_ids,
        "output_token_ids": output_token_ids,
        "dflash_output_summary": dflash_output_summary,
        "capture": capture,
    }
    if capture.get("candidate_token_ids") is not None:
        fixture["candidate_token_ids"] = capture["candidate_token_ids"]
    if capture.get("selector_selected_indices") is not None:
        fixture["selector_selected_indices"] = capture["selector_selected_indices"]
    if capture.get("selector_realized_scores") is not None:
        fixture["selector_realized_scores"] = capture["selector_realized_scores"]
    if capture.get("draft_token_ids") is not None:
        fixture["draft_token_ids"] = capture["draft_token_ids"]

    release_llm(llm)

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
    print(
        "dflash_output_summary="
        + json.dumps(dflash_output_summary, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    print(f"anchor_token_id={fixture['anchor_token_id']}", flush=True)
    print(f"oracle_capture_anchor_token_id={oracle_anchor_token_id}", flush=True)
    print(f"anchor_position={fixture['anchor_position']}", flush=True)
    print(f"oracle_anchor_position={oracle_anchor_position}", flush=True)
    print(f"proposal_token_ids={proposal_token_ids}", flush=True)
    print(f"oracle_target_lossless={oracle_target_lossless}", flush=True)
    print(f"json_output={output_path}", flush=True)
    if oracle_target_lossless:
        print("oracle_dflash2_fixture=PASS", flush=True)
    else:
        print("oracle_dflash2_fixture=DEBUG_NON_LOSSLESS", flush=True)


if __name__ == "__main__":
    main()
