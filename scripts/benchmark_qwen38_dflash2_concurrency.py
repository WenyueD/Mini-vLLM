from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if os.fspath(SRC_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SRC_ROOT))

from myvllm.sampling_parameters import SamplingParams

RESULTS_ROOT = ROOT / "results" / "qwen38_dflash2_perf"
LOGS_ROOT = ROOT / "logs" / "qwen38_dflash2_perf"

WORKLOADS = {
    "short": {"isl": 128, "osl": 64},
    "decode512": {"isl": 128, "osl": 512},
    "decode1024": {"isl": 128, "osl": 1024},
    "decode4096": {"isl": 128, "osl": 4096},
    "tasks4096": {"isl": None, "osl": 4096},
    "medium": {"isl": 512, "osl": 128},
    "long": {"isl": 1024, "osl": 128},
}

DEFAULT_PROMPT_SEEDS = [
    "Explain how speculative decoding preserves greedy output equality.",
    "Translate the following technical note into concise Chinese, then summarize it.",
    "Solve the arithmetic puzzle step by step and verify every intermediate value.",
    "Write a compact implementation checklist for a high-throughput inference server.",
    "Compare cache allocation strategies for mixed short and long generation requests.",
]

LONG_OUTPUT_TASKS = [
    "Write a complete Python 3 implementation of a persistent SQLite job queue. "
    "Include schema creation and migration, enqueue with idempotency keys, atomic "
    "leasing by multiple worker processes, lease renewal, retries with exponential "
    "backoff, dead-letter handling, and crash recovery. Use only the standard library. "
    "Provide all executable source code, followed by at least twenty concrete unittest "
    "tests covering successful execution, contention, expired leases, transaction "
    "rollback, duplicate keys, retry exhaustion, and recovery after reopening the database. "
    "Then explain each transaction boundary, demonstrate a two-worker run, and discuss "
    "correctness invariants and limitations. Do not substitute outlines, placeholders, "
    "or ellipses for the requested implementation or tests.",
    "Develop a complete Python asyncio client and server for a length-prefixed binary "
    "protocol using only the standard library. Messages carry request IDs, operation "
    "names, and JSON bodies. Implement incremental frame parsing, maximum frame sizes, "
    "timeouts, backpressure, bounded concurrency, cancellation, reconnect behavior, "
    "and graceful shutdown. Include the complete client, server, parser, and a runnable "
    "demonstration. Provide at least twenty executable tests for fragmented frames, "
    "multiple frames in a read, malformed lengths, malformed JSON, disconnects, slow "
    "clients, cancellation races, and shutdown. Finish with a detailed explanation "
    "of resource ownership, error propagation, and protocol invariants. Give actual "
    "code and assertions throughout rather than a sketch or a list of future work.",
    "Build a detailed tutorial and complete Python implementation for weighted interval "
    "scheduling with reconstruction and deterministic tie breaking. Begin with a precise "
    "problem statement covering negative weights, empty intervals, duplicate endpoints, "
    "and half-open interval semantics. Derive the dynamic-programming recurrence and "
    "prove correctness and complexity. Implement binary-search predecessor lookup, "
    "solution reconstruction, and tie breaking by the lexicographically smallest list "
    "of original interval IDs. Include a brute-force oracle and at least twenty-five "
    "fully specified tests, including exhaustive comparisons on small instances. Work "
    "through three nontrivial examples with predecessor arrays, every DP entry, and "
    "reconstructed solutions. Explain why common greedy alternatives fail, using "
    "explicit counterexamples. Supply complete runnable code without omitted helpers.",
    "Implement a production-oriented Python in-memory cache with weighted LRU eviction, "
    "per-entry TTL, a monotonic clock, and thread-safe single-flight loading. Use the "
    "standard library only and allow a fake clock for tests. Specify behavior for "
    "oversized entries, replacement, expiry, loader exceptions, reentrant calls, and "
    "concurrent invalidation. Provide the complete implementation and at least twenty-five "
    "executable unittest cases. Tests must cover weight accounting, access ordering, "
    "expiry boundaries, failed loaders, concurrent waiters, invalidation during a load, "
    "and resource cleanup. Include a realistic usage example, a proof-style discussion "
    "of invariants, and an analysis of time and space complexity. Explain concurrency "
    "choices and failure behavior in detail. Do not replace code or tests with pseudocode.",
]


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    bucket: str
    arrival_offset_s: float = 0.0
    cancel_after_tokens: int | None = None


@dataclass
class RequestTimeline:
    request_id: str
    bucket: str
    prompt_tokens: int
    max_new_tokens: int
    submit_time_s: float
    admit_return_time_s: float | None = None
    first_token_time_s: float | None = None
    finish_time_s: float | None = None
    token_ids: list[int] = field(default_factory=list)
    token_events: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    cancelled: bool = False

    @property
    def output_tokens(self) -> int:
        return len(self.token_ids)

    def to_record(self, origin_s: float) -> dict[str, Any]:
        record = asdict(self)
        for key in (
            "submit_time_s",
            "admit_return_time_s",
            "first_token_time_s",
            "finish_time_s",
        ):
            if record[key] is not None:
                record[f"{key[:-2]}relative_s"] = record[key] - origin_s
        return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible high-concurrency benchmark for Mini-vLLM Qwen3.8-27B "
            "target-only vs Mini-vLLM DFlash2."
        )
    )
    parser.add_argument(
        "--target-model", default=os.environ.get("MINIVLLM_QWEN38_PATH")
    )
    parser.add_argument(
        "--draft-model", default=os.environ.get("MINIVLLM_DFLASH2_PATH")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32")
    )
    parser.add_argument("--safe-open-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--dflash2-safe-open-device", default=None, choices=("cuda", "cpu")
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--workload",
        nargs="+",
        choices=(*WORKLOADS, "mixed"),
        default=["short", "medium", "mixed"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("target_only", "dflash2"),
        default=["target_only", "dflash2"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests-per-run", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--warmup-output-tokens", type=int, default=0,
        help="Cap only warmup generation length; zero uses the measured workload length.",
    )
    parser.add_argument(
        "--num-speculative-tokens", type=int, default=1, choices=range(1, 8)
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--target-cache-memory-fraction", type=float, default=0.05)
    parser.add_argument("--draft-memory-reservation-fraction", type=float, default=0.0)
    parser.add_argument("--cached-target-micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--replay-state-updates", action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore accepted prefixes from compact state updates instead of model replay.",
    )
    parser.add_argument(
        "--stop-verification-at-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop verifying each request after its first draft mismatch.",
    )
    parser.add_argument(
        "--parallel-target-verification", action="store_true",
        help="Verify the K+1 token block in one causal target forward.",
    )
    parser.add_argument(
        "--parallel-batched-mlp", action="store_true",
        help="Experimentally batch B=2 Qwen MLP gate/up projections.",
    )
    parser.add_argument(
        "--parallel-batched-safe-linears", action="store_true",
        help="Experimentally batch A800-observed B=2 projection shapes.",
    )
    parser.add_argument(
        "--parallel-verification-chunk-size", type=int, default=0,
        help="Verify causal blocks in chunks and skip suffix chunks after rejection.",
    )
    parser.add_argument(
        "--vectorized-target-numerics", action="store_true",
        help="Experimentally share a vectorized BF16 target path across baseline and verifier.",
    )
    parser.add_argument("--max-metrics-history", type=int, default=100000)
    parser.add_argument(
        "--lightweight-live-stats", action="store_true",
        help="Poll live counters only; export complete retained history once after timing.",
    )
    parser.add_argument(
        "--fused-gdn-target", action="store_true",
        help="Fuse the causal GDN recurrence in both vectorized target paths.",
    )
    parser.add_argument(
        "--ignore-eos", dest="ignore_eos", action="store_true", default=True
    )
    parser.add_argument("--respect-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--profile-dflash2-stages", action="store_true")
    parser.add_argument("--memory-trace", action="store_true")
    parser.add_argument("--debug-metadata", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--goodput-ttft-slo", type=float, default=5.0)
    parser.add_argument("--goodput-tpot-slo", type=float, default=0.25)
    parser.add_argument("--open-loop-rate", type=float, default=0.0)
    parser.add_argument("--burst-interval", type=float, default=0.0)
    parser.add_argument("--soak", action="store_true")
    parser.add_argument("--soak-requests", type=int, default=500)
    parser.add_argument("--soak-minutes", type=float, default=10.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--results-dir", default=os.fspath(RESULTS_ROOT))
    parser.add_argument("--logs-dir", default=os.fspath(LOGS_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.warmup_output_tokens < 0:
        parser.error("--warmup-output-tokens must be nonnegative")
    if args.fused_gdn_target and not args.vectorized_target_numerics:
        parser.error("--fused-gdn-target requires --vectorized-target-numerics")
    return args


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q < 0 or q > 100:
        raise ValueError("percentile q must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": percentile(values, 50),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_repeats(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "median": percentile(values, 50),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1024**3


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")


def prompt_for_token_count(
    tokenizer: Any, target_tokens: int, index: int
) -> tuple[str, tuple[int, ...]]:
    seed = DEFAULT_PROMPT_SEEDS[index % len(DEFAULT_PROMPT_SEEDS)]
    text = ""
    while len(tokenizer.encode(text or seed)) < target_tokens:
        text = (text + "\n" + seed).strip()
        text += f"\nRequest index {index}; keep the answer deterministic and factual."
    token_ids = tokenizer.encode(text)
    if len(token_ids) > target_tokens:
        text = tokenizer.decode(token_ids[:target_tokens])
        token_ids = tokenizer.encode(text)
    if not token_ids:
        raise RuntimeError("generated prompt encoded to zero tokens")
    return text, tuple(int(token_id) for token_id in token_ids)


def build_workload(
    tokenizer: Any,
    workload: str,
    total_requests: int,
    seed: int,
    open_loop_rate: float = 0.0,
    burst_interval: float = 0.0,
    include_soak_edges: bool = False,
) -> list[WorkloadRequest]:
    rng = random.Random(seed)
    specs: list[WorkloadRequest] = []
    if workload == "mixed":
        buckets = [
            ("short", 96, 32),
            ("short", 128, 64),
            ("medium", 384, 96),
            ("medium", 512, 128),
            ("long", 768, 128),
        ]
    else:
        bucket = WORKLOADS[workload]
        buckets = [(workload, bucket["isl"], bucket["osl"])]

    arrival = 0.0
    for index in range(total_requests):
        bucket_name, isl, osl = buckets[index % len(buckets)]
        if workload == "tasks4096":
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": LONG_OUTPUT_TASKS[index % len(LONG_OUTPUT_TASKS)]}],
                tokenize=False, add_generation_prompt=True,
            )
            token_ids = tuple(tokenizer.encode(prompt))
        else:
            jitter = rng.randint(-max(1, isl // 16), max(1, isl // 16))
            prompt, token_ids = prompt_for_token_count(
                tokenizer, max(1, isl + jitter), index
            )
        if open_loop_rate > 0:
            arrival += rng.expovariate(open_loop_rate)
        elif burst_interval > 0:
            arrival = index * burst_interval
        cancel_after = None
        if include_soak_edges and index % 97 == 0:
            cancel_after = max(1, osl // 4)
        specs.append(
            WorkloadRequest(
                request_id=f"{workload}-{index:06d}",
                prompt=prompt,
                prompt_token_ids=token_ids,
                max_new_tokens=osl,
                bucket=bucket_name,
                arrival_offset_s=arrival,
                cancel_after_tokens=cancel_after,
            )
        )
    return specs


def build_config(
    args: argparse.Namespace,
    speculative: bool,
    eos: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    from myvllm.models.qwen3_8_config import load_qwen35_config_dict

    if args.cached_target_micro_batch_size <= 0:
        raise ValueError("cached target micro-batch size must be positive")
    config = dict(load_qwen35_config_dict(args.target_model))
    max_model_length = max_prompt_tokens + max_new_tokens + 2
    max_batch_tokens = max(
        max_model_length * concurrency, max_prompt_tokens * concurrency
    )
    dflash_safe_open = args.dflash2_safe_open_device or args.safe_open_device
    config.update(
        {
            "model_name_or_path": args.target_model,
            "world_size": 1,
            "block_size": args.block_size,
            "max_num_sequences": concurrency,
            "max_num_seqs": concurrency,
            "max_num_batched_tokens": max_batch_tokens,
            "max_num_batch_tokens": max_batch_tokens,
            "max_model_length": max_model_length,
            # Exact token comparison requires every cached target path to use
            # identical BF16 CUDA kernel shapes, including accepted-token replay.
            "qwen35_cached_batch_pad_to_size": min(
                concurrency, args.cached_target_micro_batch_size
            ),
            "qwen35_cached_attention_pad_to_length": max_model_length,
            "qwen35_replay_state_updates": getattr(args, "replay_state_updates", True),
            "qwen35_stop_verification_at_mismatch": getattr(
                args, "stop_verification_at_mismatch", False
            ),
            "qwen35_parallel_verification": getattr(
                args, "parallel_target_verification", False
            ),
            "qwen35_parallel_batched_mlp": getattr(
                args, "parallel_batched_mlp", False
            ),
            "qwen35_parallel_batched_safe_linears": getattr(
                args, "parallel_batched_safe_linears", False
            ),
            "qwen35_parallel_verification_chunk_size": getattr(
                args, "parallel_verification_chunk_size", 0
            ),
            "qwen35_vectorized_target": getattr(
                args, "vectorized_target_numerics", False
            ),
            "qwen35_fused_gdn": getattr(args, "fused_gdn_target", False),
            "max_metrics_history": args.max_metrics_history,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "target_cache_memory_fraction": args.target_cache_memory_fraction,
            "draft_memory_reservation_fraction": args.draft_memory_reservation_fraction,
            "enforce_eager": True,
            "eos": int(eos),
            "dtype": args.dtype,
            "init_on_meta": True,
            "stream_weights": True,
            "safe_open_device": args.safe_open_device,
            "strict_language_weights": True,
            "dflash2_safe_open_device": dflash_safe_open,
            "dflash2_debug_metadata": args.debug_metadata,
            "debug_target_decisions": args.debug_metadata,
            "dflash2_profile_stages": args.profile_dflash2_stages,
            "memory_trace": args.memory_trace,
            "speculative_config": {
                "enabled": speculative,
                "proposer": "dflash2" if speculative else "ngram",
                "num_speculative_tokens": args.num_speculative_tokens,
                "draft_model_name_or_path": args.draft_model,
            },
        }
    )
    return config


def reset_runtime_stats(engine: Any) -> None:
    from myvllm.engine.metrics import EngineMetrics
    from myvllm.spec_decode.metrics import SpecDecodeMetrics

    def reset_counters(counters: dict[str, Any]) -> dict[str, Any]:
        reset: dict[str, Any] = {}
        for key, value in counters.items():
            if isinstance(value, bool):
                reset[key] = False
            elif isinstance(value, int):
                reset[key] = 0
            elif isinstance(value, float):
                reset[key] = 0.0
            elif isinstance(value, dict):
                reset[key] = {}
            elif isinstance(value, list):
                reset[key] = []
            elif isinstance(value, tuple):
                reset[key] = ()
            else:
                reset[key] = value
        return reset

    engine.metrics = EngineMetrics(engine.config.get("max_metrics_history", 256))
    coordinator = getattr(engine, "spec_decode_coordinator", None)
    if coordinator is not None:
        coordinator.metrics = SpecDecodeMetrics()
    runner = getattr(engine, "model_runner", None)
    if runner is not None and hasattr(runner, "_qwen35_verification_stats"):
        runner._qwen35_verification_stats = reset_counters(
            runner._qwen35_verification_stats
        )
    if runner is not None and hasattr(runner, "_dflash2_profile_stats"):
        runner._dflash2_profile_stats = reset_counters(runner._dflash2_profile_stats)
    if runner is not None and hasattr(runner, "_target_decision_diagnostics"):
        runner._target_decision_diagnostics = []


def run_summary_failed(summary: dict[str, Any]) -> bool:
    if summary.get("error") or summary.get("schema_errors"):
        return True
    if summary.get("timing", {}).get("failed_requests", 0):
        return True
    admission_failures = summary.get("failures", {}).get("admission_failures", [])
    return bool(admission_failures)


def cuda_memory_snapshot(device: Any) -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        return {
            "cuda_available": True,
            "allocated_gib": bytes_to_gib(torch.cuda.memory_allocated(device)),
            "reserved_gib": bytes_to_gib(torch.cuda.memory_reserved(device)),
            "max_allocated_gib": bytes_to_gib(torch.cuda.max_memory_allocated(device)),
            "max_reserved_gib": bytes_to_gib(torch.cuda.max_memory_reserved(device)),
            "free_gib": bytes_to_gib(free),
            "total_gib": bytes_to_gib(total),
        }
    except Exception as exc:
        return {"cuda_available": False, "error": f"{type(exc).__name__}: {exc}"}


def monitor_online(
    online: Any,
    stop: threading.Event,
    samples: list[dict[str, Any]],
    interval_s: float,
    origin_s: float,
) -> None:
    while not stop.is_set():
        try:
            stats = online.stats()
            requests = stats.get("requests", {})
            online_requests = stats.get("online_requests", {})
            samples.append(
                {
                    "time_relative_s": time.perf_counter() - origin_s,
                    "waiting": requests.get("waiting"),
                    "running": requests.get("running"),
                    "finished": requests.get("finished"),
                    "active_channels": online_requests.get("active_channels"),
                    "decode_steps": stats.get("decode_steps"),
                    "engine_steps": stats.get("engine_steps"),
                }
            )
        except Exception as exc:
            samples.append(
                {
                    "time_relative_s": time.perf_counter() - origin_s,
                    "monitor_error": f"{type(exc).__name__}: {exc}",
                }
            )
        stop.wait(interval_s)


def consume_request(
    handle: Any, spec: WorkloadRequest, timeline: RequestTimeline, timeout: float
) -> RequestTimeline:
    cumulative = 0
    cancel_sent = False
    try:
        for output in handle.iter_outputs(timeout=timeout):
            now = time.perf_counter()
            chunk = [int(token_id) for token_id in output.token_ids]
            if chunk and timeline.first_token_time_s is None:
                timeline.first_token_time_s = now
            timeline.token_ids.extend(chunk)
            cumulative += len(chunk)
            timeline.token_events.append(
                {
                    "time_s": now,
                    "token_ids": chunk,
                    "cumulative_output_tokens": cumulative,
                    "finished": output.finished,
                    "finish_reason": output.finish_reason,
                }
            )
            if output.error:
                timeline.error = output.error
            if (
                spec.cancel_after_tokens is not None
                and cumulative >= spec.cancel_after_tokens
                and not cancel_sent
                and not output.finished
            ):
                timeline.cancelled = handle.cancel(timeout=timeout)
                cancel_sent = True
            if output.finished:
                timeline.finish_time_s = now
                timeline.finish_reason = output.finish_reason
                break
    except Exception as exc:
        timeline.finish_time_s = time.perf_counter()
        timeline.error = f"{type(exc).__name__}: {exc}"
    return timeline


def submit_specs(
    online: Any,
    specs: list[WorkloadRequest],
    timeout: float,
    ignore_eos: bool,
) -> list[tuple[Any, RequestTimeline]]:
    from myvllm.online import OnlineRequestSpec

    submit_time = time.perf_counter()
    requests = [
        OnlineRequestSpec(
            spec.prompt,
            SamplingParams(
                temperature=0.0,
                max_tokens=spec.max_new_tokens,
                ignore_eos=ignore_eos,
                max_model_length=len(spec.prompt_token_ids) + spec.max_new_tokens + 2,
            ),
            spec.request_id,
        )
        for spec in specs
    ]
    handles = online.submit_requests(requests, timeout=timeout)
    admitted = time.perf_counter()
    return [
        (
            handle,
            RequestTimeline(
                request_id=spec.request_id,
                bucket=spec.bucket,
                prompt_tokens=len(spec.prompt_token_ids),
                max_new_tokens=spec.max_new_tokens,
                submit_time_s=submit_time,
                admit_return_time_s=admitted,
            ),
        )
        for handle, spec in zip(handles, specs)
    ]


def run_closed_loop(
    online: Any,
    workload: list[WorkloadRequest],
    concurrency: int,
    timeout: float,
    sample_interval_s: float,
    raw_log_path: Path,
    ignore_eos: bool = True,
    max_duration_s: float | None = None,
) -> tuple[list[RequestTimeline], list[dict[str, Any]], dict[str, Any]]:
    pending = list(workload)
    by_id = {spec.request_id: spec for spec in workload}
    completed: list[RequestTimeline] = []
    monitor_samples: list[dict[str, Any]] = []
    stop_monitor = threading.Event()
    origin = time.perf_counter()
    deadline_s = (
        origin + max_duration_s
        if max_duration_s is not None and max_duration_s > 0
        else None
    )
    monitor = threading.Thread(
        target=monitor_online,
        args=(online, stop_monitor, monitor_samples, sample_interval_s, origin),
        daemon=True,
    )
    monitor.start()
    failures: list[dict[str, Any]] = []
    deadline_reached = False
    deadline_cancelled = 0
    deadline_cancel_failures: list[dict[str, Any]] = []
    abandoned_futures = False
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    futures: dict[Any, dict[str, Any]] = {}
    try:

        def collect_completed(done: set[Any]) -> None:
            for future in done:
                active = futures.pop(future)
                try:
                    timeline = future.result()
                except Exception as exc:
                    timeline = active["timeline"]
                    timeline.finish_time_s = time.perf_counter()
                    timeline.error = f"{type(exc).__name__}: {exc}"
                completed.append(timeline)
                append_jsonl(raw_log_path, [timeline.to_record(origin)])

        def admit_next_batch(limit: int) -> None:
            nonlocal deadline_reached, pending
            if not pending or limit <= 0:
                return
            if deadline_s is not None and time.perf_counter() >= deadline_s:
                deadline_reached = True
                return
            batch: list[WorkloadRequest] = []
            now = time.perf_counter()
            while pending and len(batch) < limit:
                if pending[0].arrival_offset_s > 0:
                    target = origin + pending[0].arrival_offset_s
                    if deadline_s is not None:
                        target = min(target, deadline_s)
                    if target > now:
                        time.sleep(target - now)
                        now = time.perf_counter()
                    if deadline_s is not None and now >= deadline_s:
                        deadline_reached = True
                        break
                batch.append(pending.pop(0))
            if not batch:
                return
            try:
                for handle, timeline in submit_specs(
                    online, batch, timeout, ignore_eos
                ):
                    future = pool.submit(
                        consume_request,
                        handle,
                        by_id[handle.request_id],
                        timeline,
                        timeout,
                    )
                    futures[future] = {
                        "handle": handle,
                        "request_id": handle.request_id,
                        "timeline": timeline,
                    }
            except Exception as exc:
                for spec in batch:
                    failures.append(
                        {
                            "request_id": spec.request_id,
                            "phase": "submit",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )

        admit_next_batch(concurrency)
        while futures:
            wait_timeout = timeout
            if deadline_s is not None:
                remaining_s = deadline_s - time.perf_counter()
                if remaining_s <= 0:
                    deadline_reached = True
                    break
                wait_timeout = min(wait_timeout, remaining_s)
            done, _ = wait(
                set(futures),
                return_when=FIRST_COMPLETED,
                timeout=wait_timeout,
            )
            if not done:
                if deadline_s is not None and time.perf_counter() >= deadline_s:
                    deadline_reached = True
                    break
                raise TimeoutError(f"no request completed within {timeout:.1f}s")
            collect_completed(done)
            admit_next_batch(concurrency - len(futures))

        if deadline_reached and futures:
            cancel_timeout_s = min(max(sample_interval_s * 10.0, 1.0), 5.0)
            for future, active in list(futures.items()):
                if future.done():
                    continue
                try:
                    if active["handle"].cancel(timeout=cancel_timeout_s):
                        active["timeline"].cancelled = True
                        deadline_cancelled += 1
                except Exception as exc:
                    deadline_cancel_failures.append(
                        {
                            "request_id": active["request_id"],
                            "phase": "deadline_cancel",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
            done_after_cancel, _ = wait(
                set(futures),
                return_when=FIRST_COMPLETED,
                timeout=cancel_timeout_s,
            )
            collect_completed(done_after_cancel)
            now = time.perf_counter()
            for future, active in list(futures.items()):
                if future.done():
                    collect_completed({future})
                    continue
                timeline = active["timeline"]
                timeline.finish_time_s = now
                timeline.cancelled = True
                timeline.error = (
                    "SoakDeadlineExceeded: request still active after "
                    f"{max_duration_s:.1f}s"
                )
                completed.append(timeline)
                append_jsonl(raw_log_path, [timeline.to_record(origin)])
                futures.pop(future)
                future.cancel()
                abandoned_futures = True
    finally:
        pool.shutdown(wait=not abandoned_futures and not futures, cancel_futures=True)
        stop_monitor.set()
        monitor.join(timeout=5.0)
    failure_record = {
        "admission_failures": failures,
        "soak_time_limit": {
            "enabled": deadline_s is not None,
            "limit_s": max_duration_s,
            "reached": deadline_reached,
            "pending_requests": len(pending),
            "deadline_cancelled_requests": deadline_cancelled,
            "deadline_cancel_failures": deadline_cancel_failures,
            "elapsed_s": time.perf_counter() - origin,
        },
    }
    return completed, monitor_samples, failure_record


def token_interval_latencies(timeline: RequestTimeline) -> list[float]:
    token_times: list[float] = []
    for event in timeline.token_events:
        token_times.extend([float(event["time_s"])] * len(event["token_ids"]))
    return [
        token_times[index] - token_times[index - 1]
        for index in range(1, len(token_times))
    ]


def summarize_timelines(
    timelines: list[RequestTimeline],
    slo_ttft_s: float,
    slo_tpot_s: float,
) -> dict[str, Any]:
    successful = [
        item
        for item in timelines
        if item.finish_time_s is not None and item.error is None
    ]
    submit_times = [item.submit_time_s for item in successful]
    finish_times = [
        item.finish_time_s for item in successful if item.finish_time_s is not None
    ]
    wall_s = (
        max(finish_times) - min(submit_times) if submit_times and finish_times else 0.0
    )
    ttft = [
        item.first_token_time_s - item.submit_time_s
        for item in successful
        if item.first_token_time_s is not None
    ]
    e2e = [
        item.finish_time_s - item.submit_time_s
        for item in successful
        if item.finish_time_s is not None
    ]
    request_tpot = []
    itl = []
    for item in successful:
        itl.extend(token_interval_latencies(item))
        if item.first_token_time_s is None or item.finish_time_s is None:
            continue
        if item.output_tokens <= 1:
            request_tpot.append(0.0)
        else:
            request_tpot.append(
                (item.finish_time_s - item.first_token_time_s)
                / (item.output_tokens - 1)
            )
    output_tokens = sum(item.output_tokens for item in successful)
    prompt_tokens = sum(item.prompt_tokens for item in successful)
    good = 0
    for item in successful:
        if item.first_token_time_s is None or item.finish_time_s is None:
            continue
        item_ttft = item.first_token_time_s - item.submit_time_s
        item_tpot = (
            0.0
            if item.output_tokens <= 1
            else (item.finish_time_s - item.first_token_time_s)
            / (item.output_tokens - 1)
        )
        good += int(item_ttft <= slo_ttft_s and item_tpot <= slo_tpot_s)
    return {
        "requests": len(timelines),
        "successful_requests": len(successful),
        "failed_requests": len(timelines) - len(successful),
        "wall_time_s": wall_s,
        "request_throughput_rps": len(successful) / wall_s if wall_s > 0 else 0.0,
        "output_token_throughput_tps": output_tokens / wall_s if wall_s > 0 else 0.0,
        "total_token_throughput_tps": (
            (prompt_tokens + output_tokens) / wall_s if wall_s > 0 else 0.0
        ),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_s": latency_summary(ttft),
        "tpot_s": latency_summary(request_tpot),
        "itl_s": latency_summary(itl),
        "e2e_latency_s": latency_summary(e2e),
        "goodput": {
            "slo": {"ttft_s": slo_ttft_s, "tpot_s": slo_tpot_s},
            "requests_meeting_slo": good,
            "request_goodput_rps": good / wall_s if wall_s > 0 else 0.0,
            "fraction": good / len(successful) if successful else 0.0,
        },
        "finish_reasons": count_by(
            item.finish_reason or "unknown" for item in timelines
        ),
        "cancelled_requests": sum(int(item.cancelled) for item in timelines),
    }


def count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_monitor(samples: list[dict[str, Any]]) -> dict[str, Any]:
    waiting = [
        int(item["waiting"]) for item in samples if item.get("waiting") is not None
    ]
    running = [
        int(item["running"]) for item in samples if item.get("running") is not None
    ]
    channels = [
        int(item["active_channels"])
        for item in samples
        if item.get("active_channels") is not None
    ]
    return {
        "samples": len(samples),
        "queue_depth": {
            "max": max(waiting, default=0),
            "mean": statistics.fmean(waiting) if waiting else 0.0,
        },
        "active_batch_proxy": {
            "max": max(running, default=0),
            "mean": statistics.fmean(running) if running else 0.0,
        },
        "active_channels": {
            "max": max(channels, default=0),
            "mean": statistics.fmean(channels) if channels else 0.0,
        },
    }


def summarize_acceptance(
    stats: dict[str, Any], request_ids: set[str]
) -> dict[str, Any]:
    steps = [
        step
        for step in stats.get("recent_decode_steps", [])
        if request_ids.intersection(step.get("per_request_accepted_length", {}).keys())
    ]
    proposed = sum(int(step.get("proposed_tokens", 0)) for step in steps)
    accepted = sum(int(step.get("accepted_tokens", 0)) for step in steps)
    per_position_proposed: list[int] = []
    per_position_accepted: list[int] = []
    accepted_lengths: list[int] = []
    batch_buckets: dict[str, dict[str, Any]] = {}
    time_windows: list[dict[str, Any]] = []
    for step_index, step in enumerate(steps):
        active = int(step.get("active_batch_size", 0))
        bucket = batch_buckets.setdefault(
            str(active), {"steps": 0, "proposed": 0, "accepted": 0}
        )
        bucket["steps"] += 1
        bucket["proposed"] += int(step.get("proposed_tokens", 0))
        bucket["accepted"] += int(step.get("accepted_tokens", 0))
        for length in step.get("per_request_accepted_length", {}).values():
            accepted_lengths.append(int(length))
        for position, count in enumerate(step.get("per_position_proposed", [])):
            while len(per_position_proposed) <= position:
                per_position_proposed.append(0)
                per_position_accepted.append(0)
            per_position_proposed[position] += int(count)
        for position, count in enumerate(step.get("per_position_accepted", [])):
            while len(per_position_accepted) <= position:
                per_position_proposed.append(0)
                per_position_accepted.append(0)
            per_position_accepted[position] += int(count)
        time_windows.append(
            {
                "decode_step_index": step_index,
                "engine_step": step.get("engine_step"),
                "active_batch_size": active,
                "speculative_batch_size": step.get("speculative_batch_size"),
                "proposed": step.get("proposed_tokens", 0),
                "accepted": step.get("accepted_tokens", 0),
                "acceptance_rate": (
                    step.get("accepted_tokens", 0) / step.get("proposed_tokens", 1)
                    if step.get("proposed_tokens", 0)
                    else 0.0
                ),
            }
        )
    for bucket in batch_buckets.values():
        bucket["acceptance_rate"] = (
            bucket["accepted"] / bucket["proposed"] if bucket["proposed"] else 0.0
        )
    return {
        "steps": len(steps),
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "per_position_proposed": per_position_proposed,
        "per_position_accepted": per_position_accepted,
        "per_position_acceptance_rate": [
            accepted_count / proposed_count if proposed_count else 0.0
            for accepted_count, proposed_count in zip(
                per_position_accepted, per_position_proposed
            )
        ],
        "accepted_length_histogram": count_by(accepted_lengths),
        "per_batch_size_bucket": batch_buckets,
        "per_decode_step": time_windows,
    }


def summarize_engine_stats(stats: dict[str, Any]) -> dict[str, Any]:
    speculative = stats.get("speculative", {})
    stage_seconds = speculative.get("stage_seconds", {})
    return {
        "engine_steps": stats.get("engine_steps"),
        "decode_steps": stats.get("decode_steps"),
        "estimated_prefill_steps": (
            stats.get("engine_steps", 0) - stats.get("decode_steps", 0)
            if stats.get("engine_steps") is not None
            and stats.get("decode_steps") is not None
            else None
        ),
        "target_forward_count_estimate": stats.get("engine_steps"),
        "speculative_stage_seconds": stage_seconds,
        "qwen35_verification": stats.get("qwen35_verification", {}),
        "target_decision_diagnostics": stats.get("target_decision_diagnostics", []),
        "memory_trace": stats.get("memory_trace", []),
        "dflash2_profile": stats.get("dflash2_profile", {}),
        "requests": stats.get("requests", {}),
        "online_requests": stats.get("online_requests", {}),
    }


def validate_run_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = [
        "run_id",
        "variant",
        "workload",
        "concurrency",
        "repeat",
        "timing",
        "acceptance",
        "engine",
        "memory",
        "fairness",
        "artifacts",
    ]
    for key in required:
        if key not in summary:
            errors.append(f"missing key: {key}")
    timing = summary.get("timing", {})
    for key in (
        "request_throughput_rps",
        "output_token_throughput_tps",
        "ttft_s",
        "tpot_s",
        "e2e_latency_s",
    ):
        if key not in timing:
            errors.append(f"missing timing key: {key}")
    acceptance = summary.get("acceptance", {})
    for key in (
        "proposed_tokens",
        "accepted_tokens",
        "per_decode_step",
        "per_batch_size_bucket",
    ):
        if key not in acceptance:
            errors.append(f"missing acceptance key: {key}")
    return errors


def compare_token_outputs(
    target: dict[str, list[int]],
    candidate: dict[str, list[int]],
    sample_limit: int,
    target_diagnostics: list[dict[str, Any]] | None = None,
    candidate_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def find_decision(
        diagnostics: list[dict[str, Any]] | None,
        request_id: str,
        output_index: int,
        *,
        verifier: bool,
    ) -> dict[str, Any] | None:
        expected_offset = 0 if verifier else None
        for record in diagnostics or ():
            if (
                record.get("request_id") == request_id
                and record.get("output_index") == output_index
                and record.get("query_offset") == expected_offset
            ):
                return dict(record)
        return None

    request_ids = list(dict.fromkeys((*target, *candidate)))
    mismatches = []
    mismatch_count = 0
    for request_id in request_ids:
        target_tokens = target.get(request_id, [])
        candidate_tokens = candidate.get(request_id, [])
        if target_tokens != candidate_tokens:
            mismatch_count += 1
            if len(mismatches) >= sample_limit:
                continue
            mismatch_at = None
            for index, (left, right) in enumerate(zip(target_tokens, candidate_tokens)):
                if left != right:
                    mismatch_at = index
                    break
            if mismatch_at is None and len(target_tokens) != len(candidate_tokens):
                mismatch_at = min(len(target_tokens), len(candidate_tokens))
            mismatch = {
                "request_id": request_id,
                "first_mismatch_output_index": mismatch_at,
                "target_token": (
                    target_tokens[mismatch_at]
                    if mismatch_at is not None and mismatch_at < len(target_tokens)
                    else None
                ),
                "candidate_token": (
                    candidate_tokens[mismatch_at]
                    if mismatch_at is not None and mismatch_at < len(candidate_tokens)
                    else None
                ),
                "top2_margin": None,
                "note": (
                    "Enable --debug-metadata to attach target and candidate "
                    "top-2 decision evidence."
                ),
            }
            if mismatch_at is not None:
                target_decision = find_decision(
                    target_diagnostics,
                    request_id,
                    mismatch_at,
                    verifier=False,
                )
                candidate_decision = find_decision(
                    candidate_diagnostics,
                    request_id,
                    mismatch_at,
                    verifier=True,
                )
                if target_decision is not None or candidate_decision is not None:
                    mismatch["target_decision"] = target_decision
                    mismatch["candidate_decision"] = candidate_decision
                    mismatch["top2_margin"] = (
                        None
                        if candidate_decision is None
                        else candidate_decision.get("top2_margin")
                    )
                    mismatch["note"] = (
                        "Exact token mismatch; attached decisions are diagnostic "
                        "evidence and do not relax the fairness failure."
                    )
            mismatches.append(mismatch)
    return {
        "sampled_requests": len(request_ids),
        "compared_requests": len(request_ids),
        "mismatch_count": mismatch_count,
        "exact_token_match": mismatch_count == 0,
        "mismatches": mismatches,
    }


def code_provenance(run_dir: Path) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "-c", "core.whitespace=cr-at-eol", "diff",
         "--ignore-space-at-eol", "--binary", "--no-ext-diff"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    diff_path = run_dir / "code_diff.patch"
    diff_path.write_bytes(diff)
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True
    ).splitlines()
    sources = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    files = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    source_tree = hashlib.sha256()
    for name, digest in sorted(files.items()):
        source_tree.update(f"{name}\0{digest}\n".encode())
    archive_path = run_dir / "source_files.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    return {
        "git_status": status,
        "code_diff_patch": os.fspath(diff_path),
        "code_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_tree_sha256": source_tree.hexdigest(),
        "source_file_sha256": files,
        "source_archive": os.fspath(archive_path),
    }


def runtime_manifest(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    manifest = {
        "git_commit": git_commit(),
        **code_provenance(run_dir),
        "python": sys.version,
        "platform": platform.platform(),
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "dtype": args.dtype,
        "seed": args.seed,
        "arguments": vars(args),
        "oracle_alignment_status": (
            "PENDING: official vLLM 0.28 V2 runner is blocked by non-lossless "
            "oracle runtime behavior; this benchmark is Mini-vLLM target-only "
            "vs Mini-vLLM DFlash2 only."
        ),
    }
    try:
        import torch

        manifest.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as exc:
        manifest["torch_error"] = f"{type(exc).__name__}: {exc}"
    return manifest


def run_variant(
    args: argparse.Namespace,
    variant: str,
    workload_name: str,
    workload: list[WorkloadRequest],
    concurrency: int,
    repeat: int,
    run_dir: Path,
    log_dir: Path,
    eos: int,
) -> dict[str, Any]:
    import torch
    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.online import OnlineEngine

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_default_dtype(
        {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
    )
    max_prompt = max(len(spec.prompt_token_ids) for spec in workload)
    max_new = max(spec.max_new_tokens for spec in workload)
    config = build_config(
        args,
        speculative=variant == "dflash2",
        eos=eos,
        max_prompt_tokens=max_prompt,
        max_new_tokens=max_new,
        concurrency=concurrency,
    )
    raw_requests_path = (
        log_dir / f"{variant}_{workload_name}_c{concurrency}_r{repeat}_requests.jsonl"
    )
    monitor_path = (
        log_dir / f"{variant}_{workload_name}_c{concurrency}_r{repeat}_monitor.jsonl"
    )
    summary_path = (
        run_dir / f"{variant}_{workload_name}_c{concurrency}_r{repeat}_summary.json"
    )
    engine = None
    online = None
    native_run = None
    load_started = time.perf_counter()
    memory_before = cuda_memory_snapshot(device)
    try:
        engine = LLMEngine(config)
        target_stage_wall_seconds = {"prefill": 0.0, "target_decode": 0.0}
        native_run = engine.model_runner.run

        def timed_target_run(seqs, is_prefill):
            started = time.perf_counter()
            try:
                return native_run(seqs, is_prefill)
            finally:
                stage = "prefill" if is_prefill else "target_decode"
                target_stage_wall_seconds[stage] += time.perf_counter() - started

        engine.model_runner.run = timed_target_run
        load_elapsed = time.perf_counter() - load_started
        online = OnlineEngine(
            engine, max_terminal_history=max(args.max_metrics_history, 256),
            lightweight_stats=getattr(args, "lightweight_live_stats", False),
        )
        online.start()
        if args.warmup_requests > 0:
            warmup = workload[: min(args.warmup_requests, len(workload))]
            if getattr(args, "warmup_output_tokens", 0):
                warmup = [
                    replace(spec, max_new_tokens=min(spec.max_new_tokens, args.warmup_output_tokens))
                    for spec in warmup
                ]
            run_closed_loop(
                online,
                warmup,
                min(concurrency, len(warmup)),
                args.timeout,
                args.sample_interval,
                log_dir
                / f"{variant}_{workload_name}_c{concurrency}_r{repeat}_warmup.jsonl",
                args.ignore_eos,
            )
            reset_runtime_stats(engine)
        target_stage_wall_seconds.update(prefill=0.0, target_decode=0.0)
        measured_started = time.perf_counter()
        soak_time_limit_s = (
            args.soak_minutes * 60.0 if args.soak and args.soak_minutes > 0 else None
        )
        timelines, monitor_samples, failures = run_closed_loop(
            online,
            workload,
            concurrency,
            args.timeout,
            args.sample_interval,
            raw_requests_path,
            args.ignore_eos,
            soak_time_limit_s,
        )
        measured_elapsed = time.perf_counter() - measured_started
        append_jsonl(monitor_path, monitor_samples)
        stats = online.stats(detailed=True, timeout=args.timeout) if getattr(
            args, "lightweight_live_stats", False
        ) else online.stats()
        request_ids = {item.request_id for item in workload}
        memory_after = cuda_memory_snapshot(device)
        summary = {
            "run_id": args.run_id,
            "variant": variant,
            "workload": workload_name,
            "concurrency": concurrency,
            "repeat": repeat,
            "load_time_s": load_elapsed,
            "measured_elapsed_s": measured_elapsed,
            "config": {
                "max_model_length": config["max_model_length"],
                "max_num_sequences": config["max_num_sequences"],
                "max_num_batched_tokens": config["max_num_batched_tokens"],
                "num_speculative_tokens": args.num_speculative_tokens,
                "temperature": 0.0,
                "ignore_eos": args.ignore_eos,
                "cached_batch_pad_to_size": config.get(
                    "qwen35_cached_batch_pad_to_size"
                ),
                "cached_attention_pad_to_length": config.get(
                    "qwen35_cached_attention_pad_to_length"
                ),
                "memory_trace": args.memory_trace,
                "replay_state_updates": getattr(args, "replay_state_updates", True),
                "stop_verification_at_mismatch": getattr(
                    args, "stop_verification_at_mismatch", False
                ),
                "parallel_target_verification": getattr(
                    args, "parallel_target_verification", False
                ),
                "soak": args.soak,
                "soak_requests": args.soak_requests if args.soak else None,
                "soak_time_limit_s": soak_time_limit_s,
            },
            "timing": summarize_timelines(
                timelines, args.goodput_ttft_slo, args.goodput_tpot_slo
            ),
            "acceptance": summarize_acceptance(stats, request_ids),
            "monitor": summarize_monitor(monitor_samples),
            "engine": summarize_engine_stats(stats),
            "memory": {"before": memory_before, "after": memory_after},
            "failures": failures,
            "fairness": {
                "prompt_token_ids_fixed": True,
                "temperature": 0.0,
                "seed": args.seed,
                "startup_excluded_from_latency": True,
                "warmup_requests": args.warmup_requests,
            },
            "artifacts": {
                "request_jsonl": os.fspath(raw_requests_path),
                "monitor_jsonl": os.fspath(monitor_path),
                "summary_json": os.fspath(summary_path),
            },
            "schema_errors": [],
        }
        summary["schema_errors"] = validate_run_summary(summary)
        summary["engine"]["target_stage_wall_seconds"] = target_stage_wall_seconds
        write_json(summary_path, summary)
        return summary
    except BaseException as exc:
        failure = {
            "run_id": args.run_id,
            "variant": variant,
            "workload": workload_name,
            "concurrency": concurrency,
            "repeat": repeat,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "load_time_s": time.perf_counter() - load_started,
            "memory": {"before": memory_before, "after": cuda_memory_snapshot(device)},
        }
        write_json(summary_path, failure)
        return failure
    finally:
        if online is not None and online.is_running:
            try:
                online.close(timeout=min(args.timeout, 60.0))
            except Exception:
                pass
        elif engine is not None:
            try:
                engine.exit()
            except Exception:
                pass
        # Release the closure's bound method before collection. engine.exit()
        # has already removed model_runner, so do not access that attribute here.
        native_run = None
        del online
        del engine
        gc.collect()
        if "torch" in sys.modules:
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except Exception:
                    pass
            torch.cuda.empty_cache()


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "variant",
        "workload",
        "concurrency",
        "repeat",
        "requests",
        "request_throughput_rps",
        "output_token_throughput_tps",
        "total_token_throughput_tps",
        "ttft_p50",
        "ttft_p99",
        "tpot_p50",
        "tpot_p99",
        "e2e_p50",
        "e2e_p99",
        "acceptance_rate",
        "proposed_tokens",
        "accepted_tokens",
        "gpu_peak_allocated_gib",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            timing = summary.get("timing", {})
            acceptance = summary.get("acceptance", {})
            memory = summary.get("memory", {}).get("after", {})
            writer.writerow(
                {
                    "variant": summary.get("variant"),
                    "workload": summary.get("workload"),
                    "concurrency": summary.get("concurrency"),
                    "repeat": summary.get("repeat"),
                    "requests": timing.get("requests"),
                    "request_throughput_rps": timing.get("request_throughput_rps"),
                    "output_token_throughput_tps": timing.get(
                        "output_token_throughput_tps"
                    ),
                    "total_token_throughput_tps": timing.get(
                        "total_token_throughput_tps"
                    ),
                    "ttft_p50": timing.get("ttft_s", {}).get("p50"),
                    "ttft_p99": timing.get("ttft_s", {}).get("p99"),
                    "tpot_p50": timing.get("tpot_s", {}).get("p50"),
                    "tpot_p99": timing.get("tpot_s", {}).get("p99"),
                    "e2e_p50": timing.get("e2e_latency_s", {}).get("p50"),
                    "e2e_p99": timing.get("e2e_latency_s", {}).get("p99"),
                    "acceptance_rate": acceptance.get("acceptance_rate"),
                    "proposed_tokens": acceptance.get("proposed_tokens"),
                    "accepted_tokens": acceptance.get("accepted_tokens"),
                    "gpu_peak_allocated_gib": memory.get("max_allocated_gib"),
                    "error": summary.get("error"),
                }
            )


def aggregate_matrix(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for summary in summaries:
        if "timing" not in summary:
            continue
        key = (
            str(summary.get("variant")),
            str(summary.get("workload")),
            int(summary.get("concurrency")),
        )
        groups.setdefault(key, []).append(summary)
    aggregated = []
    for (variant, workload, concurrency), rows in sorted(groups.items()):
        aggregated.append(
            {
                "variant": variant,
                "workload": workload,
                "concurrency": concurrency,
                "repeats": len(rows),
                "request_throughput_rps": summarize_repeats(
                    [row["timing"]["request_throughput_rps"] for row in rows]
                ),
                "output_token_throughput_tps": summarize_repeats(
                    [row["timing"]["output_token_throughput_tps"] for row in rows]
                ),
                "ttft_p50_s": summarize_repeats(
                    [
                        row["timing"]["ttft_s"]["p50"]
                        for row in rows
                        if row["timing"]["ttft_s"]["p50"] is not None
                    ]
                ),
                "ttft_p99_s": summarize_repeats(
                    [
                        row["timing"]["ttft_s"]["p99"]
                        for row in rows
                        if row["timing"]["ttft_s"]["p99"] is not None
                    ]
                ),
                "tpot_p50_s": summarize_repeats(
                    [
                        row["timing"]["tpot_s"]["p50"]
                        for row in rows
                        if row["timing"]["tpot_s"]["p50"] is not None
                    ]
                ),
                "tpot_p99_s": summarize_repeats(
                    [
                        row["timing"]["tpot_s"]["p99"]
                        for row in rows
                        if row["timing"]["tpot_s"]["p99"] is not None
                    ]
                ),
                "acceptance_rate": summarize_repeats(
                    [row["acceptance"]["acceptance_rate"] for row in rows]
                ),
                "max_active_batch": max(
                    row.get("monitor", {}).get("active_batch_proxy", {}).get("max", 0)
                    for row in rows
                ),
                "max_queue_depth": max(
                    row.get("monitor", {}).get("queue_depth", {}).get("max", 0)
                    for row in rows
                ),
            }
        )
    return {"groups": aggregated}


def normalize_eos(eos_token_id: Any) -> int:
    if isinstance(eos_token_id, int):
        return eos_token_id
    if isinstance(eos_token_id, list) and eos_token_id:
        return int(eos_token_id[0])
    return 248046


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    args.run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / args.run_id
    log_dir = Path(args.logs_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = runtime_manifest(args, run_dir)
    write_json(run_dir / "manifest.json", manifest)
    if args.dry_run:
        write_json(
            run_dir / "dry_run_plan.json",
            {
                "concurrency": args.concurrency,
                "workloads": args.workload,
                "variants": args.variants,
                "repeats": args.repeats,
                "requests_per_run": args.requests_per_run,
            },
        )
        print(f"dry_run_plan={run_dir / 'dry_run_plan.json'}", flush=True)
        return
    if not args.target_model:
        raise SystemExit("Pass --target-model or set MINIVLLM_QWEN38_PATH")
    if "dflash2" in args.variants and not args.draft_model:
        raise SystemExit("Pass --draft-model or set MINIVLLM_DFLASH2_PATH")

    import torch
    from myvllm.models.qwen3_8_config import (
        load_qwen35_config_dict,
        normalize_qwen35_text_config,
    )
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real Qwen3.8 concurrency benchmark")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    text_config = normalize_qwen35_text_config(
        load_qwen35_config_dict(args.target_model)
    )
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = normalize_eos(text_config.eos_token_id)
    all_summaries: list[dict[str, Any]] = []
    fairness_failures: list[str] = []
    token_samples: dict[tuple[str, str, int, int], dict[str, list[int]]] = {}
    for workload_name in args.workload:
        for concurrency in args.concurrency:
            total_requests = args.requests_per_run or max(concurrency * 4, concurrency)
            if args.soak:
                total_requests = max(total_requests, args.soak_requests)
            workload = build_workload(
                tokenizer,
                workload_name,
                total_requests,
                args.seed + concurrency,
                open_loop_rate=args.open_loop_rate,
                burst_interval=args.burst_interval,
                include_soak_edges=args.soak,
            )
            prompt_audit = {
                spec.request_id: {
                    "bucket": spec.bucket,
                    "prompt_tokens": len(spec.prompt_token_ids),
                    "max_new_tokens": spec.max_new_tokens,
                    "prompt_token_ids": list(spec.prompt_token_ids),
                }
                for spec in workload
            }
            write_json(
                run_dir / f"workload_{workload_name}_c{concurrency}.json",
                prompt_audit,
            )
            for repeat in range(args.repeats):
                repeat_summaries = {}
                for variant in args.variants:
                    print(
                        f"run variant={variant} workload={workload_name} "
                        f"concurrency={concurrency} repeat={repeat}",
                        flush=True,
                    )
                    summary = run_variant(
                        args,
                        variant,
                        workload_name,
                        workload,
                        concurrency,
                        repeat,
                        run_dir,
                        log_dir,
                        int(eos),
                    )
                    all_summaries.append(summary)
                    repeat_summaries[variant] = summary
                    raw_path = summary.get("artifacts", {}).get("request_jsonl")
                    if raw_path and Path(raw_path).exists():
                        outputs = {}
                        with Path(raw_path).open(encoding="utf-8") as handle:
                            for line in handle:
                                record = json.loads(line)
                                outputs[record["request_id"]] = record.get(
                                    "token_ids", []
                                )
                        token_samples[(variant, workload_name, concurrency, repeat)] = (
                            outputs
                        )
                target_key = ("target_only", workload_name, concurrency, repeat)
                dflash_key = ("dflash2", workload_name, concurrency, repeat)
                if target_key in token_samples and dflash_key in token_samples:
                    fairness = compare_token_outputs(
                        token_samples[target_key],
                        token_samples[dflash_key],
                        sample_limit=8,
                        target_diagnostics=(
                            repeat_summaries.get("target_only", {})
                            .get("engine", {})
                            .get("target_decision_diagnostics", [])
                        ),
                        candidate_diagnostics=(
                            repeat_summaries.get("dflash2", {})
                            .get("engine", {})
                            .get("target_decision_diagnostics", [])
                        ),
                    )
                    fairness_path = (
                        run_dir
                        / f"fairness_{workload_name}_c{concurrency}_r{repeat}.json"
                    )
                    write_json(fairness_path, fairness)
                    if not fairness["exact_token_match"]:
                        fairness_failures.append(os.fspath(fairness_path))
    write_json(run_dir / "all_summaries.json", all_summaries)
    aggregate = aggregate_matrix(all_summaries)
    write_json(run_dir / "aggregate.json", aggregate)
    write_summary_csv(run_dir / "summary.csv", all_summaries)
    print(f"results_dir={run_dir}", flush=True)
    print(f"logs_dir={log_dir}", flush=True)
    failures = [summary for summary in all_summaries if run_summary_failed(summary)]
    if failures or fairness_failures:
        raise SystemExit(
            f"{len(failures)} benchmark cell(s) failed and "
            f"{len(fairness_failures)} fairness check(s) failed; inspect result JSON"
        )


if __name__ == "__main__":
    main()
