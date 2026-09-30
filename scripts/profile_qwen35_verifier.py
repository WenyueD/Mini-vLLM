"""Diagnostic-only operator profile; forwards remaining CLI args to benchmark.

Run performance measurements separately without this wrapper. CPU self time and
CUDA kernel/copy durations are assigned to the innermost named model scope.
CUDA launch correlations include custom Triton kernels that PyTorch operator
tables omit. The Chrome trace retains the full event hierarchy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import wraps
import json
import heapq
from pathlib import Path
import sys
import time

import benchmark_qwen38_dflash2_concurrency as benchmark
import torch
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.hybrid_cache import HybridCacheManager, FullAttentionLayerState
from myvllm.models import qwen3_8 as qwen


def summarize_trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    host = [e for e in events if e.get("cat") in {"cuda_runtime", "cuda_driver"}
            or (e.get("cat") == "user_annotation" and e["name"].startswith("qwen/"))]
    stacks = defaultdict(list)
    owners = {}
    for index, event in enumerate(sorted(host, key=lambda e: e["ts"])):
        stack = stacks[event["tid"]]
        stamp = event["ts"]
        if event.get("cat") == "user_annotation":
            heapq.heappush(stack, (-stamp, index, stamp + event["dur"], event["name"]))
        else:
            while stack and stack[0][2] <= stamp:
                heapq.heappop(stack)
            owners[event["args"]["correlation"]] = stack[0][3] if stack else "outside"
    categories = defaultdict(lambda: {"cuda_ms": 0.0, "kernels_or_copies": 0})
    kernels = defaultdict(float)
    for event in events:
        if event.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        label = owners.get(event["args"].get("correlation"), "unattributed")
        categories[label]["cuda_ms"] += event["dur"] / 1000
        categories[label]["kernels_or_copies"] += 1
        kernels[event["name"]] += event["dur"] / 1000
    return {"categories": dict(categories), "top_kernels_ms": sorted(kernels.items(), key=lambda x: -x[1])[:20]}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-output-dir", type=Path, required=True)
    parser.add_argument("--profile-skip-calls", type=int, default=5)
    parser.add_argument("--profile-calls", type=int, default=2)
    args, remaining = parser.parse_known_args()
    args.profile_output_dir.mkdir(parents=True, exist_ok=True)
    active = [False]
    count = [0]

    def annotate(owner, method, label):
        original = getattr(owner, method)

        @wraps(original)
        def wrapped(*a, **kw):
            if not active[0]:
                return original(*a, **kw)
            with torch.profiler.record_function("qwen/" + label):
                return original(*a, **kw)
        setattr(owner, method, wrapped)

    for owner, method, label in [
        (qwen.Qwen35TokenwiseLinear, "forward", "projection"),
        (qwen.Qwen35RMSNorm, "forward", "rmsnorm"),
        (qwen.Qwen35RMSNormGated, "forward", "gdn_gated_norm"),
        (qwen.Qwen35GatedDeltaNet, "_causal_conv", "gdn_conv"),
        (qwen.Qwen35GatedDeltaNet, "forward", "gdn_recurrence_and_elementwise"),
        (qwen, "fused_cached_gdn", "gdn_fused_recurrence"),
        (qwen, "l2norm", "gdn_l2norm"),
        (qwen.Qwen35Attention, "forward_with_cache_batch", "attention_core_and_cache"),
        (qwen, "_qwen35_tokenwise_matmul", "attention_matmul"),
        (qwen.Qwen35ForCausalLM, "compute_logits", "lm_head"),
        (qwen.Qwen35MLP, "forward", "mlp_activation"),
        (HybridCacheManager, "begin_transaction", "cache_transaction"),
        (FullAttentionLayerState, "append", "cache_append"),
        (ModelRunner, "_sample_qwen35_logits", "sampling"),
    ]:
        annotate(owner, method, label)

    original_verify = ModelRunner._verify_qwen35_cached

    @wraps(original_verify)
    def profile_verify(self, batch):
        count[0] += 1
        call = count[0]
        if not (args.profile_skip_calls < call <= args.profile_skip_calls + args.profile_calls):
            return original_verify(self, batch)
        torch.cuda.synchronize()
        started = time.perf_counter()
        active[0] = True
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                with torch.profiler.record_function("qwen/verifier_other"):
                    result = original_verify(self, batch)
                torch.cuda.synchronize()
        finally:
            active[0] = False
        elapsed = time.perf_counter() - started
        categories = defaultdict(lambda: {"cpu_self_ms": 0.0, "events": 0})
        for event in prof.events():
            if event.device_type != torch.autograd.DeviceType.CPU:
                continue
            parent = event
            while parent is not None and not parent.name.startswith("qwen/"):
                parent = parent.cpu_parent
            label = parent.name if parent is not None else "outside"
            row = categories[label]
            row["cpu_self_ms"] += event.self_cpu_time_total / 1000
            row["events"] += 1
        prefix = args.profile_output_dir / f"verify_{call}"
        prof.export_chrome_trace(str(prefix.with_suffix(".trace.json")))
        prefix.with_suffix(".operators.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
        )
        payload = {
            "diagnostic_only": True, "call": call, "requests": len(batch.sequences),
            "query_lengths": [len(p.token_ids) + 1 for p in batch.proposals],
            "instrumented_wall_s": elapsed,
            "categories": dict(categories),
            "cuda_launch_attribution": summarize_trace(prefix.with_suffix(".trace.json")),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))
        print("verifier_profile=" + json.dumps(payload), flush=True)
        return result

    ModelRunner._verify_qwen35_cached = profile_verify
    sys.argv = [sys.argv[0], *remaining]
    benchmark.main()


if __name__ == "__main__":
    main()
