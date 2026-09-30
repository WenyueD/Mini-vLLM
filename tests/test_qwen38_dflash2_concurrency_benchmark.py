import threading
import weakref
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("output_length", [512, 1024, 4096])
def test_long_decode_workload_keeps_short_prompts_and_extends_output(output_length):
    from scripts.benchmark_qwen38_dflash2_concurrency import build_workload

    class Tokenizer:
        def encode(self, text):
            return list(text.encode("utf-8"))

        def decode(self, tokens):
            return bytes(tokens).decode("utf-8")

    baseline = build_workload(Tokenizer(), "short", 16, 20260907)
    extended = build_workload(Tokenizer(), f"decode{output_length}", 16, 20260907)
    assert [r.prompt_token_ids for r in extended] == [r.prompt_token_ids for r in baseline]
    assert all(r.max_new_tokens == output_length for r in extended)
    assert all(120 <= len(r.prompt_token_ids) <= 136 for r in extended)


def test_run_variant_releases_timing_wrapper_before_empty_cache(monkeypatch, tmp_path):
    import torch
    from scripts import benchmark_qwen38_dflash2_concurrency as benchmark
    refs = []

    class Runner:
        def run(self, seqs, is_prefill):
            return []

    class Engine:
        def __init__(self, config):
            self.model_runner = Runner()
            refs.append(weakref.ref(self.model_runner))

        def exit(self):
            del self.model_runner

    class Online:
        is_running = False

        def __init__(self, engine, **kwargs):
            self.engine = engine

        def start(self):
            raise RuntimeError("injected after timing wrapper installation")

    def check_released():
        assert all(ref() is None for ref in refs)

    monkeypatch.setattr("myvllm.engine.llm_engine.LLMEngine", Engine)
    monkeypatch.setattr("myvllm.online.OnlineEngine", Online)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", check_released)
    monkeypatch.setattr(benchmark, "cuda_memory_snapshot", lambda _: {})
    monkeypatch.setattr(benchmark, "build_config", lambda *a, **kw: {})
    args = SimpleNamespace(device="cpu", dtype="float32", max_metrics_history=100,
                           run_id="lifecycle", timeout=1)
    request = benchmark.WorkloadRequest("a", "a", (1,), 1, "short")
    result = benchmark.run_variant(args, "target_only", "short", [request], 1, 0,
                                   tmp_path, tmp_path, 2)
    assert "injected" in result["error"]
    assert refs[0]() is None

from scripts.benchmark_qwen38_dflash2_concurrency import (
    RequestTimeline,
    WorkloadRequest,
    build_config,
    compare_token_outputs,
    latency_summary,
    percentile,
    reset_runtime_stats,
    run_closed_loop,
    run_summary_failed,
    summarize_acceptance,
    summarize_monitor,
    summarize_repeats,
    summarize_timelines,
    validate_run_summary,
)


@pytest.mark.parametrize("replay_state_updates", [False, True])
def test_build_config_pins_cached_target_shapes_for_exact_bf16_comparison(
    monkeypatch, replay_state_updates,
):
    monkeypatch.setattr(
        "myvllm.models.qwen3_8_config.load_qwen35_config_dict",
        lambda _path: {"text_config": {}},
    )
    args = SimpleNamespace(
        target_model="target",
        draft_model="draft",
        block_size=256,
        max_metrics_history=100,
        gpu_memory_utilization=0.9,
        target_cache_memory_fraction=0.05,
        draft_memory_reservation_fraction=0.0,
        cached_target_micro_batch_size=4,
        replay_state_updates=replay_state_updates,
        dtype="bfloat16",
        safe_open_device="cuda",
        dflash2_safe_open_device=None,
        debug_metadata=False,
        profile_dflash2_stages=False,
        memory_trace=True,
        num_speculative_tokens=7,
    )

    config = build_config(
        args,
        speculative=True,
        eos=0,
        max_prompt_tokens=128,
        max_new_tokens=64,
        concurrency=2,
    )

    assert config["qwen35_cached_batch_pad_to_size"] == 2
    assert config["qwen35_cached_attention_pad_to_length"] == 194
    assert config["qwen35_replay_state_updates"] is replay_state_updates

    c8_config = build_config(
        args,
        speculative=True,
        eos=0,
        max_prompt_tokens=128,
        max_new_tokens=64,
        concurrency=8,
    )
    assert c8_config["qwen35_cached_batch_pad_to_size"] == 4


def test_percentile_uses_linear_interpolation():
    assert percentile([10.0, 20.0, 30.0], 50) == 20.0
    assert percentile([0.0, 10.0], 90) == 9.0
    assert percentile([], 50) is None
    assert latency_summary([1.0, 3.0])["p95"] == pytest.approx(2.9)


def test_summarize_timelines_reports_throughput_latency_and_goodput():
    first = RequestTimeline(
        request_id="a",
        bucket="short",
        prompt_tokens=4,
        max_new_tokens=3,
        submit_time_s=100.0,
        admit_return_time_s=100.1,
        first_token_time_s=101.0,
        finish_time_s=103.0,
        token_ids=[7, 8, 9],
        token_events=[
            {"time_s": 101.0, "token_ids": [7]},
            {"time_s": 102.0, "token_ids": [8]},
            {"time_s": 103.0, "token_ids": [9]},
        ],
        finish_reason="length",
    )
    second = RequestTimeline(
        request_id="b",
        bucket="short",
        prompt_tokens=5,
        max_new_tokens=2,
        submit_time_s=100.0,
        admit_return_time_s=100.1,
        first_token_time_s=100.5,
        finish_time_s=102.0,
        token_ids=[1, 2],
        token_events=[
            {"time_s": 100.5, "token_ids": [1]},
            {"time_s": 102.0, "token_ids": [2]},
        ],
        finish_reason="length",
    )

    summary = summarize_timelines([first, second], slo_ttft_s=1.0, slo_tpot_s=1.1)

    assert summary["requests"] == 2
    assert summary["output_tokens"] == 5
    assert summary["total_token_throughput_tps"] == (9 + 5) / 3.0
    assert summary["ttft_s"]["p50"] == 0.75
    assert summary["tpot_s"]["p50"] == 1.25
    assert summary["goodput"]["requests_meeting_slo"] == 1


def test_summarize_acceptance_filters_steps_by_request_ids():
    stats = {
        "recent_decode_steps": [
            {
                "engine_step": 1,
                "active_batch_size": 2,
                "speculative_batch_size": 2,
                "proposed_tokens": 4,
                "accepted_tokens": 3,
                "per_request_accepted_length": {"a": 2, "b": 1},
                "per_position_proposed": [2, 2],
                "per_position_accepted": [2, 1],
            },
            {
                "engine_step": 2,
                "active_batch_size": 1,
                "speculative_batch_size": 1,
                "proposed_tokens": 2,
                "accepted_tokens": 0,
                "per_request_accepted_length": {"warmup": 0},
                "per_position_proposed": [1, 1],
                "per_position_accepted": [0, 0],
            },
        ]
    }

    summary = summarize_acceptance(stats, {"a", "b"})

    assert summary["steps"] == 1
    assert summary["acceptance_rate"] == 0.75
    assert summary["per_position_acceptance_rate"] == [1.0, 0.5]
    assert summary["accepted_length_histogram"] == {"1": 1, "2": 1}
    assert summary["per_batch_size_bucket"]["2"]["acceptance_rate"] == 0.75


def test_monitor_and_repeat_summaries_are_stable_for_empty_inputs():
    assert summarize_monitor([])["queue_depth"] == {"max": 0, "mean": 0.0}
    assert summarize_repeats([]) == {
        "count": 0,
        "median": None,
        "min": None,
        "max": None,
        "stdev": 0.0,
    }


def test_schema_validation_and_token_comparison_diagnostics():
    incomplete = {"timing": {}, "acceptance": {}}
    errors = validate_run_summary(incomplete)
    assert "missing key: variant" in errors
    assert "missing timing key: request_throughput_rps" in errors

    comparison = compare_token_outputs(
        {"a": [1, 2, 3], "b": [4]},
        {"a": [1, 9, 3], "b": [4]},
        sample_limit=8,
    )
    assert not comparison["exact_token_match"]
    assert comparison["mismatches"][0]["first_mismatch_output_index"] == 1
    assert comparison["mismatches"][0]["top2_margin"] is None

    diagnostic_comparison = compare_token_outputs(
        {"a": [1, 2]},
        {"a": [1, 9]},
        sample_limit=8,
        target_diagnostics=[
            {
                "request_id": "a",
                "output_index": 1,
                "query_offset": None,
                "argmax_token_id": 2,
                "top2_margin": 0.125,
            }
        ],
        candidate_diagnostics=[
            {
                "request_id": "a",
                "output_index": 1,
                "query_offset": 0,
                "argmax_token_id": 9,
                "top2_margin": 0.0,
            }
        ],
    )
    diagnostic = diagnostic_comparison["mismatches"][0]
    assert diagnostic["target_decision"]["argmax_token_id"] == 2
    assert diagnostic["candidate_decision"]["argmax_token_id"] == 9
    assert diagnostic["top2_margin"] == 0.0
    assert not diagnostic_comparison["exact_token_match"]


def test_token_comparison_checks_all_requests_and_only_caps_mismatch_details():
    target = {f"request-{index}": [index] for index in range(10)}
    candidate = dict(target)
    candidate["request-9"] = [999]

    comparison = compare_token_outputs(target, candidate, sample_limit=1)

    assert comparison["compared_requests"] == 10
    assert comparison["sampled_requests"] == 10
    assert comparison["mismatch_count"] == 1
    assert not comparison["exact_token_match"]
    assert [item["request_id"] for item in comparison["mismatches"]] == [
        "request-9"
    ]


def test_reset_runtime_stats_preserves_and_zeros_all_counter_keys():
    runner = SimpleNamespace(
        _qwen35_verification_stats={
            "calls": 7,
            "max_batch_size": 4,
            "last_active_batch_sizes": (2, 4),
        },
        _dflash2_profile_stats={
            "context_cache_update_calls": 9,
            "proposal_seconds": 1.25,
            "proposal_stage_seconds": {"forward": 0.5},
        },
    )
    engine = SimpleNamespace(
        config={"max_metrics_history": 32},
        metrics=None,
        spec_decode_coordinator=SimpleNamespace(metrics=None),
        model_runner=runner,
    )

    reset_runtime_stats(engine)

    assert runner._qwen35_verification_stats == {
        "calls": 0,
        "max_batch_size": 0,
        "last_active_batch_sizes": (),
    }
    assert runner._dflash2_profile_stats == {
        "context_cache_update_calls": 0,
        "proposal_seconds": 0.0,
        "proposal_stage_seconds": {},
    }


@pytest.mark.parametrize(
    "summary",
    [
        {"error": "boom"},
        {"schema_errors": ["missing timing"]},
        {"timing": {"failed_requests": 1}},
        {"failures": {"admission_failures": [{"request_id": "a"}]}},
    ],
)
def test_run_summary_failed_catches_cell_and_request_failures(summary):
    assert run_summary_failed(summary)


def test_run_summary_failed_accepts_clean_summary():
    assert not run_summary_failed(
        {
            "schema_errors": [],
            "timing": {"failed_requests": 0},
            "failures": {"admission_failures": []},
        }
    )


def test_run_closed_loop_honors_soak_time_limit(tmp_path):
    class BlockingHandle:
        def __init__(self, request_id):
            self.request_id = request_id
            self.cancelled = threading.Event()

        def iter_outputs(self, timeout=None):
            self.cancelled.wait(timeout=timeout)
            yield SimpleNamespace(
                token_ids=(),
                finished=True,
                finish_reason="cancelled",
                error=None,
            )

        def cancel(self, timeout=None):
            self.cancelled.set()
            return True

    class FakeOnline:
        def submit_requests(self, requests, timeout=None):
            return [BlockingHandle(request.request_id) for request in requests]

        def stats(self):
            return {
                "scheduler": {"waiting": 0, "running": 0},
                "online_requests": {"active_channels": 0},
            }

    workload = [
        WorkloadRequest(
            request_id=f"req-{index}",
            prompt="prompt",
            prompt_token_ids=(1, 2),
            max_new_tokens=8,
            bucket="short",
        )
        for index in range(3)
    ]

    timelines, _, failures = run_closed_loop(
        FakeOnline(),
        workload,
        concurrency=2,
        timeout=1.0,
        sample_interval_s=0.01,
        raw_log_path=tmp_path / "requests.jsonl",
        max_duration_s=0.02,
    )

    assert failures["soak_time_limit"]["enabled"]
    assert failures["soak_time_limit"]["reached"]
    assert failures["soak_time_limit"]["pending_requests"] == 1
    assert failures["soak_time_limit"]["deadline_cancelled_requests"] == 2
    assert len(timelines) == 2
    assert {timeline.finish_reason for timeline in timelines} == {"cancelled"}
