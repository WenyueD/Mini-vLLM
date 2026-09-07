import pytest

from scripts.benchmark_qwen38_dflash2_concurrency import (
    RequestTimeline,
    compare_token_outputs,
    latency_summary,
    percentile,
    summarize_acceptance,
    summarize_monitor,
    summarize_repeats,
    summarize_timelines,
    validate_run_summary,
)


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
