import sys

from scripts.smoke_qwen38_dflash2_online_b2 import batch_evidence, parse_args


def test_smoke_accepts_configurable_speculative_length(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke_qwen38_dflash2_online_b2.py", "--num-speculative-tokens", "3"],
    )

    assert parse_args().num_speculative_tokens == 3


def test_smoke_defaults_to_measured_best_speculative_length(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["smoke_qwen38_dflash2_online_b2.py"])

    assert parse_args().num_speculative_tokens == 1


def test_batch_evidence_aggregates_shared_batch_and_per_request_acceptance():
    stats = {
        "recent_decode_steps": [
            {
                "active_batch_size": 2,
                "speculative_batch_size": 2,
                "per_position_proposed": (2, 2, 1),
                "per_request_accepted_length": {"a": 1, "b": 0},
            },
            {
                "active_batch_size": 2,
                "speculative_batch_size": 2,
                "per_position_proposed": (2, 1),
                "per_request_accepted_length": {"a": 0, "b": 2},
            },
        ]
    }

    assert batch_evidence(stats, ["a", "b"]) == {
        "max_active_batch_size": 2,
        "max_speculative_batch_size": 2,
        "max_position_zero_proposed": 2,
        "accepted_by_request": {"a": 1, "b": 2},
    }
