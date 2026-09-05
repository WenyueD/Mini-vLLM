from scripts.smoke_qwen38_dflash2_online_b2 import batch_evidence


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
