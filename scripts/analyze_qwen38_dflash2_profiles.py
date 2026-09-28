from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Qwen3.8 DFlash2 benchmark profile JSON files."
    )
    parser.add_argument("summaries", nargs="+", type=Path)
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    timing = payload.get("timing", {})
    acceptance = payload.get("acceptance", {})
    engine = payload.get("engine", {})
    profile = engine.get("dflash2_profile", {})
    speculative_stages = engine.get("speculative_stage_seconds", {})
    proposed = int(acceptance.get("proposed_tokens", 0))
    accepted = int(acceptance.get("accepted_tokens", 0))
    proposal_calls = int(profile.get("proposal_calls", 0))
    proposal_requests = int(profile.get("proposal_requests", 0))
    emitted = proposal_requests + accepted
    stage_total = sum(float(value) for value in speculative_stages.values())
    return {
        "path": str(path),
        "variant": payload.get("variant"),
        "concurrency": payload.get("concurrency"),
        "requests": timing.get("requests"),
        "output_tps": timing.get("output_token_throughput_tps"),
        "ttft_p50_s": timing.get("ttft_s", {}).get("p50"),
        "tpot_p50_s": timing.get("tpot_s", {}).get("p50"),
        "peak_allocated_gib": payload.get("memory", {})
        .get("after", {})
        .get("max_allocated_gib"),
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": safe_ratio(accepted, proposed),
        "proposal_calls": proposal_calls,
        "proposal_requests": proposal_requests,
        "mean_proposed_per_spec_request": safe_ratio(proposed, proposal_requests),
        "mean_accepted_per_spec_request": safe_ratio(accepted, proposal_requests),
        "mean_emitted_per_spec_request": safe_ratio(emitted, proposal_requests),
        "accepted_length_histogram": acceptance.get(
            "accepted_length_histogram", {}
        ),
        "per_position_acceptance_rate": acceptance.get(
            "per_position_acceptance_rate", []
        ),
        "per_batch_size_bucket": acceptance.get("per_batch_size_bucket", {}),
        "speculative_stage_seconds": speculative_stages,
        "speculative_stage_fraction": {
            name: safe_ratio(float(seconds), stage_total)
            for name, seconds in speculative_stages.items()
        },
        "proposal_stage_seconds": profile.get("proposal_stage_seconds", {}),
        "proposal_stage_seconds_per_call": profile.get(
            "proposal_stage_seconds_per_call", {}
        ),
        "context_combine_seconds": profile.get("context_combine_seconds", 0.0),
        "context_projection_seconds": profile.get(
            "context_projection_seconds", 0.0
        ),
        "context_cache_install_seconds": profile.get(
            "context_cache_install_seconds", 0.0
        ),
        "verification": engine.get("qwen35_verification", {}),
    }


def main() -> None:
    rows = [summarize(path) for path in parse_args().summaries]
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
