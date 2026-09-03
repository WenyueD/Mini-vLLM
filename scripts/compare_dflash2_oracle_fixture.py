from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Mini-vLLM and Oracle DFlash2 proposal fixtures."
    )
    parser.add_argument("--minivllm", required=True, help="Mini-vLLM fixture JSON path.")
    parser.add_argument("--oracle", required=True, help="Oracle fixture JSON path.")
    parser.add_argument(
        "--checksum-atol",
        type=float,
        default=0.5,
        help="Absolute tolerance for optional debug checksum comparisons.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"fixture must be a JSON object: {path}")
    return value


def get_required(fixture: dict[str, Any], key: str) -> Any:
    if key not in fixture:
        raise KeyError(f"missing required fixture key: {key}")
    return fixture[key]


def optional_checksum_delta(
    left: dict[str, Any],
    right: dict[str, Any],
    key: str,
) -> float | None:
    if key not in left or key not in right:
        return None
    return abs(float(left[key]) - float(right[key]))


def main() -> None:
    args = parse_args()
    minivllm = load_json(args.minivllm)
    oracle = load_json(args.oracle)

    required_equal_keys = [
        "prompt",
        "prompt_token_ids",
        "anchor_token_id",
        "anchor_position",
        "target_layer_ids",
        "proposal_token_ids",
    ]
    for key in required_equal_keys:
        left = get_required(minivllm, key)
        right = get_required(oracle, key)
        print(f"{key}: minivllm={left} oracle={right}", flush=True)
        if left != right:
            raise SystemExit(f"dflash2_oracle_alignment=FAIL key={key}")

    for key in ("context_checksum", "proposal_hidden_checksum"):
        delta = optional_checksum_delta(minivllm, oracle, key)
        if delta is None:
            print(f"{key}: skipped", flush=True)
            continue
        print(f"{key}_delta={delta}", flush=True)
        if delta > args.checksum_atol:
            raise SystemExit(
                f"dflash2_oracle_alignment=FAIL key={key} delta={delta}"
            )

    print("dflash2_oracle_alignment=PASS", flush=True)


if __name__ == "__main__":
    main()
