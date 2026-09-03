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


def normalize_fixture_schema(
    fixture: dict[str, Any],
    *,
    label: str,
    path: str,
) -> dict[str, Any]:
    normalized = dict(fixture)
    if "fixture_schema_version" in normalized and "anchor_semantics" in normalized:
        return normalized

    has_v2_shape = all(
        key in normalized
        for key in (
            "prompt",
            "prompt_token_ids",
            "anchor_token_id",
            "anchor_position",
            "target_layer_ids",
            "proposal_token_ids",
        )
    )
    uses_minivllm_anchor = (
        normalized.get("anchor_token_mode") == "minivllm"
        and normalized.get("anchor_position_mode") == "minivllm"
    )
    if has_v2_shape and uses_minivllm_anchor:
        normalized.setdefault("fixture_schema_version", 2)
        normalized.setdefault("anchor_semantics", "target_bonus_token")
        print(
            f"legacy_fixture_schema_inferred={label} path={path} "
            "fixture_schema_version=2 anchor_semantics=target_bonus_token",
            flush=True,
        )
    return normalized


def get_required(fixture: dict[str, Any], key: str, *, label: str, path: str) -> Any:
    if key not in fixture:
        keys = ", ".join(sorted(fixture))
        raise SystemExit(
            f"dflash2_oracle_alignment=FAIL missing_key={key} fixture={label} "
            f"path={path}\n"
            f"{label}_fixture_keys=[{keys}]\n"
            "Regenerate both fixtures with the latest scripts so they contain "
            "fixture_schema_version=2 and anchor_semantics=target_bonus_token."
        )
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
    minivllm = normalize_fixture_schema(
        load_json(args.minivllm),
        label="minivllm",
        path=args.minivllm,
    )
    oracle = normalize_fixture_schema(
        load_json(args.oracle),
        label="oracle",
        path=args.oracle,
    )

    required_equal_keys = [
        "fixture_schema_version",
        "anchor_semantics",
        "prompt",
        "prompt_token_ids",
        "anchor_token_id",
        "anchor_position",
        "target_layer_ids",
        "proposal_token_ids",
    ]
    for key in required_equal_keys:
        left = get_required(minivllm, key, label="minivllm", path=args.minivllm)
        right = get_required(oracle, key, label="oracle", path=args.oracle)
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
