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
    parser.add_argument(
        "--allow-oracle-non-lossless",
        action="store_true",
        help=(
            "Compare proposal fields when a non-lossless Oracle fixture still "
            "used the same effective anchor. This is diagnostic-only and will "
            "not print the formal dflash2_oracle_alignment=PASS marker."
        ),
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
        prompt_token_ids = normalized.get("prompt_token_ids") or []
        prompt_final_anchor = prompt_token_ids[-1] if prompt_token_ids else None
        legacy_minivllm_anchor = normalized.get("minivllm_anchor_token_id")
        if (
            label == "oracle"
            and legacy_minivllm_anchor is not None
            and prompt_final_anchor is not None
            and legacy_minivllm_anchor == prompt_final_anchor
        ):
            output_token_ids = normalized.get("output_token_ids")
            raise SystemExit(
                "dflash2_oracle_alignment=FAIL "
                "legacy_oracle_fixture_not_target_bonus\n"
                f"oracle_path={path}\n"
                f"prompt_final_token_id={prompt_final_anchor}\n"
                f"oracle_minivllm_anchor_token_id={legacy_minivllm_anchor}\n"
                f"oracle_output_token_ids={output_token_ids}\n"
                "Delete this Oracle fixture and regenerate it with the latest "
                "export_oracle_dflash2_fixture.py. The latest exporter should "
                "write fixture_schema_version=2 and should refuse to write a "
                "fixture if Oracle DFlash is not target-lossless."
            )
        if "minivllm_anchor_token_id" in normalized:
            previous = normalized.get("anchor_token_id")
            normalized["anchor_token_id"] = normalized["minivllm_anchor_token_id"]
            if previous != normalized["anchor_token_id"]:
                print(
                    f"legacy_anchor_token_rewritten={label} path={path} "
                    f"old={previous} new={normalized['anchor_token_id']}",
                    flush=True,
                )
        if "minivllm_anchor_position" in normalized:
            previous = normalized.get("anchor_position")
            normalized["anchor_position"] = normalized["minivllm_anchor_position"]
            if previous != normalized["anchor_position"]:
                print(
                    f"legacy_anchor_position_rewritten={label} path={path} "
                    f"old={previous} new={normalized['anchor_position']}",
                    flush=True,
                )
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


def print_mismatch_context(fixture: dict[str, Any], *, label: str) -> None:
    debug_keys = (
        "anchor_token_id",
        "anchor_token_mode",
        "minivllm_anchor_token_id",
        "minivllm_anchor_token_source",
        "oracle_capture_anchor_token_id",
        "oracle_effective_anchor_matches_fixture",
        "prompt_final_anchor_token_id",
        "anchor_position",
        "anchor_position_mode",
        "minivllm_anchor_position",
        "oracle_anchor_position",
        "prompt_final_anchor_position",
        "oracle_target_only_token_ids",
        "raw_output_token_ids",
        "output_token_ids",
        "proposal_token_ids",
    )
    for debug_key in debug_keys:
        if debug_key in fixture:
            print(f"{label}_{debug_key}={fixture[debug_key]}", flush=True)


def print_candidate_diagnostics(fixture: dict[str, Any], *, label: str) -> None:
    for key in (
        "candidate_token_ids",
        "candidate_unary_scores",
        "selector_selected_indices",
        "selector_realized_scores",
        "draft_token_ids",
    ):
        if key not in fixture:
            continue
        value = fixture[key]
        if isinstance(value, list) and value and isinstance(value[0], list):
            first = value[0]
        else:
            first = value
        print(f"{label}_{key}={first}", flush=True)


def oracle_effective_anchor_mismatch(fixture: dict[str, Any]) -> bool:
    fixture_anchor = fixture.get("anchor_token_id")
    effective_anchor = fixture.get("oracle_capture_anchor_token_id")
    return (
        fixture_anchor is not None
        and effective_anchor is not None
        and int(fixture_anchor) != int(effective_anchor)
    )


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
    if oracle.get("oracle_target_lossless") is False:
        print_mismatch_context(oracle, label="oracle")
        print_candidate_diagnostics(oracle, label="oracle")
        if not args.allow_oracle_non_lossless:
            raise SystemExit(
                "dflash2_oracle_alignment=FAIL oracle_target_lossless=false "
                "fixture=oracle "
                f"path={args.oracle}"
            )
        print(
            "dflash2_oracle_alignment=SKIP oracle_target_lossless=false "
            "proposal_diagnostic_only=true",
            flush=True,
        )
        if oracle_effective_anchor_mismatch(oracle):
            print(
                "dflash2_oracle_proposal_diagnostic=SKIP "
                "reason=oracle_effective_anchor_mismatch "
                f"fixture_anchor_token_id={oracle['anchor_token_id']} "
                "oracle_capture_anchor_token_id="
                f"{oracle['oracle_capture_anchor_token_id']}",
                flush=True,
            )
            return

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
            print_mismatch_context(minivllm, label="minivllm")
            print_mismatch_context(oracle, label="oracle")
            if key == "proposal_token_ids":
                print_candidate_diagnostics(minivllm, label="minivllm")
                print_candidate_diagnostics(oracle, label="oracle")
            prefix = (
                "dflash2_oracle_proposal_diagnostic"
                if args.allow_oracle_non_lossless
                else "dflash2_oracle_alignment"
            )
            raise SystemExit(f"{prefix}=FAIL key={key}")

    for key in ("context_checksum", "proposal_hidden_checksum"):
        delta = optional_checksum_delta(minivllm, oracle, key)
        if delta is None:
            print(f"{key}: skipped", flush=True)
            continue
        print(f"{key}_delta={delta}", flush=True)
        if delta > args.checksum_atol:
            prefix = (
                "dflash2_oracle_proposal_diagnostic"
                if args.allow_oracle_non_lossless
                else "dflash2_oracle_alignment"
            )
            raise SystemExit(
                f"{prefix}=FAIL key={key} delta={delta}"
            )

    if args.allow_oracle_non_lossless:
        print("dflash2_oracle_proposal_diagnostic=PASS", flush=True)
    else:
        print("dflash2_oracle_alignment=PASS", flush=True)


if __name__ == "__main__":
    main()
