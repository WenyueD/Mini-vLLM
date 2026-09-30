"""Independently audit raw outputs and decompose paired benchmark timings."""
import argparse
import json
from pathlib import Path


def records(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        rid = row["request_id"]
        if rid in rows:
            raise ValueError(f"duplicate request {rid} in {path}")
        if row.get("error") or row.get("cancelled"):
            raise ValueError(f"failed or cancelled request {rid} in {path}")
        rows[rid] = row
    return rows


def audit(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    output = {"run": run_dir.name, "source_tree_sha256": manifest["source_tree_sha256"], "pairs": []}
    for path in sorted(run_dir.glob("target_only_*_summary.json")):
        target = json.loads(path.read_text())
        candidate = json.loads(path.with_name(path.name.replace("target_only_", "dflash2_", 1)).read_text())
        workload = json.loads((run_dir / f"workload_{target['workload']}_c{target['concurrency']}.json").read_text())
        left = records(target["artifacts"]["request_jsonl"])
        right = records(candidate["artifacts"]["request_jsonl"])
        if set(left) != set(workload) or set(right) != set(workload):
            raise ValueError(f"missing or extra requests in {path}")
        mismatches = []
        for rid in workload:
            if left[rid]["prompt_tokens"] != len(workload[rid]["prompt_token_ids"]) or right[rid]["prompt_tokens"] != left[rid]["prompt_tokens"]:
                raise ValueError(f"prompt length mismatch in {rid}")
            if manifest.get("arguments", {}).get("ignore_eos", False):
                for records_by_id in (left, right):
                    if len(records_by_id[rid]["token_ids"]) != workload[rid]["max_new_tokens"]:
                        raise ValueError(f"incomplete fixed-length output in {rid}")
            if left[rid]["token_ids"] != right[rid]["token_ids"]:
                mismatches.append(rid)
        target_time, spec_time = target["timing"]["wall_time_s"], candidate["timing"]["wall_time_s"]
        stages = candidate["engine"]["speculative_stage_seconds"]
        target_stages = target["engine"].get("target_stage_wall_seconds", {})
        spec_stages = candidate["engine"].get("target_stage_wall_seconds", {})
        row = {
            "concurrency": target["concurrency"], "repeat": target["repeat"],
            "requests": len(left), "output_tokens": sum(len(r["token_ids"]) for r in left.values()),
            "exact_match": not mismatches, "mismatches": mismatches,
            "target_tps": target["timing"]["output_token_throughput_tps"],
            "dflash_tps": candidate["timing"]["output_token_throughput_tps"],
            "tps_ratio": candidate["timing"]["output_token_throughput_tps"] / target["timing"]["output_token_throughput_tps"],
            "target_wall_s": target_time, "dflash_wall_s": spec_time,
            "target_stages_s": target_stages, "dflash_target_stages_s": spec_stages,
            "spec_stages_s": stages,
            "target_unattributed_wall_s": target_time - sum(target_stages.values()),
            "dflash_unattributed_wall_s": spec_time - sum(spec_stages.values()) - sum(stages.values()),
            "verify_break_even_s_if_rest_fixed": target_time - (spec_time - stages["verify"]),
            "speedup_ceiling_if_verify_free": target_time / (spec_time - stages["verify"]),
            "acceptance": {k: candidate["acceptance"][k] for k in ["accepted_tokens", "proposed_tokens", "accepted_length_histogram"]},
            "peak_gib": candidate["memory"]["after"]["max_allocated_gib"],
            "target_ttft_p50_s": target["timing"]["ttft_s"]["p50"],
            "dflash_ttft_p50_s": candidate["timing"]["ttft_s"]["p50"],
            "target_tpot_p50_s": target["timing"]["tpot_s"]["p50"],
            "dflash_tpot_p50_s": candidate["timing"]["tpot_s"]["p50"],
        }
        output["pairs"].append(row)
    (run_dir / "independent_audit.json").write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    outputs = [audit(path) for path in args.runs]
    print(json.dumps(outputs, indent=2))
    if any(not row["exact_match"] for output in outputs for row in output["pairs"]):
        raise SystemExit(1)
