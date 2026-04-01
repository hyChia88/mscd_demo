#!/usr/bin/env python3
"""Build oracle precomputed constraints from AP held-out end-to-end cases.

This is the AP-held-out counterpart to the legacy build_precomputed.py helper.
It emits JSONL files consumable by script/run.py via --precomputed.

Modes:
- default: all cases with full trusted AP held-out constraints
- --spatial-only: only cases with spatial_relations
- --no-spatial: only cases without spatial_relations
- --p1-only: keep only storey_name + ifc_class to isolate P1 filtering
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "oracle_ap_heldout"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _default_output_name(spatial_only: bool, no_spatial: bool, p1_only: bool) -> str:
    if spatial_only and p1_only:
        return "precomputed_ap_heldout_oracle_spatial_p1_only.jsonl"
    if no_spatial and p1_only:
        return "precomputed_ap_heldout_oracle_no_spatial_p1_only.jsonl"
    if spatial_only:
        return "precomputed_ap_heldout_oracle_spatial_only.jsonl"
    if no_spatial:
        return "precomputed_ap_heldout_oracle_no_spatial.jsonl"
    if p1_only:
        return "precomputed_ap_heldout_oracle_p1_only.jsonl"
    return "precomputed_ap_heldout_oracle_all.jsonl"


def _build_constraints(raw: dict, *, p1_only: bool) -> dict:
    constraints = dict(raw)
    if p1_only:
        return {
            "storey_name": constraints.get("storey_name"),
            "ifc_class": constraints.get("ifc_class"),
        }

    spatial_rels = []
    for sr in constraints.get("spatial_relations", []) or []:
        sr = dict(sr)
        sr.setdefault("subject_type", constraints.get("ifc_class", ""))
        spatial_rels.append(sr)
    constraints["spatial_relations"] = spatial_rels
    return constraints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--spatial-only", action="store_true")
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--p1-only", action="store_true")
    args = parser.parse_args()
    if args.spatial_only and args.no_spatial:
        raise SystemExit("Use at most one of --spatial-only and --no-spatial.")

    cases = _load_jsonl(args.cases)
    results: list[dict] = []

    for case in cases:
        constraints = case.get("labels", {}).get("constraints", {})
        spatial_rels = list(constraints.get("spatial_relations", []) or [])

        if args.spatial_only and not spatial_rels:
            continue
        if args.no_spatial and spatial_rels:
            continue

        results.append(
            {
                "case_id": case["case_id"],
                "status": "OK",
                "constraints": _build_constraints(constraints, p1_only=args.p1_only),
            }
        )

    out_path = args.out
    if out_path is None:
        out_path = DEFAULT_OUT_DIR / _default_output_name(
            args.spatial_only, args.no_spatial, args.p1_only
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    spatial_count = sum(1 for r in results if r["constraints"].get("spatial_relations"))
    print(f"Wrote {len(results)} AP-held-out oracle precomputed rows -> {out_path}")
    print(f"  With spatial_relations: {spatial_count}")
    print(f"  Without spatial_relations: {len(results) - spatial_count}")
    print(f"  P1-only mode: {args.p1_only}")


if __name__ == "__main__":
    main()
