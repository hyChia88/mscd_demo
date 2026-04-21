#!/usr/bin/env python3
"""
Inject Phase 6 OpenCV slot counts into an existing precomputed JSONL.

Supports two counter modes:
  --mode f3   soft-signal (no oracle): uses count_from_full_storey()
  --mode f4   oracle:                   uses count_from_full_storey_with_wall_bounds(),
                                        with wall endpoints derived from IFC via PCA
                                        + local-X alignment.

Scope: only cases whose storey has a calibrated full-storey render available
       (currently Floor 1 + Garage + Level 1 on AP). Cases outside scope pass
       through unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Load counter without triggering src/neurosym/__init__.py.
_FC_PATH = SRC_ROOT / "neurosym" / "floorplan_counter.py"
_spec = importlib.util.spec_from_file_location("_fc_mod", _FC_PATH)
_fc = importlib.util.module_from_spec(_spec)
sys.modules["_fc_mod"] = _fc
_spec.loader.exec_module(_fc)
FloorplanCounter = _fc.FloorplanCounter
FloorplanCountResult = _fc.FloorplanCountResult
merge_position_context = _fc.merge_position_context
load_storey_calibration_index = _fc.load_storey_calibration_index

# Import validator for its wall-endpoint helper (F4 mode only).
_VAL_PATH = PROJECT_ROOT / "evaluation" / "h2" / "validate_phase6_ap_canonical_counts.py"
_vspec = importlib.util.spec_from_file_location("_val_mod", _VAL_PATH)
_val = importlib.util.module_from_spec(_vspec)
sys.modules["_val_mod"] = _val
_vspec.loader.exec_module(_val)


CURATED_AP_ROOT = REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap"
SKELETONS_PATH  = CURATED_AP_ROOT / "skeletons" / "skeletons.jsonl"
FULL_DIR        = CURATED_AP_ROOT / "floorplans_full"
DEFAULT_CASES       = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
DEFAULT_PRECOMPUTED = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g8_posctx_dim__ap_eval.jsonl"
DEFAULT_OUTPUT_FMT  = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g8_posctx_dim__ap_eval_opencv_count_{mode}.jsonl"
OPENING_PREDICATES = {"FILLS", "NEXT_TO"}
OPENING_TYPES = {"IfcDoor", "IfcWindow"}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _case_map(path: Path) -> dict[str, dict]:
    rows = _load_jsonl(path)
    return {row["case_id"]: row for row in rows if row.get("case_id")}


def _skeleton_map(path: Path) -> dict[str, dict]:
    rows = _load_jsonl(path)
    return {row["id"]: row for row in rows if row.get("id")}


def _resolve_storey_name(case: dict, skeleton: Optional[dict]) -> Optional[str]:
    """Return a teacher-consistent storey name for calibration lookup."""
    if skeleton:
        band = skeleton.get("storey_band") or (skeleton.get("target_props") or {}).get("Storey")
        if band:
            return str(band)
    gt = case.get("ground_truth") or {}
    if gt.get("target_storey"):
        return str(gt["target_storey"])
    labels = (case.get("labels") or {}).get("constraints") or {}
    if labels.get("storey_name"):
        return str(labels["storey_name"])
    return None


def _resolve_calibration(calibrations: dict, storey_name: Optional[str]) -> Optional[dict]:
    if not storey_name:
        return None
    key = str(storey_name).strip().lower()
    if key in calibrations:
        return calibrations[key]
    for name, entry in calibrations.items():
        if key in name or name in key:
            return entry
    return None


def _target_world_xy(skeleton: Optional[dict]) -> Optional[tuple[float, float]]:
    if not skeleton:
        return None
    props = skeleton.get("target_props") or {}
    centre = props.get("patch_center_xyz") or skeleton.get("patch_center_xyz")
    if not centre:
        return None
    try:
        return float(centre["x"]), float(centre["y"])
    except Exception:
        return None


def _should_attempt(case: dict, entry: dict) -> bool:
    difficulty = case.get("difficulty_tags") or {}
    labels = (case.get("labels") or {}).get("constraints") or {}
    constraints = entry.get("constraints") or {}
    predicate = difficulty.get("spatial_predicate")
    if predicate in OPENING_PREDICATES:
        return True
    if labels.get("ifc_class") in OPENING_TYPES:
        return True
    if constraints.get("ifc_class") in OPENING_TYPES:
        return True
    return False


def _count_for_case(
    counter: FloorplanCounter,
    calibration: dict,
    skeleton: Optional[dict],
    target_xy: tuple[float, float],
    mode: str,
) -> Optional[FloorplanCountResult]:
    png_path = FULL_DIR / Path(calibration["png_path"]).name

    if mode == "f4":
        host_guid = _val._host_guid_from_skeleton(skeleton) if skeleton else None
        if not host_guid:
            return None
        endpoints = _val._wall_endpoints_world(host_guid)
        if endpoints is None:
            return None
        return counter.count_from_full_storey_with_wall_bounds(
            png_path, calibration, target_xy, endpoints
        )
    return counter.count_from_full_storey(png_path, calibration, target_xy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--precomputed", type=Path, default=DEFAULT_PRECOMPUTED)
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output path (defaults to *_{mode}.jsonl next to precomputed).")
    parser.add_argument("--mode", choices=("f3", "f4"), default="f4",
                        help="Counter mode: f3 = soft (no oracle), f4 = oracle wall bounds.")
    args = parser.parse_args()

    output_path = args.output or Path(str(DEFAULT_OUTPUT_FMT).format(mode=args.mode))

    case_by_id = _case_map(args.cases)
    skeletons = _skeleton_map(SKELETONS_PATH)
    calibrations = load_storey_calibration_index(FULL_DIR / "calibration.json")
    rows = _load_jsonl(args.precomputed)
    counter = FloorplanCounter()

    attempted = 0
    enriched = 0
    out_of_scope = 0
    source_usage: Counter[str] = Counter()
    storey_usage: Counter[str] = Counter()
    updated_rows: list[dict] = []

    for row in rows:
        case_id = row.get("case_id")
        case = case_by_id.get(case_id)
        if not case or not _should_attempt(case, row):
            updated_rows.append(row)
            continue

        attempted += 1
        constraints = dict(row.get("constraints") or {})
        skeleton = skeletons.get(case_id)

        storey = _resolve_storey_name(case, skeleton)
        calibration = _resolve_calibration(calibrations, storey)
        target_xy = _target_world_xy(skeleton)

        count_result: Optional[FloorplanCountResult] = None
        if calibration is not None and target_xy is not None:
            count_result = _count_for_case(counter, calibration, skeleton, target_xy, args.mode)
        else:
            out_of_scope += 1

        final_pc, pc_conf, pc_source = merge_position_context(
            constraints.get("position_context"),
            count_result,
        )

        if count_result is not None and count_result.position > 0:
            constraints["position_context"] = final_pc
            constraints["position_context_confidence"] = pc_conf
            constraints["position_context_source"] = pc_source
            constraints["phase6_opencv_mode"] = args.mode
            enriched += 1
            if pc_source:
                source_usage[pc_source] += 1
            if storey:
                storey_usage[storey] += 1
        else:
            constraints.setdefault(
                "position_context_source",
                "model" if constraints.get("position_context") else None,
            )

        updated = dict(row)
        updated["constraints"] = constraints
        updated_rows.append(updated)

    _write_jsonl(output_path, updated_rows)

    print(f"Wrote {len(updated_rows)} rows to {output_path.relative_to(REPO_ROOT)}")
    print(f"Mode: {args.mode}")
    print(f"Attempted OpenCV enrichment on {attempted} opening-oriented cases")
    print(f"Injected OpenCV-backed position_context on {enriched} cases")
    print(f"Out-of-scope (no calibration/target): {out_of_scope}")
    if source_usage:
        print(f"Position-context sources: {dict(source_usage)}")
    if storey_usage:
        print(f"Storey distribution of enriched cases: {dict(storey_usage)}")


if __name__ == "__main__":
    main()
