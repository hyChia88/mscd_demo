#!/usr/bin/env python3
"""
Phase 6 T1.1 diagnostic validator.

Runs the full-storey OpenCV counter on AP canonical cases (FILLS + NEXT_TO)
using teacher `position_context` labels from
lora6_v2_ap_eval_canonical_m_g7.jsonl as ground truth.

No IFC, no Neo4j: GT comes from teacher labels; target world coord comes from
skeletons.jsonl; full-storey renders + calibration come from
data_curation/datasets/synth_v0.5_ap/floorplans_full/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

MSCD_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MSCD_ROOT.parent
if str(MSCD_ROOT) not in sys.path:
    sys.path.insert(0, str(MSCD_ROOT))
if str(MSCD_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MSCD_ROOT / "src"))

# Import the module directly to avoid the src/neurosym/__init__.py chain
# (which pulls `common` and fails when run outside the usual runner).
import importlib.util

_FC_PATH = MSCD_ROOT / "src" / "neurosym" / "floorplan_counter.py"
_spec = importlib.util.spec_from_file_location("_fc_mod", _FC_PATH)
_fc = importlib.util.module_from_spec(_spec)
sys.modules["_fc_mod"] = _fc   # must register before exec for @dataclass
_spec.loader.exec_module(_fc)
FloorplanCounter = _fc.FloorplanCounter
load_storey_calibration_index = _fc.load_storey_calibration_index
parse_position_context_tuple = _fc.parse_position_context_tuple


AP_DATASET = REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap"
AP_IFC     = REPO_ROOT / "data_curation" / "ifc_models" / "AdvancedProject.ifc"
CANONICAL  = AP_DATASET / "train" / "lora6_v2_ap_eval_canonical_m_g7.jsonl"
SKELETONS  = AP_DATASET / "skeletons" / "skeletons.jsonl"
FULL_DIR   = AP_DATASET / "floorplans_full"
DEFAULT_OUT_FMT = MSCD_ROOT / "output" / "lora6_v2_ap_20260331" / "phase6_ap_canonical_counter_report_{mode}.jsonl"


_POS_CTX_RE = re.compile(r"\d+(?:st|nd|rd|th)\s+of\s+\d+\s+openings", re.IGNORECASE)


def _load_canonical(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_skeletons(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            index[row["id"]] = row
    return index


def _teacher_pc(case: dict) -> str | None:
    asst = next((m for m in case["messages"] if m["role"] == "assistant"), None)
    if not asst:
        return None
    try:
        target = json.loads(asst["content"])
    except Exception:
        return None
    return target.get("position_context")


def _teacher_storey_name(case: dict, skeleton: dict | None) -> str | None:
    """Prefer the skeleton's canonical `storey_band`; fall back to chat/teacher storey."""
    if skeleton:
        band = skeleton.get("storey_band") or (skeleton.get("target_props") or {}).get("Storey")
        if band:
            return band
    asst = next((m for m in case["messages"] if m["role"] == "assistant"), None)
    if asst:
        try:
            return json.loads(asst["content"]).get("storey_name")
        except Exception:
            return None
    return None


def _resolve_calibration(calibrations: dict, storey_name: str | None) -> dict | None:
    if not storey_name:
        return None
    key = str(storey_name).strip().lower()
    if key in calibrations:
        return calibrations[key]
    # loose fallback: match by substring (e.g. "1" → "1 - first floor")
    for name, entry in calibrations.items():
        if key in name or name in key:
            return entry
    return None


def _target_world_xy(skeleton: dict) -> tuple[float, float] | None:
    props = skeleton.get("target_props") or {}
    centre = props.get("patch_center_xyz") or skeleton.get("patch_center_xyz")
    if not centre:
        return None
    try:
        return float(centre["x"]), float(centre["y"])
    except Exception:
        return None


# ── F4 oracle: IFC wall-endpoint extraction (PCA over wall's 2D vertices) ─────
_IFC_CACHE: dict = {}    # lazy global IFC load
_WALL_ENDPOINTS_CACHE: dict = {}  # guid -> ((x0,y0),(x1,y1)) or None


def _load_ap_ifc():
    if "ifc" not in _IFC_CACHE:
        import ifcopenshell
        import ifcopenshell.geom
        _IFC_CACHE["ifc"] = ifcopenshell.open(str(AP_IFC))
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        _IFC_CACHE["settings"] = settings
    return _IFC_CACHE["ifc"], _IFC_CACHE["settings"]


def _wall_endpoints_world(host_guid: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Compute wall endpoints in world XY by PCA over the wall's 2D vertex cloud."""
    if host_guid in _WALL_ENDPOINTS_CACHE:
        return _WALL_ENDPOINTS_CACHE[host_guid]
    try:
        import ifcopenshell.geom
        import numpy as np
    except Exception:
        _WALL_ENDPOINTS_CACHE[host_guid] = None
        return None

    ifc, settings = _load_ap_ifc()
    try:
        wall = ifc.by_guid(host_guid)
    except Exception:
        _WALL_ENDPOINTS_CACHE[host_guid] = None
        return None
    if wall is None:
        _WALL_ENDPOINTS_CACHE[host_guid] = None
        return None

    try:
        shape = ifcopenshell.geom.create_shape(settings, wall)
        verts = np.array(shape.geometry.verts).reshape(-1, 3)[:, :2]
        if len(verts) < 2:
            _WALL_ENDPOINTS_CACHE[host_guid] = None
            return None
        center = verts.mean(axis=0)
        centered = verts - center
        # Principal axis via eigendecomposition of 2D covariance
        cov = (centered.T @ centered) / max(1, len(verts))
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, -1]  # largest eigenvalue = longest direction
        projs = centered @ axis
        lo, hi = float(projs.min()), float(projs.max())
        a = center + lo * axis
        b = center + hi * axis
        endpoints = ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        _WALL_ENDPOINTS_CACHE[host_guid] = endpoints
        return endpoints
    except Exception:
        _WALL_ENDPOINTS_CACHE[host_guid] = None
        return None


def _host_guid_from_skeleton(skeleton: dict) -> str | None:
    props = skeleton.get("target_props") or {}
    return (
        props.get("host_guid")
        or skeleton.get("host_guid")
        or props.get("anchor_guid")
        or skeleton.get("anchor_guid")
    )


def validate(limit: int | None = None, mode: str = "f3") -> dict:
    assert mode in ("f3", "f4"), f"mode must be 'f3' or 'f4', got {mode!r}"
    cases = _load_canonical(CANONICAL)
    skeletons = _load_skeletons(SKELETONS)
    calibrations = load_storey_calibration_index(FULL_DIR / "calibration.json")

    counter = FloorplanCounter()
    rows: list[dict] = []
    stats = Counter()
    exact_match = 0
    counting_denominator = 0

    for case in cases:
        case_id = case["id"]
        predicate = case.get("predicate", "?")

        # Only FILLS + NEXT_TO use "Nth of M openings" counting schema
        if predicate not in ("FILLS", "NEXT_TO"):
            rows.append({"case_id": case_id, "predicate": predicate, "status": "n/a"})
            stats["n/a_non_counting_predicate"] += 1
            continue

        teacher_pc = _teacher_pc(case)
        teacher_slot = parse_position_context_tuple(teacher_pc) if teacher_pc else None
        if not (teacher_pc and _POS_CTX_RE.search(teacher_pc) and teacher_slot):
            rows.append({
                "case_id": case_id, "predicate": predicate,
                "status": "skip_no_teacher_slot", "teacher_pc": teacher_pc,
            })
            stats["skip_no_teacher_slot"] += 1
            continue

        skeleton = skeletons.get(case_id)
        if not skeleton:
            rows.append({
                "case_id": case_id, "predicate": predicate,
                "status": "skip_no_skeleton",
            })
            stats["skip_no_skeleton"] += 1
            continue

        storey_name = _teacher_storey_name(case, skeleton)
        calibration = _resolve_calibration(calibrations, storey_name)
        if not calibration:
            rows.append({
                "case_id": case_id, "predicate": predicate,
                "status": "fail_no_calibration", "storey_name": storey_name,
            })
            stats["fail_no_calibration"] += 1
            continue

        world_xy = _target_world_xy(skeleton)
        if world_xy is None:
            rows.append({
                "case_id": case_id, "predicate": predicate,
                "status": "fail_no_world_coord",
            })
            stats["fail_no_world_coord"] += 1
            continue

        png_path = FULL_DIR / Path(calibration["png_path"]).name

        if mode == "f4":
            host_guid = _host_guid_from_skeleton(skeleton)
            if not host_guid:
                rows.append({
                    "case_id": case_id, "predicate": predicate,
                    "status": "f4_skip_no_host_guid",
                })
                stats["f4_skip_no_host_guid"] += 1
                continue
            endpoints = _wall_endpoints_world(host_guid)
            if endpoints is None:
                rows.append({
                    "case_id": case_id, "predicate": predicate,
                    "status": "f4_skip_no_wall_endpoints", "host_guid": host_guid,
                })
                stats["f4_skip_no_wall_endpoints"] += 1
                continue
            result = counter.count_from_full_storey_with_wall_bounds(
                png_path, calibration, world_xy, endpoints
            )
        else:
            result = counter.count_from_full_storey(png_path, calibration, world_xy)
        counting_denominator += 1

        if result is None:
            rows.append({
                "case_id": case_id, "predicate": predicate,
                "status": "fail_counter_returned_none",
            })
            stats["fail_counter_returned_none"] += 1
            continue

        opencv_slot = (result.position, result.total) if result.position and result.total else None
        is_exact = opencv_slot is not None and opencv_slot == teacher_slot
        if is_exact:
            exact_match += 1
            stats["ok_exact_match"] += 1
        else:
            stats[f"fail_{result.debug.get('fallback_reason') or 'counter_mismatch'}"] += 1

        rows.append({
            "case_id": case_id,
            "predicate": predicate,
            "storey_name": storey_name,
            "teacher_pc": teacher_pc,
            "teacher_slot": list(teacher_slot),
            "opencv": {
                "position": result.position,
                "total": result.total,
                "confidence": round(result.confidence, 3),
                "mode": result.mode,
                "position_context": result.position_context,
                "debug": result.debug,
            },
            "target_world_xy": list(world_xy),
            "exact_match": is_exact,
            "status": "ok_exact_match" if is_exact else "fail_counter_mismatch",
        })

        if limit and counting_denominator >= limit:
            break

    out_path = Path(str(DEFAULT_OUT_FMT).format(mode=mode))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "mode": mode,
        "total_cases": len(cases),
        "counting_denominator": counting_denominator,
        "exact_match": exact_match,
        "exact_match_rate": (exact_match / counting_denominator) if counting_denominator else 0.0,
        "status_breakdown": dict(stats),
        "output_path": str(out_path.relative_to(REPO_ROOT)),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many counting attempts")
    parser.add_argument("--mode", choices=("f3", "f4", "both"), default="both",
                        help="f3=soft (no oracle), f4=oracle wall bounds, both=run sequentially")
    args = parser.parse_args()

    modes = ("f3", "f4") if args.mode == "both" else (args.mode,)
    summaries = []
    for m in modes:
        summaries.append(validate(limit=args.limit, mode=m))

    print(json.dumps(summaries if len(summaries) > 1 else summaries[0], indent=2))


if __name__ == "__main__":
    main()
