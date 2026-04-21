#!/usr/bin/env python3
"""
Phase 6 T1.3 visualization: annotate the full-storey floorplan with the
OpenCV counter's decisions (wall line, numbered openings, target pixel, chosen
target opening) alongside a title bar comparing teacher vs opencv counts.

Output: one PNG per (case, mode) in
    mscd_demo/output/lora6_v2_ap_20260331/phase6_annotations/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


MSCD_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MSCD_ROOT.parent
if str(MSCD_ROOT) not in sys.path:
    sys.path.insert(0, str(MSCD_ROOT))
if str(MSCD_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MSCD_ROOT / "src"))

# Load counter without triggering src/neurosym/__init__.py (which pulls `common`
# and fails outside the usual runner).
_FC_PATH = MSCD_ROOT / "src" / "neurosym" / "floorplan_counter.py"
_spec = importlib.util.spec_from_file_location("_fc_mod", _FC_PATH)
_fc = importlib.util.module_from_spec(_spec)
sys.modules["_fc_mod"] = _fc
_spec.loader.exec_module(_fc)
FloorplanCounter = _fc.FloorplanCounter
load_storey_calibration_index = _fc.load_storey_calibration_index
parse_position_context_tuple = _fc.parse_position_context_tuple

# Import the validator module for its wall-endpoint helper + teacher-label logic.
_VAL_PATH = MSCD_ROOT / "evaluation" / "h2" / "validate_phase6_ap_canonical_counts.py"
_vspec = importlib.util.spec_from_file_location("_val_mod", _VAL_PATH)
_val = importlib.util.module_from_spec(_vspec)
sys.modules["_val_mod"] = _val
_vspec.loader.exec_module(_val)


AP_DATASET = REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap"
CANONICAL  = AP_DATASET / "train" / "lora6_v2_ap_eval_canonical_m_g7.jsonl"
SKELETONS  = AP_DATASET / "skeletons" / "skeletons.jsonl"
FULL_DIR   = AP_DATASET / "floorplans_full"
OUT_DIR    = MSCD_ROOT / "output" / "lora6_v2_ap_20260331" / "phase6_annotations"


_POS_CTX_RE = re.compile(r"\d+(?:st|nd|rd|th)\s+of\s+\d+\s+openings", re.IGNORECASE)


# ── Colour palette (BGR, matches cv2 convention) ─────────────────────────────
WALL_LINE_COLOR      = (180, 80, 40)      # teal
OPENING_COLOR        = (0, 165, 255)      # orange
TARGET_OPENING_COLOR = (0, 0, 220)        # red — the opening OpenCV picked
TEACHER_TARGET_COLOR = (0, 180, 0)        # green — teacher's target position (reference)
TARGET_PIXEL_COLOR   = (255, 0, 180)      # magenta — target's world-coord pixel
TEXT_COLOR           = (0, 0, 0)
TEXT_BG_OK           = (180, 240, 180)
TEXT_BG_FAIL         = (160, 160, 255)


def _teacher_pc(case: dict) -> str | None:
    asst = next((m for m in case["messages"] if m["role"] == "assistant"), None)
    if not asst:
        return None
    try:
        target = json.loads(asst["content"])
    except Exception:
        return None
    return target.get("position_context")


def _resolve_calibration(calibrations: dict, storey_name: str | None) -> dict | None:
    if not storey_name:
        return None
    key = str(storey_name).strip().lower()
    if key in calibrations:
        return calibrations[key]
    for name, entry in calibrations.items():
        if key in name or name in key:
            return entry
    return None


def _target_world_xy(skeleton: dict):
    props = skeleton.get("target_props") or {}
    centre = props.get("patch_center_xyz") or skeleton.get("patch_center_xyz")
    if not centre:
        return None
    try:
        return float(centre["x"]), float(centre["y"])
    except Exception:
        return None


def _ordinal_from_slot(slot):
    if not slot:
        return "?"
    return f"{slot[0]} / {slot[1]}"


def _draw_title_bar(img, lines: list[str], ok: bool) -> np.ndarray:
    h, w = img.shape[:2]
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    line_h = 22
    box_h = pad * 2 + line_h * len(lines)
    bar = np.full((box_h, w, 3), TEXT_BG_OK if ok else TEXT_BG_FAIL, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (pad, pad + line_h * (i + 1) - 4), font, scale, TEXT_COLOR, thick, cv2.LINE_AA)
    return np.vstack([bar, img])


def _draw_openings(img, openings, target_idx: int | None, teacher_pos: int | None):
    for i, op in enumerate(openings):
        cx, cy = int(op["cx"]), int(op["cy"])
        pos_num = i + 1
        is_opencv_target = (target_idx is not None and i == target_idx)
        is_teacher_target = (teacher_pos is not None and pos_num == teacher_pos)

        if is_opencv_target:
            color = TARGET_OPENING_COLOR
            radius = 14
            thickness = 3
        elif is_teacher_target:
            color = TEACHER_TARGET_COLOR
            radius = 14
            thickness = 2
        else:
            color = OPENING_COLOR
            radius = 10
            thickness = 2
        cv2.circle(img, (cx, cy), radius, color, thickness, cv2.LINE_AA)
        # Position number label
        label = str(pos_num)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            img,
            (cx - tw // 2 - 2, cy - radius - th - 4),
            (cx + tw // 2 + 2, cy - radius - 2),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            img, label, (cx - tw // 2, cy - radius - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )


def _draw_wall_line(img, wall_line):
    if wall_line is None:
        return
    p1 = (int(wall_line[0][0]), int(wall_line[0][1]))
    p2 = (int(wall_line[1][0]), int(wall_line[1][1]))
    cv2.line(img, p1, p2, WALL_LINE_COLOR, 2, cv2.LINE_AA)
    # Mark wall_origin (p1) with a small arrow to show direction of ordering
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / norm, dy / norm
    head = (int(p1[0] + 30 * ux), int(p1[1] + 30 * uy))
    cv2.arrowedLine(img, p1, head, WALL_LINE_COLOR, 2, cv2.LINE_AA, tipLength=0.3)


def _draw_target_pixel(img, pixel_xy):
    x, y = int(pixel_xy[0]), int(pixel_xy[1])
    # Magenta X mark + ring
    cv2.drawMarker(img, (x, y), TARGET_PIXEL_COLOR, markerType=cv2.MARKER_TILTED_CROSS,
                   markerSize=22, thickness=3)
    cv2.circle(img, (x, y), 18, TARGET_PIXEL_COLOR, 1, cv2.LINE_AA)


def annotate_case(
    case: dict,
    skeleton: dict,
    calibration: dict,
    counter: FloorplanCounter,
    mode: str,
    out_dir: Path,
) -> dict:
    case_id = case["id"]
    predicate = case.get("predicate", "?")

    teacher_pc = _teacher_pc(case)
    teacher_slot = parse_position_context_tuple(teacher_pc) if teacher_pc else None
    world_xy = _target_world_xy(skeleton)
    png_path = FULL_DIR / Path(calibration["png_path"]).name
    image = cv2.imread(str(png_path))
    if image is None or world_xy is None:
        return {"case_id": case_id, "annotated": False, "reason": "missing_inputs"}

    # Re-run counter + capture all intermediate info by calling internals directly
    tgt_px = counter._world_to_pixel(world_xy, calibration)

    wall_line = None
    openings: list = []
    if mode == "f4":
        host_guid = _val._host_guid_from_skeleton(skeleton)
        endpoints = _val._wall_endpoints_world(host_guid) if host_guid else None
        if endpoints is None:
            return {"case_id": case_id, "annotated": False, "reason": "no_wall_endpoints"}
        a_px = counter._world_to_pixel(endpoints[0], calibration)
        b_px = counter._world_to_pixel(endpoints[1], calibration)
        wall_line = ((float(a_px[0]), float(a_px[1])), (float(b_px[0]), float(b_px[1])))
        seg_len = ((b_px[0] - a_px[0]) ** 2 + (b_px[1] - a_px[1]) ** 2) ** 0.5
        raw_openings = counter._detect_openings_on_wall(image, wall_line)
        tol = 12.0
        openings = [op for op in raw_openings if -tol <= op["projection"] <= seg_len + tol]
    else:
        search_span = max(image.shape[0], image.shape[1]) * 0.25
        wall_line = counter._detect_wall_line(image, (float(tgt_px[0]), float(tgt_px[1])), search_span=search_span)
        if wall_line is not None:
            openings = counter._detect_openings_on_wall(image, wall_line)

    target_idx = None
    if openings:
        target_idx = counter._nearest_opening_index(openings, (float(tgt_px[0]), float(tgt_px[1])))
    opencv_slot = (target_idx + 1, len(openings)) if target_idx is not None else None
    ok = opencv_slot == teacher_slot

    # Draw
    annotated = image.copy()
    _draw_wall_line(annotated, wall_line)
    _draw_openings(
        annotated,
        openings,
        target_idx=target_idx,
        teacher_pos=teacher_slot[0] if teacher_slot else None,
    )
    _draw_target_pixel(annotated, tgt_px)

    title_lines = [
        f"{case_id}  [{predicate}]  mode={mode.upper()}  storey={calibration.get('storey_name', '?')}",
        f"Teacher: {_ordinal_from_slot(teacher_slot)}    OpenCV: {_ordinal_from_slot(opencv_slot)}"
        f"    {'[OK]' if ok else '[FAIL]'}",
        f"Legend: red=opencv target, green=teacher target, orange=other openings, magenta=target world_xy",
    ]
    final = _draw_title_bar(annotated, title_lines, ok=ok)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_{mode}_{'OK' if ok else 'FAIL'}.png"
    cv2.imwrite(str(out_path), final)
    return {
        "case_id": case_id,
        "annotated": True,
        "mode": mode,
        "teacher": teacher_slot,
        "opencv": opencv_slot,
        "ok": ok,
        "png": str(out_path.relative_to(REPO_ROOT)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("f3", "f4", "both"), default="f4",
                        help="Which counter mode to annotate (default: f4)")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case IDs to annotate (default: all wall-counting cases)")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    with CANONICAL.open() as f:
        cases = [json.loads(line) for line in f if line.strip()]
    with SKELETONS.open() as f:
        skeletons = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}
    calibrations = load_storey_calibration_index(FULL_DIR / "calibration.json")
    counter = FloorplanCounter()

    case_filter = set(args.cases.split(",")) if args.cases else None
    modes = ("f3", "f4") if args.mode == "both" else (args.mode,)

    summary = []
    for case in cases:
        case_id = case["id"]
        if case_filter and case_id not in case_filter:
            continue
        if case.get("predicate") not in ("FILLS", "NEXT_TO"):
            continue
        pc = _teacher_pc(case)
        if not (pc and _POS_CTX_RE.search(pc)):
            continue
        sk = skeletons.get(case_id)
        if not sk:
            continue

        storey_name = (
            sk.get("storey_band")
            or (sk.get("target_props") or {}).get("Storey")
        )
        calib = _resolve_calibration(calibrations, storey_name)
        if not calib:
            summary.append({"case_id": case_id, "annotated": False, "reason": "no_calibration"})
            continue

        for m in modes:
            out = args.out_dir / m
            res = annotate_case(case, sk, calib, counter, m, out)
            summary.append(res)

    # Aggregate stats
    by_mode: dict = {}
    for r in summary:
        if not r.get("annotated"):
            continue
        m = r["mode"]
        by_mode.setdefault(m, {"ok": 0, "fail": 0})
        by_mode[m]["ok" if r["ok"] else "fail"] += 1

    print(json.dumps({
        "annotated": sum(1 for r in summary if r.get("annotated")),
        "skipped": sum(1 for r in summary if not r.get("annotated")),
        "by_mode": by_mode,
        "out_root": str(args.out_dir.relative_to(REPO_ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
