#!/usr/bin/env python3
"""Inject Phase 6 OpenCV slot counts into an existing precomputed JSONL."""

from __future__ import annotations

import argparse
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

from src.neurosym.floorplan_counter import FloorplanCounter, FloorplanCountResult, merge_position_context


CURATED_AP_ROOT = REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap"
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
DEFAULT_PRECOMPUTED = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g8_posctx_dim__ap_eval.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g8_posctx_dim__ap_eval_opencv_count.jsonl"
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


def _candidate_scales(case_id: str, prefer_scales: list[str]) -> list[tuple[str, Path]]:
    root = CURATED_AP_ROOT / "floorplans_v2"
    found: list[tuple[str, Path]] = []
    for scale in prefer_scales:
        patch = root / f"{case_id}_scale_{scale}.png"
        if patch.exists():
            found.append((scale, patch))
    return found


def _pick_best_count(
    counter: FloorplanCounter,
    *,
    case_id: str,
    prefer_scales: list[str],
) -> tuple[Optional[FloorplanCountResult], Optional[str]]:
    full_floorplan = CURATED_AP_ROOT / "floorplans" / f"{case_id}_floorplan.png"
    if not full_floorplan.exists():
        return None, None

    best_result: Optional[FloorplanCountResult] = None
    best_scale: Optional[str] = None
    for scale, patch in _candidate_scales(case_id, prefer_scales):
        result = counter.count_from_paths(patch, full_floorplan)
        if result is None:
            continue
        if best_result is None or result.confidence > best_result.confidence:
            best_result = result
            best_scale = scale
    return best_result, best_scale


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--precomputed", type=Path, default=DEFAULT_PRECOMPUTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prefer-scale",
        action="append",
        dest="prefer_scales",
        choices=("M", "L", "S"),
        help="Patch-scale priority. May be passed multiple times. Default: M, then L, then S.",
    )
    args = parser.parse_args()

    prefer_scales = args.prefer_scales or ["M", "L", "S"]
    case_by_id = _case_map(args.cases)
    rows = _load_jsonl(args.precomputed)
    counter = FloorplanCounter(image_dir=str(REPO_ROOT / "data_curation" / "datasets"))

    attempted = 0
    enriched = 0
    scale_usage: Counter[str] = Counter()
    source_usage: Counter[str] = Counter()
    updated_rows: list[dict] = []

    for row in rows:
        case_id = row.get("case_id")
        case = case_by_id.get(case_id)
        if not case or not _should_attempt(case, row):
            updated_rows.append(row)
            continue

        attempted += 1
        constraints = dict(row.get("constraints") or {})
        count_result, chosen_scale = _pick_best_count(counter, case_id=case_id, prefer_scales=prefer_scales)
        final_position_context, pos_conf, pos_source = merge_position_context(
            constraints.get("position_context"),
            count_result,
        )

        if count_result is not None:
            constraints["position_context"] = final_position_context
            constraints["position_context_confidence"] = pos_conf
            constraints["position_context_source"] = pos_source
            constraints["phase6_patch_scale"] = chosen_scale
            enriched += 1
            if chosen_scale:
                scale_usage[chosen_scale] += 1
            if pos_source:
                source_usage[pos_source] += 1
        else:
            if constraints.get("position_context"):
                constraints.setdefault("position_context_source", "model")

        updated = dict(row)
        updated["constraints"] = constraints
        updated_rows.append(updated)

    _write_jsonl(args.output, updated_rows)

    print(f"Wrote {len(updated_rows)} rows to {args.output}")
    print(f"Attempted OpenCV enrichment on {attempted} opening-oriented cases")
    print(f"Injected OpenCV-backed position_context on {enriched} cases")
    if scale_usage:
        print(f"Patch scales used: {dict(scale_usage)}")
    if source_usage:
        print(f"Position-context sources: {dict(source_usage)}")


if __name__ == "__main__":
    main()
