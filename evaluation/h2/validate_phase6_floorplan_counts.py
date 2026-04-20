#!/usr/bin/env python3
"""Validate Phase 6 floorplan counting on 10 H2 FILLS cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.placement
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.neurosym.floorplan_counter import FloorplanCounter, FloorplanCountResult


CURATED_AP_ROOT = REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap"
DEFAULT_H2 = REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5" / "legacy" / "eval" / "h2_hard_negatives.jsonl"
DEFAULT_IFC = REPO_ROOT / "data_curation" / "ifc_models" / "AdvancedProject.ifc"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "phase6_h2_fills_exact_match.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_gt_slot_index(ifc_path: Path) -> dict[str, dict]:
    file = ifcopenshell.open(str(ifc_path))

    opening_to_host: dict[str, str] = {}
    for rel in file.by_type("IfcRelVoidsElement"):
        host = rel.RelatingBuildingElement
        opening = rel.RelatedOpeningElement
        if host and opening:
            opening_to_host[opening.GlobalId] = host.GlobalId

    wall_fillers: dict[str, list] = defaultdict(list)
    for rel in file.by_type("IfcRelFillsElement"):
        filler = rel.RelatedBuildingElement
        opening = rel.RelatingOpeningElement
        if filler and opening:
            host_guid = opening_to_host.get(opening.GlobalId)
            if host_guid:
                wall_fillers[host_guid].append(filler)

    gt_index: dict[str, dict] = {}
    for wall_guid, fillers in wall_fillers.items():
        if len(fillers) < 1:
            continue

        try:
            wall = file.by_guid(wall_guid)
            wall_mat = ifcopenshell.util.placement.get_local_placement(wall.ObjectPlacement)
            wall_dir = np.array([wall_mat[0][0], wall_mat[1][0], wall_mat[2][0]])
            wall_origin = np.array([wall_mat[0][3], wall_mat[1][3], wall_mat[2][3]])
        except Exception:
            continue

        storey_groups: dict[str, list] = defaultdict(list)
        for filler in fillers:
            container = ifcopenshell.util.element.get_container(filler)
            storey_key = container.Name if container else "_unknown"
            storey_groups[storey_key].append(filler)

        for storey_name, group in storey_groups.items():
            projections: list[tuple[float, object]] = []
            for filler in group:
                try:
                    mat = ifcopenshell.util.placement.get_local_placement(filler.ObjectPlacement)
                    centroid = np.array([mat[0][3], mat[1][3], mat[2][3]])
                    proj = float(np.dot(centroid - wall_origin, wall_dir))
                    projections.append((proj, filler))
                except Exception:
                    continue

            if not projections:
                continue

            projections.sort(key=lambda item: item[0])
            total = len(projections)
            for idx, (_, filler) in enumerate(projections, start=1):
                gt_index[filler.GlobalId] = {
                    "position": idx,
                    "total": total,
                    "storey_name": storey_name,
                    "wall_guid": wall_guid,
                }

    return gt_index


def _patch_candidates(case_id: str, prefer_scales: list[str]) -> list[tuple[str, Path]]:
    root = CURATED_AP_ROOT / "floorplans_v2"
    found: list[tuple[str, Path]] = []
    for scale in prefer_scales:
        path = root / f"{case_id}_scale_{scale}.png"
        if path.exists():
            found.append((scale, path))
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
    for scale, patch in _patch_candidates(case_id, prefer_scales):
        result = counter.count_from_paths(patch, full_floorplan)
        if result is None:
            continue
        if best_result is None or result.confidence > best_result.confidence:
            best_result = result
            best_scale = scale
    return best_result, best_scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2", type=Path, default=DEFAULT_H2)
    parser.add_argument("--ifc", type=Path, default=DEFAULT_IFC)
    parser.add_argument("--limit", type=int, default=10, help="Number of FILLS cases to validate.")
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
    rows = _load_jsonl(args.h2)
    gt_index = _build_gt_slot_index(args.ifc)
    counter = FloorplanCounter(image_dir=str(REPO_ROOT / "data_curation" / "datasets"))

    eligible: list[dict] = []
    for row in rows:
        if row.get("predicate") != "FILLS":
            continue
        case_id = f"AP_{row['skeleton_id']}"
        if not _patch_candidates(case_id, prefer_scales):
            continue
        if row.get("target_guid") not in gt_index:
            continue
        eligible.append(row)

    selected = eligible[: args.limit]
    results: list[dict] = []

    print(f"Eligible FILLS cases with AP patches: {len(eligible)}")
    print(f"Validating first {len(selected)} cases\n")
    print(f"{'H2-ID':<8} {'Case':<12} {'Scale':<5} {'Pred':<12} {'GT':<10} {'Exact':<5} {'Conf':<5}")
    print("-" * 70)

    for row in selected:
        case_id = f"AP_{row['skeleton_id']}"
        gt = gt_index[row["target_guid"]]
        pred, scale = _pick_best_count(counter, case_id=case_id, prefer_scales=prefer_scales)
        exact = bool(pred and pred.position == gt["position"] and pred.total == gt["total"])

        pred_tuple = f"{pred.position}/{pred.total}" if pred else "n/a"
        gt_tuple = f"{gt['position']}/{gt['total']}"
        conf = f"{pred.confidence:.2f}" if pred else "n/a"
        print(f"{row['h2_id']:<8} {case_id:<12} {str(scale or '-'): <5} {pred_tuple:<12} {gt_tuple:<10} {str(exact):<5} {conf:<5}")

        results.append(
            {
                "h2_id": row["h2_id"],
                "case_id": case_id,
                "target_guid": row["target_guid"],
                "chosen_scale": scale,
                "predicted": (
                    {
                        "position": pred.position,
                        "total": pred.total,
                        "confidence": pred.confidence,
                        "position_context": pred.position_context,
                    }
                    if pred
                    else None
                ),
                "ground_truth": gt,
                "exact_match": exact,
            }
        )

    _write_jsonl(args.output, results)

    n = len(results)
    exact_matches = sum(1 for row in results if row["exact_match"])
    produced_counts = sum(1 for row in results if row["predicted"] is not None)
    print()
    print(f"Counts produced: {produced_counts}/{n}")
    print(f"Exact-match rate: {exact_matches}/{n} ({(100.0 * exact_matches / n):.1f}%)" if n else "Exact-match rate: n/a")
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
