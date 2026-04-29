#!/usr/bin/env python3
"""Phase 6.1.5 — inject ResNet-predicted `size_band` into a precomputed JSONL.

Mirrors `inject_floorplan_counts.py`: reads a precomputed constraints JSONL,
runs the size_band ResNet on each held-out W/D case (using the GT element's
centroid as the oracle world_xy — same scope concession as F4's oracle wall
bounds), writes a new JSONL with `constraints.size_band` populated.

Confidence gating: when ResNet confidence < `--min-confidence`, the prediction
is *not* injected (the case falls through to soft-mode retrieval). This
preserves recall for cases where the perception modality is ambiguous (e.g.
window_L which is visually identical to neighbouring bands from above).

Scope:
  - Eval-time use only — needs GT centroid lookup from element_index.jsonl.
  - In-scope storeys only (3 of 6 AP storeys with full-floorplan renders).
  - No-op for non-W/D cases (other ifc_class) and out-of-scope storeys.

Live-inference integration is a separate task (small patch to pipeline_base.py
that calls SizeBandClassifier.predict() inline). This script is for held-out
evaluation only.

Usage:
  python mscd_demo/evaluation/analysis/inject_size_band.py \\
      --input mscd_demo/output/lora6_v2_ap_20260331/g9_opencv_cluster__ap_eval.jsonl \\
      --cases mscd_demo/evaluation/cases/cases_ap_heldout_e2e.jsonl \\
      --output mscd_demo/output/lora6_v2_ap_20260331/g9_resnet_size_band__ap_eval.jsonl \\
      --min-confidence 0.6
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "mscd_demo"))

from src.neurosym.cluster_classifier import SizeBandClassifier  # noqa: E402


def _load_element_index(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            e = json.loads(line)
            guid = e.get("global_id")
            if guid:
                out[guid] = e
    return out


def _load_cases(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            c = json.loads(line)
            cid = c.get("case_id")
            if cid:
                out[cid] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="Precomputed constraints JSONL to override (e.g. G9 emit)")
    ap.add_argument("--cases", type=Path,
                    default=REPO_ROOT / "mscd_demo/evaluation/cases/cases_ap_heldout_e2e.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "mscd_demo/models/cluster_classifier_ap/best.pt")
    ap.add_argument("--calibration", type=Path,
                    default=REPO_ROOT / "data_curation/datasets/synth_v0.5_ap/floorplans_full/calibration.json")
    ap.add_argument("--floorplans-root", type=Path,
                    default=REPO_ROOT / "data_curation")
    ap.add_argument("--element-index", type=Path,
                    default=REPO_ROOT / "data_curation/references/element_index.jsonl")
    ap.add_argument("--min-confidence", type=float, default=0.6,
                    help="Drop predictions below this — falls back to soft retrieval.")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optional path to write per-case decisions JSON.")
    args = ap.parse_args()

    cases = _load_cases(args.cases)
    elements = _load_element_index(args.element_index)
    classifier = SizeBandClassifier(
        checkpoint=args.checkpoint,
        calibration=args.calibration,
        floorplans_root=args.floorplans_root,
    )
    in_scope = set(classifier.supported_storeys())
    print(f"[init] classifier classes: {classifier.classes}", flush=True)
    print(f"[init] in-scope storeys: {sorted(in_scope)}", flush=True)
    print(f"[init] confidence gate: {args.min_confidence}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_injected = 0
    n_low_conf = 0
    n_off_storey = 0
    n_non_wd = 0
    n_no_gt = 0
    decisions = []

    with args.input.open() as f_in, args.output.open("w") as f_out:
        for line in f_in:
            rec = json.loads(line)
            n_in += 1
            cid = rec.get("case_id")
            constraints = rec.setdefault("constraints", {})
            ifc_class = constraints.get("ifc_class") or ""
            decision = {"case_id": cid, "action": None, "reason": None}

            if "Window" not in ifc_class and "Door" not in ifc_class:
                decision["action"] = "skip"; decision["reason"] = "non-W/D ifc_class"
                n_non_wd += 1
                f_out.write(json.dumps(rec) + "\n"); decisions.append(decision); continue

            case = cases.get(cid)
            if not case:
                decision["action"] = "skip"; decision["reason"] = "case missing"
                n_no_gt += 1
                f_out.write(json.dumps(rec) + "\n"); decisions.append(decision); continue
            gt_guid = case.get("ground_truth", {}).get("target_guid")
            elt = elements.get(gt_guid)
            if not elt:
                decision["action"] = "skip"; decision["reason"] = "GT element not in index"
                n_no_gt += 1
                f_out.write(json.dumps(rec) + "\n"); decisions.append(decision); continue
            storey_name = elt.get("storey_name")
            centroid = elt.get("centroid") or {}
            if storey_name not in in_scope:
                decision["action"] = "skip"; decision["reason"] = f"storey out-of-scope: {storey_name!r}"
                n_off_storey += 1
                f_out.write(json.dumps(rec) + "\n"); decisions.append(decision); continue
            if "x" not in centroid or "y" not in centroid:
                decision["action"] = "skip"; decision["reason"] = "centroid missing"
                n_no_gt += 1
                f_out.write(json.dumps(rec) + "\n"); decisions.append(decision); continue

            pred = classifier.predict(storey_name, (centroid["x"], centroid["y"]))
            if pred is None:
                decision["action"] = "skip"; decision["reason"] = "classifier returned None"
                n_off_storey += 1
                f_out.write(json.dumps(rec) + "\n"); decisions.append(decision); continue

            if pred.confidence < args.min_confidence:
                decision.update({
                    "action": "skip-low-conf",
                    "reason": f"confidence {pred.confidence:.2f} < {args.min_confidence}",
                    "predicted_band": pred.band,
                    "confidence": pred.confidence,
                })
                n_low_conf += 1
                f_out.write(json.dumps(rec) + "\n"); decisions.append(decision); continue

            constraints["size_band"] = pred.band
            constraints["size_band_confidence"] = pred.confidence
            constraints["size_band_source"] = "resnet_oracle_centroid"
            decision.update({
                "action": "inject",
                "predicted_band": pred.band,
                "confidence": pred.confidence,
                "previous_size_cluster": constraints.get("size_cluster"),
            })
            n_injected += 1
            f_out.write(json.dumps(rec) + "\n"); decisions.append(decision)

    print(f"\n[result] read {n_in} records, wrote {n_in} (overrides applied: {n_injected})", flush=True)
    print(f"  injected:        {n_injected}", flush=True)
    print(f"  low-conf skip:   {n_low_conf}", flush=True)
    print(f"  off-storey skip: {n_off_storey}", flush=True)
    print(f"  non-W/D skip:    {n_non_wd}", flush=True)
    print(f"  no-GT skip:      {n_no_gt}", flush=True)
    band_dist = Counter(d.get("predicted_band") for d in decisions if d.get("action") == "inject")
    print(f"\n[bands injected] {dict(band_dist)}", flush=True)
    if args.report:
        args.report.write_text(json.dumps({
            "n_in": n_in, "n_injected": n_injected,
            "n_low_conf": n_low_conf, "n_off_storey": n_off_storey,
            "n_non_wd": n_non_wd, "n_no_gt": n_no_gt,
            "min_confidence": args.min_confidence,
            "decisions": decisions,
        }, indent=2))
        print(f"[report] wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
