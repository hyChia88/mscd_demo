#!/usr/bin/env python3
"""Phase 6.1.3 — compute G9's apples-to-apples accuracy on the 6-way size_band.

Collapses G9's emitted full size_cluster (15-way, e.g. window_M_1500x1400) to
the band granularity (window_M) using the taxonomy, then scores against the
GT band derived from element_index.jsonl. Compares directly with the ResNet
classifier's test accuracy.

Usage:
  python mscd_demo/evaluation/analysis/score_g9_size_band.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _band_for_cluster(cluster: str | None, taxonomy: dict) -> str | None:
    if not cluster:
        return None
    for family, clusters in (taxonomy.get("clusters") or {}).items():
        props = clusters.get(cluster)
        if props is None:
            continue
        band = props.get("band")
        family_short = "window" if family == "IfcWindow" else "door" if family == "IfcDoor" else None
        return f"{family_short}_{band}" if family_short and band else None
    return None


def _gt_size_band_per_case(eval_jsonl: Path, taxonomy: dict) -> dict:
    """Pull GT size_band from the LoRA G9 eval label set (built by 9_assemble_lora9.py)."""
    import re

    gt = {}
    for line in eval_jsonl.open():
        rec = json.loads(line)
        cid = rec.get("case_id") or rec.get("id")
        msgs = rec.get("messages") or []
        a = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
        if not a:
            continue
        m = re.search(r"\{.*\}", a, flags=__import__("re").S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        cluster = obj.get("size_cluster")
        if cluster:
            gt[cid] = _band_for_cluster(cluster, taxonomy)
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g9-eval", type=Path,
                    default=REPO_ROOT / "mscd_demo/output/lora6_v2_ap_20260331/g9_opencv_cluster__ap_eval.jsonl")
    ap.add_argument("--gt-eval", type=Path,
                    default=REPO_ROOT / "data_curation/datasets/synth_v0.5_ap/train/lora6_v2_ap_eval_canonical_m_g9.jsonl")
    ap.add_argument("--taxonomy", type=Path,
                    default=REPO_ROOT / "mscd_demo/prompts/size_cluster_taxonomy.json")
    args = ap.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text())
    gt_band = _gt_size_band_per_case(args.gt_eval, taxonomy)

    # Pull G9 predictions
    g9_preds = {}
    for line in args.g9_eval.open():
        rec = json.loads(line)
        cid = rec.get("case_id")
        cluster = (rec.get("constraints") or {}).get("size_cluster")
        g9_preds[cid] = cluster

    # Compute band-level metrics
    classes = ["door_L", "door_M", "window_S", "window_M", "window_L", "window_XL"]
    cm: dict = defaultdict(lambda: Counter())  # gt → Counter(pred)
    n_total = 0
    n_correct = 0
    n_emit = 0
    n_gt = 0

    for cid, gt in gt_band.items():
        if not gt:
            continue
        n_gt += 1
        pred_cluster = g9_preds.get(cid)
        pred_band = _band_for_cluster(pred_cluster, taxonomy)
        if not pred_band:
            cm[gt]["<no_emit>"] += 1
            continue
        n_emit += 1
        n_total += 1
        cm[gt][pred_band] += 1
        if gt == pred_band:
            n_correct += 1

    # Per-class precision / recall
    print(f"=== G9 size_cluster → size_band (apples-to-apples vs ResNet 6-way) ===")
    print(f"GT cases with size_band: {n_gt}")
    print(f"G9 emit + valid band: {n_emit}")
    print(f"G9 band-correct (when emitted): {n_correct}/{n_emit} = "
          f"{100 * n_correct / n_emit:.1f}%" if n_emit else "n/a")
    print(f"G9 band-correct (over all GT): {n_correct}/{n_gt} = "
          f"{100 * n_correct / n_gt:.1f}%")
    print()
    print(f"Per-class breakdown:")
    print(f"  {'band':>10}  {'recall':>8}  (GT n)  {'precision':>10}  (pred n)")
    for c in classes:
        gt_count = sum(cm[c].values())
        recall_n = cm[c].get(c, 0)
        # Precision
        pred_n = sum(v.get(c, 0) for v in cm.values())
        prec_n = cm[c].get(c, 0)
        recall = recall_n / gt_count if gt_count else 0.0
        precision = prec_n / pred_n if pred_n else 0.0
        print(f"  {c:>10}  {recall:>6.2f}    ({gt_count})    {precision:>8.2f}    ({pred_n})")

    print()
    print(f"Confusion matrix (rows = GT, cols = G9 prediction):")
    cols = classes + ["<no_emit>"]
    header = "".join(f"{c[:9]:>10}" for c in cols)
    print(f"{'GT':>10}  {header}")
    for c in classes:
        row = "".join(f"{cm[c].get(p, 0):>10}" for p in cols)
        print(f"{c:>10}  {row}")


if __name__ == "__main__":
    main()
