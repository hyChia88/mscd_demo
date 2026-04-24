#!/usr/bin/env python3
"""LoRA6 Group 4 collapse + shortcut-like diagnostics."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.analysis.group4_common import (
    CASES_PATH,
    DEFAULT_DATE_TAG,
    EXPERIMENT_ROOT,
    G3_PRED_PATH,
    G4_PRED_PATH,
    G7_PRED_PATH,
    GT_EVAL_PATH,
    ensure_dir,
    label_signature,
    load_cases_map,
    load_gt_eval_labels,
    load_prediction_constraints,
    topology_family,
    write_csv,
    write_json,
)


DEFAULT_OUT_DIR = (
    EXPERIMENT_ROOT / "group4_post-hoc_analysis" / "model_diagnostics" / DEFAULT_DATE_TAG
)


def _count_unique(labels: Dict[str, dict], level: str) -> int:
    return len({label_signature(label, level) for label in labels.values()})


def _flatten_next_to(label: dict) -> List[dict]:
    return [
        rel
        for rel in label.get("spatial_relations", []) or []
        if str(rel.get("predicate") or "") == "NEXT_TO"
    ]


def _field_usage_stats(gt_labels: Dict[str, dict], pred_labels: Dict[str, dict], model_name: str) -> Dict[str, Any]:
    gt_next_to_total = 0
    pred_next_to_total = 0
    pred_direction_nonempty = 0
    pred_subtype_nonempty = 0
    gt_direction_total = 0
    gt_subtype_total = 0
    direction_match = 0
    subtype_match = 0

    for case_id, gt_label in gt_labels.items():
        pred_label = pred_labels.get(case_id, {})
        gt_next = _flatten_next_to(gt_label)
        pred_next = _flatten_next_to(pred_label)
        gt_next_to_total += len(gt_next)
        pred_next_to_total += len(pred_next)
        pred_direction_nonempty += sum(1 for rel in pred_next if rel.get("direction"))
        pred_subtype_nonempty += sum(1 for rel in pred_next if rel.get("object_subtype"))

        used = [False] * len(pred_next)
        for gt_rel in gt_next:
            gt_direction = str(gt_rel.get("direction") or "")
            gt_subtype = str(gt_rel.get("object_subtype") or "").strip().lower()
            if gt_direction:
                gt_direction_total += 1
            if gt_subtype:
                gt_subtype_total += 1
            for idx, pred_rel in enumerate(pred_next):
                if used[idx]:
                    continue
                if str(pred_rel.get("object_type") or "") != str(gt_rel.get("object_type") or ""):
                    continue
                used[idx] = True
                if gt_direction and str(pred_rel.get("direction") or "") == gt_direction:
                    direction_match += 1
                if gt_subtype and str(pred_rel.get("object_subtype") or "").strip().lower() == gt_subtype:
                    subtype_match += 1
                break

    return {
        "model": model_name,
        "gt_next_to_total": gt_next_to_total,
        "pred_next_to_total": pred_next_to_total,
        "pred_direction_nonempty_rate": round(pred_direction_nonempty / pred_next_to_total, 4) if pred_next_to_total else 0.0,
        "pred_object_subtype_nonempty_rate": round(pred_subtype_nonempty / pred_next_to_total, 4) if pred_next_to_total else 0.0,
        "direction_match_rate_when_gt_has_direction": round(direction_match / gt_direction_total, 4) if gt_direction_total else 0.0,
        "object_subtype_match_rate_when_gt_has_subtype": round(subtype_match / gt_subtype_total, 4) if gt_subtype_total else 0.0,
    }


def _pair_identity_rows(
    cases: Dict[str, dict],
    gt_labels: Dict[str, dict],
    model_labels: Dict[str, dict],
    model_name: str,
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for case_id, case in cases.items():
        label = gt_labels.get(case_id)
        if not label:
            continue
        family = topology_family(label.get("spatial_relations", []) or [])
        if family != "triad:FILLS+NEXT_TO+NEXT_TO":
            continue
        storey = str(label.get("storey_name") or "")
        ifc_class = str(label.get("ifc_class") or "")
        buckets[(family, storey, ifc_class)].append(case_id)

    rows: List[Dict[str, Any]] = []
    for (family, storey, ifc_class), case_ids in sorted(buckets.items()):
        for case_a, case_b in combinations(sorted(case_ids), 2):
            gt_a = gt_labels[case_a]
            gt_b = gt_labels[case_b]
            pred_a = model_labels.get(case_a, {})
            pred_b = model_labels.get(case_b, {})
            rows.append(
                {
                    "model": model_name,
                    "family": family,
                    "storey_name": storey,
                    "ifc_class": ifc_class,
                    "case_a": case_a,
                    "case_b": case_b,
                    "gt_sr_full_same": label_signature(gt_a, "sr_full") == label_signature(gt_b, "sr_full"),
                    "gt_label_full_same": label_signature(gt_a, "label_full") == label_signature(gt_b, "label_full"),
                    "sr_pred_obj_same": label_signature(pred_a, "pred_obj") == label_signature(pred_b, "pred_obj"),
                    "sr_pred_obj_dir_same": label_signature(pred_a, "pred_obj_dir") == label_signature(pred_b, "pred_obj_dir"),
                    "sr_full_same": label_signature(pred_a, "sr_full") == label_signature(pred_b, "sr_full"),
                    "label_full_same": label_signature(pred_a, "label_full") == label_signature(pred_b, "label_full"),
                }
            )
    return rows


def _identity_summary(rows: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    differing_gt = [row for row in rows if not row["gt_label_full_same"]]
    total = len(differing_gt)
    same_pred_obj = sum(1 for row in differing_gt if row["sr_pred_obj_same"])
    same_pred_obj_dir = sum(1 for row in differing_gt if row["sr_pred_obj_dir_same"])
    same_sr_full = sum(1 for row in differing_gt if row["sr_full_same"])
    same_label_full = sum(1 for row in differing_gt if row["label_full_same"])
    return {
        "model": model_name,
        "n_pairs_with_gt_difference": total,
        "pred_obj_identity_rate_when_gt_differs": round(same_pred_obj / total, 4) if total else 0.0,
        "pred_obj_dir_identity_rate_when_gt_differs": round(same_pred_obj_dir / total, 4) if total else 0.0,
        "sr_full_identity_rate_when_gt_differs": round(same_sr_full / total, 4) if total else 0.0,
        "label_full_identity_rate_when_gt_differs": round(same_label_full / total, 4) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--gt-eval", type=Path, default=GT_EVAL_PATH)
    parser.add_argument("--g3-pred", type=Path, default=G3_PRED_PATH)
    parser.add_argument("--g4-pred", type=Path, default=G4_PRED_PATH)
    parser.add_argument("--g7-pred", type=Path, default=G7_PRED_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    cases = load_cases_map(args.cases)
    gt_labels = load_gt_eval_labels(args.gt_eval)
    model_inputs: List[Tuple[str, Path]] = [
        ("G3", args.g3_pred),
        ("G4", args.g4_pred),
    ]
    if args.g7_pred.exists():
        model_inputs.append(("G7", args.g7_pred))

    model_labels = {
        model_name: load_prediction_constraints(path)
        for model_name, path in model_inputs
    }

    diversity_rows = []
    for name, labels in [("GT", gt_labels), *model_labels.items()]:
        diversity_rows.append(
            {
                "source": name,
                "n_cases": len(labels),
                "predicate_only": _count_unique(labels, "predicate_only"),
                "pred_obj": _count_unique(labels, "pred_obj"),
                "pred_obj_dir": _count_unique(labels, "pred_obj_dir"),
                "sr_full": _count_unique(labels, "sr_full"),
                "label_full": _count_unique(labels, "label_full"),
            }
        )

    field_usage_rows = [
        _field_usage_stats(gt_labels, labels, model_name)
        for model_name, labels in model_labels.items()
    ]

    pair_rows = []
    for model_name, labels in model_labels.items():
        pair_rows.extend(_pair_identity_rows(cases, gt_labels, labels, model_name))

    pair_summary_rows = [
        _identity_summary(
            [row for row in pair_rows if row["model"] == model_name],
            model_name,
        )
        for model_name, _ in model_inputs
    ]

    write_csv(
        out_dir / "collapse_diversity.csv",
        diversity_rows,
        ["source", "n_cases", "predicate_only", "pred_obj", "pred_obj_dir", "sr_full", "label_full"],
    )
    write_json(out_dir / "collapse_diversity.json", diversity_rows)
    write_csv(
        out_dir / "field_usage_summary.csv",
        field_usage_rows,
        [
            "model",
            "gt_next_to_total",
            "pred_next_to_total",
            "pred_direction_nonempty_rate",
            "pred_object_subtype_nonempty_rate",
            "direction_match_rate_when_gt_has_direction",
            "object_subtype_match_rate_when_gt_has_subtype",
        ],
    )
    write_csv(
        out_dir / "matched_pair_identity.csv",
        pair_rows,
        [
            "model",
            "family",
            "storey_name",
            "ifc_class",
            "case_a",
            "case_b",
            "gt_sr_full_same",
            "gt_label_full_same",
            "sr_pred_obj_same",
            "sr_pred_obj_dir_same",
            "sr_full_same",
            "label_full_same",
        ],
    )
    write_json(out_dir / "matched_pair_identity_summary.json", pair_summary_rows)

    diversity_table = [
        [row["source"], row["predicate_only"], row["pred_obj"], row["pred_obj_dir"], row["sr_full"], row["label_full"]]
        for row in diversity_rows
    ]
    pair_table = [
        [
            row["model"],
            row["n_pairs_with_gt_difference"],
            f"{row['pred_obj_identity_rate_when_gt_differs'] * 100:.1f}%",
            f"{row['pred_obj_dir_identity_rate_when_gt_differs'] * 100:.1f}%",
            f"{row['sr_full_identity_rate_when_gt_differs'] * 100:.1f}%",
            f"{row['label_full_identity_rate_when_gt_differs'] * 100:.1f}%",
        ]
        for row in pair_summary_rows
    ]

    md = [
        "# Model Behavior Summary",
        "",
        "## Collapse Diversity",
        "",
        "| Source | Predicate Only | Pred+Obj | Pred+Obj+Dir | SR Full | Label Full |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in diversity_table:
        md.append("| " + " | ".join(str(cell) for cell in row) + " |")
    md.extend(
        [
            "",
            "## Matched-Case Identity When GT Differs",
            "",
            "| Model | Pairs | Pred+Obj Same | Pred+Obj+Dir Same | SR Full Same | Label Full Same |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in pair_table:
        md.append("| " + " | ".join(str(cell) for cell in row) + " |")
    md.extend(
        [
            "",
            "Interpretation:",
            "- `prediction diversity << GT diversity` indicates residual template collapse.",
            "- High matched-pair identity when GT differs indicates position-insensitive shortcut-like behavior.",
            "- `Pred+Obj+Dir` and `SR Full` are the most useful columns for LoRA6, because they test whether direction and subtype are actually being used.",
            "",
        ]
    )
    (out_dir / "model_behavior_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
