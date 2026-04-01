#!/usr/bin/env python3
"""Score LoRA6-v2 AP held-out predictions against assembled AP eval GT.

This script is the Track A scorer for the LoRA6-v2 dual-track evaluation plan.
It reproduces the same metric family used during training-time inference checks
while adding:
  - per-predicate breakdown
  - per-storey breakdown
  - confusion summaries
  - winner selection across G0~G3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.track_registry import (
    G_SERIES_ORDER,
    GROUP_DISPLAY,
    METRICS_DIR,
    TRACK_A_ORDER,
)
DEFAULT_GT = (
    PROJECT_ROOT.parent
    / "data_curation"
    / "datasets"
    / "synth_v0.5_ap"
    / "train"
    / "lora6_v2_ap_eval_canonical_m.jsonl"
)


def _parse_kv_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected KEY=VALUE, got: {value}"
        )
    key, raw = value.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key or not raw:
        raise argparse.ArgumentTypeError(
            f"Expected KEY=VALUE, got: {value}"
        )
    return key, raw


def _normalize_spatial_relations(value: Any) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        return [{"predicate": value}]
    if isinstance(value, list):
        normalized: List[dict] = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"predicate": item})
        return normalized
    return []


def _relation_predicate(rel: Any) -> str:
    if isinstance(rel, dict):
        pred = rel.get("predicate", "")
        return pred if isinstance(pred, str) else ""
    if isinstance(rel, str):
        return rel
    return ""


def _relation_signature(rel: Any) -> Tuple[str, str, str]:
    if not isinstance(rel, dict):
        return ("", "", "")
    return (
        str(rel.get("predicate") or ""),
        str(rel.get("object_type") or ""),
        str(rel.get("direction") or ""),
    )


def _safe_ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _json_default(value: Any):
    if isinstance(value, Counter):
        return dict(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _extract_assistant_label(case: dict) -> dict:
    for message in case.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return json.loads(content)
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return json.loads(first["text"])
    raise ValueError(f"Assistant label missing for eval case: {case.get('id')}")


def load_gt(gt_path: Path) -> Tuple[List[str], Dict[str, dict]]:
    case_order: List[str] = []
    gt_by_case: Dict[str, dict] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("id") or row.get("base_case_id") or row.get("case_id")
            if not case_id:
                raise ValueError(f"Missing case id in GT row: {row.keys()}")
            label = _extract_assistant_label(row)
            gt_by_case[case_id] = {
                "label": label,
                "predicate": row.get("predicate"),
                "base_case_id": row.get("base_case_id"),
                "scale": row.get("scale"),
                "modality": row.get("modality"),
                "text_tier": row.get("text_tier"),
            }
            case_order.append(case_id)
    return case_order, gt_by_case


def load_predictions(pred_path: Path) -> Dict[str, dict]:
    preds: Dict[str, dict] = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("case_id")
            if case_id:
                preds[case_id] = row
    return preds


def score_group(
    group_key: str,
    gt_order: List[str],
    gt_by_case: Dict[str, dict],
    pred_path: Path,
    *,
    gt_path: Path,
    eval_loss: Optional[float] = None,
) -> dict:
    preds = load_predictions(pred_path)

    n_total = len(gt_order)
    n_valid_json = 0
    n_class_match = 0
    n_storey_match = 0
    n_spatial_match = 0
    n_spatial_total = 0
    n_hop2_match = 0
    n_hop2_total = 0
    n_false_positive = 0
    n_attr_only = 0
    n_pred_match = 0
    n_pred_total = 0
    n_predicted_total = 0
    n_direction_match = 0
    n_direction_total = 0
    n_over_pred = 0
    n_under_pred = 0
    missing_predictions = 0

    per_pred_total: Counter = Counter()
    per_pred_correct: Counter = Counter()
    per_storey = defaultdict(lambda: {"cases": 0, "class_match": 0, "storey_match": 0, "hop1_match": 0})
    predicate_confusion = defaultdict(Counter)
    class_confusion = defaultdict(Counter)
    direction_confusion = defaultdict(Counter)

    parse_fail_ids: List[str] = []
    missing_ids: List[str] = []

    for case_id in gt_order:
        gt_row = gt_by_case[case_id]
        gt = gt_row["label"]

        pred_entry = preds.get(case_id)
        parsed: Optional[dict] = None
        if pred_entry is None:
            missing_predictions += 1
            missing_ids.append(case_id)
        else:
            if pred_entry.get("status") == "OK" and isinstance(pred_entry.get("constraints"), dict):
                parsed = pred_entry["constraints"]
                n_valid_json += 1
            else:
                parse_fail_ids.append(case_id)

        gt_class = gt.get("ifc_class")
        pred_class = parsed.get("ifc_class") if parsed else None
        class_confusion[str(gt_class or "__NULL__")][str(pred_class or "__NONE__")] += 1
        if parsed and pred_class == gt_class:
            n_class_match += 1

        gt_storey = gt.get("storey_name")
        pred_storey = parsed.get("storey_name") if parsed else None
        if parsed and pred_storey == gt_storey:
            n_storey_match += 1

        storey_key = str(gt_storey or "__NULL__")
        per_storey[storey_key]["cases"] += 1
        if parsed and pred_class == gt_class:
            per_storey[storey_key]["class_match"] += 1
        if parsed and pred_storey == gt_storey:
            per_storey[storey_key]["storey_match"] += 1

        gt_rels = _normalize_spatial_relations(gt.get("spatial_relations", []))
        pred_rels = _normalize_spatial_relations(parsed.get("spatial_relations", [])) if parsed else []

        if gt_rels:
            n_spatial_total += 1
            gt_sig1 = _relation_signature(gt_rels[0])
            pred_sig1 = _relation_signature(pred_rels[0]) if pred_rels else ("", "", "")
            gt_pred1 = gt_sig1[0] or "__NONE__"
            pred_pred1 = pred_sig1[0] or "__NONE__"
            predicate_confusion[gt_pred1][pred_pred1] += 1
            per_pred_total[gt_pred1] += 1
            if gt_sig1 == pred_sig1:
                n_spatial_match += 1
                per_pred_correct[gt_pred1] += 1
                per_storey[storey_key]["hop1_match"] += 1

            if len(gt_rels) >= 2:
                n_hop2_total += 1
                gt_sig2 = _relation_signature(gt_rels[1])
                pred_sig2 = _relation_signature(pred_rels[1]) if len(pred_rels) >= 2 else ("", "", "")
                gt_pred2 = gt_sig2[0] or "__NONE__"
                per_pred_total[f"{gt_pred2}_hop2"] += 1
                if gt_sig2 == pred_sig2:
                    n_hop2_match += 1
                    per_pred_correct[f"{gt_pred2}_hop2"] += 1

            gt_pred_counts = Counter(_relation_predicate(rel) for rel in gt_rels)
            pred_pred_counts = Counter(_relation_predicate(rel) for rel in pred_rels)
            common = sum(min(gt_pred_counts[p], pred_pred_counts[p]) for p in gt_pred_counts)
            n_pred_match += common
            n_pred_total += sum(gt_pred_counts.values())
            n_predicted_total += sum(pred_pred_counts.values())

            for gt_rel in gt_rels:
                gt_dir = gt_rel.get("direction")
                if not gt_dir:
                    continue
                n_direction_total += 1
                gt_predicate = gt_rel.get("predicate")
                gt_object_type = gt_rel.get("object_type")
                pred_dirs = [
                    rel.get("direction")
                    for rel in pred_rels
                    if rel.get("predicate") == gt_predicate
                    and rel.get("object_type") == gt_object_type
                ]
                chosen_pred_dir = next((d for d in pred_dirs if d), "__NONE__")
                direction_confusion[str(gt_dir)][str(chosen_pred_dir)] += 1
                if gt_dir in pred_dirs:
                    n_direction_match += 1

            n_over_pred += max(0, len(pred_rels) - len(gt_rels))
            n_under_pred += max(0, len(gt_rels) - len(pred_rels))
        else:
            n_attr_only += 1
            if pred_rels:
                n_false_positive += 1

    per_predicate = {}
    for pred_name in sorted(per_pred_total):
        total = per_pred_total[pred_name]
        correct = per_pred_correct.get(pred_name, 0)
        per_predicate[pred_name] = {
            "correct": correct,
            "total": total,
            "hop1_acc": _safe_ratio(correct, total),
        }

    per_storey_summary = {}
    for storey, counts in sorted(per_storey.items()):
        total = counts["cases"]
        per_storey_summary[storey] = {
            **counts,
            "class_acc": _safe_ratio(counts["class_match"], total),
            "storey_acc": _safe_ratio(counts["storey_match"], total),
            "hop1_acc": _safe_ratio(counts["hop1_match"], total),
        }

    metrics = {
        "group": group_key,
        "display_name": GROUP_DISPLAY.get(group_key, group_key),
        "prediction_path": str(pred_path),
        "gt_path": str(gt_path),
        "eval_loss": eval_loss,
        "n_cases": n_total,
        "n_predictions": len(preds),
        "missing_predictions": missing_predictions,
        "missing_prediction_ids": missing_ids,
        "json_parse_rate": _safe_ratio(n_valid_json, n_total),
        "class_acc": _safe_ratio(n_class_match, n_total),
        "storey_acc": _safe_ratio(n_storey_match, n_total),
        "hop1_acc": _safe_ratio(n_spatial_match, n_spatial_total),
        "hop2_acc": _safe_ratio(n_hop2_match, n_hop2_total),
        "predicate_precision": _safe_ratio(n_pred_match, n_predicted_total),
        "predicate_recall": _safe_ratio(n_pred_match, n_pred_total),
        "direction_acc": _safe_ratio(n_direction_match, n_direction_total),
        "over_pred_rate": _safe_ratio(n_over_pred, n_predicted_total),
        "under_pred_rate": _safe_ratio(n_under_pred, n_pred_total),
        "legacy_fp_rate": _safe_ratio(n_false_positive, n_attr_only),
        "counts": {
            "valid_json": n_valid_json,
            "class_match": n_class_match,
            "storey_match": n_storey_match,
            "hop1_match": n_spatial_match,
            "hop1_total": n_spatial_total,
            "hop2_match": n_hop2_match,
            "hop2_total": n_hop2_total,
            "predicate_match": n_pred_match,
            "predicate_total": n_pred_total,
            "predicted_total": n_predicted_total,
            "direction_match": n_direction_match,
            "direction_total": n_direction_total,
            "over_pred": n_over_pred,
            "under_pred": n_under_pred,
            "attr_only_cases": n_attr_only,
            "attr_only_false_positive": n_false_positive,
        },
        "per_predicate": per_predicate,
        "per_storey": per_storey_summary,
        "confusions": {
            "predicate": {k: dict(v) for k, v in sorted(predicate_confusion.items())},
            "ifc_class": {k: dict(v) for k, v in sorted(class_confusion.items())},
            "direction": {k: dict(v) for k, v in sorted(direction_confusion.items())},
        },
        "parse_fail_case_ids": parse_fail_ids,
    }
    return metrics


def _winner_sort_key(metrics: dict) -> Tuple[float, float, float, float, float, float, float]:
    per_pred = metrics.get("per_predicate", {})
    next_to = per_pred.get("NEXT_TO", {}).get("hop1_acc", 0.0)
    adjacent = per_pred.get("ADJACENT_TO", {}).get("hop1_acc", 0.0)
    eval_loss = metrics.get("eval_loss")
    eval_loss_score = -float(eval_loss) if isinstance(eval_loss, (float, int)) else float("-inf")
    return (
        float(metrics.get("hop1_acc", 0.0)),
        float(metrics.get("predicate_recall", 0.0)),
        float(next_to),
        float(adjacent),
        float(metrics.get("hop2_acc", 0.0)),
        float(metrics.get("direction_acc", 0.0)),
        eval_loss_score,
    )


def choose_g_series_winner(metrics_by_group: Dict[str, dict]) -> Optional[str]:
    candidates = [
        metrics_by_group[group]
        for group in G_SERIES_ORDER
        if group in metrics_by_group
    ]
    if not candidates:
        return None
    ranked = sorted(candidates, key=_winner_sort_key, reverse=True)
    return ranked[0]["group"]


def _summary_rows(metrics_by_group: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for group in TRACK_A_ORDER:
        metrics = metrics_by_group.get(group)
        if not metrics:
            continue
        per_pred = metrics.get("per_predicate", {})
        rows.append(
            {
                "group": group,
                "display_name": metrics["display_name"],
                "json_parse_rate": metrics["json_parse_rate"],
                "class_acc": metrics["class_acc"],
                "storey_acc": metrics["storey_acc"],
                "hop1_acc": metrics["hop1_acc"],
                "hop2_acc": metrics["hop2_acc"],
                "predicate_precision": metrics["predicate_precision"],
                "predicate_recall": metrics["predicate_recall"],
                "direction_acc": metrics["direction_acc"],
                "next_to_hop1": per_pred.get("NEXT_TO", {}).get("hop1_acc", 0.0),
                "adjacent_to_hop1": per_pred.get("ADJACENT_TO", {}).get("hop1_acc", 0.0),
                "eval_loss": metrics.get("eval_loss"),
                "missing_predictions": metrics.get("missing_predictions", 0),
            }
        )
    return rows


def write_group_metrics(metrics_by_group: Dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for group, metrics in metrics_by_group.items():
        path = out_dir / f"{group}__ap_metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)


def write_summary_files(metrics_by_group: Dict[str, dict], out_dir: Path) -> Optional[str]:
    rows = _summary_rows(metrics_by_group)
    if not rows:
        return None

    winner = choose_g_series_winner(metrics_by_group)
    md_path = out_dir / "track_a_summary.md"
    csv_path = out_dir / "track_a_summary.csv"
    winner_path = out_dir / "track_a_winner.json"

    with winner_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "winner": winner,
                "winner_display_name": GROUP_DISPLAY.get(winner, winner) if winner else None,
                "ranking_rule": {
                    "primary": "hop1_acc",
                    "tie_breakers": [
                        "predicate_recall",
                        "NEXT_TO hop1",
                        "ADJACENT_TO hop1",
                        "hop2_acc",
                        "direction_acc",
                        "eval_loss (lower is better, optional)",
                    ],
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    header = [
        "group",
        "display_name",
        "json_parse_rate",
        "class_acc",
        "storey_acc",
        "hop1_acc",
        "hop2_acc",
        "predicate_precision",
        "predicate_recall",
        "direction_acc",
        "next_to_hop1",
        "adjacent_to_hop1",
        "eval_loss",
        "missing_predictions",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Track A — AP Held-out Intermediate Evaluation",
        "",
        f"- GT: `{DEFAULT_GT}`",
        f"- Winner: `{winner}` ({GROUP_DISPLAY.get(winner, winner)})" if winner else "- Winner: not available",
        "",
        "| Group | Parse | Class | Storey | Hop-1 | Hop-2 | Pred P | Pred R | Dir | NEXT_TO | ADJ | Eval Loss | Missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        eval_loss = row["eval_loss"]
        eval_loss_str = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else "-"
        lines.append(
            f"| {row['display_name']} | "
            f"{row['json_parse_rate']:.1%} | "
            f"{row['class_acc']:.1%} | "
            f"{row['storey_acc']:.1%} | "
            f"{row['hop1_acc']:.1%} | "
            f"{row['hop2_acc']:.1%} | "
            f"{row['predicate_precision']:.1%} | "
            f"{row['predicate_recall']:.1%} | "
            f"{row['direction_acc']:.1%} | "
            f"{row['next_to_hop1']:.1%} | "
            f"{row['adjacent_to_hop1']:.1%} | "
            f"{eval_loss_str} | "
            f"{row['missing_predictions']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT,
        help="Assembled AP held-out GT JSONL.",
    )
    parser.add_argument(
        "--pred",
        action="append",
        default=[],
        metavar="GROUP=PATH",
        help="Prediction JSONL to score. Can be repeated.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Optional directory to auto-discover *_ap_eval.jsonl files.",
    )
    parser.add_argument(
        "--eval-loss",
        action="append",
        default=[],
        metavar="GROUP=FLOAT",
        help="Optional eval loss for winner tie-break metadata.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=METRICS_DIR,
        help="Directory for per-group metrics JSON and summary files.",
    )
    args = parser.parse_args()

    pred_map: Dict[str, Path] = {}
    for raw in args.pred:
        key, value = _parse_kv_arg(raw)
        pred_map[key] = Path(value)

    if args.pred_dir:
        for path in sorted(args.pred_dir.glob("*__ap_eval.jsonl")):
            group_key = path.name.replace("__ap_eval.jsonl", "")
            pred_map.setdefault(group_key, path)

    if not pred_map:
        raise SystemExit("No prediction files provided. Use --pred or --pred-dir.")

    eval_losses: Dict[str, float] = {}
    for raw in args.eval_loss:
        key, value = _parse_kv_arg(raw)
        eval_losses[key] = float(value)

    gt_order, gt_by_case = load_gt(args.gt)
    metrics_by_group: Dict[str, dict] = {}
    for group_key, pred_path in sorted(pred_map.items(), key=lambda kv: TRACK_A_ORDER.index(kv[0]) if kv[0] in TRACK_A_ORDER else 999):
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction JSONL not found: {pred_path}")
        metrics = score_group(
            group_key,
            gt_order,
            gt_by_case,
            pred_path,
            gt_path=args.gt,
            eval_loss=eval_losses.get(group_key),
        )
        metrics_by_group[group_key] = metrics
        print(
            f"[{group_key}] parse={metrics['json_parse_rate']:.1%} "
            f"class={metrics['class_acc']:.1%} "
            f"storey={metrics['storey_acc']:.1%} "
            f"hop1={metrics['hop1_acc']:.1%} "
            f"hop2={metrics['hop2_acc']:.1%} "
            f"pred_recall={metrics['predicate_recall']:.1%}"
        )

    write_group_metrics(metrics_by_group, args.out_dir)
    winner = write_summary_files(metrics_by_group, args.out_dir)

    print("\nTrack A scoring complete.")
    print(f"  Metrics dir: {args.out_dir}")
    if winner:
        print(f"  Winner: {winner} ({GROUP_DISPLAY.get(winner, winner)})")


if __name__ == "__main__":
    main()
