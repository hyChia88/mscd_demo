#!/usr/bin/env python3
"""LoRA6 Group 4 minimal ablation + index builder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.analysis.group4_common import (
    DEFAULT_DATE_TAG,
    EXPERIMENT_ROOT,
    GROUP4_ROOT,
    GT_EVAL_PATH,
    METRICS_DIR,
    ensure_dir,
    extract_assistant_label,
    label_signature,
    load_jsonl,
    markdown_table,
    write_csv,
)


DEFAULT_OUT_DIR = (
    EXPERIMENT_ROOT / "group4_post-hoc_analysis" / "minimal_ablation" / DEFAULT_DATE_TAG
)
ORACLE_SUMMARY_CSV = EXPERIMENT_ROOT / "oracle_phase3_fixed" / "summary_20260401_191910_v2_lora_p0_union_p1.csv"

CURRENT_JSONL = {
    "train_canonical": GT_EVAL_PATH.parent / "lora6_v2_ap_train_canonical_m.jsonl",
    "train_aug": GT_EVAL_PATH.parent / "lora6_v2_ap_train_aug.jsonl",
    "eval_canonical": GT_EVAL_PATH,
}
RICHER_JSONL = {
    "train_canonical": GT_EVAL_PATH.parent / "lora6_v2_ap_train_canonical_m_g7.jsonl",
    "train_aug": GT_EVAL_PATH.parent / "lora6_v2_ap_train_aug_g7.jsonl",
    "eval_canonical": GT_EVAL_PATH.parent / "lora6_v2_ap_eval_canonical_m_g7.jsonl",
}


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label_file(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in load_jsonl(path):
        out[row["id"]] = extract_assistant_label(row)
    return out


def _oracle_track_b_metrics(path: Path) -> Dict[str, Any]:
    rows = path.read_text(encoding="utf-8").splitlines()
    out: Dict[str, Any] = {"top10_pct": None, "top1_pct": None, "mrr": None}
    for idx, line in enumerate(rows):
        if line.strip() != "=== OVERALL METRICS ===":
            continue
        for metric_line in rows[idx + 2 :]:
            if not metric_line.strip():
                break
            metric, value = metric_line.split(",", 1)
            raw = value.strip()
            if metric == "Top-10 Accuracy":
                out["top10_pct"] = round(float(raw) * 100, 1)
            elif metric == "Top-1 Accuracy":
                out["top1_pct"] = round(float(raw) * 100, 1)
            elif metric == "MRR@10":
                out["mrr"] = round(float(raw), 4)
    return out


def _dataset_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "n_records": 0,
            "direction_cov": 0.0,
            "object_subtype_cov": 0.0,
            "position_context_cov": 0.0,
            "sr_full_unique": 0,
            "label_full_unique": 0,
        }
    labels = _load_label_file(path)
    next_to_total = 0
    direction = 0
    subtype = 0
    position = 0
    for label in labels.values():
        rels = label.get("spatial_relations", []) or []
        next_to = [rel for rel in rels if str(rel.get("predicate") or "") == "NEXT_TO"]
        next_to_total += len(next_to)
        direction += sum(1 for rel in next_to if rel.get("direction"))
        subtype += sum(1 for rel in next_to if rel.get("object_subtype"))
        if label.get("position_context"):
            position += 1
    return {
        "path": str(path),
        "exists": True,
        "n_records": len(labels),
        "direction_cov": round(direction / next_to_total, 4) if next_to_total else 0.0,
        "object_subtype_cov": round(subtype / next_to_total, 4) if next_to_total else 0.0,
        "position_context_cov": round(position / len(labels), 4) if labels else 0.0,
        "sr_full_unique": len({label_signature(label, "sr_full") for label in labels.values()}),
        "label_full_unique": len({label_signature(label, "label_full") for label in labels.values()}),
        "labels": labels,
    }


def _compare_stats(current: Dict[str, Any], richer: Dict[str, Any], split: str) -> Dict[str, Any]:
    record_change_pct = 0.0
    if current.get("exists") and richer.get("exists"):
        current_labels = current["labels"]
        richer_labels = richer["labels"]
        shared_ids = sorted(set(current_labels) & set(richer_labels))
        if shared_ids:
            changed = sum(
                1
                for case_id in shared_ids
                if label_signature(current_labels[case_id], "label_full")
                != label_signature(richer_labels[case_id], "label_full")
            )
            record_change_pct = round(changed / len(shared_ids), 4)
    return {
        "split": split,
        "current_exists": current["exists"],
        "richer_exists": richer["exists"],
        "current_direction_cov": current["direction_cov"],
        "richer_direction_cov": richer["direction_cov"],
        "current_object_subtype_cov": current["object_subtype_cov"],
        "richer_object_subtype_cov": richer["object_subtype_cov"],
        "current_position_context_cov": current["position_context_cov"],
        "richer_position_context_cov": richer["position_context_cov"],
        "current_sr_full_unique": current["sr_full_unique"],
        "richer_sr_full_unique": richer["sr_full_unique"],
        "current_label_full_unique": current["label_full_unique"],
        "richer_label_full_unique": richer["label_full_unique"],
        "record_change_pct": record_change_pct,
    }


def _gate_decision(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not all(row["richer_exists"] for row in rows):
        return {
            "gate_pass": False,
            "reason": "richer_label_files_missing",
        }
    best_change = max(row["record_change_pct"] for row in rows)
    best_unique_gain = max(
        row["richer_label_full_unique"] - row["current_label_full_unique"] for row in rows
    )
    if best_change < 0.05 and best_unique_gain <= 0:
        return {
            "gate_pass": False,
            "reason": "richer_labels_not_materially_different",
        }
    return {
        "gate_pass": True,
        "reason": "richer_labels_materially_different",
    }


def _write_group4_index(root: Path) -> None:
    sections = []
    for category in ("oracle_ceiling", "model_diagnostics", "minimal_ablation"):
        category_root = root / category / DEFAULT_DATE_TAG
        files = sorted(p.name for p in category_root.glob("*") if p.is_file())
        sections.append((category, category_root, files))

    lines = [
        "# Group 4 Index",
        "",
        "This index links the three Group 4 output bundles.",
        "",
    ]
    for category, category_root, files in sections:
        lines.extend(
            [
                f"## {category}",
                "",
                f"- Directory: `{category_root.relative_to(root.parent.parent)}`",
                "",
            ]
        )
        if files:
            for filename in files:
                lines.append(f"- `{filename}`")
        else:
            lines.append("- No files found.")
        lines.append("")

    (root / f"group4_index_{DEFAULT_DATE_TAG}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--write-index", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    track_pairs = [
        ("G3", METRICS_DIR / "g3_fullaug_r32__ap_metrics.json", METRICS_DIR / "g3_fullaug_r32__ap_e2e_metrics.json"),
        ("G4", METRICS_DIR / "g4_ultimate__ap_metrics.json", METRICS_DIR / "g4_ultimate__ap_e2e_metrics.json"),
    ]
    g7_track_a = METRICS_DIR / "g7_position_context__ap_metrics.json"
    g7_track_b = METRICS_DIR / "g7_position_context__ap_e2e_metrics.json"
    if g7_track_a.exists() and g7_track_b.exists():
        track_pairs.append(("G7", g7_track_a, g7_track_b))

    track_a = {name: _load_metrics(track_a_path) for name, track_a_path, _ in track_pairs}
    track_b = {name: _load_metrics(track_b_path) for name, _, track_b_path in track_pairs}
    oracle_b = _oracle_track_b_metrics(ORACLE_SUMMARY_CSV)

    tension_rows = [
        {
            "system": "Oracle phase3 fixed",
            "hop1_acc": "",
            "predicate_recall": "",
            "direction_acc": "",
            "top10_pct": oracle_b["top10_pct"],
            "top1_pct": oracle_b["top1_pct"],
            "mrr": oracle_b["mrr"],
        }
    ]
    for model, _, _ in track_pairs:
        tension_rows.append(
            {
                "system": model,
                "hop1_acc": round(track_a[model]["hop1_acc"] * 100, 1),
                "predicate_recall": round(track_a[model]["predicate_recall"] * 100, 1),
                "direction_acc": round(track_a[model]["direction_acc"] * 100, 1),
                "top10_pct": track_b[model]["overall"]["top10_pct"],
                "top1_pct": track_b[model]["overall"]["top1_pct"],
                "mrr": track_b[model]["overall"]["mrr"],
            }
        )

    current_stats = {split: _dataset_stats(path) for split, path in CURRENT_JSONL.items()}
    richer_stats = {split: _dataset_stats(path) for split, path in RICHER_JSONL.items()}
    audit_rows = [
        _compare_stats(current_stats[split], richer_stats[split], split)
        for split in ("train_canonical", "train_aug", "eval_canonical")
    ]
    gate = _gate_decision(audit_rows)

    write_csv(
        out_dir / "g3_g4_tension_table.csv",
        tension_rows,
        ["system", "hop1_acc", "predicate_recall", "direction_acc", "top10_pct", "top1_pct", "mrr"],
    )
    write_csv(
        out_dir / "label_richness_audit.csv",
        audit_rows,
        [
            "split",
            "current_exists",
            "richer_exists",
            "current_direction_cov",
            "richer_direction_cov",
            "current_object_subtype_cov",
            "richer_object_subtype_cov",
            "current_position_context_cov",
            "richer_position_context_cov",
            "current_sr_full_unique",
            "richer_sr_full_unique",
            "current_label_full_unique",
            "richer_label_full_unique",
            "record_change_pct",
        ],
    )

    tension_md = [
        "# Minimal Ablation Tension Summary",
        "",
        markdown_table(
            ["System", "Hop-1", "Pred R", "Dir", "Top-10", "Top-1", "MRR@10"],
            [
                [
                    row["system"],
                    row["hop1_acc"] if row["hop1_acc"] != "" else "-",
                    row["predicate_recall"] if row["predicate_recall"] != "" else "-",
                    row["direction_acc"] if row["direction_acc"] != "" else "-",
                    row["top10_pct"],
                    row["top1_pct"],
                    row["mrr"],
                ]
                for row in tension_rows
            ],
        ),
        "",
        "Interpretation:",
        "- `G4` wins intermediate extraction metrics but still trails `G3` on downstream retrieval.",
        "- If present, `G7` shows whether planner-aware richer fingerprint training closes that gap.",
        "- The oracle row provides the bug-fixed Track B-2 ceiling for the same AP held-out benchmark.",
        "",
    ]
    (out_dir / "g3_g4_tension_summary.md").write_text("\n".join(tension_md), encoding="utf-8")

    audit_md = [
        "# Label Richness Audit",
        "",
        markdown_table(
            [
                "Split",
                "Current Exists",
                "Richer Exists",
                "Current Label Full",
                "Richer Label Full",
                "Changed Records",
            ],
            [
                [
                    row["split"],
                    row["current_exists"],
                    row["richer_exists"],
                    row["current_label_full_unique"],
                    row["richer_label_full_unique"],
                    f"{row['record_change_pct'] * 100:.1f}%",
                ]
                for row in audit_rows
            ],
        ),
        "",
        f"- Gate pass: `{gate['gate_pass']}`",
        f"- Reason: `{gate['reason']}`",
        "",
    ]
    (out_dir / "label_richness_audit.md").write_text("\n".join(audit_md), encoding="utf-8")

    minimal_md = [
        "# Minimal Ablation Summary",
        "",
        f"- Gate pass: `{gate['gate_pass']}`",
        f"- Reason: `{gate['reason']}`",
        "",
    ]
    if gate["gate_pass"] and "G7" in track_a and "G7" in track_b:
        g7_a = track_a["G7"]
        g7_b = track_b["G7"]["overall"]
        minimal_md.extend(
            [
                "The richer-label `G7_position_context` run has been evaluated and added to the tension table.",
                (
                    "Current result: "
                    f"`Hop-1 {g7_a['hop1_acc'] * 100:.1f}`, "
                    f"`Pred R {g7_a['predicate_recall'] * 100:.1f}`, "
                    f"`Dir {g7_a['direction_acc'] * 100:.1f}`, "
                    f"`Top-10 {g7_b['top10_pct']:.1f}`, "
                    f"`MRR@10 {g7_b['mrr']:.4f}`."
                ),
                "",
            ]
        )
    elif gate["gate_pass"]:
        minimal_md.extend(
            [
                "A richer-label `G7_position_context` run is warranted, but this script currently only prepares the audit bundle and baseline summaries.",
                "",
            ]
        )
    else:
        minimal_md.extend(
            [
                "No richer-label training should be launched yet.",
                "Current output bundle stops at audit + baseline tension summary.",
                "",
            ]
        )
    (out_dir / "minimal_ablation_summary.md").write_text("\n".join(minimal_md), encoding="utf-8")

    if args.write_index:
        _write_group4_index(GROUP4_ROOT)

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
