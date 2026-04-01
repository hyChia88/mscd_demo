#!/usr/bin/env python3
"""Build combined dual-track evaluation summaries from per-group metrics JSONs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.track_registry import METRICS_DIR


def _load_metrics(metrics_dir: Path, suffix: str) -> Dict[str, dict]:
    result = {}
    for path in sorted(metrics_dir.glob(f"*{suffix}")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        result[payload["group"]] = payload
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    args = parser.parse_args()

    track_a = _load_metrics(args.metrics_dir, "__ap_metrics.json")
    track_b = _load_metrics(args.metrics_dir, "__unified_metrics.json")
    track_b2 = _load_metrics(args.metrics_dir, "__ap_e2e_metrics.json")

    winner_info = None
    winner_path = args.metrics_dir / "track_a_winner.json"
    if winner_path.exists():
        with winner_path.open("r", encoding="utf-8") as f:
            winner_info = json.load(f)

    md_lines: List[str] = ["# LoRA6-v2 Dual-Track Comparison Summary", ""]
    csv_rows: List[dict] = []

    if winner_info:
        winner = winner_info.get("winner")
        md_lines.append(f"- Track A winner: `{winner}`")
        md_lines.append("")

    if track_a:
        md_lines.extend(
            [
                "## Track A — AP Held-out Intermediate Evaluation",
                "",
                "| Group | Parse | Class | Storey | Hop-1 | Hop-2 | Pred P | Pred R | Dir |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, metrics in sorted(track_a.items()):
            md_lines.append(
                f"| {metrics['display_name']} | "
                f"{metrics['json_parse_rate']:.1%} | "
                f"{metrics['class_acc']:.1%} | "
                f"{metrics['storey_acc']:.1%} | "
                f"{metrics['hop1_acc']:.1%} | "
                f"{metrics['hop2_acc']:.1%} | "
                f"{metrics['predicate_precision']:.1%} | "
                f"{metrics['predicate_recall']:.1%} | "
                f"{metrics['direction_acc']:.1%} |"
            )
            csv_rows.append(
                {
                    "track": "track_a",
                    "group": group,
                    "display_name": metrics["display_name"],
                    "metric_1": metrics["hop1_acc"],
                    "metric_2": metrics["predicate_recall"],
                    "metric_3": metrics["direction_acc"],
                }
            )
        md_lines.append("")

    if track_b:
        md_lines.extend(
            [
                "## Track B-1 — Unified End-to-End Evaluation",
                "",
                "| Group | GT-in-Pool | Top-10 | Top-1 | MRR@10 | Avg Pool |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, metrics in sorted(track_b.items()):
            overall = metrics["overall"]
            md_lines.append(
                f"| {metrics['display_name']} | "
                f"{overall['gt_in_pct']:.1f}% | "
                f"{overall['top10_pct']:.1f}% | "
                f"{overall['top1_pct']:.1f}% | "
                f"{overall['mrr']:.4f} | "
                f"{overall['avg_pool']:.1f} |"
            )
            csv_rows.append(
                {
                    "track": "track_b",
                    "group": group,
                    "display_name": metrics["display_name"],
                    "metric_1": overall["gt_in_pct"],
                    "metric_2": overall["top1_pct"],
                    "metric_3": overall["mrr"],
                }
            )
        md_lines.append("")

    if track_b2:
        md_lines.extend(
            [
                "## Track B-2 — AP Held-out End-to-End Evaluation",
                "",
                "| Group | GT-in-Pool | Top-10 | Top-1 | MRR@10 | Avg Pool |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, metrics in sorted(track_b2.items()):
            overall = metrics["overall"]
            md_lines.append(
                f"| {metrics['display_name']} | "
                f"{overall['gt_in_pct']:.1f}% | "
                f"{overall['top10_pct']:.1f}% | "
                f"{overall['top1_pct']:.1f}% | "
                f"{overall['mrr']:.4f} | "
                f"{overall['avg_pool']:.1f} |"
            )
            csv_rows.append(
                {
                    "track": "track_b2",
                    "group": group,
                    "display_name": metrics["display_name"],
                    "metric_1": overall["gt_in_pct"],
                    "metric_2": overall["top1_pct"],
                    "metric_3": overall["mrr"],
                }
            )
        md_lines.append("")

    md_path = args.metrics_dir / "comparison_summary.md"
    csv_path = args.metrics_dir / "comparison_summary.csv"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["track", "group", "display_name", "metric_1", "metric_2", "metric_3"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
