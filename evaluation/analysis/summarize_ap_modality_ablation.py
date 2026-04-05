#!/usr/bin/env python3
"""Summarize and plot Track A modality ablation for G3/G4/G7/Gemini."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_main"
DEFAULT_METRICS_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "modality_ablation_trackA" / "metrics"

MODEL_ORDER = ["g3_fullaug_r32", "g4_ultimate", "g7_position_context", "gemini_ap_v2"]
SLICE_ORDER = ["MC", "MC4D", "FP", "SITE", "MA"]
DISPLAY = {
    "g3_fullaug_r32": "G3",
    "g4_ultimate": "G4",
    "g7_position_context": "G7",
    "gemini_ap_v2": "Gemini AP v2",
    "MC": "Site + Floorplan + Chat",
    "MC4D": "Site + Floorplan + Chat + 4D Metadata",
    "FP": "Floorplan + Chat",
    "SITE": "Site + Chat",
    "MA": "Chat Only",
}
COLORS = {
    "g3_fullaug_r32": "#D32F2F",
    "g4_ultimate": "#B71C1C",
    "g7_position_context": "#6A1B9A",
    "gemini_ap_v2": "#1565C0",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(metrics_root: Path) -> List[dict]:
    rows: List[dict] = []
    for slice_key in SLICE_ORDER:
        slice_dir = metrics_root / slice_key
        for model_key in MODEL_ORDER:
            path = slice_dir / f"{model_key}__ap_metrics.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing metrics file: {path}")
            metrics = _load_json(path)
            rows.append(
                {
                    "slice": slice_key,
                    "model": model_key,
                    "parse_rate": metrics["json_parse_rate"] * 100.0,
                    "class_acc": metrics["class_acc"] * 100.0,
                    "storey_acc": metrics["storey_acc"] * 100.0,
                    "one_hop_spatial_accuracy": metrics["hop1_acc"] * 100.0,
                    "predicate_recall": metrics["predicate_recall"] * 100.0,
                    "direction_accuracy": metrics["direction_acc"] * 100.0,
                }
            )
    return rows


def write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slice",
                "model",
                "parse_rate",
                "class_acc",
                "storey_acc",
                "one_hop_spatial_accuracy",
                "predicate_recall",
                "direction_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: List[dict], out_path: Path) -> None:
    lines = [
        "# Track A Modality Ablation Summary",
        "",
        "| Slice | Model | Parse | Class | Storey | One-hop Spatial Accuracy | Predicate Recall | Direction Accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for slice_key in SLICE_ORDER:
        slice_rows = [r for r in rows if r["slice"] == slice_key]
        slice_rows.sort(key=lambda r: MODEL_ORDER.index(r["model"]))
        for row in slice_rows:
            lines.append(
                f"| {DISPLAY[slice_key]} | {DISPLAY[row['model']]} | "
                f"{row['parse_rate']:.1f} | {row['class_acc']:.1f} | {row['storey_acc']:.1f} | "
                f"{row['one_hop_spatial_accuracy']:.1f} | {row['predicate_recall']:.1f} | {row['direction_accuracy']:.1f} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: List[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharex=True)
    metrics = [
        ("one_hop_spatial_accuracy", "One-hop Spatial Accuracy"),
        ("predicate_recall", "Predicate Recall"),
        ("direction_accuracy", "Direction Accuracy"),
    ]
    x = list(range(len(SLICE_ORDER)))

    for ax, (metric_key, title) in zip(axes, metrics):
        for model_key in MODEL_ORDER:
            y = [next(r[metric_key] for r in rows if r["slice"] == slice_key and r["model"] == model_key) for slice_key in SLICE_ORDER]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=COLORS[model_key],
                label=DISPLAY[model_key],
            )
        ax.set_title(title)
        ax.set_xticks(x, [DISPLAY[s] for s in SLICE_ORDER], rotation=20, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(frameon=False, loc="lower left")
    fig.suptitle("Track A Modality Ablation on AP Held-out (G3, G4, G7, Gemini AP v2)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()

    rows = load_rows(args.metrics_root)
    write_csv(rows, args.out_dir / "fig09_trackA_modality_ablation_summary.csv")
    write_md(rows, args.out_dir / "fig09_trackA_modality_ablation_summary.md")
    plot(rows, args.out_dir / "fig09_trackA_modality_ablation.png")
    print(f"Wrote modality ablation summary to {args.out_dir}")


if __name__ == "__main__":
    main()
