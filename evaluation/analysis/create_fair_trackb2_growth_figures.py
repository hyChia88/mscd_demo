#!/usr/bin/env python3
"""Create mixed-regime AP held-out growth figures for thesis milestones."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPECIAL_ROOT = PROJECT_ROOT / "output" / "ap_lora2_vs_lora5_floorplan_only"
CANONICAL_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331"
MAIN_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_main"
APPENDIX_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_appendix"

MIXED_MODELS = [
    {
        "key": "lora5_fp",
        "label": "LoRA5-r32\n(FP only)",
        "display": "LoRA5-r32 FP",
        "input_regime": "Floorplan + Chat (no site)",
        "source_type": "special_csv",
        "summary": SPECIAL_ROOT / "e2e" / "lora5r32_apheldout_FP" / "summary_20260405_030157_v2_lora_p0_union_p1.csv",
        "precomputed": SPECIAL_ROOT / "precomputed" / "lora5r32_apheldout_FP.jsonl",
        "color": "#FB8C00",
    },
    {
        "key": "gemini_mm",
        "label": "Gemini AP\n(MM)",
        "display": "Gemini AP (MM)",
        "input_regime": "Canonical multimodal",
        "source_type": "canonical_json",
        "e2e_json": CANONICAL_ROOT / "metrics" / "gemini_ap__ap_e2e_metrics.json",
        "precomputed": CANONICAL_ROOT / "gemini_ap__ap_eval_v2.jsonl",
        "color": "#1565C0",
    },
    {
        "key": "g3_mm",
        "label": "G3\n(MM)",
        "display": "G3 FullAug r32 (MM)",
        "input_regime": "Canonical multimodal",
        "source_type": "canonical_json",
        "e2e_json": CANONICAL_ROOT / "metrics" / "g3_fullaug_r32__ap_e2e_metrics.json",
        "precomputed": CANONICAL_ROOT / "g3_fullaug_r32__ap_eval.jsonl",
        "color": "#D32F2F",
    },
    {
        "key": "g4_mm",
        "label": "G4\n(MM)",
        "display": "G4 Ultimate (MM)",
        "input_regime": "Canonical multimodal",
        "source_type": "canonical_json",
        "e2e_json": CANONICAL_ROOT / "metrics" / "g4_ultimate__ap_e2e_metrics.json",
        "precomputed": CANONICAL_ROOT / "g4_ultimate__ap_eval.jsonl",
        "color": "#8E2424",
    },
    {
        "key": "g7_mm",
        "label": "G7\n(MM)",
        "display": "G7 Position Context (MM)",
        "input_regime": "Canonical multimodal",
        "source_type": "canonical_json",
        "e2e_json": CANONICAL_ROOT / "metrics" / "g7_position_context__ap_e2e_metrics.json",
        "precomputed": CANONICAL_ROOT / "g7_position_context__ap_eval.jsonl",
        "color": "#6A1B9A",
    },
]


def _load_summary_metrics(path: Path) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_metrics = False
        for row in reader:
            if not row:
                continue
            if row[0].startswith("=== OVERALL METRICS"):
                in_metrics = True
                continue
            if row[0].startswith("=== V2 DIAGNOSTIC"):
                break
            if not in_metrics or row[0] == "Metric":
                continue
            if len(row) >= 2:
                metrics[row[0]] = row[1]
    return metrics


def _parse_gt_in_pool(text: str) -> float:
    pct = text.split("(")[-1].rstrip(")")
    return float(pct.rstrip("%"))


def _load_special_row(spec: Dict[str, str]) -> Dict[str, float | str]:
    metrics = _load_summary_metrics(Path(spec["summary"]))
    return {
        "top10": float(metrics["Top-10 Accuracy"]) * 100.0,
        "top1": float(metrics["Top-1 Accuracy"]) * 100.0,
        "mrr10": float(metrics["MRR@10"]),
        "gt_in_pool": _parse_gt_in_pool(metrics["GT-in-Pool"]),
    }


def _load_canonical_row(spec: Dict[str, str]) -> Dict[str, float | str]:
    with Path(spec["e2e_json"]).open("r", encoding="utf-8") as f:
        data = json.load(f)
    overall = data["overall"]
    return {
        "top10": float(overall["top10_pct"]),
        "top1": float(overall["top1_pct"]),
        "mrr10": float(overall["mrr"]),
        "gt_in_pool": float(overall["gt_in_pct"]),
    }


def _precomputed_stats(path: Path) -> Dict[str, float]:
    total = 0
    sr = 0
    hop2 = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            spatial = row.get("constraints", {}).get("spatial_relations") or []
            if spatial:
                sr += 1
            if len(spatial) >= 2:
                hop2 += 1
    if total == 0:
        return {"sr_pct": 0.0, "hop2_pct": 0.0}
    return {"sr_pct": 100.0 * sr / total, "hop2_pct": 100.0 * hop2 / total}


def build_rows() -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for spec in MIXED_MODELS:
        if spec["source_type"] == "special_csv":
            metrics = _load_special_row(spec)
        else:
            metrics = _load_canonical_row(spec)
        pre = _precomputed_stats(Path(spec["precomputed"]))
        rows.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "display": spec["display"],
                "input_regime": spec["input_regime"],
                "color": spec["color"],
                "top10": metrics["top10"],
                "top1": metrics["top1"],
                "mrr10": metrics["mrr10"],
                "gt_in_pool": metrics["gt_in_pool"],
                "sr_pct": pre["sr_pct"],
                "hop2_pct": pre["hop2_pct"],
            }
        )
    return rows


def _summary_csv_text(rows: List[Dict[str, float | str]]) -> str:
    lines = ["system,input_regime,top10,top1,mrr10,gt_in_pool,sr_pct,hop2_pct"]
    for row in rows:
        lines.append(
            f"{row['display']},{row['input_regime']},{row['top10']:.1f},{row['top1']:.1f},{row['mrr10']:.4f},{row['gt_in_pool']:.1f},{row['sr_pct']:.1f},{row['hop2_pct']:.1f}"
        )
    return "\n".join(lines) + "\n"


def _summary_md_text(rows: List[Dict[str, float | str]]) -> str:
    lines = [
        "# Mixed-Regime AP Held-out Milestone Growth",
        "",
        "This milestone table is intentionally thesis-oriented rather than fully fair. Only `LoRA5-r32 FP` is kept in the no-site `Floorplan + Chat` regime to preserve the older pre-LoRA6 baseline; later rows use their canonical multimodal AP held-out setup.",
        "",
        "| System | Input regime | Top-10 | Top-1 | MRR@10 | GT-in-Pool | SR extracted | 2-hop SR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['display']} | {row['input_regime']} | {row['top10']:.1f}% | {row['top1']:.1f}% | {row['mrr10']:.4f} | {row['gt_in_pool']:.1f}% | {row['sr_pct']:.1f}% | {row['hop2_pct']:.1f}% |"
        )
    best_top10 = max(rows, key=lambda r: float(r["top10"]))
    best_top1 = max(rows, key=lambda r: float(r["top1"]))
    best_mrr = max(rows, key=lambda r: float(r["mrr10"]))
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- `LoRA5-r32 FP` remains the historical no-site anchor. It already extracts AP-compatible topology (`SR 100.0%`, `2-hop 40.0%`) and reaches `Top-10 {rows[0]['top10']:.1f}%`, but it still has `Top-1 0.0%`.",
            f"- Among the later canonical multimodal milestones, the strongest strict `Top-10` retriever is `{best_top10['display']}` at `{best_top10['top10']:.1f}%`.",
            f"- The strongest early-rank system is `{best_top1['display']}`, with `Top-1 {best_top1['top1']:.1f}%` and `MRR@10 {best_mrr['mrr10']:.4f}`.",
            "- The growth narrative is therefore not monotonic in every metric, but it is structurally progressive: later multimodal LoRA6 variants improve either shortlist coverage (`G3`) or early-rank quality (`G7`) over the older pre-LoRA6 baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_summary(prefix_path: Path, rows: List[Dict[str, float | str]]) -> None:
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    (prefix_path.parent / f"{prefix_path.stem}_summary.csv").write_text(_summary_csv_text(rows), encoding="utf-8")
    (prefix_path.parent / f"{prefix_path.stem}_summary.md").write_text(_summary_md_text(rows), encoding="utf-8")


def plot_fig02(out_path: Path, rows: List[Dict[str, float | str]]) -> None:
    labels = [str(r["label"]) for r in rows]
    colors = [str(r["color"]) for r in rows]
    x = list(range(len(rows)))
    top10 = [float(r["top10"]) for r in rows]
    top1 = [float(r["top1"]) for r in rows]
    mrr = [float(r["mrr10"]) for r in rows]
    gt = [float(r["gt_in_pool"]) for r in rows]
    sr = [float(r["sr_pct"]) for r in rows]
    hop2 = [float(r["hop2_pct"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.0), constrained_layout=False)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.23, wspace=0.28)

    ax = axes[0]
    bars = ax.bar(x, top10, color=colors, edgecolor="white", linewidth=0.9, width=0.62, label="Top-10", zorder=3)
    ax.plot(x, gt, color="#111111", marker="s", linestyle="--", linewidth=2.1, markersize=7, label="GT-in-Pool", zorder=4)
    ax_r = ax.twinx()
    ax_r.plot(x, [v * 1000.0 for v in mrr], color="#1565C0", marker="D", linestyle=":", linewidth=2.4, markersize=7, label="MRR@10 (×1000)", zorder=4)
    for bar, val in zip(bars, top10):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    for xx, val in zip(x, gt):
        ax.text(xx, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#111111")
    for xx, raw in zip(x, mrr):
        ax_r.text(xx, raw * 1000.0 + 1.5, f"{raw:.4f}", ha="center", va="bottom", fontsize=9, color="#1565C0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylim(0, 110)
    ax_r.set_ylim(0, max(v * 1000.0 for v in mrr) * 1.18)
    ax.set_ylabel("Percent")
    ax_r.set_ylabel("MRR@10 ×1000")
    ax.set_title("A. Mixed-regime end-to-end milestone growth", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)

    ax = axes[1]
    width = 0.22
    ax.bar([i - width for i in x], sr, width=width, color="#7C3AED", label="SR extracted")
    ax.bar(x, hop2, width=width, color="#A78BFA", label="2-hop SR")
    ax.bar([i + width for i in x], top1, width=width, color="#DDD6FE", label="Top-1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylim(0, max(110.0, max(sr + hop2 + top1) + 12))
    ax.set_ylabel("Percent")
    ax.set_title("B. Topology signal available to retrieval", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)

    fig.suptitle(
        "Figure 2v2. Thesis growth trajectory with LoRA5 as the no-site anchor and later multimodal milestones",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.07,
        "Only `LoRA5-r32` is kept in the no-site `Floorplan + Chat` regime. All later milestones remain in their canonical multimodal AP held-out setting, "
        "so this figure should be read as a thesis-growth narrative rather than a single strict fairness leaderboard.",
        fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fig03(out_path: Path, rows: List[Dict[str, float | str]]) -> None:
    labels = [str(r["label"]) for r in rows]
    colors = [str(r["color"]) for r in rows]
    x = list(range(len(rows)))
    gt = [float(r["gt_in_pool"]) for r in rows]
    top10 = [float(r["top10"]) for r in rows]
    top1 = [float(r["top1"]) for r in rows]
    mrr = [float(r["mrr10"]) * 1000.0 for r in rows]

    fig, ax = plt.subplots(figsize=(15.0, 6.8), constrained_layout=False)
    plt.subplots_adjust(left=0.07, right=0.94, top=0.86, bottom=0.23)
    width = 0.18
    ax_r = ax.twinx()
    ax_r.grid(False)

    bars1 = ax.bar([i - width for i in x], gt, width=width, color=colors, alpha=1.0, edgecolor="white", linewidth=0.8, label="GT-in-Pool", zorder=3)
    bars2 = ax.bar(x, top10, width=width, color=colors, alpha=0.60, edgecolor="white", linewidth=0.8, label="Top-10", zorder=3)
    bars3 = ax.bar([i + width for i in x], top1, width=width, color=colors, alpha=0.28, edgecolor="white", linewidth=0.8, label="Top-1", zorder=3)
    line = ax_r.plot(x, mrr, color="#1565C0", marker="D", linestyle="--", linewidth=2.3, markersize=7, label="MRR@10 (×1000)", zorder=4)[0]

    for bars, vals, fmt in [(bars1, gt, "{:.1f}%"), (bars2, top10, "{:.1f}%"), (bars3, top1, "{:.1f}%")]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, fmt.format(val), ha="center", va="bottom", fontsize=8.8)
    for xx, raw in zip(x, [float(r["mrr10"]) for r in rows]):
        ax_r.text(xx, raw * 1000.0 + 1.5, f"{raw:.4f}", ha="center", va="bottom", fontsize=8.8, color="#1565C0")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax_r.set_ylabel("MRR@10 ×1000")
    ax.set_ylim(0, 112)
    ax_r.set_ylim(0, max(mrr) * 1.20)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Figure 3. Track B-2 milestone comparison under the final thesis narrative")
    ax.legend([bars1, bars2, bars3, line], ["GT-in-Pool", "Top-10", "Top-1", "MRR@10 (×1000)"], loc="upper left", frameon=True)

    best_top10_idx = max(range(len(rows)), key=lambda i: top10[i])
    best_early_idx = max(range(len(rows)), key=lambda i: (top1[i], mrr[i]))
    ax.axvspan(best_top10_idx - 0.42, best_top10_idx + 0.42, color="#EDE7F6", zorder=0)
    ax.text(best_top10_idx, ax.get_ylim()[1] * 0.965, f"Best Top-10: {labels[best_top10_idx].replace(chr(10), ' ')}", ha="center", va="top", fontsize=10, weight="bold", color="#5B21B6")
    ax.text(best_early_idx, ax.get_ylim()[1] * 0.905, f"Best early-rank: {labels[best_early_idx].replace(chr(10), ' ')}", ha="center", va="top", fontsize=9.5, weight="bold", color="#1565C0")

    fig.suptitle("Track B-2: milestone growth from LoRA5 to the late multimodal LoRA6 systems", fontsize=14, y=0.98)
    fig.text(
        0.07,
        0.07,
        "Only `LoRA5-r32` is evaluated without the on-site image. The later milestone rows keep their canonical multimodal AP held-out setup, "
        "so this figure emphasizes thesis progression rather than strict single-regime fairness.",
        fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_appendix(out_path: Path, rows: List[Dict[str, float | str]]) -> None:
    labels = [str(r["label"]) for r in rows]
    colors = [str(r["color"]) for r in rows]
    x = list(range(len(rows)))
    top10 = [float(r["top10"]) for r in rows]
    top1 = [float(r["top1"]) for r in rows]
    mrr = [float(r["mrr10"]) for r in rows]
    gt = [float(r["gt_in_pool"]) for r in rows]
    sr = [float(r["sr_pct"]) for r in rows]
    hop2 = [float(r["hop2_pct"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(15.0, 9.3), constrained_layout=False)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.14, hspace=0.40)

    ax = axes[0]
    width = 0.22
    ax.bar([i - width for i in x], gt, width=width, color=colors, alpha=1.0, edgecolor="white", linewidth=0.8, label="GT-in-Pool")
    ax.bar(x, top10, width=width, color=colors, alpha=0.60, edgecolor="white", linewidth=0.8, label="Top-10")
    ax.bar([i + width for i in x], top1, width=width, color=colors, alpha=0.28, edgecolor="white", linewidth=0.8, label="Top-1")
    ax_r = ax.twinx()
    ax_r.plot(x, [v * 1000.0 for v in mrr], color="#1565C0", marker="D", linestyle="--", linewidth=2.3, markersize=7, label="MRR@10 (×1000)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax_r.set_ylabel("MRR@10 ×1000")
    ax.set_ylim(0, 112)
    ax_r.set_ylim(0, max(v * 1000.0 for v in mrr) * 1.20)
    ax.set_title("A. Mixed-regime Track B-2 milestone trajectory", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)

    ax = axes[1]
    width = 0.22
    ax.bar([i - width for i in x], sr, width=width, color="#7C3AED", label="SR extracted")
    ax.bar(x, hop2, width=width, color="#A78BFA", label="2-hop SR")
    ax.bar([i + width for i in x], top10, width=width, color="#DDD6FE", label="Top-10")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 110)
    ax.set_title("B. Intermediate topology signal behind the trajectory", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)

    fig.suptitle("Appendix A9. Mixed-regime AP milestone growth", fontsize=14, fontweight="bold")
    fig.text(
        0.07,
        0.045,
        "The appendix companion mirrors the mixed-regime main-text comparison and makes the intermediate topology signal explicit. "
        "Only `LoRA5-r32` is shown in the no-site floorplan regime; later rows remain canonical multimodal milestones.",
        fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    rows = build_rows()

    fig02 = MAIN_DIR / "fig02_v2_extraction_vs_downstream_tradeoff.png"
    fig03 = MAIN_DIR / "fig03_trackB2_strict_downstream.png"
    figA9 = APPENDIX_DIR / "figA9_fair_trackb2_growth.png"

    _write_summary(fig02, rows)
    _write_summary(fig03, rows)
    _write_summary(figA9, rows)

    plot_fig02(fig02, rows)
    plot_fig03(fig03, rows)
    plot_appendix(figA9, rows)

    print(f"Wrote {fig02}")
    print(f"Wrote {fig03}")
    print(f"Wrote {figA9}")


if __name__ == "__main__":
    main()
