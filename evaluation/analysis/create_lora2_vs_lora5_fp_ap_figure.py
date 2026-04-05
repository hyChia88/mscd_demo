#!/usr/bin/env python3
"""Create a thesis-ready LoRA2 vs LoRA5 AP floorplan-only comparison figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULT_ROOT = PROJECT_ROOT / "output" / "ap_lora2_vs_lora5_floorplan_only"
DOCS_MAIN_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_main"
DOCS_APPENDIX_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_appendix"

LORA2_SUMMARY = RESULT_ROOT / "e2e" / "lora2_apheldout_FP" / "summary_20260405_025930_v2_lora_p0_union_p1.csv"
LORA5_SUMMARY = RESULT_ROOT / "e2e" / "lora5r32_apheldout_FP" / "summary_20260405_030157_v2_lora_p0_union_p1.csv"
LORA2_PRECOMP = RESULT_ROOT / "precomputed" / "lora2_apheldout_FP.jsonl"
LORA5_PRECOMP = RESULT_ROOT / "precomputed" / "lora5r32_apheldout_FP.jsonl"

OUT_DEFAULT = RESULT_ROOT / "fig13_lora2_vs_lora5_fp_ap.png"
MAIN_OUT_DEFAULT = DOCS_MAIN_DIR / "fig13_lora2_vs_lora5_fp_ap.png"
APPENDIX_OUT_DEFAULT = DOCS_APPENDIX_DIR / "figA8_lora2_vs_lora5_fp_ap.png"


def _load_summary_metrics(path: Path) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        current_section = None
        for row in reader:
            if not row:
                continue
            if row[0].startswith("==="):
                current_section = row[0]
                continue
            if row[0] == "Metric":
                continue
            if current_section and len(row) >= 2:
                metrics[row[0]] = row[1]
    return metrics


def _parse_gt_in_pool(text: str) -> float:
    pct = text.split("(")[-1].rstrip(")")
    return float(pct.rstrip("%"))


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
        return {"cases": 0.0, "sr_pct": 0.0, "hop2_pct": 0.0}
    return {
        "cases": float(total),
        "sr_pct": 100.0 * sr / total,
        "hop2_pct": 100.0 * hop2 / total,
    }


def _build_rows() -> List[Dict[str, float | str]]:
    lora2 = _load_summary_metrics(LORA2_SUMMARY)
    lora5 = _load_summary_metrics(LORA5_SUMMARY)
    lora2_pre = _precomputed_stats(LORA2_PRECOMP)
    lora5_pre = _precomputed_stats(LORA5_PRECOMP)
    return [
        {
            "system": "LoRA2 FP",
            "top10": float(lora2["Top-10 Accuracy"]) * 100.0,
            "top1": float(lora2["Top-1 Accuracy"]) * 100.0,
            "mrr10": float(lora2["MRR@10"]),
            "gt_in_pool": _parse_gt_in_pool(lora2["GT-in-Pool"]),
            "sr_pct": lora2_pre["sr_pct"],
            "hop2_pct": lora2_pre["hop2_pct"],
            "parse_rate": float(lora2["Constraints Parse Rate"]) * 100.0,
        },
        {
            "system": "LoRA5-r32 FP",
            "top10": float(lora5["Top-10 Accuracy"]) * 100.0,
            "top1": float(lora5["Top-1 Accuracy"]) * 100.0,
            "mrr10": float(lora5["MRR@10"]),
            "gt_in_pool": _parse_gt_in_pool(lora5["GT-in-Pool"]),
            "sr_pct": lora5_pre["sr_pct"],
            "hop2_pct": lora5_pre["hop2_pct"],
            "parse_rate": float(lora5["Constraints Parse Rate"]) * 100.0,
        },
    ]


def _summary_csv_text(rows: List[Dict[str, float | str]]) -> str:
    lines = ["system,top10,top1,mrr10,gt_in_pool,parse_rate,sr_pct,hop2_pct"]
    for row in rows:
        lines.append(
            f"{row['system']},{row['top10']:.1f},{row['top1']:.1f},{row['mrr10']:.4f},{row['gt_in_pool']:.1f},{row['parse_rate']:.1f},{row['sr_pct']:.1f},{row['hop2_pct']:.1f}"
        )
    return "\n".join(lines) + "\n"


def _summary_md_text(rows: List[Dict[str, float | str]]) -> str:
    lines = [
        "# LoRA2 vs LoRA5-r32 on AP Held-out, Floorplan-only",
        "",
        "Both systems were evaluated on the same AP held-out 60-case benchmark using floorplan-only input. Each model kept its native intermediate extraction format (`LoRA2` prompt style vs `LoRA5` prompt style), and both were fed into the same downstream `v2_lora + p0_union_p1` end-to-end pipeline.",
        "",
        "| System | Top-10 | Top-1 | MRR@10 | GT-in-Pool | Parse | SR extracted | 2-hop SR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['top10']:.1f}% | {row['top1']:.1f}% | {row['mrr10']:.4f} | {row['gt_in_pool']:.1f}% | {row['parse_rate']:.1f}% | {row['sr_pct']:.1f}% | {row['hop2_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- `LoRA5-r32 FP` improves strict AP end-to-end retrieval over `LoRA2 FP` on `Top-10` (`21.7%` vs `16.7%`) and `GT-in-Pool` (`100.0%` vs `98.3%`).",
            "- Neither model solves last-mile ranking under this older floorplan-only setup: both remain at `Top-1 = 0.0%`.",
            "- The main intermediate-format difference is structural. `LoRA2 FP` produced no usable `spatial_relations` on this AP held-out benchmark, whereas `LoRA5-r32 FP` produced `SR` in all `60/60` cases and `2-hop` relations in `24/60` cases.",
            "- This makes the comparison thesis-safe: the downstream gain from `LoRA5-r32` comes from a fair end-to-end run under the same AP benchmark and the same retrieval stack, not from a custom downstream path.",
        ]
    )
    return "\n".join(lines) + "\n"


def _companion_prefix(out_path: Path) -> str:
    return out_path.stem


def write_companion_files(out_path: Path, rows: List[Dict[str, float | str]]) -> None:
    prefix = _companion_prefix(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.parent / f"{prefix}_summary.csv").write_text(_summary_csv_text(rows), encoding="utf-8")
    (out_path.parent / f"{prefix}_summary.md").write_text(_summary_md_text(rows), encoding="utf-8")


def plot_figure(out_path: Path, rows: List[Dict[str, float | str]]) -> None:
    labels = [str(r["system"]) for r in rows]
    top10 = [float(r["top10"]) for r in rows]
    top1 = [float(r["top1"]) for r in rows]
    mrr = [float(r["mrr10"]) for r in rows]
    gt_in_pool = [float(r["gt_in_pool"]) for r in rows]
    sr = [float(r["sr_pct"]) for r in rows]
    hop2 = [float(r["hop2_pct"]) for r in rows]
    parse = [float(r["parse_rate"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), constrained_layout=False)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.16, wspace=0.28)

    x = list(range(len(labels)))
    width = 0.22

    ax = axes[0]
    ax.bar([i - width for i in x], top10, width=width, color="#0F766E", label="Top-10")
    ax.bar(x, top1, width=width, color="#14B8A6", label="Top-1")
    ax.bar([i + width for i in x], gt_in_pool, width=width, color="#99F6E4", label="GT-in-Pool")
    ax.set_title("A. Fair AP end-to-end comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax_r = ax.twinx()
    ax_r.plot(x, mrr, color="#D97706", marker="o", linewidth=2, label="MRR@10")
    ax_r.set_ylabel("MRR@10")
    ax_r.set_ylim(0, max(0.12, max(mrr) * 2.2))

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)

    ax = axes[1]
    ax.bar([i - width for i in x], parse, width=width, color="#7C3AED", label="Parse")
    ax.bar(x, sr, width=width, color="#A78BFA", label="SR extracted")
    ax.bar([i + width for i in x], hop2, width=width, color="#DDD6FE", label="2-hop SR")
    ax.set_title("B. Native intermediate-format coverage", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "LoRA2 vs LoRA5-r32 on AP held-out, floorplan-only: native-format extraction and fair end-to-end accuracy",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.05,
        "Both runs use the same AP held-out 60-case benchmark and the same downstream v2_lora + p0_union_p1 pipeline. "
        "Only the upstream LoRA family and its native intermediate extraction format differ.",
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    rows = _build_rows()
    for out_path in [OUT_DEFAULT, MAIN_OUT_DEFAULT, APPENDIX_OUT_DEFAULT]:
        write_companion_files(out_path, rows)
        plot_figure(out_path, rows)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
