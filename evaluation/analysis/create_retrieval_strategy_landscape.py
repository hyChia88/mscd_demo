#!/usr/bin/env python3
"""Create a thesis-ready retrieval strategy landscape figure."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase4_plot_style import HIGHLIGHT_COLORS, METRIC_COLORS


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331"
DOCS_MAIN_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_main"
DOCS_APPENDIX_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_appendix"
LANDSCAPE_OUT_DIR = OUTPUT_ROOT / "strategy_landscape" / "20260405"


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


TRACK_B2_G3 = OUTPUT_ROOT / "metrics" / "track_b2_phase3_fixed_summary.csv"
TRACK_B2_G7 = OUTPUT_ROOT / "metrics" / "track_b2_phase5_g7_summary.csv"
GRAPH_RAG_G7_AT10 = _first_existing(
    OUTPUT_ROOT / "graph_rag_rerank" / "20260405_g7_formal_v3" / "graph_rag_rerank_summary.json",
    OUTPUT_ROOT / "graph_rag_rerank" / "legacy" / "20260405_g7_formal_v3" / "graph_rag_rerank_summary.json",
    OUTPUT_ROOT / "graph_rag_rerank" / "20260407_g7_phase5_v1" / "graph_rag_rerank_summary.json",
)
GRAPH_RAG_P1_AT10 = _first_existing(
    OUTPUT_ROOT / "graph_rag_rerank" / "20260404_p1_formal_v2" / "graph_rag_rerank_summary.json",
    OUTPUT_ROOT / "graph_rag_rerank" / "legacy" / "20260404_p1_formal_v2" / "graph_rag_rerank_summary.json",
    OUTPUT_ROOT / "graph_rag_rerank" / "20260407_g7_phase5_v1" / "graph_rag_rerank_summary.json",
)
GRAPH_RAG_G7_AT15 = _first_existing(
    OUTPUT_ROOT / "graph_rag_rerank" / "20260405_top15_g7_v1" / "graph_rag_rerank_summary.json",
    OUTPUT_ROOT / "graph_rag_rerank" / "legacy" / "20260405_top15_g7_v1" / "graph_rag_rerank_summary.json",
    OUTPUT_ROOT / "graph_rag_rerank" / "20260407_g7_phase5_v1" / "graph_rag_rerank_summary.json",
)
GRAPH_RAG_P1_AT15 = _first_existing(
    OUTPUT_ROOT / "graph_rag_rerank" / "20260405_top15_p1_v1" / "graph_rag_rerank_summary.json",
    OUTPUT_ROOT / "graph_rag_rerank" / "legacy" / "20260405_top15_p1_v1" / "graph_rag_rerank_summary.json",
    OUTPUT_ROOT / "graph_rag_rerank" / "20260407_g7_phase5_v1" / "graph_rag_rerank_summary.json",
)
FINGERPRINT_CSV = OUTPUT_ROOT / "group4_post-hoc_analysis" / "oracle_ceiling" / "20260404" / "fingerprint_loss_by_level.csv"

OUT_DEFAULT = LANDSCAPE_OUT_DIR / "fig12_retrieval_strategy_landscape.png"
MAIN_OUT_DEFAULT = DOCS_MAIN_DIR / "fig12_retrieval_strategy_landscape.png"
APPENDIX_OUT_DEFAULT = DOCS_APPENDIX_DIR / "figA7_retrieval_strategy_landscape.png"


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalise_modes_key(d: dict) -> dict:
    """Renamed g7_pipeline → full_topology; tolerate legacy summary JSONs."""
    modes = d.get("modes")
    if isinstance(modes, dict) and "g7_pipeline" in modes and "full_topology" not in modes:
        modes["full_topology"] = modes.pop("g7_pipeline")
    return d


def _load_json(path: Path) -> dict:
    import json

    return _normalise_modes_key(json.loads(path.read_text(encoding="utf-8")))


def _track_row(path: Path, group: str) -> Dict[str, str]:
    rows = _load_csv_rows(path)
    for row in rows:
        if row.get("group") == group:
            return row
    raise KeyError(f"Missing group {group} in {path}")


def _fingerprint_rows(path: Path, slice_name: str) -> List[Dict[str, str]]:
    rows = _load_csv_rows(path)
    return [row for row in rows if row.get("slice") == slice_name]


def _build_executable_rows() -> List[Dict[str, float | str]]:
    g3 = _track_row(TRACK_B2_G3, "g3_fullaug_r32")
    g7 = _track_row(TRACK_B2_G7, "g7_position_context")
    return [
        {"system": "P1-only upper bound", "top10": 16.7, "top1": 0.0, "mrr10": 0.0392},
        {"system": "Oracle p0_union_p1", "top10": 40.0, "top1": 5.0, "mrr10": 0.1279},
        {
            "system": "G3 FullAug r32",
            "top10": float(g3["top10_pct"]),
            "top1": float(g3["top1_pct"]),
            "mrr10": float(g3["mrr"]),
        },
        {
            "system": "G7 Position Context",
            "top10": float(g7["top10_pct"]),
            "top1": float(g7["top1_pct"]),
            "mrr10": float(g7["mrr"]),
        },
    ]


def _build_rerank_rows() -> List[Dict[str, float | str]]:
    g7_10 = _load_json(GRAPH_RAG_G7_AT10)
    p1_10 = _load_json(GRAPH_RAG_P1_AT10)
    g7_15 = _load_json(GRAPH_RAG_G7_AT15)
    p1_15 = _load_json(GRAPH_RAG_P1_AT15)
    return [
        {
            "system": "Full-topology (G7)",
            "top1": float(g7_15["modes"]["full_topology"]["baseline"]["top1_pct"]),
            "mrr10": float(g7_15["modes"]["full_topology"]["baseline"]["mrr10"]),
        },
        {
            "system": "Full-topology (G7) + rerank@10",
            "top1": float(g7_10["modes"]["full_topology"]["reranked"]["top1_pct"]),
            "mrr10": float(g7_10["modes"]["full_topology"]["reranked"]["mrr10"]),
        },
        {
            "system": "Full-topology (G7) + rerank@15",
            "top1": float(g7_15["modes"]["full_topology"]["reranked"]["top1_pct"]),
            "mrr10": float(g7_15["modes"]["full_topology"]["reranked"]["mrr10"]),
        },
        {
            "system": "P1-only (G7 coarse)",
            "top1": float(p1_15["modes"]["p1_only"]["baseline"]["top1_pct"]),
            "mrr10": float(p1_15["modes"]["p1_only"]["baseline"]["mrr10"]),
        },
        {
            "system": "P1-only (G7 coarse) + rerank@10",
            "top1": float(p1_10["modes"]["p1_only"]["reranked"]["top1_pct"]),
            "mrr10": float(p1_10["modes"]["p1_only"]["reranked"]["mrr10"]),
        },
        {
            "system": "P1-only (G7 coarse) + rerank@15",
            "top1": float(p1_15["modes"]["p1_only"]["reranked"]["top1_pct"]),
            "mrr10": float(p1_15["modes"]["p1_only"]["reranked"]["mrr10"]),
        },
    ]


def _build_fingerprint_rows() -> List[Dict[str, float | str]]:
    rows = []
    for row in _fingerprint_rows(FINGERPRINT_CSV, "position_sensitive_subset"):
        rows.append(
            {
                "level": row["level"],
                "avg_pool": float(row["avg_pool"]),
                "top10": float(row["top10_rate"]) * 100.0,
                "top1": float(row["top1_rate"]) * 100.0,
            }
        )
    return rows


def _summary_csv_text(
    executable_rows: List[Dict[str, float | str]],
    rerank_rows: List[Dict[str, float | str]],
    fingerprint_rows: List[Dict[str, float | str]],
) -> str:
    lines = ["section,label,metric_a,metric_b,metric_c"]
    for row in executable_rows:
        lines.append(
            f"executable,{row['system']},{row['top10']:.1f},{row['top1']:.1f},{row['mrr10']:.4f}"
        )
    for row in rerank_rows:
        lines.append(
            f"graph_rag,{row['system']},,{row['top1']:.1f},{row['mrr10']:.4f}"
        )
    for row in fingerprint_rows:
        lines.append(
            f"fingerprint,{row['level']},{row['avg_pool']:.3f},{row['top10']:.1f},{row['top1']:.1f}"
        )
    return "\n".join(lines) + "\n"


def _summary_md_text(
    executable_rows: List[Dict[str, float | str]],
    rerank_rows: List[Dict[str, float | str]],
    fingerprint_rows: List[Dict[str, float | str]],
) -> str:
    lines = [
        "# Retrieval Strategy Landscape Summary",
        "",
        "## Executable systems",
        "",
        "| System | Top-10 | Top-1 | MRR@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in executable_rows:
        lines.append(
            f"| {row['system']} | {row['top10']:.1f}% | {row['top1']:.1f}% | {row['mrr10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Graph-RAG paired follow-ups",
            "",
            "| System | Top-1 | MRR@10 |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in rerank_rows:
        lines.append(f"| {row['system']} | {row['top1']:.1f}% | {row['mrr10']:.4f} |")
    lines.extend(
        [
            "",
            "## Fingerprint ladder (position-sensitive subset)",
            "",
            "| Level | Avg pool | Ideal Top-10 | Ideal Top-1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in fingerprint_rows:
        lines.append(
            f"| {row['level']} | {row['avg_pool']:.3f} | {row['top10']:.1f}% | {row['top1']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- `Oracle p0_union_p1` remains the strongest executable symbolic planner.",
            "- `G3 FullAug r32` remains the strongest learned Top-10 retriever, while `G7 Position Context` is the strongest learned early-rank system.",
            "- Graph-RAG helps the coarse `P1-only (G7 coarse)` shortlist more than the hard-negative `Full-topology (G7)` shortlist.",
            "- The largest remaining ceiling lies in richer fingerprint consumption at `L3/L4`, especially subtype and exact slot identity.",
        ]
    )
    return "\n".join(lines) + "\n"


def _companion_prefix(out_path: Path) -> str:
    return out_path.stem


def _write_companion_files(
    out_path: Path,
    executable_rows: List[Dict[str, float | str]],
    rerank_rows: List[Dict[str, float | str]],
    fingerprint_rows: List[Dict[str, float | str]],
) -> None:
    prefix = _companion_prefix(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.parent / f"{prefix}_summary.csv").write_text(
        _summary_csv_text(executable_rows, rerank_rows, fingerprint_rows),
        encoding="utf-8",
    )
    (out_path.parent / f"{prefix}_summary.md").write_text(
        _summary_md_text(executable_rows, rerank_rows, fingerprint_rows),
        encoding="utf-8",
    )


def _plot_figure(
    executable_rows: List[Dict[str, float | str]],
    rerank_rows: List[Dict[str, float | str]],
    fingerprint_rows: List[Dict[str, float | str]],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 10.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    ax_exec = fig.add_subplot(gs[0, 0])
    ax_rerank = fig.add_subplot(gs[0, 1])
    ax_fp = fig.add_subplot(gs[1, :])

    # Panel A
    exec_labels = [str(row["system"]) for row in executable_rows]
    exec_top10 = [float(row["top10"]) for row in executable_rows]
    exec_top1 = [float(row["top1"]) for row in executable_rows]
    exec_mrr = [float(row["mrr10"]) for row in executable_rows]
    x = list(range(len(exec_labels)))
    width = 0.34
    ax_exec.bar([i - width / 2 for i in x], exec_top10, width=width, color=METRIC_COLORS["top10"], label="Top-10")
    ax_exec.bar([i + width / 2 for i in x], exec_top1, width=width, color=METRIC_COLORS["top1"], label="Top-1")
    ax_exec.set_title("A. Executable systems", fontsize=12, fontweight="bold")
    ax_exec.set_ylabel("Percent")
    ax_exec.set_xticks(x)
    ax_exec.set_xticklabels(exec_labels, rotation=18, ha="right")
    ax_exec.grid(axis="y", alpha=0.25)
    ax_exec_r = ax_exec.twinx()
    ax_exec_r.plot(x, exec_mrr, color=METRIC_COLORS["mrr"], marker="o", linewidth=2, label="MRR@10")
    ax_exec_r.set_ylabel("MRR@10")
    for i, value in enumerate(exec_top10):
        ax_exec.text(i - width / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(exec_top1):
        ax_exec.text(i + width / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(exec_mrr):
        ax_exec_r.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color=HIGHLIGHT_COLORS["winner_amber_text"])
    lines1, labels1 = ax_exec.get_legend_handles_labels()
    lines2, labels2 = ax_exec_r.get_legend_handles_labels()
    ax_exec.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Panel B
    rerank_labels = [str(row["system"]) for row in rerank_rows]
    rerank_top1 = [float(row["top1"]) for row in rerank_rows]
    rerank_mrr = [float(row["mrr10"]) for row in rerank_rows]
    x2 = list(range(len(rerank_labels)))
    colors = [
        METRIC_COLORS["graph_rag_full"],
        METRIC_COLORS["graph_rag_full_rerank10"],
        METRIC_COLORS["graph_rag_full_rerank15"],
        METRIC_COLORS["graph_rag_p1"],
        METRIC_COLORS["graph_rag_p1_rerank10"],
        METRIC_COLORS["graph_rag_p1_rerank15"],
    ]
    ax_rerank.bar(x2, rerank_top1, color=colors, label="Top-1")
    ax_rerank.set_title("B. Graph-RAG paired follow-ups", fontsize=12, fontweight="bold")
    ax_rerank.set_ylabel("Top-1 (%)")
    ax_rerank.set_xticks(x2)
    ax_rerank.set_xticklabels(rerank_labels, rotation=20, ha="right")
    ax_rerank.grid(axis="y", alpha=0.25)
    ax_rerank_r = ax_rerank.twinx()
    ax_rerank_r.plot(x2, rerank_mrr, color=HIGHLIGHT_COLORS["rerank_orange"], marker="o", linewidth=2, label="MRR@10")
    ax_rerank_r.set_ylabel("MRR@10")
    for i, value in enumerate(rerank_top1):
        ax_rerank.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(rerank_mrr):
        ax_rerank_r.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color=HIGHLIGHT_COLORS["winner_amber_text"])
    lines1, labels1 = ax_rerank.get_legend_handles_labels()
    lines2, labels2 = ax_rerank_r.get_legend_handles_labels()
    ax_rerank.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Panel C
    fp_labels = [str(row["level"]) for row in fingerprint_rows]
    fp_pool = [float(row["avg_pool"]) for row in fingerprint_rows]
    fp_top10 = [float(row["top10"]) for row in fingerprint_rows]
    fp_top1 = [float(row["top1"]) for row in fingerprint_rows]
    x3 = list(range(len(fp_labels)))
    ax_fp.bar(x3, fp_pool, color=METRIC_COLORS["avg_pool"], alpha=0.88, label="Avg pool")
    ax_fp.set_title("C. Fingerprint ladder (position-sensitive subset)", fontsize=12, fontweight="bold")
    ax_fp.set_ylabel("Average pool")
    ax_fp.set_xticks(x3)
    ax_fp.set_xticklabels(fp_labels)
    ax_fp.grid(axis="y", alpha=0.25)
    ax_fp_r = ax_fp.twinx()
    ax_fp_r.plot(x3, fp_top10, color=METRIC_COLORS["ideal_top10"], marker="o", linewidth=2, label="Ideal Top-10")
    ax_fp_r.plot(x3, fp_top1, color=METRIC_COLORS["ideal_top1"], marker="o", linewidth=2, label="Ideal Top-1")
    ax_fp_r.set_ylabel("Percent")
    for i, value in enumerate(fp_pool):
        ax_fp.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(fp_top10):
        ax_fp_r.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9, color=HIGHLIGHT_COLORS["safe_green_text"])
    for i, value in enumerate(fp_top1):
        ax_fp_r.text(i, value + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=9, color=HIGHLIGHT_COLORS["winner_amber_text"])
    lines1, labels1 = ax_fp.get_legend_handles_labels()
    lines2, labels2 = ax_fp_r.get_legend_handles_labels()
    ax_fp.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle(
        "Retrieval strategy landscape on AP held-out",
        fontsize=16,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.94, top=0.92, bottom=0.18, hspace=0.36, wspace=0.28)
    fig.text(
        0.5,
        0.06,
        "Panel A compares the strongest executable symbolic and learned systems. "
        "Panel B shows that Graph-RAG helps the coarse G7-derived shortlist more than the learned hard-negative shortlist. "
        "Panel C shows that the largest remaining ceiling lies in richer fingerprint consumption at L3/L4.",
        ha="center",
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    executable_rows = _build_executable_rows()
    rerank_rows = _build_rerank_rows()
    fingerprint_rows = _build_fingerprint_rows()

    for out_path in [OUT_DEFAULT, MAIN_OUT_DEFAULT, APPENDIX_OUT_DEFAULT]:
        _plot_figure(executable_rows, rerank_rows, fingerprint_rows, out_path)
        _write_companion_files(out_path, executable_rows, rerank_rows, fingerprint_rows)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
