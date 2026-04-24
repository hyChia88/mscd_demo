#!/usr/bin/env python3
"""Create a final Graph-RAG comparison figure for thesis plots."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase4_plot_style import GRAPH_RAG_COLORS as PHASE4_COLORS


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_RAG_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "graph_rag_rerank"
DOCS_MAIN_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_main"
DOCS_APPENDIX_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_appendix"

G7_SUMMARY_DEFAULT = GRAPH_RAG_ROOT / "20260407_g7_phase5_v1" / "graph_rag_rerank_summary.json"
G8_SUMMARY_DEFAULT = GRAPH_RAG_ROOT / "20260407_g8_phase5_v1" / "graph_rag_rerank_summary.json"
G7_RESULTS_DEFAULT = GRAPH_RAG_ROOT / "20260407_g7_phase5_v1" / "graph_rag_rerank_results.jsonl"
G8_RESULTS_DEFAULT = GRAPH_RAG_ROOT / "20260407_g8_phase5_v1" / "graph_rag_rerank_results.jsonl"
# Keep legacy P1 alias (same as G7 p1_only mode)
P1_SUMMARY_DEFAULT = G7_SUMMARY_DEFAULT
P1_RESULTS_DEFAULT = G7_RESULTS_DEFAULT
OUT_DEFAULT = GRAPH_RAG_ROOT / "20260407_g8_phase5_v1" / "graph_rag_rerank_comparison.png"
MAIN_OUT_DEFAULT = DOCS_MAIN_DIR / "fig11_graph_rag_rerank_comparison.png"
APPENDIX_OUT_DEFAULT = DOCS_APPENDIX_DIR / "figA6_graph_rag_rerank_comparison.png"

# Match the canonical Phase 4 palette so this figure reads as part of the
# same G-series/P1/oracle visual system.
VARIANT_HATCHES = {
    "baseline": "",
    "single_shot": "///",
    "cot": "xxx",
    "rerank": "///",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _subset_stats(rows: List[Dict[str, Any]], top_k: int, mode: str = "g7_pipeline") -> Dict[str, int]:
    subset = [
        r
        for r in rows
        if r.get("mode") == mode and 2 <= int(r.get("base_rank", -1)) <= top_k
    ]
    rescued = sum(1 for r in subset if int(r.get("reranked_rank", -1)) == 1)
    worsened = sum(
        1
        for r in subset
        if int(r.get("reranked_rank", -1)) > int(r.get("base_rank", -1)) > 0
    )
    return {"n": len(subset), "rescued": rescued, "worsened": worsened}


def _changed_cases(rows: List[Dict[str, Any]], mode: str, output_mode: str | None = None) -> List[Dict[str, Any]]:
    changed: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("mode") != mode:
            continue
        base_rank = int(row.get("base_rank", -1))
        reranked_rank = int(row.get("reranked_rank", -1))
        if base_rank <= 0 or reranked_rank <= 0 or base_rank == reranked_rank:
            continue
        status = "improved" if reranked_rank < base_rank else "worsened"
        if reranked_rank == 1 and base_rank > 1:
            status = "rescued_to_top1"
        changed.append(
            {
                "mode": output_mode or mode,
                "case_id": row.get("case_id", ""),
                "family": row.get("family", ""),
                "base_rank": base_rank,
                "reranked_rank": reranked_rank,
                "delta": reranked_rank - base_rank,
                "status": status,
            }
        )
    return changed


def build_rows(
    g7_summary: Dict[str, Any],
    g8_summary: Dict[str, Any],
    *,
    g7_label: str,
    g7_rerank_label: str,
    g8_label: str,
    g8_rerank_label: str,
    p1_label: str,
    p1_rerank_label: str,
    p1_g8_rerank_label: str | None,
    oracle_label: str,
) -> List[Dict[str, float]]:
    """Build comparison rows: G7, G7+GR, G8, G8+GR, P1-only (G7 coarse), P1+GR, Oracle."""
    p1 = g7_summary["modes"]["p1_only"]   # P1-only pool is identical for G7/G8 unless comparing prompt modes
    rows = [
        {
            "system": g7_label,
            "top10": g7_summary["modes"]["g7_pipeline"]["baseline"]["top10_pct"],
            "top1": g7_summary["modes"]["g7_pipeline"]["baseline"]["top1_pct"],
            "mrr10": g7_summary["modes"]["g7_pipeline"]["baseline"]["mrr10"],
        },
        {
            "system": g7_rerank_label,
            "top10": g7_summary["modes"]["g7_pipeline"]["reranked"]["top10_pct"],
            "top1": g7_summary["modes"]["g7_pipeline"]["reranked"]["top1_pct"],
            "mrr10": g7_summary["modes"]["g7_pipeline"]["reranked"]["mrr10"],
        },
        {
            "system": p1_label,
            "top10": p1["baseline"]["top10_pct"],
            "top1": p1["baseline"]["top1_pct"],
            "mrr10": p1["baseline"]["mrr10"],
        },
        {
            "system": p1_rerank_label,
            "top10": p1["reranked"]["top10_pct"],
            "top1": p1["reranked"]["top1_pct"],
            "mrr10": p1["reranked"]["mrr10"],
        },
    ]
    g7_baseline = g7_summary["modes"]["g7_pipeline"]["baseline"]
    g8_baseline = g8_summary["modes"]["g7_pipeline"]["baseline"]
    if g8_label and (
        g8_label != g7_label
        or g8_baseline["top10_pct"] != g7_baseline["top10_pct"]
        or g8_baseline["top1_pct"] != g7_baseline["top1_pct"]
        or g8_baseline["mrr10"] != g7_baseline["mrr10"]
    ):
        rows.extend(
            [
                {
                    "system": g8_label,
                    "top10": g8_baseline["top10_pct"],
                    "top1": g8_baseline["top1_pct"],
                    "mrr10": g8_baseline["mrr10"],
                },
                {
                    "system": g8_rerank_label,
                    "top10": g8_summary["modes"]["g7_pipeline"]["reranked"]["top10_pct"],
                    "top1": g8_summary["modes"]["g7_pipeline"]["reranked"]["top1_pct"],
                    "mrr10": g8_summary["modes"]["g7_pipeline"]["reranked"]["mrr10"],
                },
            ]
        )
    else:
        rows.append(
            {
                "system": g8_rerank_label,
                "top10": g8_summary["modes"]["g7_pipeline"]["reranked"]["top10_pct"],
                "top1": g8_summary["modes"]["g7_pipeline"]["reranked"]["top1_pct"],
                "mrr10": g8_summary["modes"]["g7_pipeline"]["reranked"]["mrr10"],
            }
        )
    if p1_g8_rerank_label:
        rows.append(
            {
                "system": p1_g8_rerank_label,
                "top10": g8_summary["modes"]["p1_only"]["reranked"]["top10_pct"],
                "top1": g8_summary["modes"]["p1_only"]["reranked"]["top1_pct"],
                "mrr10": g8_summary["modes"]["p1_only"]["reranked"]["mrr10"],
            }
        )
    rows.append({"system": oracle_label, "top10": 40.0, "top1": 5.0, "mrr10": 0.1279})
    return rows


def _summary_csv_text(rows: List[Dict[str, float]]) -> str:
    lines = ["system,top10,top1,mrr10"]
    for row in rows:
        lines.append(
            f"{row['system']},{row['top10']:.1f},{row['top1']:.1f},{row['mrr10']:.4f}"
        )
    return "\n".join(lines) + "\n"


def _summary_md_text(
    rows: List[Dict[str, float]],
    g7_subset: Dict[str, int],
    p1_subset: Dict[str, int],
    top_k: int,
    *,
    subset_left_label: str,
    subset_p1_label: str,
) -> str:
    lines = [
        "# Graph-RAG Canonical Summary",
        "",
        "| System | Top-10 | Top-1 | MRR@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['top10']:.1f}% | {row['top1']:.1f}% | {row['mrr10']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"## Target subsets (Top-{top_k} but not Top-1 before rerank)",
            "",
            f"- `{subset_left_label}`: `{g7_subset['rescued']}/{g7_subset['n']}` rescued to Top-1, `{g7_subset['worsened']}` worsened",
            f"- `{subset_p1_label}`: `{p1_subset['rescued']}/{p1_subset['n']}` rescued to Top-1, `{p1_subset['worsened']}` worsened",
        ]
    )
    return "\n".join(lines) + "\n"


def _changed_cases_csv_text(rows: List[Dict[str, Any]]) -> str:
    lines = ["mode,case_id,family,base_rank,reranked_rank,delta,status"]
    for row in rows:
        lines.append(
            f"{row['mode']},{row['case_id']},{row['family']},{row['base_rank']},{row['reranked_rank']},{row['delta']},{row['status']}"
        )
    return "\n".join(lines) + "\n"


def _changed_cases_md_text(rows: List[Dict[str, Any]]) -> str:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)

    lines = ["# Significant Graph-RAG Rerank Cases", ""]
    for mode in grouped:
        lines.append(f"## {mode}")
        lines.append("")
        lines.append("| Case | Family | Base rank | Reranked rank | Delta | Status |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in grouped.get(mode, []):
            lines.append(
                f"| {row['case_id']} | {row['family']} | {row['base_rank']} | {row['reranked_rank']} | {row['delta']} | {row['status']} |"
            )
        if not grouped.get(mode):
            lines.append("| - | - | - | - | - | - |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _companion_prefix(out_path: Path) -> str:
    return out_path.stem.replace("_comparison", "")


def _style_for_system(system: str) -> Dict[str, str]:
    label = system.lower()
    if "p1" in label:
        color = PHASE4_COLORS["p1"]
    elif "oracle" in label:
        color = PHASE4_COLORS["oracle"]
    elif "g8" in label or "opencv f4" in label:
        color = PHASE4_COLORS["g8"]
    elif "g7" in label:
        color = PHASE4_COLORS["g7"]
    else:
        color = PHASE4_COLORS["fallback"]

    if "cot" in label:
        hatch = VARIANT_HATCHES["cot"]
    elif "single-shot" in label or "single_shot" in label:
        hatch = VARIANT_HATCHES["single_shot"]
    elif "rerank" in label:
        hatch = VARIANT_HATCHES["rerank"]
    else:
        hatch = VARIANT_HATCHES["baseline"]

    return {"color": color, "hatch": hatch}


def _apply_bar_styles(bars: Any, styles: List[Dict[str, str]]) -> None:
    for bar, style in zip(bars, styles):
        bar.set_hatch(style["hatch"])
        bar.set_edgecolor("#252525")
        bar.set_linewidth(0.75)


def write_companion_files(
    *,
    out_path: Path,
    rows: List[Dict[str, float]],
    g7_subset: Dict[str, int],
    p1_subset: Dict[str, int],
    changed_rows: List[Dict[str, Any]],
    top_k: int,
    subset_left_label: str,
    subset_p1_label: str,
) -> None:
    prefix = _companion_prefix(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.parent / f"{prefix}_summary.csv").write_text(_summary_csv_text(rows), encoding="utf-8")
    (out_path.parent / f"{prefix}_summary.md").write_text(
        _summary_md_text(
            rows,
            g7_subset,
            p1_subset,
            top_k,
            subset_left_label=subset_left_label,
            subset_p1_label=subset_p1_label,
        ),
        encoding="utf-8",
    )
    (out_path.parent / f"{prefix}_significant_cases.csv").write_text(
        _changed_cases_csv_text(changed_rows),
        encoding="utf-8",
    )
    (out_path.parent / f"{prefix}_significant_cases.md").write_text(
        _changed_cases_md_text(changed_rows),
        encoding="utf-8",
    )


def plot_figure(
    *,
    rows: List[Dict[str, float]],
    g7_subset: Dict[str, int],
    p1_subset: Dict[str, int],
    out_path: Path,
    top_k: int,
    figure_title: str,
    figure_note: str,
    subset_left_label: str,
    subset_p1_label: str,
) -> None:
    systems = [row["system"] for row in rows]
    top10 = [row["top10"] for row in rows]
    top1 = [row["top1"] for row in rows]
    mrr = [row["mrr10"] for row in rows]
    styles = [_style_for_system(system) for system in systems]
    colors = [style["color"] for style in styles]

    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.06,
        right=0.98,
        top=0.80 if figure_note else 0.86,
        bottom=0.15,
        hspace=0.85,
        wspace=0.62,
    )

    ax_top10 = fig.add_subplot(gs[0, 0])
    ax_top1 = fig.add_subplot(gs[0, 1])
    ax_mrr = fig.add_subplot(gs[0, 2])
    ax_subset = fig.add_subplot(gs[1, :])

    for ax, values, title, fmt in (
        (ax_top10, top10, "A. Top-10", "{:.1f}%"),
        (ax_top1, top1, "B. Top-1", "{:.1f}%"),
        (ax_mrr, mrr, "C. MRR@10", "{:.4f}"),
    ):
        bars = ax.bar(range(len(systems)), values, color=colors)
        _apply_bar_styles(bars, styles)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(systems, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    subset_labels = [
        f"{subset_left_label} top-{top_k}\u2260top-1",
        f"{subset_p1_label} top-{top_k}\u2260top-1",
    ]
    rescued = [g7_subset["rescued"], p1_subset["rescued"]]
    total = [g7_subset["n"], p1_subset["n"]]
    worsened = [g7_subset["worsened"], p1_subset["worsened"]]

    x = list(range(len(subset_labels)))
    subset_colors = [
        _style_for_system(subset_left_label)["color"],
        _style_for_system(subset_p1_label)["color"],
    ]
    bars1 = ax_subset.bar(
        x,
        rescued,
        width=0.42,
        color=subset_colors,
        edgecolor="#252525",
        linewidth=0.75,
        label="rescued to Top-1",
    )
    bars2 = ax_subset.bar(
        [i + 0.42 for i in x],
        worsened,
        width=0.42,
        color=subset_colors,
        alpha=0.35,
        edgecolor="#252525",
        linewidth=0.75,
        hatch="xxx",
        label="worsened inside top-10",
    )
    ax_subset.set_title(
        f"D. Reranker effect on Top-{top_k}-but-not-Top-1 target subsets",
        fontsize=12,
        fontweight="bold",
    )
    ax_subset.set_xticks([i + 0.21 for i in x])
    ax_subset.set_xticklabels(subset_labels)
    ax_subset.set_ylabel("Cases")
    ax_subset.grid(axis="y", alpha=0.25)
    ax_subset.legend(loc="upper right")

    for i, (r, t) in enumerate(zip(rescued, total)):
        ax_subset.text(i, r + 0.15, f"{r}/{t}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i, w in enumerate(worsened):
        ax_subset.text(i + 0.42, w + 0.15, str(w), ha="center", va="bottom", fontsize=10)

    fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.98)
    if figure_note:
        fig.text(
            0.5,
            0.915,
            textwrap.fill(figure_note, width=150),
            ha="center",
            fontsize=9.5,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g7-summary", type=Path, default=G7_SUMMARY_DEFAULT)
    parser.add_argument("--g8-summary", type=Path, default=G8_SUMMARY_DEFAULT)
    parser.add_argument("--g7-results", type=Path, default=G7_RESULTS_DEFAULT)
    parser.add_argument("--g8-results", type=Path, default=G8_RESULTS_DEFAULT)
    # Legacy aliases (ignored — kept for backward compat)
    parser.add_argument("--p1-summary", type=Path, default=None)
    parser.add_argument("--p1-results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--main-out", type=Path, default=MAIN_OUT_DEFAULT)
    parser.add_argument("--appendix-out", type=Path, default=APPENDIX_OUT_DEFAULT)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--g7-label", default="Full-topology (G7)")
    parser.add_argument("--g7-rerank-label", default="Full-topology (G7) + Graph-RAG rerank")
    parser.add_argument("--g8-label", default="Full-topology (G8)")
    parser.add_argument("--g8-rerank-label", default="Full-topology (G8) + Graph-RAG rerank")
    parser.add_argument("--p1-label", default="P1-only (G7 coarse)")
    parser.add_argument("--p1-rerank-label", default="P1-only (G7 coarse) + Graph-RAG rerank")
    parser.add_argument("--p1-g8-rerank-label", default=None)
    parser.add_argument("--oracle-label", default="Oracle")
    parser.add_argument("--subset-left-label", default="Full-topology (G7)")
    parser.add_argument("--subset-p1-label", default="P1-only (G7 coarse)")
    parser.add_argument(
        "--figure-title",
        default="Graph-RAG reranking at top-{top_k}: G7→G8 gain (+3pp Top-10), but reranking hurts full-topology; P1-only benefits",
    )
    parser.add_argument(
        "--figure-note",
        default=(
            "Top-10 remains fixed because the benchmark still scores Top-10/MRR@10, but reranking now operates inside the top-{top_k} shortlist. "
            "The canonical comparison uses phase5 (enriched graph) traces for G7 and G8. The effect remains asymmetric: "
            "Full-topology pipelines degrade under reranking, while the coarse P1-only (G7 coarse) shortlist improves clearly."
        ),
    )
    args = parser.parse_args()

    g7_summary = _load_json(args.g7_summary)
    g8_summary = _load_json(args.g8_summary)
    g7_rows = _load_jsonl(args.g7_results)
    g8_rows = _load_jsonl(args.g8_results)

    rows = build_rows(
        g7_summary,
        g8_summary,
        g7_label=args.g7_label,
        g7_rerank_label=args.g7_rerank_label,
        g8_label=args.g8_label,
        g8_rerank_label=args.g8_rerank_label,
        p1_label=args.p1_label,
        p1_rerank_label=args.p1_rerank_label,
        p1_g8_rerank_label=args.p1_g8_rerank_label,
        oracle_label=args.oracle_label,
    )
    g7_subset = _subset_stats(g7_rows, args.top_k, mode="g7_pipeline")
    p1_subset = _subset_stats(g7_rows, args.top_k, mode="p1_only")
    changed_rows = (
        _changed_cases(g7_rows, "g7_pipeline", args.g7_rerank_label)
        + _changed_cases(g8_rows, "g7_pipeline", args.g8_rerank_label)
        + _changed_cases(g7_rows, "p1_only", args.p1_rerank_label)
        + _changed_cases(g8_rows, "p1_only", args.p1_g8_rerank_label or "P1-only second reranker")
    )

    for out_path in [args.out, args.main_out, args.appendix_out]:
        plot_figure(
            rows=rows,
            g7_subset=g7_subset,
            p1_subset=p1_subset,
            out_path=out_path,
            top_k=args.top_k,
            figure_title=args.figure_title.format(top_k=args.top_k),
            figure_note=args.figure_note.format(top_k=args.top_k),
            subset_left_label=args.subset_left_label,
            subset_p1_label=args.subset_p1_label,
        )
        write_companion_files(
            out_path=out_path,
            rows=rows,
            g7_subset=g7_subset,
            p1_subset=p1_subset,
            changed_rows=changed_rows,
            top_k=args.top_k,
            subset_left_label=args.subset_left_label,
            subset_p1_label=args.subset_p1_label,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
