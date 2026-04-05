#!/usr/bin/env python3
"""Create a final Graph-RAG comparison figure for thesis plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_RAG_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "graph_rag_rerank"
DOCS_MAIN_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_main"
DOCS_APPENDIX_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_appendix"

G7_SUMMARY_DEFAULT = GRAPH_RAG_ROOT / "20260405_top15_g7_v1" / "graph_rag_rerank_summary.json"
P1_SUMMARY_DEFAULT = GRAPH_RAG_ROOT / "20260405_top15_p1_v1" / "graph_rag_rerank_summary.json"
G7_RESULTS_DEFAULT = GRAPH_RAG_ROOT / "20260405_top15_g7_v1" / "graph_rag_rerank_results.jsonl"
P1_RESULTS_DEFAULT = GRAPH_RAG_ROOT / "20260405_top15_p1_v1" / "graph_rag_rerank_results.jsonl"
OUT_DEFAULT = GRAPH_RAG_ROOT / "20260405_top15_g7_v1" / "graph_rag_rerank_comparison.png"
MAIN_OUT_DEFAULT = DOCS_MAIN_DIR / "fig11_graph_rag_rerank_comparison.png"
APPENDIX_OUT_DEFAULT = DOCS_APPENDIX_DIR / "figA6_graph_rag_rerank_comparison.png"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _subset_stats(rows: List[Dict[str, Any]], top_k: int) -> Dict[str, int]:
    subset = [r for r in rows if 2 <= int(r.get("base_rank", -1)) <= top_k]
    rescued = sum(1 for r in subset if int(r.get("reranked_rank", -1)) == 1)
    worsened = sum(
        1
        for r in subset
        if int(r.get("reranked_rank", -1)) > int(r.get("base_rank", -1)) > 0
    )
    return {"n": len(subset), "rescued": rescued, "worsened": worsened}


def _changed_cases(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    changed: List[Dict[str, Any]] = []
    for row in rows:
        base_rank = int(row.get("base_rank", -1))
        reranked_rank = int(row.get("reranked_rank", -1))
        if base_rank <= 0 or reranked_rank <= 0 or base_rank == reranked_rank:
            continue
        status = "improved" if reranked_rank < base_rank else "worsened"
        if reranked_rank == 1 and base_rank > 1:
            status = "rescued_to_top1"
        changed.append(
            {
                "mode": mode,
                "case_id": row.get("case_id", ""),
                "family": row.get("family", ""),
                "base_rank": base_rank,
                "reranked_rank": reranked_rank,
                "delta": reranked_rank - base_rank,
                "status": status,
            }
        )
    return changed


def build_rows(g7_summary: Dict[str, Any], p1_summary: Dict[str, Any]) -> List[Dict[str, float]]:
    return [
        {
            "system": "Full-topology (G7)",
            "top10": g7_summary["modes"]["g7_pipeline"]["baseline"]["top10_pct"],
            "top1": g7_summary["modes"]["g7_pipeline"]["baseline"]["top1_pct"],
            "mrr10": g7_summary["modes"]["g7_pipeline"]["baseline"]["mrr10"],
        },
        {
            "system": "Full-topology (G7) + Graph-RAG rerank",
            "top10": g7_summary["modes"]["g7_pipeline"]["reranked"]["top10_pct"],
            "top1": g7_summary["modes"]["g7_pipeline"]["reranked"]["top1_pct"],
            "mrr10": g7_summary["modes"]["g7_pipeline"]["reranked"]["mrr10"],
        },
        {
            "system": "P1-only (G7 coarse)",
            "top10": p1_summary["modes"]["p1_only"]["baseline"]["top10_pct"],
            "top1": p1_summary["modes"]["p1_only"]["baseline"]["top1_pct"],
            "mrr10": p1_summary["modes"]["p1_only"]["baseline"]["mrr10"],
        },
        {
            "system": "P1-only (G7 coarse) + Graph-RAG rerank",
            "top10": p1_summary["modes"]["p1_only"]["reranked"]["top10_pct"],
            "top1": p1_summary["modes"]["p1_only"]["reranked"]["top1_pct"],
            "mrr10": p1_summary["modes"]["p1_only"]["reranked"]["mrr10"],
        },
        {"system": "Oracle", "top10": 40.0, "top1": 5.0, "mrr10": 0.1279},
    ]


def _summary_csv_text(rows: List[Dict[str, float]]) -> str:
    lines = ["system,top10,top1,mrr10"]
    for row in rows:
        lines.append(
            f"{row['system']},{row['top10']:.1f},{row['top1']:.1f},{row['mrr10']:.4f}"
        )
    return "\n".join(lines) + "\n"


def _summary_md_text(
    rows: List[Dict[str, float]], g7_subset: Dict[str, int], p1_subset: Dict[str, int], top_k: int
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
            f"- `Full-topology (G7)`: `{g7_subset['rescued']}/{g7_subset['n']}` rescued to Top-1, `{g7_subset['worsened']}` worsened",
            f"- `P1-only (G7 coarse)`: `{p1_subset['rescued']}/{p1_subset['n']}` rescued to Top-1, `{p1_subset['worsened']}` worsened",
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

    mode_titles = {
        "g7_pipeline": "Full-topology (G7) + Graph-RAG rerank changed cases",
        "p1_only": "P1-only (G7 coarse) + Graph-RAG rerank changed cases",
    }

    lines = ["# Significant Graph-RAG Rerank Cases", ""]
    for mode in ["g7_pipeline", "p1_only"]:
        lines.append(f"## {mode_titles.get(mode, mode)}")
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


def write_companion_files(
    *,
    out_path: Path,
    rows: List[Dict[str, float]],
    g7_subset: Dict[str, int],
    p1_subset: Dict[str, int],
    changed_rows: List[Dict[str, Any]],
    top_k: int,
) -> None:
    prefix = _companion_prefix(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.parent / f"{prefix}_summary.csv").write_text(_summary_csv_text(rows), encoding="utf-8")
    (out_path.parent / f"{prefix}_summary.md").write_text(
        _summary_md_text(rows, g7_subset, p1_subset, top_k),
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
) -> None:
    systems = [row["system"] for row in rows]
    top10 = [row["top10"] for row in rows]
    top1 = [row["top1"] for row in rows]
    mrr = [row["mrr10"] for row in rows]

    fig = plt.figure(figsize=(15.5, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_top10 = fig.add_subplot(gs[0, 0])
    ax_top1 = fig.add_subplot(gs[0, 1])
    ax_mrr = fig.add_subplot(gs[0, 2])
    ax_subset = fig.add_subplot(gs[1, :])

    palette = ["#0F766E", "#14B8A6", "#7C3AED", "#A78BFA", "#D97706"]

    for ax, values, title, fmt in (
        (ax_top10, top10, "A. Top-10", "{:.1f}%"),
        (ax_top1, top1, "B. Top-1", "{:.1f}%"),
        (ax_mrr, mrr, "C. MRR@10", "{:.4f}"),
    ):
        bars = ax.bar(range(len(systems)), values, color=palette)
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

    subset_labels = ["Full-topology (G7) target subset", "P1-only (G7 coarse) target subset"]
    rescued = [g7_subset["rescued"], p1_subset["rescued"]]
    total = [g7_subset["n"], p1_subset["n"]]
    worsened = [g7_subset["worsened"], p1_subset["worsened"]]

    x = list(range(len(subset_labels)))
    bars1 = ax_subset.bar(x, rescued, width=0.42, color=["#14B8A6", "#A78BFA"], label="rescued to Top-1")
    bars2 = ax_subset.bar(
        [i + 0.42 for i in x],
        worsened,
        width=0.42,
        color=["#99F6E4", "#E9D5FF"],
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

    fig.suptitle(
        f"Graph-RAG reranking at top-{top_k}: weak on Full-topology (G7), stronger on P1-only (G7 coarse)",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        f"Top-10 remains fixed because the benchmark still scores Top-10/MRR@10, but reranking now operates inside the top-{top_k} shortlist. "
        "The canonical comparison uses the 2026-04-05 top-15 rerank runs. The effect remains asymmetric: "
        "Full-topology (G7) degrades under reranking, while the coarse P1-only (G7 coarse) shortlist improves clearly.",
        ha="center",
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g7-summary", type=Path, default=G7_SUMMARY_DEFAULT)
    parser.add_argument("--p1-summary", type=Path, default=P1_SUMMARY_DEFAULT)
    parser.add_argument("--g7-results", type=Path, default=G7_RESULTS_DEFAULT)
    parser.add_argument("--p1-results", type=Path, default=P1_RESULTS_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--main-out", type=Path, default=MAIN_OUT_DEFAULT)
    parser.add_argument("--appendix-out", type=Path, default=APPENDIX_OUT_DEFAULT)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    g7_summary = _load_json(args.g7_summary)
    p1_summary = _load_json(args.p1_summary)
    g7_rows = _load_jsonl(args.g7_results)
    p1_rows = _load_jsonl(args.p1_results)

    rows = build_rows(g7_summary, p1_summary)
    g7_subset = _subset_stats(g7_rows, args.top_k)
    p1_subset = _subset_stats(p1_rows, args.top_k)
    changed_rows = _changed_cases(g7_rows, "g7_pipeline") + _changed_cases(p1_rows, "p1_only")

    for out_path in [args.out, args.main_out, args.appendix_out]:
        plot_figure(rows=rows, g7_subset=g7_subset, p1_subset=p1_subset, out_path=out_path, top_k=args.top_k)
        write_companion_files(
            out_path=out_path,
            rows=rows,
            g7_subset=g7_subset,
            p1_subset=p1_subset,
            changed_rows=changed_rows,
            top_k=args.top_k,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
