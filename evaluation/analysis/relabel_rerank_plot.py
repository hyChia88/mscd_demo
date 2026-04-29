#!/usr/bin/env python3
"""Temporary: relabel an existing graph_rag_rerank_summary.json's plot + markdown
without re-running Gemini. Reads the summary JSON, rebuilds comparison_rows with
a new --label, rewrites comparison.png and summary.md in place.

Usage:
  python evaluation/analysis/relabel_rerank_plot.py \\
      --rerank-dir mscd_demo/output/.../phase6_1_g9_soft_top30_v2 \\
      --label G9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.experiments.graph_rag_rerank_ap import (  # noqa: E402
    REFERENCE_ROWS,
    _build_mode_labels,
    _plot_comparison,
    _write_summary_md,
)


def _normalise_modes(d: dict) -> dict:
    modes = d.get("modes")
    if isinstance(modes, dict) and "g7_pipeline" in modes and "full_topology" not in modes:
        modes["full_topology"] = modes.pop("g7_pipeline")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rerank-dir", type=Path, required=True,
                    help="Dir containing graph_rag_rerank_summary.json")
    ap.add_argument("--label", required=True, help="System label, e.g. G9")
    args = ap.parse_args()

    summary_path = args.rerank_dir / "graph_rag_rerank_summary.json"
    summary = _normalise_modes(json.loads(summary_path.read_text(encoding="utf-8")))
    top_k = int(summary.get("meta", {}).get("top_k", 10))

    base_labels, rerank_labels = _build_mode_labels(args.label)
    modes = summary.get("modes", {})

    def cell(mode: str, kind: str, key: str) -> float:
        return float(modes.get(mode, {}).get(kind, {}).get(key, 0.0))

    comparison_rows = [
        {"system": base_labels["full_topology"],
         "top10": cell("full_topology", "baseline", "top10_pct"),
         "top1":  cell("full_topology", "baseline", "top1_pct"),
         "mrr10": cell("full_topology", "baseline", "mrr10")},
        {"system": rerank_labels["full_topology"],
         "top10": cell("full_topology", "reranked", "top10_pct"),
         "top1":  cell("full_topology", "reranked", "top1_pct"),
         "mrr10": cell("full_topology", "reranked", "mrr10")},
        {"system": base_labels["p1_only"],
         "top10": cell("p1_only", "baseline", "top10_pct"),
         "top1":  cell("p1_only", "baseline", "top1_pct"),
         "mrr10": cell("p1_only", "baseline", "mrr10")},
        {"system": rerank_labels["p1_only"],
         "top10": cell("p1_only", "reranked", "top10_pct"),
         "top1":  cell("p1_only", "reranked", "top1_pct"),
         "mrr10": cell("p1_only", "reranked", "mrr10")},
        *REFERENCE_ROWS,
    ]

    _plot_comparison(args.rerank_dir / "graph_rag_rerank_comparison.png",
                     comparison_rows, top_k, system_label=args.label)
    _write_summary_md(args.rerank_dir / "graph_rag_rerank_summary.md",
                      summary, comparison_rows, top_k, system_label=args.label)
    print(f"Re-rendered plot + summary in {args.rerank_dir} with label={args.label!r}")


if __name__ == "__main__":
    main()
