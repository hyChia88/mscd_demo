"""Sankey + waterfall figure illustrating the P0 ∪ P1 → top-K → rerank flow.

Standalone helper that visualises why P0's 'shortlist moat' survives only at the
top-K cut, not inside the rerank room itself. Uses the same trace case as
fig00 (AP_SK_228) so numbers match the rest of the plot suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_PATH = REPO_ROOT / "docs/plots/final/data/symbolic_reasoning_trace_case.json"
OUT_DIR = REPO_ROOT / "docs/plots/final"
OUT_BASE = OUT_DIR / "fig00c_p0_topk_flow"

# ── Palette (matches fig00) ─────────────────────────────────────────────────
COL_P0 = "#059669"          # green — P0 topology winners
COL_P1_ONLY = "#7C3AED"     # purple — P1-only fillers
COL_DROP = "#CBD5E1"        # grey — P1-only dropped at top-K cut
COL_GT = "#16A34A"          # GT trajectory


def _load_case() -> dict:
    return json.loads(CASE_PATH.read_text(encoding="utf-8"))


def _ribbon(ax, x0: float, x1: float, y0_l: float, y1_l: float,
            y0_r: float, y1_r: float, color: str, alpha: float = 0.55) -> None:
    """Draw a Bezier-curved Sankey ribbon between two stages."""
    midx = (x0 + x1) / 2.0
    verts = [
        (x0, y0_l), (midx, y0_l), (midx, y0_r), (x1, y0_r),  # top edge
        (x1, y1_r), (midx, y1_r), (midx, y1_l), (x0, y1_l),  # bottom edge
        (x0, y0_l),
    ]
    codes = [MplPath.MOVETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    patch = PathPatch(MplPath(verts, codes), facecolor=color,
                      edgecolor="none", alpha=alpha, zorder=2)
    ax.add_patch(patch)


def _stage_block(ax, x: float, y_top: float, h: float, label: str,
                 count_text: str, color: str, width: float = 0.04,
                 label_y: float | None = None,
                 count_y: float | None = None) -> None:
    rect = FancyBboxPatch((x - width / 2, y_top - h), width, h,
                          boxstyle="round,pad=0.002,rounding_size=0.005",
                          linewidth=0.0, facecolor=color, edgecolor="none",
                          alpha=0.95, zorder=4)
    ax.add_patch(rect)
    if label:
        ly = label_y if label_y is not None else y_top + 0.025
        ax.text(x, ly, label, ha="center", va="bottom",
                fontsize=10.2, fontweight="bold", color="#0F172A")
    if count_text:
        cy = count_y if count_y is not None else y_top - h - 0.03
        ax.text(x, cy, count_text, ha="center", va="top",
                fontsize=9.3, color="#334155")


def _plot(case: dict) -> plt.Figure:
    p0 = int(case["p0_pool_size"])                  # 3
    p1_total = int(case["p1_pool_size"])            # 33
    union = int(case["union_pool_size"])            # 33  (P0 ⊆ P1 here)
    p1_only = union - p0                            # 30
    top_k = 10
    p0_in_topk = min(p0, top_k)                     # 3
    p1_only_in_topk = top_k - p0_in_topk            # 7
    p1_only_dropped = p1_only - p1_only_in_topk     # 23
    top5 = 5
    gt_base = case.get("base_rank")                 # 3
    gt_rerank = case.get("reranked_rank")           # 1
    case_id = case["case_id"]
    all_ifc = int(case.get("initial_pool_size", 1666))

    fig = plt.figure(figsize=(14.0, 9.0), constrained_layout=False)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.85, 1.0],
                            hspace=0.42, left=0.05, right=0.97,
                            top=0.88, bottom=0.07)
    ax_top = fig.add_subplot(grid[0])
    ax_bot = fig.add_subplot(grid[1])

    # ── Title ───────────────────────────────────────────────────────────────
    fig.suptitle(
        "P0 ∪ P1 → Top-K → Rerank: Where the P0 advantage is preserved vs lost",
        fontsize=15.5, fontweight="bold", y=0.965)
    fig.text(0.05, 0.928,
             f"Trace: {case_id}  |  Query: {case['query_text']}  |  "
             f"GT rank {gt_base} → {gt_rerank} after rerank",
             ha="left", va="top", fontsize=9.6, color="#475569")

    # ── TOP PANEL: Sankey flow ──────────────────────────────────────────────
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    # Layout zones (y-axis):
    #   0.78 - 1.00 : annotation boxes
    #   0.66 - 0.74 : stage labels (header strip)
    #   0.28 - 0.62 : main sankey (live pool of 33)
    #   0.05 - 0.22 : dropped P1-only branch
    UNIT = 0.34 / union               # height per candidate in main sankey
    pool_top_y = 0.62
    pool_bot_y = pool_top_y - union * UNIT  # = 0.28

    X = {
        "p1_pool": 0.13,
        "split": 0.36,
        "topk": 0.62,
        "rerank_out": 0.87,
    }

    # ── Stage labels (header strip, y=0.69) ──
    label_y = 0.69
    for key, txt in [("p1_pool", "P1 valid pool"),
                     ("split", "P0  /  P1-only"),
                     ("topk", "Top-K (rerank input)"),
                     ("rerank_out", "Final Top-5")]:
        ax_top.text(X[key], label_y, txt, ha="center", va="bottom",
                    fontsize=10.6, fontweight="bold", color="#0F172A")

    # ── Stage 1: P1 valid pool ──
    h_pool = union * UNIT
    _stage_block(ax_top, X["p1_pool"], pool_top_y, h_pool, "",
                 f"{union} candidates\n(storey + IfcWindow)",
                 COL_P1_ONLY, width=0.045,
                 count_y=pool_bot_y - 0.02)

    # ── Stage 2: P0 / P1-only split ──
    h_p0 = p0 * UNIT
    h_p1only = p1_only * UNIT
    p0_top_split = pool_top_y
    p0_bot_split = p0_top_split - h_p0
    p1_top_split = p0_bot_split
    p1_bot_split = p1_top_split - h_p1only

    _stage_block(ax_top, X["split"], p0_top_split, h_p0, "",
                 "", COL_P0, width=0.045)
    _stage_block(ax_top, X["split"], p1_top_split, h_p1only, "",
                 "", COL_P1_ONLY, width=0.045)
    # Inline count callouts to right of split blocks
    ax_top.text(X["split"] + 0.030, p0_bot_split + h_p0 / 2,
                f"P0 = {p0}  ← GT @ rank {gt_base}",
                ha="left", va="center", fontsize=9.2,
                color=COL_P0, fontweight="bold")
    ax_top.text(X["split"] + 0.030, p1_bot_split + h_p1only / 2,
                f"P1-only = {p1_only}\n(no topology hit)",
                ha="left", va="center", fontsize=9.0,
                color=COL_P1_ONLY)

    # Ribbon: P1 pool → P0 + P1-only
    _ribbon(ax_top,
            X["p1_pool"] + 0.022, X["split"] - 0.022,
            pool_top_y, pool_top_y - h_p0,
            p0_top_split, p0_bot_split, COL_P0, alpha=0.50)
    _ribbon(ax_top,
            X["p1_pool"] + 0.022, X["split"] - 0.022,
            pool_top_y - h_p0, pool_bot_y,
            p1_top_split, p1_bot_split, COL_P1_ONLY, alpha=0.42)

    # ── Stage 3: Top-K cut ──
    h_p0_topk = p0_in_topk * UNIT
    h_p1only_topk = p1_only_in_topk * UNIT
    h_dropped = p1_only_dropped * UNIT

    p0_top_topk = pool_top_y
    p0_bot_topk = p0_top_topk - h_p0_topk
    p1k_top = p0_bot_topk
    p1k_bot = p1k_top - h_p1only_topk

    _stage_block(ax_top, X["topk"], p0_top_topk, h_p0_topk, "",
                 "", COL_P0, width=0.045)
    _stage_block(ax_top, X["topk"], p1k_top, h_p1only_topk, "",
                 "", COL_P1_ONLY, width=0.045)
    ax_top.text(X["topk"] + 0.030, p0_bot_topk + h_p0_topk / 2 + 0.01,
                f"K = {top_k}\n= {p0_in_topk} P0 + {p1_only_in_topk} P1-only",
                ha="left", va="center", fontsize=9.0, color="#0F172A")

    # Dropped P1-only branch (curves down to a grey "discarded" bar)
    drop_y_top = 0.20
    drop_y_bot = drop_y_top - h_dropped
    _stage_block(ax_top, X["topk"], drop_y_top, h_dropped, "",
                 "", COL_DROP, width=0.045)
    ax_top.text(X["topk"] + 0.030, drop_y_top - h_dropped / 2,
                f"{p1_only_dropped} P1-only DROPPED\n(never seen by rerank)",
                ha="left", va="center", fontsize=9.0,
                color="#475569", fontweight="bold")

    # Ribbons: P0 carries all 3 forward; P1-only splits 7 forward + 23 dropped
    _ribbon(ax_top,
            X["split"] + 0.022, X["topk"] - 0.022,
            p0_top_split, p0_bot_split,
            p0_top_topk, p0_bot_topk, COL_P0, alpha=0.55)
    _ribbon(ax_top,
            X["split"] + 0.022, X["topk"] - 0.022,
            p1_top_split, p1_top_split - h_p1only_topk,
            p1k_top, p1k_bot, COL_P1_ONLY, alpha=0.55)
    _ribbon(ax_top,
            X["split"] + 0.022, X["topk"] - 0.022,
            p1_top_split - h_p1only_topk, p1_bot_split,
            drop_y_top, drop_y_bot, COL_DROP, alpha=0.65)

    # ── Stage 4: Final Top-5 ──
    h_top5 = top5 * UNIT
    top5_top = pool_top_y
    top5_bot = top5_top - h_top5
    _stage_block(ax_top, X["rerank_out"], top5_top, h_top5, "",
                 "", COL_GT, width=0.045)
    ax_top.text(X["rerank_out"], top5_bot - 0.025,
                f"Top-{top5}  (after Gemini rerank)\nGT promoted to #1",
                ha="center", va="top", fontsize=9.2,
                color="#065F46", fontweight="bold")

    _ribbon(ax_top,
            X["topk"] + 0.022, X["rerank_out"] - 0.022,
            p0_top_topk, p1k_bot,
            top5_top, top5_bot, "#34D399", alpha=0.45)

    # ── Annotations: where P0 wins / loses ──
    ax_top.text(0.13, 0.965,
                "✅ P0 advantage PRESERVED",
                ha="left", va="top", fontsize=10.0, fontweight="bold",
                color="#065F46")
    ax_top.text(0.13, 0.93,
                "Top-K cut keeps P0 candidates at the front;\n"
                f"{p1_only_dropped} P1-only candidates are eliminated\n"
                "before the rerank ever sees them.",
                ha="left", va="top", fontsize=8.9, color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5",
                          edgecolor="#059669", linewidth=0.8))

    ax_top.text(0.55, 0.965,
                "⚠️ P0 advantage LOST inside the rerank room",
                ha="left", va="top", fontsize=10.0, fontweight="bold",
                color="#9A3412")
    ax_top.text(0.55, 0.93,
                "fusion_score = band + slot only — source tag is\n"
                "dropped, so a P1-only candidate with strong band+slot\n"
                "match can outrank a P0 candidate inside top-K.",
                ha="left", va="top", fontsize=8.9, color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF7ED",
                          edgecolor="#EA580C", linewidth=0.8))

    # Pre-stage label: All IFC → P1 backstop
    ax_top.text(0.02, (pool_top_y + pool_bot_y) / 2,
                f"All IFC\n{all_ifc}\n──filter──→",
                ha="left", va="center", fontsize=9.0, color="#475569",
                fontweight="bold")

    # ── BOTTOM PANEL: Waterfall with source stratification ─────────────────
    stages = ["All IFC", "P1 valid", "P0 ∪ P1\n(ordered)",
              "Top-K\n(rerank input)", "Final Top-5"]
    p0_counts = [0, p0, p0, p0_in_topk, 0]              # P0 stratum (we don't know top5 split)
    p1_counts = [0, p1_only, p1_only, p1_only_in_topk, 0]
    drop_counts = [all_ifc - union, 0, 0, p1_only_dropped, top_k - top5]
    total_counts = [all_ifc, union, union, top_k, top5]

    x_idx = np.arange(len(stages))
    width = 0.55

    # Stack: P0 (green) bottom, P1-only (purple) middle, dropped (grey) top
    ax_bot.bar(x_idx, p0_counts, width=width, color=COL_P0,
               label="P0 (topology hit)", zorder=3)
    ax_bot.bar(x_idx, p1_counts, width=width, bottom=p0_counts,
               color=COL_P1_ONLY, label="P1-only (storey+class)", zorder=3)
    ax_bot.bar(x_idx, drop_counts, width=width,
               bottom=np.array(p0_counts) + np.array(p1_counts),
               color=COL_DROP, label="Dropped / not retrieved",
               alpha=0.7, zorder=3)

    # Total labels
    for i, total in enumerate(total_counts):
        ax_bot.text(i, total * 1.05 if total > 0 else 1, str(total),
                    ha="center", va="bottom", fontsize=10.5,
                    fontweight="bold", color="#0F172A", zorder=5)

    # Highlight the top-K cut event
    ax_bot.annotate("", xy=(3, p1_only_in_topk + p0_in_topk + 1.5),
                    xytext=(2.55, p1_only_in_topk + p0_in_topk + 1.5),
                    arrowprops=dict(arrowstyle="->", color="#EA580C", lw=1.4))
    ax_bot.text(2.78, p1_only_in_topk + p0_in_topk + 4,
                f"Top-K cut\n(K=10)\n−{p1_only_dropped} P1-only",
                ha="center", va="bottom", fontsize=8.7, color="#9A3412",
                fontweight="bold")

    ax_bot.set_yscale("symlog", linthresh=1)
    ax_bot.set_ylabel("Candidates (symlog)")
    ax_bot.set_xticks(x_idx)
    ax_bot.set_xticklabels(stages)
    ax_bot.set_ylim(0, all_ifc * 2.0)
    ax_bot.grid(axis="y", which="major", alpha=0.25, zorder=0)
    ax_bot.set_title("Candidate compression — coloured by source", fontsize=11.5)
    ax_bot.legend(loc="upper right", fontsize=8.8, framealpha=0.95,
                  edgecolor="#94A3B8")

    return fig


def main() -> None:
    case = _load_case()
    fig = _plot(case)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_BASE.with_suffix(".png")
    pdf = OUT_BASE.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
