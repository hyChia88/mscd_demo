#!/usr/bin/env python3
"""Presentation plot suite — 7 figures matching the SR narrative arc.

Each figure corresponds to a single "beat" of the talk and is sized for
16:9 slide use (1920×1080 at 220 dpi):

  Beat 1 — Attribute entropy bottleneck (problem)
  Beat 2 — Relational signal lives in the graph (insight)
  Beat 3 — Mechanism: P0/P1/top-K Sankey + waterfall
  Beat 4 — Evidence: 4-model Top-1 vs baseline
  Beat 5 — Oracle ceiling (system capability vs current LoRA)
  Beat 6 — Per-predicate SSR honesty (where SR fails)
  Beat 7 — Forward-looking: the top-K bottleneck

Reads from the curated CSVs already produced by
generate_final_plot_suite.py (under docs/plots/final/data/) plus a few
hardcoded numbers documented in MEMORY.md.

Usage:
    python generate_final_plot_suite_short.py
    python generate_final_plot_suite_short.py --only beat3 beat4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mscd_demo_matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from phase4_plot_style import GRAPH_RAG_COLORS, HIGHLIGHT_COLORS, METRIC_COLORS, MODELS, STRATEGIES

# ── Paths ───────────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent.parent
FINAL_DATA_DIR = PROJECT_ROOT / "docs" / "plots" / "final" / "data"
OUT_DIR = PROJECT_ROOT / "docs" / "plots" / "presentation"

# ── Palette (shared Phase 4 source of truth) ───────────────────────────────
COL_P0 = STRATEGIES.get("p0_spatial_relation", "#059669")
COL_P1_ONLY = STRATEGIES.get("p1_storey_class", "#7C3AED")
COL_DROP = "#CBD5E1"
COL_GT = METRIC_COLORS.get("ideal_top10", "#16A34A")
COL_BASELINE = MODELS.get("g6_baseline", "#90A4AE")
COL_HIGHLIGHT = HIGHLIGHT_COLORS.get("rerank_orange", "#EA580C")
COL_INK = "#0F172A"
COL_MUTED = "#475569"
COL_GEMINI = MODELS.get("gemini_ap_v2", "#1565C0")
COL_G7 = MODELS.get("g7_position_context", "#6A1B9A")
COL_G9_BASE = MODELS.get("g9_resnet_f4", "#0F766E")
COL_G9_RERANK = HIGHLIGHT_COLORS.get("rerank_orange", "#EA580C")
COL_ORACLE = GRAPH_RAG_COLORS.get("oracle", "#190433")
COL_MRR = METRIC_COLORS.get("mrr_track", "#1565C0")
COL_SAFE_TEXT = HIGHLIGHT_COLORS.get("safe_green_text", "#166534")
COL_ORACLE_TEXT = HIGHLIGHT_COLORS.get("oracle_text", "#5B21B6")
COL_AMBER_TEXT = HIGHLIGHT_COLORS.get("winner_amber_text", "#9A3412")

SLIDE_FIGSIZE = (10.8, 8.1)  # 4:3 at 100 dpi → matches FIGSIZE_4X3_TIGHT

P0_INLINE = "P0 (spatial relation)"
P1_INLINE = "P1 (storey + IFC class)"
P1_ONLY_INLINE = "P1-only (storey + IFC class)"
G7_SHORT_LABEL = "G7\nposition\ncontext"
G9_BASE_SHORT_LABEL = "G9\nOpenCV/\nResNet cues"
G9_RERANK_SHORT_LABEL = "G9\nOpenCV/ResNet\n+ rerank"

# ── Beat-4 G-series rows (read from lora_vs_gemini.csv at run time) ────────
# Order = narrative order (worst → best learned → ceiling)
BEAT4_SYSTEMS = [
    ("Gemini AP (MM)", "Gemini\n(zero-shot)", COL_GEMINI),
    ("G7 Position Context (MM)", G7_SHORT_LABEL, COL_G7),
    ("G9 + OpenCV F4 + ResNet", G9_BASE_SHORT_LABEL, COL_G9_BASE),
    ("G9 + OpenCV F4 + ResNet + Graph-RAG", G9_RERANK_SHORT_LABEL, COL_G9_RERANK),
    ("Oracle ceiling", "Oracle\nceiling", COL_ORACLE),
]

PER_PREDICATE_SSR = [
    ("ADJACENT_TO", 30.0, 34, "Heterogeneous anchors\n(Door↔Window)"),
    ("CONTINUOUS", 65.0, 21, "Multi-storey wall spans"),
    ("FILLS", 0.0, 28, "Same-storey windows all\nFILL walls — degenerate"),
]


# ── Tiny IO helpers ─────────────────────────────────────────────────────────
def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, base: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {png.relative_to(PROJECT_ROOT)}")


def _configure_mpl() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Beat 1 — Attribute entropy bottleneck
# ─────────────────────────────────────────────────────────────────────────────
def beat1_entropy_bottleneck() -> plt.Figure:
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=SLIDE_FIGSIZE, gridspec_kw={"width_ratios": [1.0, 1.15]})
    fig.suptitle("The attribute entropy bottleneck",
                 fontsize=18, fontweight="bold", y=0.96)
    fig.text(0.5, 0.918,
             "BIM duplicates identical elements by design.",
             ha="center", va="top", fontsize=12, color=COL_MUTED)

    # Left: schematic of 46 identical windows on one floor
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 10)
    ax_left.axis("off")
    ax_left.set_title("AdvancedProject.ifc — Floor 2",
                      fontsize=12, color=COL_INK, loc="left", pad=12)

    # Wall
    ax_left.plot([0.5, 9.5], [3.5, 3.5], color=COL_INK, lw=2.0)
    ax_left.plot([0.5, 9.5], [6.5, 6.5], color=COL_INK, lw=2.0)
    # 46 windows arranged in 2 rows of 23
    rng = np.linspace(0.8, 9.2, 23)
    for x in rng:
        ax_left.add_patch(FancyBboxPatch(
            (x - 0.13, 4.6), 0.26, 0.8,
            boxstyle="round,pad=0.01", linewidth=0.6,
            facecolor="#DBEAFE", edgecolor="#1D4ED8"))
        ax_left.add_patch(FancyBboxPatch(
            (x - 0.13, 5.6), 0.26, 0.8,
            boxstyle="round,pad=0.01", linewidth=0.6,
            facecolor="#DBEAFE", edgecolor="#1D4ED8"))
    ax_left.text(5.0, 7.7, "46 identical IfcWindows on this floor",
                 ha="center", va="bottom", fontsize=11.5,
                 fontweight="bold", color=COL_INK)
    ax_left.annotate("Query: 'the window next to the door'",
                     xy=(5.0, 6.4), xytext=(5.0, 1.7),
                     ha="center", fontsize=11, color=COL_HIGHLIGHT,
                     arrowprops=dict(arrowstyle="->", color=COL_HIGHLIGHT, lw=1.4))
    ax_left.text(5.0, 0.8,
                 "Same IfcType, same storey, same material —\n"
                 "attribute predicates can't disambiguate.",
                 ha="center", va="top", fontsize=10.5, color=COL_MUTED)

    # Right: bar chart of expected accuracy
    ax_right.set_title("Top-1 retrieval accuracy",
                       fontsize=12, color=COL_INK, loc="left", pad=12)
    bars_data = [
        ("Random pick (1/46)", 2.2, COL_DROP),
        ("Attribute filter only", 3.0, COL_BASELINE),
        ("LoRA + SR (this work)", 53.4, COL_P0),
    ]
    ys = np.arange(len(bars_data))[::-1]
    for y, (label, val, color) in zip(ys, bars_data):
        ax_right.barh(y, val, height=0.55, color=color, zorder=3)
        ax_right.text(val + 1.5, y, f"{val:.1f}%",
                      va="center", fontsize=12, fontweight="bold", color=COL_INK)
        ax_right.text(-1, y, label, va="center", ha="right",
                      fontsize=11.5, color=COL_INK)
    ax_right.set_xlim(-30, 70)
    ax_right.set_ylim(-0.6, len(bars_data) - 0.4)
    ax_right.set_yticks([])
    ax_right.set_xticks([0, 25, 50])
    ax_right.set_xticklabels(["0%", "25%", "50%"])
    ax_right.grid(axis="x", alpha=0.25, zorder=0)
    ax_right.spines["left"].set_visible(False)
    ax_right.text(53.4, -0.45,
                  "  17.8× lift over attribute baseline",
                  ha="left", va="top", fontsize=10.5, color=COL_P0,
                  fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Beat 2 — Relational signal lives in the graph
# ─────────────────────────────────────────────────────────────────────────────
def beat2_relational_signal() -> plt.Figure:
    case = _read_json(FINAL_DATA_DIR / "symbolic_reasoning_trace_case.json")
    fig = plt.figure(figsize=SLIDE_FIGSIZE)
    fig.suptitle("Spatial relations in IFC are typed and queryable",
                 fontsize=18, fontweight="bold", y=0.96)
    fig.text(0.5, 0.905,
             "Language → SpatialTriplet (LoRA) → Cypher (Neo4j) → "
             "IFC-valid GUIDs — every step is inspectable.",
             ha="center", va="top", fontsize=11.5, color=COL_MUTED)

    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    import textwrap
    box_w, box_h, y0 = 0.205, 0.62, 0.13
    boxes = [
        (0.020, "1. Natural language",      "#EFF6FF", "#2563EB"),
        (0.252, "2. SpatialTriplet (LoRA)", "#F5F3FF", "#7C3AED"),
        (0.504, "3. Cypher (Neo4j)",        "#ECFDF5", "#059669"),
        (0.756, "4. IFC GUIDs",             "#FFF7ED", COL_HIGHLIGHT),
    ]
    for x, title, face, edge in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y0), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.6, facecolor=face, edgecolor=edge))
        ax.text(x + 0.012, y0 + box_h - 0.035, title,
                ha="left", va="top", fontsize=11.5,
                fontweight="bold", color=COL_INK)

    for (lx, *_), (rx, *_) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch(
            (lx + box_w + 0.003, y0 + box_h / 2),
            (rx - 0.003, y0 + box_h / 2),
            arrowstyle="-|>", mutation_scale=15,
            linewidth=1.6, color="#94A3B8"))

    # Box 1: query (wrapped)
    wrapped_query = "\n".join(textwrap.wrap(f'"{case["query_text"]}"', width=24))
    ax.text(boxes[0][0] + 0.012, y0 + box_h - 0.10,
            wrapped_query,
            ha="left", va="top", fontsize=9.8, color=COL_INK,
            fontstyle="italic")

    # Box 2: SpatialTriplet (compact; abbreviate subtype)
    rels = case["topology_relations"]
    triplet_lines = []
    for r in rels[:3]:
        sub = r["subject_type"].replace("Ifc", "")
        obj = r["object_type"].replace("Ifc", "")[:11]
        d = f" {r['direction'][0].upper()}" if r.get("direction") else ""
        triplet_lines.append(f"{sub}\n  -[{r['predicate']}{d}]->\n  {obj}")
    ax.text(boxes[1][0] + 0.012, y0 + box_h - 0.10,
            "\n\n".join(triplet_lines),
            ha="left", va="top", fontsize=8.2, color=COL_INK,
            family="monospace")

    # Box 3: Cypher snippet (kept short to fit)
    cypher = (
        "MATCH (w:Elem)\n"
        " WHERE w.ifc='IfcWin'\n"
        "MATCH (w)-[:NEXT_TO]\n"
        "  ->(d {ifc:'IfcDoor'})\n"
        "MATCH (w)-[:FILLS]\n"
        "  ->(:Wall)\n"
        "WHERE w.storey='1'\n"
        "RETURN w.guid"
    )
    ax.text(boxes[2][0] + 0.012, y0 + box_h - 0.10,
            cypher, ha="left", va="top", fontsize=8.2,
            color=COL_INK, family="monospace")

    # Box 4: result GUIDs
    top5 = case["top5_guids"][:5]
    gt = case["gt_guid"]
    lines = []
    for i, g in enumerate(top5, 1):
        marker = "  ← GT" if g == gt else ""
        lines.append(f"{i}. {g[:7]}…{marker}")
    ax.text(boxes[3][0] + 0.012, y0 + box_h - 0.10,
            "\n".join(lines), ha="left", va="top",
            fontsize=9.0, color=COL_INK, family="monospace")
    ax.text(boxes[3][0] + 0.012, y0 + 0.04,
            "Every output is an\nexisting IFC GUID —\nno hallucination.",
            ha="left", va="bottom", fontsize=9.0, color="#9A3412",
            fontstyle="italic")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Beat 3 — Mechanism: P0 / P1 / top-K flow (Sankey + waterfall)
# ─────────────────────────────────────────────────────────────────────────────
def _ribbon(ax, x0, x1, y0_l, y1_l, y0_r, y1_r, color, alpha=0.55):
    midx = (x0 + x1) / 2.0
    verts = [
        (x0, y0_l), (midx, y0_l), (midx, y0_r), (x1, y0_r),
        (x1, y1_r), (midx, y1_r), (midx, y1_l), (x0, y1_l),
        (x0, y0_l)]
    codes = [MplPath.MOVETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           edgecolor="none", alpha=alpha, zorder=2))


def _block(ax, x, y_top, h, color, w=0.045):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y_top - h), w, h,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        linewidth=0.0, facecolor=color, alpha=0.95, zorder=4))


def beat3_p0_topk_flow() -> plt.Figure:
    case = _read_json(FINAL_DATA_DIR / "symbolic_reasoning_trace_case.json")
    p0 = int(case["p0_pool_size"])
    union = int(case["union_pool_size"])
    p1_only = union - p0
    K = 10
    p0_tk = min(p0, K)
    p1_tk = K - p0_tk
    dropped = p1_only - p1_tk
    top5 = 5
    gt_base = case.get("base_rank")
    gt_rerank = case.get("reranked_rank")
    all_ifc = int(case.get("initial_pool_size", 1666))

    fig = plt.figure(figsize=SLIDE_FIGSIZE)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.85, 1.0],
                            hspace=0.42, left=0.05, right=0.97,
                            top=0.86, bottom=0.07)
    ax_top = fig.add_subplot(grid[0])
    ax_bot = fig.add_subplot(grid[1])

    fig.suptitle("How P0 Spatial-Relation Hits + P1 Storey/Class Collapse into the Rerank Shortlist",
                 fontsize=17, fontweight="bold", y=0.965)
    fig.text(0.05, 0.918,
             f"Single trace: {case['case_id']} | GT rank {gt_base} → {gt_rerank} after rerank",
             ha="left", fontsize=10.5, color=COL_MUTED)

    # Top: Sankey
    ax_top.set_xlim(0, 1); ax_top.set_ylim(0, 1); ax_top.axis("off")
    UNIT = 0.34 / union
    pool_top, pool_bot = 0.62, 0.62 - union * UNIT
    X = {"p1_pool": 0.13, "split": 0.36, "topk": 0.62, "rerank": 0.87}

    for key, lbl in [("p1_pool", "P1 storey+class pool"),
                     ("split", "P0 spatial-relation / P1-only"),
                     ("topk", "Top-K (rerank input)"),
                     ("rerank", "Final Top-5")]:
        ax_top.text(X[key], 0.69, lbl, ha="center", va="bottom",
                    fontsize=10.6, fontweight="bold", color=COL_INK)

    # Stage 1
    h_pool = union * UNIT
    _block(ax_top, X["p1_pool"], pool_top, h_pool, COL_P1_ONLY)
    ax_top.text(X["p1_pool"], pool_bot - 0.02,
                f"{union} candidates\n(storey + IFC class)",
                ha="center", va="top", fontsize=9.3, color=COL_MUTED)

    # Stage 2: split
    h_p0, h_p1 = p0 * UNIT, p1_only * UNIT
    p0_top_s = pool_top; p0_bot_s = p0_top_s - h_p0
    p1_top_s = p0_bot_s; p1_bot_s = p1_top_s - h_p1
    _block(ax_top, X["split"], p0_top_s, h_p0, COL_P0)
    _block(ax_top, X["split"], p1_top_s, h_p1, COL_P1_ONLY)
    ax_top.text(X["split"] + 0.030, p0_bot_s + h_p0 / 2,
                f"P0 spatial-relation = {p0}  ← GT @ rank {gt_base}",
                va="center", fontsize=9.2, color=COL_P0, fontweight="bold")
    ax_top.text(X["split"] + 0.030, p1_bot_s + h_p1 / 2,
                f"{P1_ONLY_INLINE} = {p1_only}\n(no topology hit)",
                va="center", fontsize=9.0, color=COL_P1_ONLY)

    _ribbon(ax_top, X["p1_pool"] + 0.022, X["split"] - 0.022,
            pool_top, pool_top - h_p0, p0_top_s, p0_bot_s, COL_P0, 0.50)
    _ribbon(ax_top, X["p1_pool"] + 0.022, X["split"] - 0.022,
            pool_top - h_p0, pool_bot, p1_top_s, p1_bot_s, COL_P1_ONLY, 0.42)

    # Stage 3: top-K
    h_p0_tk, h_p1_tk, h_drop = p0_tk * UNIT, p1_tk * UNIT, dropped * UNIT
    p0_top_tk = pool_top; p0_bot_tk = p0_top_tk - h_p0_tk
    p1_top_tk = p0_bot_tk; p1_bot_tk = p1_top_tk - h_p1_tk
    _block(ax_top, X["topk"], p0_top_tk, h_p0_tk, COL_P0)
    _block(ax_top, X["topk"], p1_top_tk, h_p1_tk, COL_P1_ONLY)
    ax_top.text(X["topk"] + 0.030, p0_bot_tk + h_p0_tk / 2 + 0.01,
                f"K = {K}\n= {p0_tk} P0 spatial-relation + {p1_tk} P1-only",
                va="center", fontsize=9.0, color=COL_INK)

    drop_top = 0.20; drop_bot = drop_top - h_drop
    _block(ax_top, X["topk"], drop_top, h_drop, COL_DROP)
    ax_top.text(X["topk"] + 0.030, drop_top - h_drop / 2,
                f"{dropped} P1-only DROPPED\n(storey+class never reranked)",
                va="center", fontsize=9.0, color=COL_MUTED, fontweight="bold")

    _ribbon(ax_top, X["split"] + 0.022, X["topk"] - 0.022,
            p0_top_s, p0_bot_s, p0_top_tk, p0_bot_tk, COL_P0, 0.55)
    _ribbon(ax_top, X["split"] + 0.022, X["topk"] - 0.022,
            p1_top_s, p1_top_s - h_p1_tk, p1_top_tk, p1_bot_tk, COL_P1_ONLY, 0.55)
    _ribbon(ax_top, X["split"] + 0.022, X["topk"] - 0.022,
            p1_top_s - h_p1_tk, p1_bot_s, drop_top, drop_bot, COL_DROP, 0.65)

    # Stage 4
    h_t5 = top5 * UNIT
    _block(ax_top, X["rerank"], pool_top, h_t5, COL_GT)
    ax_top.text(X["rerank"], pool_top - h_t5 - 0.03,
                f"Top-{top5}  (after rerank)\nGT promoted to #1",
                ha="center", va="top", fontsize=9.2,
                color="#065F46", fontweight="bold")
    _ribbon(ax_top, X["topk"] + 0.022, X["rerank"] - 0.022,
            p0_top_tk, p1_bot_tk, pool_top, pool_top - h_t5, "#34D399", 0.45)

    # Annotation banners
    ax_top.text(0.13, 0.96, "✓ P0 spatial-relation advantage PRESERVED",
                ha="left", va="top", fontsize=10.0,
                fontweight="bold", color="#065F46")
    ax_top.text(0.13, 0.925,
                f"Top-K cut keeps P0 spatial-relation hits at the front;\n"
                f"{dropped} P1-only storey/class candidates eliminated\n"
                f"before rerank ever sees them.",
                ha="left", va="top", fontsize=9.0, color=COL_INK,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="#ECFDF5", edgecolor="#059669"))
    ax_top.text(0.55, 0.96, "⚠ P0 spatial-relation advantage LOST inside rerank",
                ha="left", va="top", fontsize=10.0,
                fontweight="bold", color="#9A3412")
    ax_top.text(0.55, 0.925,
                "fusion_score = band + slot only —\n"
                "source tag dropped, so P1-only storey/class rows with strong\n"
                "band+slot cues can outrank P0 inside top-K.",
                ha="left", va="top", fontsize=9.0, color=COL_INK,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="#FFF7ED", edgecolor=COL_HIGHLIGHT))

    ax_top.text(0.02, (pool_top + pool_bot) / 2,
                f"All IFC\n{all_ifc}\n──filter──→",
                va="center", fontsize=9.0, color=COL_MUTED, fontweight="bold")

    # Bottom: stacked waterfall
    # Final Top-5 split derived from trace: top5_guids = first 3 P0 + 2 P1-only
    # (the 3 P0 always survive top-K; rerank promoted 2 of the 7 P1-only).
    cand = case.get("candidate_guids", [])
    top5_guids = case.get("top5_guids", [])[:top5]
    p0_in_top5 = sum(1 for g in top5_guids if g in cand[:p0])
    p1_in_top5 = top5 - p0_in_top5

    stages = ["All IFC", "P1 storey/class", "P0 ∪ P1\n(ordered)",
              "Top-K\n(rerank input)", "Final Top-5"]
    p0_b = [0, p0, p0, p0_tk, p0_in_top5]
    p1_b = [0, p1_only, p1_only, p1_tk, p1_in_top5]
    drop_b = [all_ifc - union, 0, 0, dropped, 0]
    totals = [all_ifc, union, union, K, top5]

    xs = np.arange(len(stages))
    ax_bot.bar(xs, p0_b, 0.55, color=COL_P0, label="P0 (spatial-relation hit)", zorder=3)
    ax_bot.bar(xs, p1_b, 0.55, bottom=p0_b, color=COL_P1_ONLY,
               label="P1-only (storey + IFC class)", zorder=3)
    ax_bot.bar(xs, drop_b, 0.55,
               bottom=np.array(p0_b) + np.array(p1_b),
               color=COL_DROP, alpha=0.7, label="Dropped / not retrieved",
               zorder=3)
    for i, t in enumerate(totals):
        ax_bot.text(i, t * 1.05 if t else 1, str(t),
                    ha="center", va="bottom", fontsize=10.5,
                    fontweight="bold", color=COL_INK, zorder=5)

    ax_bot.set_yscale("symlog", linthresh=1)
    ax_bot.set_ylabel("Candidates (symlog)")
    ax_bot.set_xticks(xs); ax_bot.set_xticklabels(stages)
    ax_bot.set_ylim(0, all_ifc * 2.0)
    ax_bot.grid(axis="y", which="major", alpha=0.25, zorder=0)
    ax_bot.set_title("Candidate compression — coloured by source",
                     fontsize=12)
    ax_bot.legend(loc="upper right", fontsize=9.0, framealpha=0.95)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Beat 4 — Evidence: 4-model Top-1
# ─────────────────────────────────────────────────────────────────────────────
def beat4_top1_evidence() -> plt.Figure:
    """G-series tight comparison — mirrors fig03_lora_vs_gemini_tight."""
    rows_lvg = {r["system"]: r for r in
                _read_csv(FINAL_DATA_DIR / "lora_vs_gemini.csv")}
    rows_grag = {r["system"]: r for r in
                 _read_csv(FINAL_DATA_DIR / "graph_rag_evidence_dependent.csv")}

    plotted = []
    for system_key, label, color in BEAT4_SYSTEMS:
        src = rows_lvg.get(system_key) or rows_grag.get(system_key)
        if src is None:
            continue
        plotted.append({
            "label": label,
            "color": color,
            "top10": float(src.get("top10") or 0),
            "top1": float(src.get("top1") or 0),
            "mrr10": float(src.get("mrr10") or 0) * 100.0,
            "gt_in_pool": float(src.get("gt_in_pool") or src.get("top10") or 0),
        })

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE, constrained_layout=True)
    x = np.arange(len(plotted))
    width = 0.30
    colors = [p["color"] for p in plotted]
    top10 = [p["top10"] for p in plotted]
    top1 = [p["top1"] for p in plotted]
    gt = [p["gt_in_pool"] for p in plotted]
    mrr = [p["mrr10"] for p in plotted]

    # Best learned + ceiling shading
    best_idx = next(i for i, p in enumerate(plotted)
                    if p["label"].startswith("G9\nOpenCV/ResNet"))
    ceil_idx = next(i for i, p in enumerate(plotted)
                    if p["label"].startswith("Oracle"))
    ax.axvspan(best_idx - 0.52, best_idx + 0.52,
               color="#DCFCE7", alpha=0.95, zorder=0)
    ax.axvspan(ceil_idx - 0.52, ceil_idx + 0.52,
               color="#F3E8FF", alpha=0.55, zorder=0)

    gt_bars = ax.bar(x - width / 2, gt, width=width, color=colors,
                      alpha=0.30, label="GT-in-pool", zorder=3)
    t10_bars = ax.bar(x + width / 2, top10, width=width, color=colors,
                       alpha=0.92, label="Top-10", zorder=3)

    for bar, val in zip(gt_bars, gt):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=10, color=COL_MUTED)
    for bar, val in zip(t10_bars, top10):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=COL_INK)

    # Top-1 in pill labels above
    for idx, val in enumerate(top1):
        ax.text(idx, 117, f"T1 {val:.1f}",
                ha="center", va="center", fontsize=11,
                color=colors[idx], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor="white", edgecolor=colors[idx],
                          linewidth=1.4))

    ax.set_xticks(x); ax.set_xticklabels([p["label"] for p in plotted],
                                          fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 128)
    ax.grid(axis="y", alpha=0.22, zorder=0)

    # MRR line on twinx
    ax2 = ax.twinx()
    line, = ax2.plot(x, mrr, "o-", color=COL_MRR, lw=3.0,
                      markersize=10, zorder=4, label="MRR@10 ×100")
    for xx, val in zip(x, mrr):
        ax2.text(xx, val + 0.5, f"{val:.1f}",
                  ha="center", va="bottom", fontsize=10,
                  color=line.get_color())
    ax2.set_ylabel("MRR@10 ×100", fontsize=12, color=COL_MRR)
    ax2.set_ylim(0, max(mrr) * 1.45)
    ax2.spines["top"].set_visible(False)

    # Best/ceiling banners
    ax.text(best_idx, 124, "Best learned",
            ha="center", fontsize=11, fontweight="bold", color=COL_SAFE_TEXT)
    ax.text(ceil_idx, 124, "Ceiling",
            ha="center", fontsize=11, fontweight="bold", color=COL_ORACLE_TEXT)

    # Annotation: Graph-RAG lift (placed below pill row, near Gemini)
    g9_idx = next(i for i, p in enumerate(plotted)
                  if p["label"].startswith("G9\nOpenCV/"))
    delta = top1[best_idx] - top1[g9_idx]
    ax.annotate(
        f"Rerank\n+{delta:.1f}pp Top-1   "
        f"+{mrr[best_idx] - mrr[g9_idx]:.1f} MRR×100",
        xy=(best_idx, top10[best_idx] + 4),
        xytext=(best_idx - 1.4, 65),
        fontsize=10.5, fontweight="bold", color=COL_AMBER_TEXT,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.32",
                  facecolor="#FFF7ED", edgecolor=COL_HIGHLIGHT),
        arrowprops=dict(arrowstyle="->", color=COL_HIGHLIGHT, lw=1.4))

    ax.legend([t10_bars, gt_bars, line],
              ["Top-10", "GT-in-pool", "MRR@10 ×100"],
              loc="lower right", frameon=False, fontsize=10.5)
    fig.suptitle("Strict retrieval: from Gemini to G9 OpenCV/ResNet + rerank",
                 fontsize=17, fontweight="bold")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Beat 5 — Oracle ceiling
# ─────────────────────────────────────────────────────────────────────────────
def beat5_oracle_ceiling() -> plt.Figure:
    """Single panel — bars on left axis (accuracy), line on right axis (pool)."""
    rows = _read_csv(FINAL_DATA_DIR / "oracle_symbolic_ceiling.csv")

    short_names = {
        "Storey + class": "Storey\n+ class",
        "+ relation + target": "+ relation\n+ target",
        "+ direction": "+ direction",
        "+ subtype": "+ subtype",
        "+ position slot": "+ position\nslot",
    }
    short_labels = [
        f"L{i}\n{short_names.get(r['level_name'], r['level_name'])}"
        for i, r in enumerate(rows)
    ]
    top1 = [float(r["top1"]) for r in rows]
    top10 = [float(r["top10"]) for r in rows]
    median_pool = [float(r["median_pool"]) for r in rows]

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE, constrained_layout=True)
    xs = np.arange(len(rows))
    w = 0.36

    bars_t10 = ax.bar(xs - w / 2, top10, width=w, color=COL_P1_ONLY,
                       alpha=0.90, label="Top-10", zorder=3)
    bars_t1 = ax.bar(xs + w / 2, top1, width=w, color=COL_P0,
                      alpha=0.95, label="Top-1", zorder=3)
    for b, v in zip(bars_t10, top10):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.0f}",
                ha="center", fontsize=10.5, fontweight="bold", color=COL_INK)
    for b, v in zip(bars_t1, top1):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.0f}",
                ha="center", fontsize=10.5, fontweight="bold", color=COL_INK)

    ax.set_xticks(xs); ax.set_xticklabels(short_labels, fontsize=10.5)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 118)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=11)

    # Secondary axis — median pool size (log scale, 76 → 1)
    ax2 = ax.twinx()
    line, = ax2.plot(xs, median_pool, "o-", color=COL_HIGHLIGHT,
                      lw=3.0, markersize=11, zorder=4,
                      label="Median pool size")
    for x, v in zip(xs, median_pool):
        ax2.text(x, v * 1.18, f"{v:.0f}", ha="center",
                  fontsize=10.5, fontweight="bold",
                  color=COL_HIGHLIGHT)
    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 200)
    ax2.set_ylabel("Median pool size (log)", fontsize=12,
                    color=COL_HIGHLIGHT)
    ax2.tick_params(axis="y", colors=COL_HIGHLIGHT)
    ax2.spines["right"].set_color(COL_HIGHLIGHT)
    ax2.axhline(1, ls="--", color=COL_GT, lw=1.0, alpha=0.7)
    ax2.text(len(rows) - 0.5, 1.15, "ideal = 1",
              ha="right", fontsize=10, color=COL_GT, fontweight="bold")

    # Combined legend
    ax.legend([bars_t10, bars_t1, line],
              ["Top-10", "Top-1", "Median pool (log)"],
              loc="upper left", framealpha=0.95, fontsize=11)

    fig.suptitle("Oracle ceiling — adding symbolic query fields lifts Top-1, shrinks the pool",
                 fontsize=16, fontweight="bold")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Beat 6 — Per-predicate SSR honesty
# ─────────────────────────────────────────────────────────────────────────────
def beat6_predicate_ssr() -> plt.Figure:
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    fig.suptitle("Not all predicates discriminate equally",
                 fontsize=18, fontweight="bold", y=0.96)
    fig.text(0.5, 0.918,
             "H2 hard-negative SSR by predicate (n=83).",
             ha="center", fontsize=12, color=COL_MUTED)

    predicates = [r[0] for r in PER_PREDICATE_SSR]
    ssr = [r[1] for r in PER_PREDICATE_SSR]
    n = [r[2] for r in PER_PREDICATE_SSR]
    notes = [r[3] for r in PER_PREDICATE_SSR]
    colors = [COL_P0, "#0EA5E9", COL_DROP]

    xs = np.arange(len(predicates))
    bars = ax.bar(xs, ssr, width=0.55, color=colors, zorder=3)
    for x, v, ni, note in zip(xs, ssr, n, notes):
        ax.text(x, v + 2, f"{v:.0f}%\n(n={ni})",
                ha="center", fontsize=11, fontweight="bold", color=COL_INK)
        ax.text(x, -8, note, ha="center", va="top",
                fontsize=10, color=COL_MUTED)

    ax.axhline(3.0, ls="--", color=COL_BASELINE, lw=1.2, zorder=2)
    ax.text(len(predicates) - 0.5, 4.5, "attribute baseline (3.0%)",
            ha="right", fontsize=10, color=COL_BASELINE)

    ax.set_xticks(xs); ax.set_xticklabels(predicates, fontsize=12,
                                            fontweight="bold")
    ax.set_ylim(-25, 105)
    ax.set_ylabel("Single-Step Reduction (SSR, %)")
    ax.set_title("Symbolic discrimination by predicate type",
                 fontsize=12, loc="left", pad=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    # Takeaway box (placed below x-axis labels in lower-right)
    ax.text(0.99, 0.99,
            "Design lesson:\n"
            "• Heterogeneous anchors (Door↔Window)\n   beat homogeneous (Window↔Window)\n"
            "• Multi-storey CONTINUOUS is the\n   strongest discriminator\n"
            "• FILLS = 0 signal when every same-\n   storey window FILLS a wall",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#F8FAFC", edgecolor="#94A3B8"))

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Beat 7 — What's next: top-K is the new bottleneck
# ─────────────────────────────────────────────────────────────────────────────
def beat7_next_bottleneck() -> plt.Figure:
    fig = plt.figure(figsize=SLIDE_FIGSIZE)
    fig.suptitle("Next bottleneck: the top-K cut",
                 fontsize=18, fontweight="bold", y=0.96)
    fig.text(0.5, 0.918,
             "Three failure modes — three concrete interventions.",
             ha="center", fontsize=12, color=COL_MUTED)

    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 3 columns: Failure mode → Intervention → Cost/Benefit
    columns = [
        {
            "title": "A. GT not in P0 spatial relation ∪ P1 backstop",
            "subtitle": "(recall miss)",
            "color_face": "#FEF2F2", "color_edge": "#DC2626",
            "diag": "rare but fatal\n(e.g. BasicHouse 0/23 before fix)",
            "fix": "P_safety net:\n  type_only across\n  all storeys when\n  P1 backstop returns empty",
            "cb": "Low cost,\nhigh upside",
        },
        {
            "title": "C. GT pushed past K",
            "subtitle": "(P0 spatial relation ∪ P1 unsorted)",
            "color_face": "#FFF7ED", "color_edge": COL_HIGHLIGHT,
            "diag": "most common —\nP1-only storey/class ordering is\nnear-random by GUID",
            "fix": "Pre-K hybrid score:\n  P0 spatial-relation prior\n  + size_band + slot match\n  before top-K cut",
            "cb": "Highest ROI,\nmedium effort",
        },
        {
            "title": "B. P0 spatial relation too narrow",
            "subtitle": "(LoRA misses triplet)",
            "color_face": "#F5F3FF", "color_edge": COL_P1_ONLY,
            "diag": "LoRA hallucinates\nor misses predicate;\nP0 spatial relation = 0",
            "fix": "Adaptive K:\n  K = max(K_min,\n         2·|P0 spatial relation|);\n  confidence gate\n  on P0 entry",
            "cb": "Tunable,\nrequires care",
        },
    ]
    n = len(columns)
    margin = 0.04
    col_w = (1 - 2 * margin - 0.05 * (n - 1)) / n
    y_top, y_bot = 0.82, 0.10

    for i, col in enumerate(columns):
        x = margin + i * (col_w + 0.05)
        ax.add_patch(FancyBboxPatch(
            (x, y_bot), col_w, y_top - y_bot,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.6, facecolor=col["color_face"],
            edgecolor=col["color_edge"]))

        # Title
        ax.text(x + 0.012, y_top - 0.04,
                col["title"], fontsize=12, fontweight="bold", color=COL_INK,
                ha="left", va="top")
        ax.text(x + 0.012, y_top - 0.08,
                col["subtitle"], fontsize=10, fontstyle="italic",
                color=COL_MUTED, ha="left", va="top")

        # Diagnosis
        ax.text(x + 0.012, y_top - 0.16,
                "What goes wrong:", fontsize=9.5, fontweight="bold",
                color=COL_HIGHLIGHT, ha="left", va="top")
        ax.text(x + 0.012, y_top - 0.20, col["diag"],
                fontsize=10, color=COL_INK, ha="left", va="top")

        # Fix
        ax.text(x + 0.012, y_top - 0.36,
                "Intervention:", fontsize=9.5, fontweight="bold",
                color=COL_P0, ha="left", va="top")
        ax.text(x + 0.012, y_top - 0.40, col["fix"],
                fontsize=10, color=COL_INK, ha="left", va="top",
                family="monospace")

        # Cost/benefit pill
        ax.add_patch(FancyBboxPatch(
            (x + 0.012, y_bot + 0.04), col_w - 0.024, 0.06,
            boxstyle="round,pad=0.005,rounding_size=0.015",
            linewidth=0.8, facecolor="white", edgecolor=col["color_edge"]))
        ax.text(x + col_w / 2, y_bot + 0.07, col["cb"],
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=col["color_edge"])

    # Closing line
    ax.text(0.5, 0.04,
            "Diagnostic first: instrument GT-in-top-K and per-mode counts "
            "before tuning. Measure twice, cut once.",
            ha="center", va="center", fontsize=11,
            fontstyle="italic", color=COL_MUTED)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
BEATS = {
    "beat1": ("beat1_attribute_entropy_bottleneck", beat1_entropy_bottleneck),
    "beat2": ("beat2_relational_signal", beat2_relational_signal),
    "beat3": ("beat3_p0_topk_flow", beat3_p0_topk_flow),
    "beat4": ("beat4_top1_evidence", beat4_top1_evidence),
    "beat5": ("beat5_oracle_ceiling", beat5_oracle_ceiling),
    "beat6": ("beat6_predicate_ssr", beat6_predicate_ssr),
    "beat7": ("beat7_topk_bottleneck", beat7_next_bottleneck),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=list(BEATS.keys()),
                        help="Render a subset of beats (default: all)")
    args = parser.parse_args()

    _configure_mpl()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = args.only or list(BEATS.keys())
    print(f"Rendering {len(targets)} beat(s) to {OUT_DIR.relative_to(PROJECT_ROOT)}/")
    for key in targets:
        fname, builder = BEATS[key]
        print(f"[{key}] {fname}")
        fig = builder()
        _save(fig, OUT_DIR / fname)
    print("Done.")


if __name__ == "__main__":
    main()
