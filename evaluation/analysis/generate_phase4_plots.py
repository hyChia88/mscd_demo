#!/usr/bin/env python3
"""Generate the six main Phase 4 LoRA6 figures for thesis writing.

Outputs PNG figures into:
    mscd_demo/docs/plots/phase4_lora6_main/

The plots are sourced from the latest structured evaluation artifacts under:
    mscd_demo/output/lora6_v2_ap_20260331/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from phase4_plot_style import (
    COLORS as SHARED_COLORS,
    FAMILY_TO_UNIVERSE as SHARED_FAMILY_TO_UNIVERSE,
    FINGERPRINT_WATERFALL_COLORS,
    HIGHLIGHT_COLORS as SHARED_HIGHLIGHT_COLORS,
    METRIC_COLORS as SHARED_METRIC_COLORS,
    RELATION_FAMILY_COLORS,
    STRATEGY_META as SHARED_STRATEGY_META,
    UNIVERSE_META as SHARED_UNIVERSE_META,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331"
METRICS_DIR = EXPERIMENT_ROOT / "metrics"
ORACLE_DIR_CANDIDATES = [
    EXPERIMENT_ROOT / "oracle_ap_heldout",
    EXPERIMENT_ROOT / "legacy" / "oracle_ap_heldout",
    PROJECT_ROOT / "logs" / "evaluation_output" / "lora6_v2_ap_20260331" / "oracle_ap_heldout",
    PROJECT_ROOT / "logs" / "evaluation_output" / "lora6_v2_ap_20260331" / "legacy" / "oracle_ap_heldout",
]
ORACLE_PHASE3_DIRS = [
    EXPERIMENT_ROOT / "oracle_phase3b",
    EXPERIMENT_ROOT / "legacy" / "oracle_phase3b",
    EXPERIMENT_ROOT / "oracle_phase3_fixed",
]
GT_PATH = (
    REPO_ROOT
    / "data_curation"
    / "datasets"
    / "synth_v0.5_ap"
    / "train"
    / "lora6_v2_ap_eval_canonical_m.jsonl"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_main"
DEFAULT_APPENDIX_OUT_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_appendix"

MODEL_ORDER = [
    "g0_canonical",
    "g1_fullaug",
    "g2_fullaug_lowlr",
    "g3_fullaug_r32",
    "g4_ultimate",
    "g7_position_context",
    "g8_posctx_dim",
    "g6_baseline",
    "gemini_ap_v2",
]

TRACK_A_FILES = {
    "g0_canonical": METRICS_DIR / "g0_canonical__ap_metrics.json",
    "g1_fullaug": METRICS_DIR / "g1_fullaug__ap_metrics.json",
    "g2_fullaug_lowlr": METRICS_DIR / "g2_fullaug_lowlr__ap_metrics.json",
    "g3_fullaug_r32": METRICS_DIR / "g3_fullaug_r32__ap_metrics.json",
    "g4_ultimate": METRICS_DIR / "g4_ultimate__ap_metrics.json",
    "g7_position_context": METRICS_DIR / "g7_position_context__ap_metrics.json",
    "g8_posctx_dim": METRICS_DIR / "g8_posctx_dim__ap_metrics.json",
    "g6_baseline": METRICS_DIR / "g6_baseline__ap_metrics.json",
    "gemini_ap_v2": METRICS_DIR / "gemini_ap_v2__ap_metrics.json",
}

# Track B2 (downstream e2e) metrics:
# G1-G6: phase3_fixed (enriched graph has no effect — don't use new fields)
# G7/G8/Gemini: phase5 (enriched graph, v2_lora profile for all, correct comparison)
# G0: phase3 (only phases prior to G8 training)
TRACK_B2_FILES = {
    "g0_canonical": METRICS_DIR / "g0_canonical__ap_e2e_metrics.json",
    "g1_fullaug": METRICS_DIR / "g1_fullaug__ap_e2e_phase3_metrics.json",
    "g2_fullaug_lowlr": METRICS_DIR / "g2_fullaug_lowlr__ap_e2e_phase3_metrics.json",
    "g3_fullaug_r32": METRICS_DIR / "g3_fullaug_r32__ap_e2e_phase3_metrics.json",
    "g4_ultimate": METRICS_DIR / "g4_ultimate__ap_e2e_phase3_metrics.json",
    "g7_position_context": METRICS_DIR / "g7_position_context__ap_e2e_phase5_metrics.json",
    "g8_posctx_dim": METRICS_DIR / "g8_posctx_dim__ap_e2e_phase5_metrics.json",
    "g6_baseline": METRICS_DIR / "g6_baseline__ap_e2e_phase3_metrics.json",
    "gemini_ap_v2": METRICS_DIR / "gemini_ap_v2__ap_e2e_phase5_metrics.json",
}

DISPLAY = {
    "g0_canonical": "G0",
    "g1_fullaug": "G1",
    "g2_fullaug_lowlr": "G2",
    "g3_fullaug_r32": "G3",
    "g4_ultimate": "G4",
    "g7_position_context": "G7",
    "g8_posctx_dim": "G8",
    "g6_baseline": "G6",
    "gemini_ap_v2": "Gemini v2",
    "oracle_phase3": "Oracle P3",
}

COLORS = {
    "g0_canonical": "#E65100",
    "g1_fullaug": "#F57C00",
    "g2_fullaug_lowlr": "#F5A623",
    "g3_fullaug_r32": "#D32F2F",
    "g4_ultimate": "#B71C1C",
    "g7_position_context": "#6A1B9A",
    "g8_posctx_dim": "#3E1080",
    "g6_baseline": "#90A4AE",
    "gemini_ap_v2": "#1565C0",
    "gemini_unified": "#1565C0",
    "oracle_phase3": "#4A148C",
    "p0_only": "#E65100",
    "p1_only_strategy": "#F5A623",
    "p0_intersect_p1": "#D32F2F",
    "p0_union_p1": "#1565C0",
    "p1_only_upper_bound": "#F5A623",
    "full_topology_union": "#1565C0",
}

METRIC_COLORS = {
    "gt": "#E65100",
    "top10": "#F4A261",
    "top1": "#F7D7C1",
    "pool": "#000000",
    "mrr": "#7B1FA2",
    "pred_r": "#7B1FA2",
    "reduction": "#000000",
    "mrr_aux": "#7B1FA2",
    "gt_line": "#000000",
    "mrr_track": "#1565C0",
}

HIGHLIGHT_COLORS = {
    "safe_green_fill": "#E8F5E9",
    "safe_green_text": "#166534",
    "winner_amber_fill": "#FFF3E0",
    "winner_amber_text": "#9A3412",
    "oracle_fill": "#EDE7F6",
    "oracle_text": "#5B21B6",
}

FAMILY_ORDER = [
    "singleton:CONNECTS_TO",
    "singleton:ADJACENT_TO",
    "paired:FILLS+NEXT_TO",
    "triad:FILLS+NEXT_TO+NEXT_TO",
    "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)",
    "singleton:FILLS",
]

UNIVERSE_META = {
    "U1": {"label": "U1 CONNECTS_TO", "color": "#1565C0"},
    "U2": {"label": "U2 ADJACENT_TO", "color": "#2A9D8F"},
    "U3": {"label": "U3 Opening-paired", "color": "#F5A623"},
    "U4": {"label": "U4 Symmetric triad", "color": "#D32F2F"},
    "U5": {"label": "U5 Mixed triad", "color": "#7B1FA2"},
    "U6": {"label": "U6 FILLS only", "color": "#B0BEC5"},
}

FAMILY_TO_UNIVERSE = {
    "singleton:CONNECTS_TO": "U1",
    "singleton:ADJACENT_TO": "U2",
    "paired:FILLS+NEXT_TO": "U3",
    "triad:FILLS+NEXT_TO+NEXT_TO": "U4",
    "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)": "U5",
    "singleton:FILLS": "U6",
}

STRATEGY_META = {
    "p0_only": {"label": "p0_only", "color": COLORS["p0_only"]},
    "p1_only_strategy": {"label": "p1_only", "color": COLORS["p1_only_strategy"]},
    "p0_intersect_p1": {"label": "p0∩p1", "color": COLORS["p0_intersect_p1"]},
    "p0_union_p1": {"label": "p0∪p1", "color": COLORS["p0_union_p1"]},
}

# Keep the JSON palette authoritative while preserving the older inline values
# above as a readable fallback for anyone skimming this standalone script.
COLORS = dict(SHARED_COLORS)
METRIC_COLORS = dict(SHARED_METRIC_COLORS)
HIGHLIGHT_COLORS = dict(SHARED_HIGHLIGHT_COLORS)
UNIVERSE_META = dict(SHARED_UNIVERSE_META)
FAMILY_TO_UNIVERSE = dict(SHARED_FAMILY_TO_UNIVERSE)
STRATEGY_META = dict(SHARED_STRATEGY_META)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_oracle_topology_dir() -> Path:
    for candidate in ORACLE_DIR_CANDIDATES:
        if (candidate / "oracle_topology_metrics.json").exists():
            return candidate
    raise FileNotFoundError("No oracle_topology_metrics.json found in canonical or legacy oracle directories.")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_assistant_label(case: dict) -> dict:
    for message in case.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return json.loads(content)
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return json.loads(first["text"])
    raise ValueError(f"Assistant label missing for eval case: {case.get('id')}")


def _classify_family(relations: List[dict]) -> str:
    preds = Counter(str(r.get("predicate") or "") for r in relations)
    if len(relations) == 1:
        return f"singleton:{relations[0].get('predicate')}"
    if len(relations) == 2 and preds == Counter({"FILLS": 1, "NEXT_TO": 1}):
        return "paired:FILLS+NEXT_TO"
    if len(relations) == 3 and preds == Counter({"FILLS": 1, "NEXT_TO": 2}):
        next_to_types = {
            str(r.get("object_type") or "")
            for r in relations
            if str(r.get("predicate") or "") == "NEXT_TO"
        }
        if len(next_to_types) > 1:
            return "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)"
        return "triad:FILLS+NEXT_TO+NEXT_TO"
    return "other"


def _signature(relations: List[dict]) -> str:
    parts = []
    for rel in relations:
        pred = str(rel.get("predicate") or "")
        obj = str(rel.get("object_type") or "")
        direction = str(rel.get("direction") or "")
        piece = f"{pred}:{obj}"
        if direction:
            piece += f":{direction}"
        parts.append(piece)
    return " | ".join(parts)


def load_topology_counts() -> dict:
    multiplicity = Counter()
    family = Counter()
    signatures = Counter()
    n_cases = 0
    with GT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            label = _extract_assistant_label(row)
            relations = label.get("spatial_relations") or []
            n_cases += 1
            multiplicity[f"{len(relations)}-rel"] += 1
            family[_classify_family(relations)] += 1
            signatures[_signature(relations)] += 1
    return {
        "n_cases": n_cases,
        "multiplicity": multiplicity,
        "family": family,
        "n_signatures": len(signatures),
    }


def load_track_a_rows() -> List[dict]:
    rows = []
    for key in MODEL_ORDER:
        data = _load_json(TRACK_A_FILES[key])
        rows.append(
            {
                "key": key,
                "label": DISPLAY[key],
                "hop1": data["hop1_acc"] * 100.0,
                "pred_p": data["predicate_precision"] * 100.0,
                "pred_r": data["predicate_recall"] * 100.0,
                "dir": data["direction_acc"] * 100.0,
            }
        )
    return rows


def load_track_b2_rows() -> List[dict]:
    rows = []
    for key in MODEL_ORDER:
        data = _load_json(TRACK_B2_FILES[key])
        overall = data["overall"]
        pool = data.get("pool_stats") or {}
        rows.append(
            {
                "key": key,
                "label": DISPLAY[key],
                "gt": overall["gt_in_pct"],
                "top10": overall["top10_pct"],
                "top1": overall["top1_pct"],
                "mrr": overall["mrr"],
                "avg_pool": overall["avg_pool"],
                "med_pool": pool.get("median_final_pool"),
                "reduction": pool.get("avg_search_space_reduction", 0.0) * 100.0
                if pool.get("avg_search_space_reduction") is not None
                else None,
            }
        )
    return rows


def load_oracle_strategy_rows() -> List[dict]:
    data = _load_json(_pick_oracle_topology_dir() / "oracle_topology_metrics.json")
    rows = []
    for key in ["p0_only", "p1_only_strategy", "p0_intersect_p1", "p0_union_p1"]:
        overall = data["overall"][key]["overall"]
        pool = data["overall"][key]["pool_stats"]
        rows.append(
            {
                "key": key,
                "label": key.replace("_strategy", ""),
                "gt": overall["gt_in_pct"],
                "top10": overall["top10_pct"],
                "top1": overall["top1_pct"],
                "mrr": overall["mrr"],
                "avg_pool": pool["avg_final_pool"],
                "med_pool": pool["median_final_pool"],
                "reduction": pool["avg_search_space_reduction"] * 100.0,
            }
        )
    return rows


def load_oracle_topology_rows() -> dict:
    data = _load_json(_pick_oracle_topology_dir() / "oracle_topology_metrics.json")
    overall = data["overall"]
    universe = data["sliced"]["universe"]
    multiplicity = data["sliced"]["multiplicity"]
    return {
        "overall": {
            "p1_only_upper_bound": overall["p1_only_upper_bound"],
            "full_topology_union": overall["full_topology_union"],
        },
        "universe": universe,
        "multiplicity": multiplicity,
    }


def _pick_latest_oracle_phase3_summary() -> Path:
    candidates: List[Path] = []
    for root in ORACLE_PHASE3_DIRS:
        candidates.extend(sorted(root.glob("summary_*_v2_lora_p0_union_p1.csv")))
    if not candidates:
        raise FileNotFoundError("No oracle Phase 3 summary CSV found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_oracle_phase3_metrics() -> dict:
    path = _pick_latest_oracle_phase3_summary()
    wanted = {
        "Top-1 Accuracy": "top1",
        "Top-10 Accuracy": "top10",
        "MRR@10": "mrr",
        "GT-in-Pool": "gt",
        "Avg Search-Space Reduction": "reduction",
    }
    out: Dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("===") or line == "Metric,Value":
            continue
        if "," not in line:
            continue
        key, value = line.split(",", 1)
        key = key.strip()
        value = value.strip()
        if key not in wanted:
            continue
        target = wanted[key]
        if key == "GT-in-Pool":
            pct = value.split("(")[-1].rstrip(")")
            out[target] = float(pct.rstrip("%"))
        elif key == "Avg Search-Space Reduction":
            out[target] = float(value) * 100.0
        else:
            out[target] = float(value) * 100.0 if key.startswith("Top-") else float(value)
    out["path"] = str(path)
    return out


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#d9d9e3",
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "grid.color": "#e9e7ef",
            "grid.linewidth": 0.8,
        }
    )


def _annotate_bars(ax, bars: Iterable, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _style_axes(fig, axes) -> None:
    axes_list = axes if isinstance(axes, (list, tuple)) else [axes]
    for ax in axes_list:
        if hasattr(ax, "spines"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#d9d9e3")
            ax.spines["bottom"].set_color("#d9d9e3")
            ax.set_facecolor("#ffffff")
            ax.grid(True, linestyle="--", alpha=0.35)


def _callout_bbox(color: str) -> dict:
    return {"facecolor": color, "edgecolor": "#111111", "alpha": 0.96, "pad": 0.35}


def _add_note(fig, text: str) -> None:
    return


def _highlight_best(ax, bars: Iterable, values: List[float]) -> None:
    if not values:
        return
    best = max(values)
    for bar, value in zip(bars, values):
        if value == best:
            bar.set_linewidth(1.8)
            bar.set_edgecolor("#111111")


def plot_topology_overview(out_path: Path) -> None:
    counts = load_topology_counts()
    _ensure_parent(out_path)
    fig = plt.figure(figsize=(15.8, 5.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.14)
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    _style_axes(fig, [ax2, ax3])

    mult_order = ["1-rel", "2-rel", "3-rel"]
    mult_vals = [counts["multiplicity"].get(k, 0) for k in mult_order]

    family_map = {
        "singleton:CONNECTS_TO": (UNIVERSE_META["U1"]["label"], UNIVERSE_META["U1"]["color"], "1-rel"),
        "singleton:ADJACENT_TO": (UNIVERSE_META["U2"]["label"], UNIVERSE_META["U2"]["color"], "1-rel"),
        "singleton:FILLS": (UNIVERSE_META["U6"]["label"], UNIVERSE_META["U6"]["color"], "1-rel"),
        "paired:FILLS+NEXT_TO": (UNIVERSE_META["U3"]["label"], UNIVERSE_META["U3"]["color"], "2-rel"),
        "triad:FILLS+NEXT_TO+NEXT_TO": (UNIVERSE_META["U4"]["label"], UNIVERSE_META["U4"]["color"], "3-rel"),
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)": (UNIVERSE_META["U5"]["label"], UNIVERSE_META["U5"]["color"], "3-rel"),
    }
    bottoms = {k: 0.0 for k in mult_order}
    totals = {k: counts["multiplicity"].get(k, 0) for k in mult_order}
    x_positions = list(range(len(mult_order)))
    x_lookup = {bucket: idx for idx, bucket in enumerate(mult_order)}
    for family_key in [
        "singleton:CONNECTS_TO",
        "singleton:ADJACENT_TO",
        "singleton:FILLS",
        "paired:FILLS+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)",
    ]:
        display, color, bucket = family_map[family_key]
        value = counts["family"].get(family_key, 0)
        if value <= 0:
            continue
        pct = value / totals[bucket] * 100.0 if totals[bucket] else 0.0
        bars = ax2.bar(
            x_lookup[bucket],
            pct,
            bottom=bottoms[bucket],
            color=color,
            edgecolor="white",
            linewidth=1.0,
            label=display,
            width=0.62,
        )
        if pct >= 12:
            ax2.text(
                x_lookup[bucket],
                bottoms[bucket] + pct / 2,
                f"{pct:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                weight="bold",
            )
        bottoms[bucket] += pct
    ax2.set_title("Composition within each multiplicity bucket")
    ax2.set_ylabel("Share within bucket (%)")
    ax2.set_ylim(0, 100)
    xtick_labels = []
    for bucket in mult_order:
        count = totals[bucket]
        overall_pct = count / counts["n_cases"] * 100.0 if counts["n_cases"] else 0.0
        xtick_labels.append(f"{bucket}\n(n={count}, {overall_pct:.0f}%)")
    ax2.set_xticks(x_positions, xtick_labels)
    ax2.set_xlim(-0.6, len(mult_order) - 0.4)
    ax2.text(
        0.01,
        1.04,
        f"n = {counts['n_cases']} cases | {counts['n_signatures']} unique signatures",
        transform=ax2.transAxes,
        fontsize=10,
        color="#555555",
    )
    handles, labels = ax2.get_legend_handles_labels()
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    ax2.legend([h for h, _ in uniq], [l for _, l in uniq], frameon=True, loc="lower right", fontsize=9)

    universe_order = [
        "singleton:CONNECTS_TO",
        "singleton:ADJACENT_TO",
        "paired:FILLS+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)",
        "singleton:FILLS",
    ]
    universe_labels = [family_map[k][0] for k in universe_order]
    universe_vals = [counts["family"].get(k, 0) for k in universe_order]
    universe_colors = [family_map[k][1] for k in universe_order]
    bars = ax3.barh(universe_labels, universe_vals, color=universe_colors, edgecolor="black", linewidth=0.6)
    ax3.set_title("Universe share in AP held-out benchmark")
    ax3.set_xlabel("Cases")
    ax3.invert_yaxis()
    for tick, family_key in zip(ax3.get_yticklabels(), universe_order):
        tick.set_color(family_map[family_key][1])
        tick.set_fontweight("semibold")
    for bar, value in zip(bars, universe_vals):
        pct = value / counts["n_cases"] * 100.0
        ax3.text(value + 0.35, bar.get_y() + bar.get_height() / 2, f"{value:.0f} ({pct:.0f}%)", va="center", fontsize=9.5)

    fig.suptitle("Figure 1. AP held-out topology overview", fontsize=16, y=1.02)
    _add_note(
        fig,
        "Data source: AP held-out eval ground-truth labels (n=60), extracted from assistant JSON in "
        "lora6_v2_ap_eval_canonical_m.jsonl. No model inference is involved here. This figure describes "
        "the benchmark topology itself: a flat multi-anchor relation benchmark rather than a labeled deep-hop benchmark.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_track_a_comparison(out_path: Path) -> None:
    rows = load_track_a_rows()
    metrics = [("hop1", "Hop-1"), ("pred_p", "Pred P"), ("pred_r", "Pred R"), ("dir", "Dir")]
    labels = [r["label"] for r in rows]
    values = [[r[key] for key, _ in metrics] for r in rows]
    fig, ax = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    _style_axes(fig, ax)

    image = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(metrics)), [title for _, title in metrics])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Track A metrics (%)")
    ax.set_ylabel("Models")

    for row_idx, row in enumerate(values):
        for col_idx, value in enumerate(row):
            text_color = "#111111" if value < 72 else "white"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                fontweight="bold" if value == max(row) else None,
            )

    for col_idx in range(len(metrics)):
        col_vals = [row[col_idx] for row in values]
        best_idx = col_vals.index(max(col_vals))
        ax.add_patch(
            plt.Rectangle(
                (col_idx - 0.5, best_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#111111",
                linewidth=2.0,
            )
        )

    cbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Score (%)")
    fig.suptitle("Figure 2. Track A intermediate extraction matrix", fontsize=16, y=1.03)
    _add_note(
        fig,
        "Benchmark: Track A / AP held-out eval (n=60). Inputs are site photo + floorplan(M) + chat. "
        "Metrics come from score_ap_track.py. Rows are models, columns are Hop-1, predicate precision, predicate recall, and direction accuracy. "
        "Dark outlined cells mark the best score in each column. G0–G6 are LoRA adapters; Gemini AP v2 is the repaired zero-shot baseline using the shared schema and prompt, but no fine-tuning.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_track_b2_comparison(out_path: Path) -> None:
    rows = load_track_b2_rows()
    labels = [r["label"] for r in rows]
    model_keys = [r["key"] for r in rows]
    x = list(range(len(rows)))
    width = 0.18
    gt_vals = [r["gt"] for r in rows]
    top10_vals = [r["top10"] for r in rows]
    top1_vals = [r["top1"] for r in rows]
    pool_vals = [r["avg_pool"] for r in rows]
    mrr_vals = [r["mrr"] * 1000.0 for r in rows]
    model_colors = [COLORS[k] for k in model_keys]

    fig, ax = plt.subplots(figsize=(14.8, 6.8), constrained_layout=True)
    _style_axes(fig, ax)
    ax_r = ax.twinx()
    ax_r.grid(False)

    bars1 = ax.bar(
        [i - width for i in x],
        gt_vals,
        width=width,
        color=model_colors,
        alpha=1.0,
        edgecolor="white",
        linewidth=0.8,
        label="GT-in-Pool",
        zorder=3,
    )
    bars2 = ax.bar(
        x,
        top10_vals,
        width=width,
        color=model_colors,
        alpha=0.58,
        edgecolor="white",
        linewidth=0.8,
        label="Top-10",
        zorder=3,
    )
    bars3 = ax.bar(
        [i + width for i in x],
        top1_vals,
        width=width,
        color=model_colors,
        alpha=0.30,
        edgecolor="white",
        linewidth=0.8,
        label="Top-1",
        zorder=3,
    )
    line1 = ax_r.plot(x, pool_vals, color=METRIC_COLORS["pool"], marker="o", linewidth=2.2, markersize=8, label="Avg Pool Size", zorder=4)[0]
    line2 = ax_r.plot(x, mrr_vals, color=METRIC_COLORS["mrr"], marker="s", linestyle="--", linewidth=2.4, markersize=8, label="MRR@10 (×1000)", zorder=4)[0]

    for bars, vals, fmt in [(bars1, gt_vals, "{:.1f}%"), (bars2, top10_vals, "{:.1f}%"), (bars3, top1_vals, "{:.1f}%")]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, fmt.format(val), ha="center", va="bottom", fontsize=9)
    for xx, val in zip(x, pool_vals):
        ax_r.text(xx, val + 1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=9, color=METRIC_COLORS["pool"])
    for xx, val, raw in zip(x, mrr_vals, [r["mrr"] for r in rows]):
        ax_r.text(xx, val + 1.8, f"{raw:.4f}", ha="center", va="bottom", fontsize=9, color=METRIC_COLORS["mrr"])

    ax.set_xticks(x, labels)
    for tick, key in zip(ax.get_xticklabels(), model_keys):
        tick.set_color(COLORS[key])
        tick.set_fontweight("semibold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(max(gt_vals), max(top10_vals), max(top1_vals)) + 14)
    ax_r.set_ylabel("Pool Size / MRR@10 ×1000")
    ax_r.set_ylim(0, max(max(pool_vals), max(mrr_vals)) * 1.18)
    ax.set_title("Figure 3. Track B-2 strict downstream evaluation")
    legend_handles = [bars1, bars2, bars3, line1, line2]
    legend_labels = ["GT-in-Pool", "Top-10", "Top-1", "Avg Pool Size", "MRR@10 (×1000)"]
    ax.legend(legend_handles, legend_labels, loc="upper right", frameon=True)

    best_top10 = max(rows, key=lambda r: r["top10"])
    idx = labels.index(best_top10["label"])
    ax.axvspan(idx - 0.42, idx + 0.42, color=HIGHLIGHT_COLORS["safe_green_fill"], zorder=0)
    ax.text(idx, ax.get_ylim()[1] * 0.97, f"Best Top-10: {best_top10['label']}", ha="center", va="top", fontsize=10, weight="bold", color=HIGHLIGHT_COLORS["safe_green_text"])

    fig.suptitle("Track B-2: GT-in-Pool / Top-10 / Top-1", fontsize=15, y=1.02)
    _add_note(
        fig,
        "Benchmark: Track B-2 / AP held-out end-to-end (n=60). Pipeline: v2_lora with p0_union_p1. "
        "P0 is the spatial planner branch; P1 is the storey + ifc_class fallback. These bars use the current canonical Phase 4 planner settings, so they reflect strict AP downstream retrieval, not only extraction quality.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_strategy_sweep(out_path: Path) -> None:
    rows = load_oracle_strategy_rows()
    strategy_keys = [r["key"] for r in rows]
    labels = [STRATEGY_META[k]["label"] for k in strategy_keys]
    x = list(range(len(rows)))
    width = 0.18
    gt_vals = [r["gt"] for r in rows]
    top10_vals = [r["top10"] for r in rows]
    top1_vals = [r["top1"] for r in rows]
    pool_vals = [r["avg_pool"] for r in rows]
    mrr_vals = [r["mrr"] * 1000.0 for r in rows]

    fig, ax = plt.subplots(figsize=(13.8, 6.4), constrained_layout=True)
    _style_axes(fig, ax)
    ax_r = ax.twinx()
    ax_r.grid(False)

    bars1 = ax.bar([i - width for i in x], gt_vals, width=width, color=METRIC_COLORS["gt"], edgecolor="white", linewidth=0.8, label="GT-in-Pool", zorder=3)
    bars2 = ax.bar(x, top10_vals, width=width, color=METRIC_COLORS["top10"], edgecolor="white", linewidth=0.8, label="Top-10", zorder=3)
    bars3 = ax.bar([i + width for i in x], top1_vals, width=width, color=METRIC_COLORS["top1"], edgecolor="white", linewidth=0.8, label="Top-1", zorder=3)
    line1 = ax_r.plot(x, pool_vals, color=METRIC_COLORS["pool"], marker="o", linewidth=2.2, markersize=8, label="Avg Pool Size", zorder=4)[0]
    line2 = ax_r.plot(x, mrr_vals, color=METRIC_COLORS["mrr"], marker="s", linestyle="--", linewidth=2.4, markersize=8, label="MRR@10 (×1000)", zorder=4)[0]

    for bars, vals, fmt in [(bars1, gt_vals, "{:.1f}%"), (bars2, top10_vals, "{:.1f}%"), (bars3, top1_vals, "{:.1f}%")]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, fmt.format(val), ha="center", va="bottom", fontsize=9)
    for xx, val in zip(x, pool_vals):
        ax_r.text(xx, val + 1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=9, color=METRIC_COLORS["pool"])
    for xx, val, raw in zip(x, mrr_vals, [r["mrr"] for r in rows]):
        ax_r.text(xx, val + 1.8, f"{raw:.4f}", ha="center", va="bottom", fontsize=9, color=METRIC_COLORS["mrr"])

    ax.set_xticks(x, labels)
    for tick, key in zip(ax.get_xticklabels(), strategy_keys):
        tick.set_color(STRATEGY_META[key]["color"])
        tick.set_fontweight("semibold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(max(gt_vals), max(top10_vals), max(top1_vals)) + 16)
    ax_r.set_ylabel("Pool Size / MRR@10 ×1000")
    ax_r.set_ylim(0, max(max(pool_vals), max(mrr_vals)) * 1.18)
    ax.legend([bars1, bars2, bars3, line1, line2], ["GT-in-Pool", "Top-10", "Top-1", "Avg Pool Size", "MRR@10 (×1000)"], loc="upper right", frameon=True)
    ax.set_title("Figure 4. Oracle strategy sweep on AP held-out")

    safest = max(rows, key=lambda r: (r["gt"], r["top10"], r["mrr"]))
    idx = strategy_keys.index(safest["key"])
    ax.axvspan(idx - 0.42, idx + 0.42, color="#E3F2FD", zorder=0)
    ax.text(idx, ax.get_ylim()[1] * 0.97, f"Safest: {labels[idx]}", ha="center", va="top", fontsize=10, weight="bold", color=STRATEGY_META[safest["key"]]["color"])

    fig.suptitle("Oracle strategy sweep: GT-in-Pool / Top-10 / Top-1", fontsize=15, y=1.02)
    _add_note(
        fig,
        "Benchmark: oracle full-topology precomputed constraints on the same AP held-out end-to-end cases (n=60). "
        "This is the Phase 2A current-system strategy search. p0 = current executable spatial planner branch; "
        "p1 = storey + ifc_class fallback. The selection criterion is not just Top-10/MRR, but GT-in-Pool safety as well.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_p1_vs_full_topology(out_path: Path) -> None:
    data = load_oracle_topology_rows()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
    _style_axes(fig, list(axes))

    left = axes[0]
    metrics = ["Top-10", "Top-1", "MRR@10"]
    p1_vals = [
        data["overall"]["p1_only_upper_bound"]["overall"]["top10_pct"],
        data["overall"]["p1_only_upper_bound"]["overall"]["top1_pct"],
        data["overall"]["p1_only_upper_bound"]["overall"]["mrr"] * 100.0,
    ]
    full_vals = [
        data["overall"]["full_topology_union"]["overall"]["top10_pct"],
        data["overall"]["full_topology_union"]["overall"]["top1_pct"],
        data["overall"]["full_topology_union"]["overall"]["mrr"] * 100.0,
    ]
    y = list(range(len(metrics)))
    for idx, metric in enumerate(metrics):
        left.plot([p1_vals[idx], full_vals[idx]], [idx, idx], color="#9e9e9e", linewidth=2)
    left.scatter(p1_vals, y, color=COLORS["p1_only_upper_bound"], s=90, label="P1-only")
    left.scatter(full_vals, y, color=COLORS["full_topology_union"], s=90, label="Full-topology")
    left.set_yticks(y, metrics)
    left.set_xlabel("Score (%)")
    left.set_title("Overall gain from topology-aware oracle constraints")
    left.legend(frameon=True)

    right = axes[1]
    universe_map = data["universe"]
    universe_order = ["U1", "U2", "U3", "U4", "U5", "U6"]
    deltas = []
    for u in universe_order:
        p1 = universe_map[u]["p1_only_upper_bound"]["overall"]["top10_pct"]
        full = universe_map[u]["full_topology_union"]["overall"]["top10_pct"]
        deltas.append(full - p1)
    universe_colors = [UNIVERSE_META[u]["color"] for u in universe_order]
    bars = right.bar(universe_order, deltas, color=universe_colors, edgecolor="black", linewidth=0.6)
    right.axhline(0, color="black", linewidth=1)
    right.set_ylabel("Top-10 delta (pts)")
    right.set_title("Benefit-of-spatial by topology universe")
    for tick, u in zip(right.get_xticklabels(), universe_order):
        tick.set_color(UNIVERSE_META[u]["color"])
        tick.set_fontweight("semibold")
    _annotate_bars(right, bars, "{:.1f}")

    fig.suptitle("Figure 5. P1-only vs full-topology oracle benefit", fontsize=16, y=1.02)
    _add_note(
        fig,
        "Left: global oracle gain from P1-only to full-topology constraints under the same AP held-out end-to-end benchmark. "
        "Right: Top-10 delta by topology universe (U1–U6). U3 Opening-Paired and U5 Mixed-Triad are the main planner-opportunity families.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_vs_model_gap(out_path: Path) -> None:
    oracle = load_oracle_phase3_metrics()
    rows = load_track_b2_rows()
    model_rows = [
        next(r for r in rows if r["key"] == key)
        for key in ["g3_fullaug_r32", "g4_ultimate", "g7_position_context", "g2_fullaug_lowlr", "gemini_ap_v2"]
    ]
    compare_rows = [
        {
            "label": "Oracle P3",
            "gt": oracle["gt"],
            "top10": oracle["top10"],
            "top1": oracle["top1"],
            "mrr": oracle["mrr"],
            "reduction": oracle["reduction"],
            "key": "oracle_phase3",
        }
    ] + model_rows
    labels = [r["label"] for r in compare_rows]
    model_keys = [r["key"] for r in compare_rows]
    x = list(range(len(compare_rows)))
    width = 0.18
    gt_vals = [r["gt"] for r in compare_rows]
    top10_vals = [r["top10"] for r in compare_rows]
    top1_vals = [r["top1"] for r in compare_rows]
    reduction_vals = [r["reduction"] for r in compare_rows]
    mrr_vals = [r["mrr"] * 1000.0 for r in compare_rows]
    model_colors = [COLORS[k] for k in model_keys]

    fig, ax = plt.subplots(figsize=(14.6, 6.6), constrained_layout=True)
    _style_axes(fig, ax)
    ax_r = ax.twinx()
    ax_r.grid(False)

    bars1 = ax.bar(
        [i - width for i in x],
        gt_vals,
        width=width,
        color=model_colors,
        alpha=1.0,
        edgecolor="white",
        linewidth=0.8,
        label="GT-in-Pool",
        zorder=3,
    )
    bars2 = ax.bar(
        x,
        top10_vals,
        width=width,
        color=model_colors,
        alpha=0.58,
        edgecolor="white",
        linewidth=0.8,
        label="Top-10",
        zorder=3,
    )
    bars3 = ax.bar(
        [i + width for i in x],
        top1_vals,
        width=width,
        color=model_colors,
        alpha=0.30,
        edgecolor="white",
        linewidth=0.8,
        label="Top-1",
        zorder=3,
    )
    line1 = ax_r.plot(x, reduction_vals, color=METRIC_COLORS["reduction"], marker="o", linewidth=2.2, markersize=8, label="Reduction (%)", zorder=4)[0]
    line2 = ax_r.plot(x, mrr_vals, color=METRIC_COLORS["mrr_aux"], marker="s", linestyle="--", linewidth=2.4, markersize=8, label="MRR@10 (×1000)", zorder=4)[0]

    for bars, vals, fmt in [(bars1, gt_vals, "{:.1f}%"), (bars2, top10_vals, "{:.1f}%"), (bars3, top1_vals, "{:.1f}%")]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, fmt.format(val), ha="center", va="bottom", fontsize=9)
    for xx, val in zip(x, reduction_vals):
        ax_r.text(xx, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=METRIC_COLORS["reduction"])
    for xx, val, raw in zip(x, mrr_vals, [r["mrr"] for r in compare_rows]):
        ax_r.text(xx, val + 1.8, f"{raw:.4f}", ha="center", va="bottom", fontsize=9, color=METRIC_COLORS["mrr_aux"])

    ax.set_xticks(x, labels)
    for tick, key in zip(ax.get_xticklabels(), model_keys):
        tick.set_color(COLORS[key])
        tick.set_fontweight("semibold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(max(gt_vals), max(top10_vals), max(top1_vals)) + 16)
    ax_r.set_ylabel("Reduction / MRR@10 ×1000")
    ax_r.set_ylim(0, max(max(reduction_vals), max(mrr_vals)) * 1.18)
    ax.legend([bars1, bars2, bars3, line1, line2], ["GT-in-Pool", "Top-10", "Top-1", "Reduction (%)", "MRR@10 (×1000)"], loc="upper right", frameon=True)
    ax.set_title("Figure 6. Oracle vs model gap after planner optimization")

    ax.axvspan(-0.42, 0.42, color=HIGHLIGHT_COLORS["oracle_fill"], zorder=0)
    best_model = max(model_rows, key=lambda r: (r["top10"], r["mrr"]))
    best_idx = labels.index(best_model["label"])
    ax.axvspan(best_idx - 0.42, best_idx + 0.42, color=HIGHLIGHT_COLORS["safe_green_fill"], zorder=0)
    ax.text(0, ax.get_ylim()[1] * 0.97, "Oracle ceiling", ha="center", va="top", fontsize=10, weight="bold", color=HIGHLIGHT_COLORS["oracle_text"])
    ax.text(best_idx, ax.get_ylim()[1] * 0.90, f"Closest model: {best_model['label']}", ha="center", va="top", fontsize=10, weight="bold", color=HIGHLIGHT_COLORS["safe_green_text"])

    fig.suptitle("Oracle-to-model comparison after G7 correction", fontsize=15, y=1.02)
    _add_note(
        fig,
        "Oracle Phase3 is the symbolic ceiling under perfect extraction with the upgraded planner and p0_union_p1. "
        "Model bars use the same strict AP held-out downstream benchmark. The residual gap therefore isolates extraction-side loss after planner optimization.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_waterfall_progression(out_path: Path) -> None:
    oracle_topo = load_oracle_topology_rows()
    oracle_phase3 = load_oracle_phase3_metrics()
    track_b2 = {r["key"]: r for r in load_track_b2_rows()}
    best_model = track_b2["g3_fullaug_r32"]

    stages = ["P1-only\noracle", "Full-topology\noracle", "Oracle\nPhase3", "Best model\n(G3)"]
    top10 = [
        oracle_topo["overall"]["p1_only_upper_bound"]["overall"]["top10_pct"],
        oracle_topo["overall"]["full_topology_union"]["overall"]["top10_pct"],
        oracle_phase3["top10"],
        best_model["top10"],
    ]
    mrr = [
        oracle_topo["overall"]["p1_only_upper_bound"]["overall"]["mrr"],
        oracle_topo["overall"]["full_topology_union"]["overall"]["mrr"],
        oracle_phase3["mrr"],
        best_model["mrr"],
    ]
    colors = [COLORS["p1_only_upper_bound"], COLORS["full_topology_union"], COLORS["oracle_phase3"], COLORS["g3_fullaug_r32"]]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), constrained_layout=True)
    _style_axes(fig, list(axes))
    x = list(range(len(stages)))

    axes[0].plot(x, top10, color="#333333", linewidth=2.0, marker="o")
    bars = axes[0].bar(x, top10, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(x, stages)
    for tick, color in zip(axes[0].get_xticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("semibold")
    axes[0].set_ylim(0, 45)
    axes[0].set_ylabel("Top-10 (%)")
    axes[0].set_title("Top-10 progression")
    _annotate_bars(axes[0], bars, "{:.1f}")
    axes[0].annotate("+13.3", xy=(1, top10[1]), xytext=(0.55, 33), arrowprops=dict(arrowstyle="->", color="#666"))
    axes[0].annotate("+10.0", xy=(2, top10[2]), xytext=(1.45, 41), arrowprops=dict(arrowstyle="->", color="#666"))
    axes[0].annotate("-13.3", xy=(3, top10[3]), xytext=(2.45, 31), arrowprops=dict(arrowstyle="->", color="#666"))

    axes[1].plot(x, mrr, color="#333333", linewidth=2.0, marker="o")
    bars = axes[1].bar(x, mrr, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(x, stages)
    for tick, color in zip(axes[1].get_xticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("semibold")
    axes[1].set_ylim(0, 0.15)
    axes[1].set_ylabel("MRR@10")
    axes[1].set_title("MRR progression")
    _annotate_bars(axes[1], bars, "{:.3f}")

    fig.suptitle("Figure 7. Retrieval progression from P1 baseline to oracle ceiling and best realized model", fontsize=16, y=1.03)
    _add_note(
        fig,
        "This progression compresses the full story into four stages on the same AP held-out benchmark: "
        "P1-only oracle -> +topology-aware oracle constraints -> +Phase3 planner ceiling -> best realized model (G3). "
        "It separates the benefit of spatial relations from the remaining extraction gap.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_v2_extraction_downstream_tradeoff(out_path: Path) -> None:
    track_a = {r["key"]: r for r in load_track_a_rows()}
    track_b2 = {r["key"]: r for r in load_track_b2_rows()}
    keys = [k for k in MODEL_ORDER if k in track_a and k in track_b2]
    labels = [DISPLAY[k] for k in keys]
    x = list(range(len(keys)))
    top10_vals = [track_b2[k]["top10"] for k in keys]
    pred_r_vals = [track_a[k]["pred_r"] for k in keys]
    gt_vals = [track_b2[k]["gt"] for k in keys]
    mrr_vals = [track_b2[k]["mrr"] * 1000.0 for k in keys]
    colors = [COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(14.8, 6.7), constrained_layout=True)
    _style_axes(fig, ax)
    ax_r = ax.twinx()
    ax_r.grid(False)

    bars = ax.bar(x, top10_vals, color=colors, edgecolor="white", linewidth=0.9, width=0.62, label="Track B-2 Top-10", zorder=3)
    line1 = ax.plot(x, pred_r_vals, color=METRIC_COLORS["pred_r"], marker="o", linewidth=2.6, markersize=8, label="Track A Pred Recall", zorder=4)[0]
    line2 = ax.plot(x, gt_vals, color=METRIC_COLORS["gt_line"], marker="s", linestyle="--", linewidth=2.2, markersize=7, label="Track B-2 GT-in-Pool", zorder=4)[0]
    line3 = ax_r.plot(x, mrr_vals, color=METRIC_COLORS["mrr_track"], marker="D", linestyle=":", linewidth=2.4, markersize=7, label="Track B-2 MRR@10 (×1000)", zorder=4)[0]

    for bar, val in zip(bars, top10_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.55, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    for xx, val in zip(x, pred_r_vals):
        ax.text(xx, val + 1.1, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=METRIC_COLORS["pred_r"])
    for xx, val in zip(x, gt_vals):
        ax.text(xx, val + 1.1, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=METRIC_COLORS["gt_line"])
    for xx, val, raw in zip(x, mrr_vals, [track_b2[k]["mrr"] for k in keys]):
        ax_r.text(xx, val + 1.5, f"{raw:.4f}", ha="center", va="bottom", fontsize=9, color=METRIC_COLORS["mrr_track"])

    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy / Recall (%)")
    ax.set_ylim(0, max(max(gt_vals), max(pred_r_vals), max(top10_vals)) + 16)
    ax_r.set_ylabel("MRR@10 ×1000")
    ax_r.set_ylim(0, max(mrr_vals) * 1.18)
    ax.set_title("Figure 2v2. Dual-track model selection dashboard")
    ax.legend([bars, line1, line2, line3], ["Track B-2 Top-10", "Track A Pred Recall", "Track B-2 GT-in-Pool", "Track B-2 MRR@10 (×1000)"], loc="upper right", frameon=True)

    winner_a = max(keys, key=lambda k: track_a[k]["pred_r"])
    winner_b = max(keys, key=lambda k: track_b2[k]["top10"])
    idx_a = keys.index(winner_a)
    idx_b = keys.index(winner_b)
    ax.axvspan(idx_a - 0.34, idx_a + 0.34, color=HIGHLIGHT_COLORS["winner_amber_fill"], zorder=0)
    ax.axvspan(idx_b - 0.34, idx_b + 0.34, color=HIGHLIGHT_COLORS["safe_green_fill"], zorder=0)
    ax.text(idx_a, ax.get_ylim()[1] * 0.97, f"Best Pred Recall: {DISPLAY[winner_a]}", ha="center", va="top", fontsize=10, weight="bold", color=HIGHLIGHT_COLORS["winner_amber_text"])
    ax.text(idx_b, ax.get_ylim()[1] * 0.90, f"Best Top-10: {DISPLAY[winner_b]}", ha="center", va="top", fontsize=10, weight="bold", color=HIGHLIGHT_COLORS["safe_green_text"])

    fig.suptitle("Track A vs Track B-2: Pred Recall / Top-10 / GT-in-Pool", fontsize=15, y=1.02)
    _add_note(
        fig,
        "Left: Track A Hop-1 vs Track B-2 Top-10, with bubble size proportional to GT-in-Pool on the AP held-out strict downstream benchmark. "
        "Right: compact dual-track matrix using Hop-1, predicate recall, Top-10, MRRx100, GT-in-Pool, and search-space reduction. "
        "This replaces separate Track A/Track B-2 reading with a single model-selection dashboard.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_v2_oracle_dashboard(out_path: Path) -> None:
    strategy_rows = load_oracle_strategy_rows()
    topo = load_oracle_topology_rows()

    fig = plt.figure(figsize=(16.0, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.18, 1.0], wspace=0.10)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    _style_axes(fig, [ax1, ax2, ax3])

    strat_keys = ["p0_only", "p1_only_strategy", "p0_intersect_p1", "p0_union_p1"]
    strat_labels = [STRATEGY_META[k]["label"] for k in strat_keys]
    strat_map = {r["key"]: r for r in strategy_rows}
    strat_matrix = [
        [
            strat_map[k]["gt"],
            strat_map[k]["top10"],
            strat_map[k]["top1"],
            strat_map[k]["mrr"] * 100.0,
            strat_map[k]["reduction"],
        ]
        for k in strat_keys
    ]
    strat_cols = ["GT", "Top-10", "Top-1", "MRRx100", "Red."]
    image = ax1.imshow(strat_matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax1.grid(False)
    ax1.set_xticks(range(len(strat_cols)), strat_cols)
    ax1.set_yticks(range(len(strat_labels)), strat_labels)
    for tick, key in zip(ax1.get_yticklabels(), strat_keys):
        tick.set_color(STRATEGY_META[key]["color"])
        tick.set_fontweight("semibold")
    ax1.set_title("Current-system strategy sweep")
    for i, row in enumerate(strat_matrix):
        for j, value in enumerate(row):
            ax1.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9, color="white" if value > 60 else "#111111")
    safest_idx = strat_keys.index("p0_union_p1")
    ax1.add_patch(
        plt.Rectangle((-0.5, safest_idx - 0.5), len(strat_cols), 1, fill=False, edgecolor=COLORS["p0_union_p1"], linewidth=2.2)
    )
    ax1.text(
        len(strat_cols) - 0.2,
        safest_idx - 0.62,
        "Safest executable strategy",
        ha="right",
        va="bottom",
        fontsize=9,
        color=COLORS["p0_union_p1"],
        weight="bold",
    )
    fig.colorbar(image, ax=ax1, shrink=0.82, pad=0.02)

    overall = topo["overall"]
    metrics = ["Top-10", "Top-1", "MRRx100"]
    p1_vals = [
        overall["p1_only_upper_bound"]["overall"]["top10_pct"],
        overall["p1_only_upper_bound"]["overall"]["top1_pct"],
        overall["p1_only_upper_bound"]["overall"]["mrr"] * 100.0,
    ]
    full_vals = [
        overall["full_topology_union"]["overall"]["top10_pct"],
        overall["full_topology_union"]["overall"]["top1_pct"],
        overall["full_topology_union"]["overall"]["mrr"] * 100.0,
    ]
    y = list(range(len(metrics)))
    for idx in range(len(metrics)):
        ax2.plot([p1_vals[idx], full_vals[idx]], [idx, idx], color="#9e9e9e", linewidth=2)
    ax2.scatter(p1_vals, y, color=COLORS["p1_only_upper_bound"], s=95, label="P1-only")
    ax2.scatter(full_vals, y, color=COLORS["full_topology_union"], s=95, label="Full-topology")
    for idx in range(len(metrics)):
        delta = full_vals[idx] - p1_vals[idx]
        ax2.text(full_vals[idx] + 0.6, idx, f"+{delta:.1f}", va="center", fontsize=10, color=COLORS["full_topology_union"], weight="bold")
    ax2.set_yticks(y, metrics)
    ax2.set_xlabel("Score")
    ax2.set_title("Benefit of spatial topology")
    ax2.legend(frameon=True, loc="lower right", bbox_to_anchor=(0.98, 0.04), borderpad=0.8)

    universe = topo["universe"]
    universe_order = ["U1", "U2", "U3", "U4", "U5", "U6"]
    top10_delta = []
    mrr_delta = []
    for u in universe_order:
        p1 = universe[u]["p1_only_upper_bound"]["overall"]
        full = universe[u]["full_topology_union"]["overall"]
        top10_delta.append(full["top10_pct"] - p1["top10_pct"])
        mrr_delta.append((full["mrr"] - p1["mrr"]) * 100.0)

    y = list(range(len(universe_order)))
    width = 0.34
    for u in ["U3", "U5"]:
        idx = universe_order.index(u)
        ax3.axhspan(idx - 0.55, idx + 0.55, color=UNIVERSE_META[u]["color"], alpha=0.10, zorder=0)

    bars1 = ax3.barh(
        [yy - width / 2 for yy in y],
        top10_delta,
        height=width,
        color=COLORS["g4_ultimate"],
        edgecolor="black",
        linewidth=0.6,
        label="Top-10 Δ",
        zorder=3,
    )
    bars2 = ax3.barh(
        [yy + width / 2 for yy in y],
        mrr_delta,
        height=width,
        color=COLORS["gemini_ap_v2"],
        edgecolor="black",
        linewidth=0.6,
        label="MRRx100 Δ",
        zorder=3,
    )
    ax3.axvline(0, color="#666666", linewidth=1.0)
    ax3.set_yticks(y, universe_order)
    for tick, u in zip(ax3.get_yticklabels(), universe_order):
        tick.set_color(UNIVERSE_META[u]["color"])
        tick.set_fontweight("semibold")
    ax3.set_xlabel("Improvement over P1-only (points)")
    ax3.set_title("Topology benefit by universe slice (U1–U6)")
    ax3.legend(frameon=True, loc="lower right", ncol=2)
    ax3.set_xlim(min(min(top10_delta), min(mrr_delta), 0) - 5, max(max(top10_delta), max(mrr_delta)) + 8)
    ax3.invert_yaxis()
    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_width()
            ax3.text(
                value + (0.8 if value >= 0 else -0.8),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
            )
    ax3.text(
        max(max(top10_delta), max(mrr_delta)) + 3.4,
        universe_order.index("U3"),
        "Paired\nlift",
        va="center",
        fontsize=10,
        weight="bold",
        color=UNIVERSE_META["U3"]["color"],
    )
    ax3.text(
        max(max(top10_delta), max(mrr_delta)) + 3.4,
        universe_order.index("U5"),
        "Mixed-triad\nlift",
        va="center",
        fontsize=10,
        weight="bold",
        color=UNIVERSE_META["U5"]["color"],
    )

    fig.suptitle("Figure 4v2. Oracle dashboard for planner selection and topology benefit", fontsize=16, y=1.02)
    _add_note(
        fig,
        "Top-left: Phase 2A strategy search under current executable planner logic. Top-right: overall gain from P1-only to full-topology oracle constraints. "
        "Bottom: topology benefit by universe slice. This compact dashboard supports two claims: p0_union_p1 is the safest executable strategy, and U3/U5 are the clearest planner-opportunity families.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_appendix_universe_benefit(out_path: Path) -> None:
    data = load_oracle_topology_rows()["universe"]
    order = ["U1", "U2", "U3", "U4", "U5", "U6"]
    p1 = [data[u]["p1_only_upper_bound"]["overall"]["top10_pct"] for u in order]
    full = [data[u]["full_topology_union"]["overall"]["top10_pct"] for u in order]
    x = range(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    _style_axes(fig, ax)
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        p1,
        width=width,
        color=COLORS["p1_only_upper_bound"],
        edgecolor="black",
        linewidth=0.6,
        label="P1-only",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        full,
        width=width,
        color=COLORS["full_topology_union"],
        edgecolor="black",
        linewidth=0.6,
        label="Full-topology",
    )
    ax.set_xticks(list(x), order)
    for tick, u in zip(ax.get_xticklabels(), order):
        tick.set_color(UNIVERSE_META[u]["color"])
        tick.set_fontweight("semibold")
    ax.set_ylabel("Top-10 (%)")
    ax.set_ylim(0, 70)
    ax.set_title("Appendix A1. Benefit-of-spatial by topology universe")
    ax.legend(frameon=True, loc="lower right", borderpad=0.8)
    _annotate_bars(ax, bars1, "{:.1f}")
    _annotate_bars(ax, bars2, "{:.1f}")
    _add_note(
        fig,
        "Appendix benchmark: Phase 2B oracle comparison on AP held-out cases. "
        "Bars show Top-10 under P1-only vs full-topology oracle constraints for each universe U1–U6.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_appendix_multiplicity_benefit(out_path: Path) -> None:
    data = load_oracle_topology_rows()["multiplicity"]
    order = ["1-rel", "2-rel", "3-rel"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    _style_axes(fig, list(axes))

    p1_top10 = [data[k]["p1_only_upper_bound"]["overall"]["top10_pct"] for k in order]
    full_top10 = [data[k]["full_topology_union"]["overall"]["top10_pct"] for k in order]
    x = range(len(order))
    width = 0.38
    bars1 = axes[0].bar(
        [i - width / 2 for i in x], p1_top10, width=width,
        color=COLORS["p1_only_upper_bound"], edgecolor="black", linewidth=0.6, label="P1-only"
    )
    bars2 = axes[0].bar(
        [i + width / 2 for i in x], full_top10, width=width,
        color=COLORS["full_topology_union"], edgecolor="black", linewidth=0.6, label="Full-topology"
    )
    axes[0].set_xticks(list(x), order)
    axes[0].set_title("Top-10 by relation multiplicity")
    axes[0].set_ylabel("Top-10 (%)")
    axes[0].set_ylim(0, 70)
    axes[0].legend(frameon=True)
    _annotate_bars(axes[0], bars1, "{:.1f}")
    _annotate_bars(axes[0], bars2, "{:.1f}")

    p1_mrr = [data[k]["p1_only_upper_bound"]["overall"]["mrr"] for k in order]
    full_mrr = [data[k]["full_topology_union"]["overall"]["mrr"] for k in order]
    bars1 = axes[1].bar(
        [i - width / 2 for i in x], p1_mrr, width=width,
        color=COLORS["p1_only_upper_bound"], edgecolor="black", linewidth=0.6
    )
    bars2 = axes[1].bar(
        [i + width / 2 for i in x], full_mrr, width=width,
        color=COLORS["full_topology_union"], edgecolor="black", linewidth=0.6
    )
    axes[1].set_xticks(list(x), order)
    axes[1].set_title("MRR@10 by relation multiplicity")
    axes[1].set_ylabel("MRR@10")
    axes[1].set_ylim(0, 0.20)
    _annotate_bars(axes[1], bars1, "{:.3f}")
    _annotate_bars(axes[1], bars2, "{:.3f}")

    fig.suptitle("Appendix A2. Benefit-of-spatial by multiplicity", fontsize=16, y=1.03)
    _add_note(
        fig,
        "Appendix benchmark: Phase 2B oracle comparison on AP held-out cases. "
        "The 2-rel bucket is where topology awareness has the clearest ranking gain, while 3-rel still benefits but remains planner-limited.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_appendix_track_b1(out_path: Path) -> None:
    rows = []
    with (METRICS_DIR / "track_b_summary.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    order = ["g2_fullaug_lowlr", "g4_ultimate", "g6_baseline", "gemini_unified"]
    rows = [next(r for r in rows if r["group"] == group) for group in order]
    labels = [DISPLAY.get(r["group"], r["display_name"]) for r in rows]
    colors = [COLORS.get(r["group"], "#777777") for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.3), constrained_layout=True)
    axes = axes.flatten()
    _style_axes(fig, list(axes))
    metrics = [
        ("gt_in_pct", "GT-in-Pool", (0, 50), "{:.1f}%"),
        ("top10_pct", "Top-10", (0, 12), "{:.1f}%"),
        ("top1_pct", "Top-1", (0, 4), "{:.1f}%"),
        ("mrr", "MRR@10", (0, 0.05), "{:.3f}"),
    ]
    for ax, (key, title, ylim, fmt) in zip(axes, metrics):
        vals = [float(r[key]) for r in rows]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=20)
        _annotate_bars(ax, bars, fmt)
        _highlight_best(ax, bars, vals)

    fig.suptitle("Appendix A3. Track B-1 external generalization", fontsize=16, y=1.02)
    _add_note(
        fig,
        "Benchmark: Track B-1 unified end-to-end (n=116), modality MC, prompt style lora5, retrieval strategy p0_union_p1. "
        "This is the external / mixed-source benchmark and is secondary to Track B-2 for strict AP claims.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_appendix_phase4_zoom(out_path: Path) -> None:
    track_a = {r["key"]: r for r in load_track_a_rows()}
    track_b2 = {r["key"]: r for r in load_track_b2_rows()}
    order = ["g3_fullaug_r32", "g4_ultimate", "g7_position_context", "g6_baseline"]
    labels = [DISPLAY[k] for k in order]
    colors = [COLORS[k] for k in order]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    _style_axes(fig, list(axes))

    left_metrics = [track_a[k]["hop1"] for k in order]
    right_metrics = [track_a[k]["pred_r"] for k in order]
    x = range(len(order))
    width = 0.38
    bars1 = axes[0].bar(
        [i - width / 2 for i in x], left_metrics, width=width,
        color=colors, edgecolor="black", linewidth=0.6, label="Hop-1"
    )
    bars2 = axes[0].bar(
        [i + width / 2 for i in x], right_metrics, width=width,
        color=colors, alpha=0.45, edgecolor="black", linewidth=0.6, label="Pred Recall"
    )
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Track A zoom-in")
    axes[0].legend(frameon=True, loc="lower right", borderpad=0.8)
    _annotate_bars(axes[0], bars1, "{:.1f}")
    _annotate_bars(axes[0], bars2, "{:.1f}")

    left_metrics = [track_b2[k]["top10"] for k in order]
    right_metrics = [track_b2[k]["mrr"] * 100.0 for k in order]
    bars1 = axes[1].bar(
        [i - width / 2 for i in x], left_metrics, width=width,
        color=colors, edgecolor="black", linewidth=0.6, label="Top-10"
    )
    bars2 = axes[1].bar(
        [i + width / 2 for i in x], right_metrics, width=width,
        color=colors, alpha=0.45, edgecolor="black", linewidth=0.6, label="MRR x100"
    )
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylim(0, 35)
    axes[1].set_title("Track B-2 zoom-in")
    axes[1].legend(frameon=True, loc="lower right", borderpad=0.8)
    _annotate_bars(axes[1], bars1, "{:.1f}")
    _annotate_bars(axes[1], bars2, "{:.1f}")

    fig.suptitle("Appendix A4. Phase 4 new-model zoom-in (G3 vs G4 vs G7 vs G6)", fontsize=16, y=1.03)
    _add_note(
        fig,
        "Left: Track A extraction comparison. Right: Track B-2 strict downstream comparison. "
        "This zoom-in is useful for explaining why G7 now leads Track A and early-rank metrics, while G3 still retains the strongest Top-10 hit rate.",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_appendix_planner_explainer(out_path: Path) -> None:
    _ensure_parent(out_path)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.2, 7.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]},
    )

    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def add_box(ax, x, y, w, h, title, lines, facecolor, *, fontsize=9.5, family=None):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#111111",
            facecolor=facecolor,
            alpha=0.97,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.02,
            y + h - 0.04,
            title,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#111111",
        )
        ax.text(
            x + 0.02,
            y + h - 0.085,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=fontsize,
            color="#222222",
            family=family,
            linespacing=1.25,
        )
        return patch

    def add_arrow(ax, start, end, color="#444444"):
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    def add_node(ax, x, y, label, facecolor, size=1500, text_color="#111111"):
        ax.scatter([x], [y], s=size, color=facecolor, edgecolors="#111111", linewidths=1.2, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.2, color=text_color, zorder=4)

    def add_callout(ax, x, y, text, facecolor="#FFFFFF", ha="left"):
        ax.text(
            x,
            y,
            text,
            ha=ha,
            va="center",
            fontsize=8.9,
            color="#1F2937",
            bbox={
                "boxstyle": "round,pad=0.22,rounding_size=0.02",
                "facecolor": facecolor,
                "edgecolor": "#CBD5E1",
                "alpha": 0.98,
            },
            zorder=5,
        )

    fig.suptitle(
        "Appendix A5. Planner explainer: multi-anchor + multi-chain + fingerprint-aware filter",
        fontsize=16,
        y=1.02,
    )

    # Left column: extracted constraints / planner inputs
    ax = axes[0]
    ax.text(0.02, 0.96, "1. Extracted constraints", fontsize=13, fontweight="bold", va="top")
    add_box(
        ax,
        0.04,
        0.69,
        0.90,
        0.18,
        "Base target attributes",
        [
            "storey_name = Level 1",
            "ifc_class = IfcWindow",
            "space_name = Living",
            "target_keyword = BALANS 10M",
        ],
        "#FFF3E0",
        fontsize=9.2,
    )
    add_box(
        ax,
        0.04,
        0.40,
        0.90,
        0.22,
        "Spatial relations (multi-anchor)",
        [
            "SR1 FILLS -> IfcWall",
            "SR2 NEXT_TO -> IfcDoor (direction = left)",
            "SR3 NEXT_TO -> IfcWindow (subtype = BATHROOM)",
        ],
        "#E3F2FD",
        fontsize=9.0,
    )
    add_box(
        ax,
        0.04,
        0.18,
        0.90,
        0.12,
        "Position fingerprint",
        [
            "position_context = 3rd of 17 openings",
            "on the same wall",
        ],
        "#E8F5E9",
        fontsize=9.2,
    )
    ax.text(
        0.05,
        0.04,
        "Planner request:\nattribute_only -> topology_only -> relation_fingerprint -> exact_slot",
        fontsize=9.2,
        color="#333333",
        va="bottom",
    )

    # Middle column: deterministic chain execution
    ax = axes[1]
    ax.text(0.02, 0.96, "2. Execution path", fontsize=13, fontweight="bold", va="top")
    add_box(
        ax,
        0.09,
        0.82,
        0.84,
        0.10,
        "Seed pool",
        [
            "P1 base pool = storey_name + ifc_class",
            "Target-rooted candidate set",
        ],
        "#FFF3E0",
    )

    # Graph traversal panel
    graph_panel = FancyBboxPatch(
        (0.06, 0.32),
        0.88,
        0.43,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#111111",
        facecolor="#F8FAFC",
        alpha=0.98,
    )
    ax.add_patch(graph_panel)
    ax.text(0.09, 0.72, "Deterministic graph traversal templates", fontsize=11, fontweight="bold", va="top")
    ax.text(0.09, 0.685, "Example: FILLS + NEXT_TO + NEXT_TO", fontsize=9.5, color="#374151", va="top")

    # nodes
    add_node(ax, 0.50, 0.54, "target", "#FDE68A", size=1500)
    add_node(ax, 0.50, 0.43, "wall", "#BFDBFE", size=1350)
    add_node(ax, 0.24, 0.54, "A", "#C7D2FE", size=1250)
    add_node(ax, 0.76, 0.54, "B", "#C7D2FE", size=1250)
    add_node(ax, 0.24, 0.40, "dir", "#E9D5FF", size=1050)
    add_node(ax, 0.76, 0.40, "sub", "#E9D5FF", size=1050)
    add_callout(ax, 0.50, 0.61, "target = IfcWindow", facecolor="#FFFBEB", ha="center")
    add_callout(ax, 0.50, 0.35, "filled wall fi0", facecolor="#EFF6FF", ha="center")
    add_callout(ax, 0.12, 0.60, "anchor A:\nIfcDoor", facecolor="#EEF2FF")
    add_callout(ax, 0.80, 0.60, "anchor B:\nIfcWindow", facecolor="#EEF2FF")
    add_callout(ax, 0.08, 0.42, "direction = left", facecolor="#F5F3FF")
    add_callout(ax, 0.81, 0.42, "subtype =\nBATHROOM", facecolor="#F5F3FF")

    # edges and labels
    add_arrow(ax, (0.50, 0.51), (0.50, 0.46), color="#1F2937")
    add_arrow(ax, (0.46, 0.54), (0.30, 0.54), color="#1F2937")
    add_arrow(ax, (0.54, 0.54), (0.70, 0.54), color="#1F2937")
    add_arrow(ax, (0.24, 0.50), (0.24, 0.44), color="#6D28D9")
    add_arrow(ax, (0.76, 0.50), (0.76, 0.44), color="#6D28D9")
    ax.text(0.50, 0.485, "FILLS", fontsize=9.2, ha="center", va="center", color="#111827")
    ax.text(0.37, 0.57, "chain 1", fontsize=8.8, ha="center", va="center", color="#111827")
    ax.text(0.63, 0.57, "chain 2", fontsize=8.8, ha="center", va="center", color="#111827")
    ax.text(0.50, 0.28, "same-wall pin via wall_guid\nAND distinct-neighbor filter", fontsize=9.1, ha="center", va="center", color="#334155")

    add_box(
        ax,
        0.09,
        0.12,
        0.84,
        0.09,
        "Multi-anchor AND filter",
        [
            "All extracted relations remain hard constraints",
            "Only candidates satisfying every anchored chain survive",
        ],
        "#E3F2FD",
        fontsize=8.9,
    )
    add_box(
        ax,
        0.09,
        0.04,
        0.84,
        0.09,
        "Output",
        [
            "Candidate pool after graph matching",
            "Then rank / report Top-10",
        ],
        "#EDE7F6",
        fontsize=8.9,
    )
    add_arrow(ax, (0.50, 0.82), (0.50, 0.76))
    add_arrow(ax, (0.50, 0.31), (0.50, 0.21))
    add_arrow(ax, (0.50, 0.12), (0.50, 0.095))

    # Right column: fingerprint-aware narrowing and fallbacks
    ax = axes[2]
    ax.text(0.02, 0.96, "3. Fingerprint-aware narrowing", fontsize=13, fontweight="bold", va="top")
    add_box(
        ax,
        0.06,
        0.77,
        0.88,
        0.11,
        "full_fingerprint",
        [
            "topology + direction + subtype + exact slot",
        ],
        "#D1FAE5",
        fontsize=9.0,
    )
    add_box(
        ax,
        0.06,
        0.61,
        0.88,
        0.10,
        "no_position",
        [
            "drop exact slot, keep relation fingerprint",
        ],
        "#E8F5E9",
        fontsize=9.0,
    )
    add_box(
        ax,
        0.06,
        0.46,
        0.88,
        0.10,
        "topology_only",
        [
            "drop direction / subtype",
        ],
        "#FFF3E0",
        fontsize=9.0,
    )
    add_box(
        ax,
        0.06,
        0.31,
        0.88,
        0.10,
        "no_storey",
        [
            "retry without storey restriction",
        ],
        "#FFE0B2",
        fontsize=9.0,
    )
    add_box(
        ax,
        0.06,
        0.15,
        0.88,
        0.10,
        "relaxed",
        [
            "drop weakest relation one-by-one",
        ],
        "#FFCDD2",
        fontsize=9.0,
    )
    add_arrow(ax, (0.50, 0.77), (0.50, 0.71), color="#166534")
    add_arrow(ax, (0.50, 0.61), (0.50, 0.56), color="#2E7D32")
    add_arrow(ax, (0.50, 0.46), (0.50, 0.41), color="#9A3412")
    add_arrow(ax, (0.50, 0.31), (0.50, 0.26), color="#9A3412")
    ax.text(
        0.06,
        0.03,
        "Interpretation:\nMore specific fingerprints shrink the pool earlier.\nFallback keeps recall when full fingerprint returns empty.",
        fontsize=9.5,
        color="#333333",
        va="bottom",
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─── Modality Ablation (Track-A, 6 conditions) ────────────────────────────────
_MODALITY_METRICS_ROOT = EXPERIMENT_ROOT / "modality_ablation_trackA" / "metrics"
_MODALITY_RERANK_ROOT  = EXPERIMENT_ROOT / "graph_rag_rerank" / "20260407_g8_phase5_v1"

_SLICE_ORDER = ["MC", "MC4D", "FP", "SITE", "FPSITE", "MA"]
_SLICE_LABEL = {
    "MC":     "MC\n(Site+FP+Chat)",
    "MC4D":   "MC4D\n(+4D scan)",
    "FP":     "FP\n(FP+Chat)",
    "SITE":   "SITE\n(Site+Chat)",
    "FPSITE": "FPSITE\n(Visual only)",
    "MA":     "MA\n(Chat only)",
}
_MODAL_MODELS  = ["g7_position_context", "g8_posctx_dim", "gemini_ap_v2"]
_MODAL_COLOR = {
    "g7_position_context": COLORS["g7_position_context"],
    "g8_posctx_dim": COLORS["g8_posctx_dim"],
    "gemini_ap_v2": COLORS["gemini_ap_v2"],
}
_MODAL_LS      = {"g7_position_context": "-", "g8_posctx_dim": "--", "gemini_ap_v2": ":"}
_MODAL_MK      = {"g7_position_context": "o", "g8_posctx_dim": "s", "gemini_ap_v2": "^"}
_MODAL_DISPLAY = {"g7_position_context": "G7", "g8_posctx_dim": "G8", "gemini_ap_v2": "Gemini v2"}


def _load_modality_metrics() -> dict:
    data: dict = {}
    for sl in _SLICE_ORDER:
        for m in _MODAL_MODELS:
            p = _MODALITY_METRICS_ROOT / sl / f"{m}__ap_metrics.json"
            if not p.exists():
                continue
            d = _load_json(p)
            data[(m, sl)] = {
                "hop1":    d["hop1_acc"] * 100,
                "pred_r":  d["predicate_recall"] * 100,
                "dir_acc": d["direction_acc"] * 100,
            }
    return data


def plot_modality_ablation(out_path: Path) -> None:
    """3-panel line chart: hop1 / predicate recall / direction accuracy × 6 conditions."""
    import numpy as np
    import matplotlib.patches as mpatches

    data = _load_modality_metrics()
    metric_specs = [
        ("hop1",    "One-hop Spatial Accuracy (%)"),
        ("pred_r",  "Predicate Recall (%)"),
        ("dir_acc", "Direction Accuracy (%)"),
    ]
    n_slices = len(_SLICE_ORDER)
    xs       = list(range(n_slices))
    xlabels  = [_SLICE_LABEL[s] for s in _SLICE_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    fig.suptitle(
        "Track-A Modality Ablation on AP Held-out (n=60 per condition)",
        fontsize=14, fontweight="bold",
    )

    for ax, (metric_key, ylabel) in zip(axes, metric_specs):
        for m in _MODAL_MODELS:
            ys = [data.get((m, sl), {}).get(metric_key, float("nan"))
                  for sl in _SLICE_ORDER]
            valid_xs = [i for i, y in enumerate(ys) if not np.isnan(y)]
            valid_ys = [y for y in ys if not np.isnan(y)]
            if not valid_ys:
                continue
            ax.plot(valid_xs, valid_ys,
                    marker=_MODAL_MK[m], linestyle=_MODAL_LS[m],
                    linewidth=2.2, markersize=7,
                    color=_MODAL_COLOR[m], label=_MODAL_DISPLAY[m])
            for xi, yi in zip(valid_xs, valid_ys):
                ax.annotate(f"{yi:.0f}",
                            (xi, yi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7.5,
                            color=_MODAL_COLOR[m])

        # shade FPSITE column
        fpsite_idx = _SLICE_ORDER.index("FPSITE")
        ax.axvspan(fpsite_idx - 0.45, fpsite_idx + 0.45,
                   color="gold", alpha=0.12, zorder=0)

        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=_MODAL_COLOR[m], label=_MODAL_DISPLAY[m])
               for m in _MODAL_MODELS]
    fpsite_patch = mpatches.Patch(color="gold", alpha=0.5, label="FPSITE (visual-only)")
    axes[-1].legend(handles=handles + [fpsite_patch], frameon=False,
                    fontsize=9, loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_multimodal_weak_proof(out_path: Path) -> None:
    """2-panel figure: scatter (G7 vs P1 per-case rank) + bar (retrieval top-1 by system)."""
    import numpy as np
    import matplotlib.patches as mpatches

    rerank_f = _MODALITY_RERANK_ROOT / "graph_rag_rerank_results.jsonl"
    if not rerank_f.exists():
        print(f"[SKIP] rerank file not found: {rerank_f}")
        return
    rows  = [json.loads(l) for l in rerank_f.open()]
    g7    = {r["case_id"]: r for r in rows if r["mode"] in ("full_topology", "g7_pipeline")}
    p1    = {r["case_id"]: r for r in rows if r["mode"] == "p1_only"}
    common = sorted(set(g7) & set(p1))
    n      = len(common)

    g7_base = [g7[c]["base_rank"] for c in common]
    p1_base = [p1[c]["base_rank"] for c in common]
    g7_rr   = [g7[c].get("reranked_rank", g7[c]["base_rank"]) for c in common]
    p1_rr   = [p1[c].get("reranked_rank", p1[c]["base_rank"]) for c in common]

    helps = sum(1 for g, p in zip(g7_base, p1_base) if g < p)
    hurts = sum(1 for g, p in zip(g7_base, p1_base) if g > p)
    equal = sum(1 for g, p in zip(g7_base, p1_base) if g == p)

    g7_top1_base = sum(1 for r in g7_base if r == 1)
    p1_top1_base = sum(1 for r in p1_base if r == 1)
    g7_top1_rr   = sum(1 for r in g7_rr   if r == 1)
    p1_top1_rr   = sum(1 for r in p1_rr   if r == 1)
    random_pct   = 1 / 128.9 * 100          # avg pool size from H2 eval

    families = [g7[c]["family"].split(":")[-1] for c in common]
    fam_colors = RELATION_FAMILY_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Multimodal Proof: Neuro-Symbolic (G7 P0∪P1) vs Text-Only (P1)\non AP Held-out (n=60)",
        fontsize=13, fontweight="bold",
    )

    # ── Panel A: scatter ──────────────────────────────────────────────────────
    ax = axes[0]
    for cid, g7b, p1b, fam in zip(common, g7_base, p1_base, families):
        c = fam_colors.get(fam, "#999999")
        ax.scatter(p1b, g7b, color=c, alpha=0.75, s=45, zorder=3,
                   edgecolors="white", linewidths=0.5)

    lim = max(max(g7_base), max(p1_base)) + 20
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.4)
    ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.06, color="#43A047")   # below diagonal = spatial helps
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.06, color="#E53935")  # above = hurts

    ax.text(lim * 0.08, lim * 0.78, f"Spatial helps\n({helps} cases)",
            color="#43A047", fontsize=9, fontweight="bold")
    ax.text(lim * 0.55, lim * 0.15, f"Spatial hurts\n({hurts} cases)",
            color="#E53935", fontsize=9, fontweight="bold")

    ax.set_xlabel("P1 Text-Only Base Rank", fontsize=11)
    ax.set_ylabel("G7 Spatial Base Rank", fontsize=11)
    ax.set_title(f"Per-case Rank Comparison (equal={equal})", fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fam_set     = sorted(set(families))
    fam_handles = [mpatches.Patch(color=fam_colors.get(f, "#999999"), label=f)
                   for f in fam_set]
    ax.legend(handles=fam_handles, fontsize=7.5, frameon=False, loc="upper left",
              title="Relation family", title_fontsize=8)

    # ── Panel B: bar ──────────────────────────────────────────────────────────
    ax2  = axes[1]
    lbls = ["Random\nbaseline", "P1 text-only\n(base)",
            "G7 spatial\n(P0∪P1 base)", "G7+GraphRAG\n(rerank)",
            "P1+GraphRAG\n(rerank)"]
    vals = [random_pct,
            p1_top1_base / n * 100,
            g7_top1_base / n * 100,
            g7_top1_rr   / n * 100,
            p1_top1_rr   / n * 100]
    bar_colors = ["#BDBDBD", "#90CAF9", "#1B5E20", "#66BB6A", "#1565C0"]
    xpos = list(range(len(lbls)))
    bars = ax2.bar(xpos, vals, color=bar_colors, width=0.55,
                   edgecolor="white", linewidth=0.8)

    raw_ns = ["~0.8%", "0/60", f"{g7_top1_base}/60", f"{g7_top1_rr}/60", f"{p1_top1_rr}/60"]
    for bar, v, rn in zip(bars, vals, raw_ns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax2.text(bar.get_x() + bar.get_width() / 2, -0.9,
                 rn, ha="center", va="top", fontsize=8, color="grey")

    # proof arrow P1→G7
    ax2.annotate("", xy=(2, vals[2] + 0.15), xytext=(1, vals[1] + 0.15),
                 arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=2))
    ax2.text(1.5, max(vals[1], vals[2]) + 1.8,
             "Weak positive\nproof ✓", color="#1B5E20", fontsize=8.5,
             ha="center", fontweight="bold")

    ax2.set_xticks(xpos)
    ax2.set_xticklabels(lbls, fontsize=9)
    ax2.set_ylim(0, 16)
    ax2.set_ylabel("Top-1 Retrieval Accuracy (%)", fontsize=11)
    ax2.set_title("Top-1 Retrieval Rate by System\n(Spatial P0∪P1 > pure text P1)", fontsize=10)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


_FINGERPRINT_CSV = (
    EXPERIMENT_ROOT
    / "group4_post-hoc_analysis"
    / "oracle_ceiling"
    / "20260404"
    / "fingerprint_loss_by_level.csv"
)

# CSV internal level codes → display order for the waterfall
_FP_LEVEL_ORDER = [
    "L0_p1_only",
    "L1_pred_obj",
    "L3_pred_obj_dir_sub",
    "L4_full_fingerprint",
]
_FP_LEVEL_LABEL = {
    "L0_p1_only":           ("L1", "(Storey + IFC Class)  —  Attribute Baseline"),
    "L1_pred_obj":          ("L2", "(Topology Type  —  Predicate + Object)"),
    "L3_pred_obj_dir_sub":  ("L3", "(Fingerprint  —  Direction + Subtype)"),
    "L4_full_fingerprint":  ("L4", "(Exact Position Slot  —  FILLS / NEXT_TO)"),
}


def _load_fingerprint_rows(slice_name: str = "all_cases") -> List[dict]:
    """Return rows from fingerprint_loss_by_level.csv for a given slice."""
    import csv as _csv
    rows = list(_csv.DictReader(_FINGERPRINT_CSV.read_text(encoding="utf-8").splitlines()))
    return [r for r in rows if r["slice"] == slice_name]


def plot_fingerprint_waterfall(out_path: Path) -> None:
    """Graph B: Fingerprint information-loss waterfall.

    Loads median pool sizes and Top-1 accuracy from the oracle ceiling
    experiment (fingerprint_loss_by_level.csv, all_cases slice, n=60).

    Layer mapping:
      L0  No Filter                 → 1,257  (oracle total_pool, constant)
      L1  Storey + IFC Class        → CSV L0_p1_only       (median pool)
      L2  Predicate + Object        → CSV L1_pred_obj       (median pool)
      L3  Direction + Subtype       → CSV L3_pred_obj_dir_sub (median pool)
      L4  Exact Position Slot       → CSV L4_full_fingerprint (median pool)

    Note: CSV L2_pred_obj_dir (direction-only, 55% coverage) is omitted for
    a clean 5-level waterfall consistent with the reference figure.
    """
    _ensure_parent(out_path)

    # ── Load oracle-ceiling data ──────────────────────────────────────────────
    rows_by_level = {r["level"]: r for r in _load_fingerprint_rows("all_cases")}

    def _med(level_key: str) -> float:
        return float(rows_by_level[level_key]["median_pool"])

    def _top1(level_key: str) -> float:
        return float(rows_by_level[level_key]["top1_rate"]) * 100

    def _cov(level_key: str) -> float:
        return float(rows_by_level[level_key]["coverage"]) * 100

    # ── Stage data ────────────────────────────────────────────────────────────
    # (short_code, desc, median_pool, top1_pct, coverage_pct)
    STAGES: List[tuple] = [
        ("L0", "(No Filter)", 1_257, 0.0, 100.0),
    ]
    for lvl in _FP_LEVEL_ORDER:
        lx, desc = _FP_LEVEL_LABEL[lvl]
        STAGES.append((lx, desc, _med(lvl), _top1(lvl), _cov(lvl)))

    N = len(STAGES)
    MAX_COUNT = float(STAGES[0][2])

    # ── Palette ───────────────────────────────────────────────────────────────
    BAR_BLUE = FINGERPRINT_WATERFALL_COLORS["bar_blue"]
    BAR_ORANGE = FINGERPRINT_WATERFALL_COLORS["bar_orange"]
    ARROW_CLR = FINGERPRINT_WATERFALL_COLORS["arrow"]
    LABEL_CLR = FINGERPRINT_WATERFALL_COLORS["label"]
    DESC_CLR = FINGERPRINT_WATERFALL_COLORS["description"]
    TOP1_CLR = FINGERPRINT_WATERFALL_COLORS["top1_badge"]
    PARTIAL_CLR = FINGERPRINT_WATERFALL_COLORS["partial_coverage"]

    # ── Layout constants (axes-fraction units) ────────────────────────────────
    ROW_H  = 0.60
    ROW_GAP= 0.36
    STEP   = ROW_H + ROW_GAP

    LX_X   = 0.000
    DESC_X = 0.060
    BAR_X0 = 0.000

    # Bars: sqrt-scaled for visual balance; minimum width so thin bars are visible
    MAX_BAR_W = 1.0
    MIN_BAR_W = 0.010
    bar_widths = [
        max((pool / MAX_COUNT) ** 0.5 * MAX_BAR_W, MIN_BAR_W)
        for _, _, pool, _, _ in STAGES
    ]

    # ── Figure: label panel (left) + bar panel (right) ───────────────────────
    fig = plt.figure(figsize=(14.0, 5.0))
    fig.patch.set_facecolor("#ffffff")

    # Left: 38% for row labels; Right: 57% for bars + annotations
    ax_lbl = fig.add_axes([0.01, 0.09, 0.37, 0.80])
    ax_bar = fig.add_axes([0.38, 0.09, 0.57, 0.80])

    for ax in (ax_lbl, ax_bar):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5 * STEP, (N - 0.5) * STEP)

    def yc(i: int) -> float:
        """y-centre for stage i (i=0 → L0 at top)."""
        return (N - 1 - i) * STEP + ROW_H / 2

    # ── Labels (left panel) ───────────────────────────────────────────────────
    for i, (lx, desc, pool, top1, cov) in enumerate(STAGES):
        yv    = yc(i)
        is_l4 = (i == N - 1)
        ax_lbl.text(
            LX_X, yv, lx,
            ha="left", va="center",
            fontsize=12.5, fontweight="bold",
            color=BAR_ORANGE if is_l4 else LABEL_CLR,
        )
        ax_lbl.text(
            DESC_X, yv, fill(desc, width=48),
            ha="left", va="center",
            fontsize=9.2, color=DESC_CLR,
            linespacing=1.28,
        )

    # ── Bars, count labels, Top-1 badges, arrows (right panel) ───────────────
    for i, (lx, desc, pool, top1, cov) in enumerate(STAGES):
        yv    = yc(i)
        y0    = yv - ROW_H / 2
        bw    = bar_widths[i]
        is_l4 = (i == N - 1)
        color = BAR_ORANGE if is_l4 else BAR_BLUE

        # Bar
        rect = FancyBboxPatch(
            (BAR_X0, y0), bw, ROW_H,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            linewidth=0, facecolor=color, alpha=0.87, zorder=2,
        )
        ax_bar.add_patch(rect)

        # Pool-size label
        if pool == MAX_COUNT:
            count_str = f"{int(pool):,} elements"
        elif pool <= 1.5:
            count_str = "median 1  (exact match)"
        else:
            count_str = f"median {int(pool)}"
        ax_bar.text(
            BAR_X0 + bw + 0.018, yv,
            count_str,
            ha="left", va="center",
            fontsize=10.5, fontweight="bold",
            color=BAR_ORANGE if is_l4 else LABEL_CLR,
        )

        # Top-1 accuracy badge (skip L0 which has no top-1 meaning)
        if i > 0 and top1 > 0:
            badge_x = BAR_X0 + bw + 0.018
            badge_y = yv - ROW_H * 0.38
            ax_bar.text(
                badge_x, badge_y,
                f"Top-1: {top1:.0f}%",
                ha="left", va="center",
                fontsize=8.2, color=TOP1_CLR,
                fontstyle="italic",
            )

        # Partial-coverage note (coverage < 100%)
        if cov < 99.9 and i > 0:
            cov_x = BAR_X0 + bw + 0.018
            cov_y = yv + ROW_H * 0.42
            ax_bar.text(
                cov_x, cov_y,
                f"({cov:.0f}% cases)",
                ha="left", va="center",
                fontsize=7.8, color=PARTIAL_CLR,
            )

        # Downward arrow to next row
        if i < N - 1:
            ax_bar.annotate(
                "",
                xy=(BAR_X0 + 0.028, yc(i + 1) + ROW_H / 2),
                xytext=(BAR_X0 + 0.028, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=ARROW_CLR, lw=1.8, mutation_scale=11,
                ),
                annotation_clip=False, zorder=3,
            )

    # ── Title ────────────────────────────────────────────────────────────────
    fig.text(
        0.01, 0.965,
        "Graph B.  Fingerprint information-loss waterfall",
        ha="left", va="top",
        fontsize=14, fontweight="bold", color="#111827",
    )

    # ── Footer note ──────────────────────────────────────────────────────────
    fig.text(
        0.01, 0.01,
        "Median candidate pool across n=60 oracle eval cases (AdvancedProject.ifc).  "
        "Bar widths: √(median / 1,257) — square-root scale for visual balance.  "
        "Top-1 = oracle accuracy at that layer.",
        ha="left", va="bottom",
        fontsize=7.8, color="#6B7280", style="italic",
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--appendix-out-dir", type=Path, default=DEFAULT_APPENDIX_OUT_DIR)
    args = parser.parse_args()

    _setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        "fig01_topology_overview.png": plot_topology_overview,
        "fig02_trackA_intermediate_comparison.png": plot_track_a_comparison,
        "fig02_v2_extraction_vs_downstream_tradeoff.png": plot_v2_extraction_downstream_tradeoff,
        "fig03_trackB2_strict_downstream.png": plot_track_b2_comparison,
        "fig04_oracle_strategy_sweep.png": plot_oracle_strategy_sweep,
        "fig04_v2_oracle_dashboard.png": plot_v2_oracle_dashboard,
        "fig05_p1_vs_full_topology_benefit.png": plot_p1_vs_full_topology,
        "fig06_oracle_vs_model_gap.png": plot_oracle_vs_model_gap,
        "fig07_oracle_progression_waterfall.png": plot_oracle_waterfall_progression,
        "fig08_modality_ablation.png": plot_modality_ablation,
        "fig09_multimodal_weak_proof.png": plot_multimodal_weak_proof,
        "figB_fingerprint_waterfall.png": plot_fingerprint_waterfall,
    }

    manifest = {"out_dir": str(args.out_dir), "generated": []}
    for filename, fn in plots.items():
        path = args.out_dir / filename
        _ensure_parent(path)
        fn(path)
        manifest["generated"].append(str(path))
        print(f"Wrote {path}")

    manifest_path = args.out_dir / "phase4_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")

    appendix_plots = {
        "figA1_topology_slice_benefit_by_universe.png": plot_appendix_universe_benefit,
        "figA2_topology_slice_benefit_by_multiplicity.png": plot_appendix_multiplicity_benefit,
        "figA3_trackB1_external_generalization.png": plot_appendix_track_b1,
        "figA4_phase4_new_models_zoomin.png": plot_appendix_phase4_zoom,
        "figA5_planner_multianchor_multichain_explainer.png": plot_appendix_planner_explainer,
    }
    args.appendix_out_dir.mkdir(parents=True, exist_ok=True)
    appendix_manifest = {"out_dir": str(args.appendix_out_dir), "generated": []}
    for filename, fn in appendix_plots.items():
        path = args.appendix_out_dir / filename
        _ensure_parent(path)
        fn(path)
        appendix_manifest["generated"].append(str(path))
        print(f"Wrote {path}")

    appendix_manifest_path = args.appendix_out_dir / "phase4_appendix_plot_manifest.json"
    appendix_manifest_path.write_text(json.dumps(appendix_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {appendix_manifest_path}")


if __name__ == "__main__":
    main()
