#!/usr/bin/env python3
"""
Unified Evaluation Analysis & U-Series Plot Generation

Reads v3 trace JSONL files (strategy_ablation_v3/) and precomputed constraint
files to produce the U-series thesis plots and summary text.

Usage:
  # Generate all U-series plots from latest v3 traces
  python evaluation/analysis/compare_results.py

  # Specify trace files explicitly
  python evaluation/analysis/compare_results.py \
    --traces logs/.../traces_lora5r32.jsonl --label "LoRA5-r32" \
    --traces logs/.../traces_gemini.jsonl   --label "Gemini"

  # Custom output directory
  python evaluation/analysis/compare_results.py --output plots/v4/
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
warnings.filterwarnings("ignore", message=".*constrained_layout.*")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Model display config: {key: (display_label, color, hatch)}
# Order: LoRA5-r32, LoRA5-r16, LoRA2, Gemini
MODEL_STYLE = OrderedDict([
    ("lora5r32", ("LoRA₅-r32", "#E65100", None)),     # deep orange
    ("lora5r16", ("LoRA₅-r16", "#F5A623", None)),     # amber/yellow
    ("lora2",    ("LoRA₂",    "#D32F2F", None)),       # red
    ("gemini",   ("Gemini",    "#1565C0", None)),      # blue
])

# Precomputed constraint file patterns (FP condition)
PRECOMPUTED_MAP = {
    "lora5r32": "eval_constraints_lora5r32_FP.jsonl",
    "lora5r16": "eval_constraints_lora5r16_FP.jsonl",
    "gemini":   "eval_constraints_final_FP.jsonl",
    "lora2":    "eval_constraints_lora2_FP.jsonl",
}

# MC variants
PRECOMPUTED_MAP_MC = {
    "lora5r32": "eval_constraints_lora5r32_MC.jsonl",
    "lora5r16": "eval_constraints_lora5r16_MC.jsonl",
    "gemini":   "eval_constraints_final_MC.jsonl",
    "lora2":    "eval_constraints_lora2_MC.jsonl",
}

P0_STRATEGIES = {"spatial_triplet", "continuous_span"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_traces(path: str) -> List[dict]:
    """Load evaluation traces from a JSONL file."""
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def load_precomputed(path: str) -> List[dict]:
    """Load precomputed constraints JSONL."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def classify_model(case_id: str) -> str:
    """Classify case into IFC model by case_id prefix."""
    if "_BH_" in case_id:
        return "BH"
    if "_DXA_" in case_id:
        return "DXA"
    return "AP"


def classify_tier(trace: dict, tier_lookup: Dict[str, str] = None) -> str:
    """Extract tier from trace or external lookup.

    Tier info is in the cases file (difficulty_tags.tier), not the trace.
    Pass tier_lookup = {case_id: "T1"|"T2"|"T3"} built from cases.
    Falls back to embedded scenario data if available.
    """
    case_id = trace.get("scenario_id", trace.get("scenario", {}).get("id", ""))
    if tier_lookup and case_id in tier_lookup:
        return tier_lookup[case_id]

    # Fallback: check embedded scenario
    scenario = trace.get("scenario", {})
    gt = scenario.get("ground_truth", {})
    tags = gt.get("difficulty_tags", scenario.get("difficulty_tags", {}))
    tier = tags.get("tier", "")
    if "1" in tier:
        return "T1"
    if "2" in tier:
        return "T2"
    if "3" in tier:
        return "T3"
    return "T?"


def build_tier_lookup(cases: List[dict]) -> Dict[str, str]:
    """Build {case_id: tier} mapping from cases file."""
    lookup = {}
    for c in cases:
        cid = c.get("case_id", "")
        tags = c.get("difficulty_tags", {})
        tier = tags.get("tier", "")
        if "1" in tier:
            lookup[cid] = "T1"
        elif "2" in tier:
            lookup[cid] = "T2"
        elif "3" in tier:
            lookup[cid] = "T3"
    return lookup


def classify_dataset(case_id: str) -> str:
    """v04 vs v05 dataset."""
    if case_id.startswith("V05_") or case_id.startswith("TEST_"):
        return "v05"
    return "v04"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(traces: List[dict]) -> dict:
    """Compute thesis metrics from a trace list.

    GT-in-Pool = GT GUID in rr[0] candidates (first retrieval strategy's pool).
    Top-1/5/10 = GT rank in interpreter_output.candidates.
    """
    n = len(traces)
    if n == 0:
        return {k: 0 for k in [
            "n", "gt_in_pool", "gt_in_pct", "top1", "top1_pct",
            "top5", "top5_pct", "top10", "top10_pct", "mrr",
            "avg_pool", "p0_fired", "p0_pct", "sr_extracted", "sr_pct",
        ]}

    gt_in_pool = top1 = top5 = top10 = p0_fired = sr_extracted = 0
    rrs: List[float] = []
    pools: List[int] = []

    for t in traces:
        gt_guid = t.get("scenario", {}).get("ground_truth", {}).get("target_guid", "")
        internals = t.get("internals", {})
        rr_list = internals.get("retrieval_results", [])
        io_cands = [c.get("guid", "") for c in
                    t.get("interpreter_output", {}).get("candidates", [])]
        constraints = internals.get("constraints", {})

        # GT-in-pool: rr[0] candidates
        if rr_list:
            rr0_guids = [c.get("guid", "") for c in
                         rr_list[0].get("candidates", [])]
            if gt_guid in rr0_guids:
                gt_in_pool += 1

        # Top-K / MRR from reranked output
        if gt_guid in io_cands:
            rank = io_cands.index(gt_guid) + 1
            rrs.append(1.0 / rank)
            if rank == 1:
                top1 += 1
            if rank <= 5:
                top5 += 1
            if rank <= 10:
                top10 += 1
        else:
            rrs.append(0.0)

        # P0 fired
        for rr in rr_list:
            strat = rr.get("query_plan_used", {}).get("strategy", "")
            if strat in P0_STRATEGIES and rr.get("pool_size", 0) > 0:
                p0_fired += 1
                break

        # Spatial relations extracted
        if constraints.get("spatial_relations"):
            sr_extracted += 1

        pools.append(t.get("final_pool_size", 0))

    avg_pool = sum(pools) / n if pools else 0
    mrr = sum(rrs) / n if rrs else 0

    return {
        "n": n,
        "gt_in_pool": gt_in_pool,
        "gt_in_pct": round(gt_in_pool / n * 100, 1),
        "top1": top1,
        "top1_pct": round(top1 / n * 100, 1),
        "top5": top5,
        "top5_pct": round(top5 / n * 100, 1),
        "top10": top10,
        "top10_pct": round(top10 / n * 100, 1),
        "mrr": round(mrr, 4),
        "avg_pool": round(avg_pool, 1),
        "p0_fired": p0_fired,
        "p0_pct": round(p0_fired / n * 100, 1),
        "sr_extracted": sr_extracted,
        "sr_pct": round(sr_extracted / n * 100, 1),
    }


def compute_field_accuracy(
    precomputed: List[dict], cases: List[dict]
) -> dict:
    """Compute per-field extraction accuracy vs ground truth.

    Returns dict with storey_acc, ifc_class_acc, sr_rate.
    """
    n = 0
    storey_match = ifc_match = sr_count = 0

    gt_map = {}
    for c in cases:
        cid = c.get("case_id", c.get("scenario_id", ""))
        gt = c.get("ground_truth", {})
        gt_map[cid] = gt

    for entry in precomputed:
        cid = entry.get("case_id", "")
        gt = gt_map.get(cid, {})
        if not gt:
            continue
        n += 1
        constraints = entry.get("constraints", {})

        # Storey match: extracted storey_name resolves to GT storey
        pred_storey = (constraints.get("storey_name") or "").lower().strip()
        gt_storey = (gt.get("target_storey") or "").lower().strip()
        if pred_storey and gt_storey:
            # Flexible: number match or substring
            import re
            pred_nums = set(re.findall(r'-?\d+', pred_storey))
            gt_nums = set(re.findall(r'-?\d+', gt_storey))
            if pred_nums and gt_nums and pred_nums & gt_nums:
                storey_match += 1
            elif pred_storey in gt_storey or gt_storey in pred_storey:
                storey_match += 1

        # IFC class match
        pred_class = (constraints.get("ifc_class") or "").lower()
        gt_class = (gt.get("target_ifc_class") or "").lower()
        if pred_class and gt_class:
            if pred_class == gt_class or gt_class.startswith(pred_class):
                ifc_match += 1

        # Spatial relations extracted
        if constraints.get("spatial_relations"):
            sr_count += 1

    if n == 0:
        return {"storey_acc": 0, "ifc_class_acc": 0, "sr_rate": 0, "n": 0}
    return {
        "storey_acc": round(storey_match / n * 100, 1),
        "ifc_class_acc": round(ifc_match / n * 100, 1),
        "sr_rate": round(sr_count / n * 100, 1),
        "n": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Auto-discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_v3_traces(
    ablation_dir: Path,
) -> Dict[str, Tuple[str, List[dict]]]:
    """Auto-discover v3 trace files mapped to model keys.

    Heuristic: the 4 most recent full-size (>100KB) p0_union_p1 traces
    are LoRA5-r32, LoRA5-r16, LoRA2, Gemini (in chronological order).
    This matches the run order from the evaluation script.
    """
    pattern = "traces_*_p0_union_p1.jsonl"
    candidates = sorted(ablation_dir.glob(pattern))

    # Filter out small files (test runs) and MC condition traces
    full = [f for f in candidates
            if f.stat().st_size > 100_000 and "_MC_" not in f.name]

    if len(full) < 4:
        print(f"⚠️  Found only {len(full)} full traces (need 4). Using all available.")

    # Take the LAST 4 (most recent run of each model)
    latest_4 = full[-4:] if len(full) >= 4 else full

    # Match to models by precomputed constraints content
    precomputed_dir = ablation_dir.parent
    model_map: Dict[str, Tuple[str, List[dict]]] = {}

    for trace_path in latest_4:
        traces = load_traces(str(trace_path))
        if not traces:
            continue

        # Identify model by checking SR extraction rate
        sr_count = sum(
            1 for t in traces
            if t.get("internals", {}).get("constraints", {}).get("spatial_relations")
        )
        sr_rate = sr_count / len(traces) if traces else 0

        # Also check storey accuracy pattern to distinguish LoRA5 vs Gemini
        # LoRA5 has ~82% storey acc, Gemini ~68%, LoRA2 ~67%
        # Use p0_fired as distinguisher: LoRA2 = 0%, LoRA5 = 100%, Gemini = ~93%
        p0_count = 0
        for t in traces:
            for rr in t.get("internals", {}).get("retrieval_results", []):
                strat = rr.get("query_plan_used", {}).get("strategy", "")
                if strat in P0_STRATEGIES and rr.get("pool_size", 0) > 0:
                    p0_count += 1
                    break

        if sr_rate < 0.01:
            model_key = "lora2"
        elif sr_rate < 0.97:
            model_key = "gemini"
        else:
            # Both LoRA5 variants have 100% SR — distinguish by file order
            # r32 is run first, r16 second (alphabetical adapter path order)
            if "lora5r32" not in model_map:
                model_key = "lora5r32"
            else:
                model_key = "lora5r16"

        model_map[model_key] = (str(trace_path), traces)
        print(f"  {model_key:<12} ← {trace_path.name}  "
              f"(n={len(traces)}, SR={sr_rate:.0%})")

    return model_map


# ─────────────────────────────────────────────────────────────────────────────
# U-Series Plot Functions
# ─────────────────────────────────────────────────────────────────────────────

def _fig_style():
    """Common plot style."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })


def _bar_label(ax, bars, fmt="{:.1f}%"):
    """Add value labels on top of bars."""
    # Use proportional offset based on y-axis range
    y_lo, y_hi = ax.get_ylim()
    offset = (y_hi - y_lo) * 0.01 if y_hi > y_lo else 0.5
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    fmt.format(h), ha="center", va="bottom", fontsize=8)


def _plot_multi_metric_grouped(
    model_metrics_map: Dict[str, dict],
    title: str,
    output_path: Path,
    subtitle: str = "",
    show_pool_line: bool = False,
):
    """Grouped bar chart: GT-in-Pool / Top-10 / Top-1 / MRR@10 per model.

    model_metrics_map: {model_key: metrics_dict} in MODEL_STYLE order.
    Bars: GT-in-Pool, Top-10, Top-1 (graduated opacity).
    MRR@10 shown as text annotation below each model label (different scale).
    If show_pool_line=True, overlay avg pool size as a secondary y-axis line.
    """
    _fig_style()

    keys = [k for k in MODEL_STYLE if k in model_metrics_map]
    if not keys:
        return

    metric_defs = [
        ("gt_in_pct",  "GT-in-Pool"),
        ("top10_pct",  "Top-10"),
        ("top1_pct",   "Top-1"),
    ]
    # Graduated alpha: darkest for GT-in-Pool, lightest for Top-1
    alphas = [1.0, 0.55, 0.30]

    n_models = len(keys)
    n_metrics = len(metric_defs)
    group_width = 0.7
    bar_width = group_width / n_metrics

    fig, ax = plt.subplots(figsize=(max(9, n_models * 2.4), 6.5))
    x = np.arange(n_models)

    for j, (mkey, mlabel) in enumerate(metric_defs):
        vals = [model_metrics_map[k].get(mkey, 0) for k in keys]
        colors = [MODEL_STYLE[k][1] for k in keys]
        offset = (j - n_metrics / 2 + 0.5) * bar_width

        for i, (val, color) in enumerate(zip(vals, colors)):
            bar = ax.bar(x[i] + offset, val, bar_width,
                         color=color, alpha=alphas[j], edgecolor="white",
                         label=mlabel if i == 0 else "")
            if val > 0:
                ax.text(x[i] + offset, val + 0.5, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=8)

    # X-axis: model name
    xlabels = [MODEL_STYLE[k][0] for k in keys]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=13)
    if subtitle:
        ax.text(0.5, 0.96, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="gray")

    max_val = max(model_metrics_map[k].get("gt_in_pct", 0) for k in keys)
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 100)

    # Secondary y-axis: MRR@10 line + optional Avg Pool Size line
    if show_pool_line:
        mrr_vals = [model_metrics_map[k].get("mrr", 0) for k in keys]
        pool_vals = [model_metrics_map[k].get("avg_pool", 0) for k in keys]

        ax2 = ax.twinx()
        # MRR@10 line (scaled ×100 to share axis with pool)
        mrr_scaled = [v * 1000 for v in mrr_vals]
        ax2.plot(x, pool_vals, "ko-", markersize=8, label="Avg Pool Size")
        ax2.plot(x, mrr_scaled, "s--", color="#7B1FA2", markersize=8,
                 linewidth=2, label="MRR@10 (×1000)")
        for i, (p, m, mr) in enumerate(zip(pool_vals, mrr_scaled, mrr_vals)):
            ax2.text(i, p + 2, f"{p:.0f}", ha="center", va="bottom",
                     fontsize=8)
            ax2.text(i, m + 2, f"{mr:.4f}", ha="center", va="bottom",
                     fontsize=8, color="#7B1FA2")
        ax2.set_ylabel("Pool Size / MRR@10 ×1000")
        y_max = max(max(pool_vals), max(mrr_scaled)) * 1.4
        ax2.set_ylim(0, y_max if y_max > 0 else 200)

    # De-duplicate legend (combine both axes)
    handles, labels_leg = ax.get_legend_handles_labels()
    if show_pool_line:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels_leg += l2
    seen = set()
    unique = []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique.append((h, l))
    ax.legend([h for h, l in unique], [l for h, l in unique],
              loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_u1_gt_in_pool(
    model_metrics: Dict[str, dict], output_dir: Path
):
    """U1: GT-in-Pool / Top-10 / Top-1 grouped by model + avg pool line."""
    _plot_multi_metric_grouped(
        model_metrics,
        title="Unified Evaluation: GT-in-Pool / Top-10 / Top-1  (n=116, p0∪p1)",
        output_path=output_dir / "U1_overall_metrics.png",
        show_pool_line=True,
    )
    print("  ✓ U1_overall_metrics.png")


def plot_u2_gt_by_ifc_model(
    model_traces: Dict[str, List[dict]], output_dir: Path
):
    """U2: GT-in-Pool / Top-10 / Top-1 by IFC model — one subplot per model."""
    _fig_style()

    ifc_models = ["AP", "BH", "DXA"]
    ifc_n = {}
    keys = [k for k in MODEL_STYLE if k in model_traces]

    # Compute per-IFC-model metrics
    data = {}  # {model_key: {ifc: metrics}}
    for key in keys:
        for ifc_m in ifc_models:
            subset = [t for t in model_traces[key]
                      if classify_model(t["scenario_id"]) == ifc_m]
            if ifc_m not in ifc_n:
                ifc_n[ifc_m] = len(subset)
            data.setdefault(key, {})[ifc_m] = compute_metrics(subset)

    # Subplot per IFC model
    fig, axes = plt.subplots(1, len(ifc_models), figsize=(14, 5), sharey=True)
    for ax_i, ifc_m in enumerate(ifc_models):
        ax = axes[ax_i]
        sub_metrics = {k: data[k][ifc_m] for k in keys}
        metric_defs = [("gt_in_pct", "GT-in-Pool"), ("top10_pct", "Top-10"), ("top1_pct", "Top-1")]
        alphas = [1.0, 0.55, 0.30]
        n_met = len(metric_defs)
        bar_w = 0.7 / n_met
        x = np.arange(len(keys))

        for j, (mkey, mlabel) in enumerate(metric_defs):
            offset = (j - n_met / 2 + 0.5) * bar_w
            for i, k in enumerate(keys):
                val = sub_metrics[k].get(mkey, 0)
                bar = ax.bar(x[i] + offset, val, bar_w,
                             color=MODEL_STYLE[k][1], alpha=alphas[j],
                             edgecolor="white",
                             label=mlabel if i == 0 and ax_i == 0 else "")
                if val > 0:
                    ax.text(x[i] + offset, val + 0.8, f"{val:.1f}%",
                            ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_STYLE[k][0] for k in keys], fontsize=8, rotation=20)
        ax.set_title(f"{ifc_m} (n={ifc_n.get(ifc_m, '?')})")
        ax.set_ylim(0, 105)
        if ax_i == 0:
            ax.set_ylabel("Accuracy (%)")

    handles, labels_leg = axes[0].get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels_leg) if l not in seen and not seen.add(l)]
    fig.legend([h for h, l in unique], [l for h, l in unique],
               loc="upper right", fontsize=9, bbox_to_anchor=(0.98, 0.98))

    fig.suptitle("GT-in-Pool / Top-10 / Top-1 by IFC Model (FP, p0∪p1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "U2_metrics_by_ifc_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ U2_metrics_by_ifc_model.png")


def plot_u3_gt_by_tier(
    model_traces: Dict[str, List[dict]],
    output_dir: Path,
    tier_lookup: Dict[str, str] = None,
):
    """U3: GT-in-Pool / Top-10 / Top-1 by tier — one subplot per tier."""
    _fig_style()

    tiers = ["T1", "T2", "T3"]
    tier_labels_map = {"T1": "T1 (easy)", "T2": "T2 (medium)", "T3": "T3 (hard)"}
    tier_n = {}
    keys = [k for k in MODEL_STYLE if k in model_traces]

    data = {}  # {model_key: {tier: metrics}}
    for key in keys:
        for tier in tiers:
            subset = [t for t in model_traces[key]
                      if classify_tier(t, tier_lookup) == tier]
            if tier not in tier_n:
                tier_n[tier] = len(subset)
            data.setdefault(key, {})[tier] = compute_metrics(subset)

    fig, axes = plt.subplots(1, len(tiers), figsize=(14, 5), sharey=True)
    for ax_i, tier in enumerate(tiers):
        ax = axes[ax_i]
        metric_defs = [("gt_in_pct", "GT-in-Pool"), ("top10_pct", "Top-10"), ("top1_pct", "Top-1")]
        alphas = [1.0, 0.55, 0.30]
        n_met = len(metric_defs)
        bar_w = 0.7 / n_met
        x = np.arange(len(keys))

        for j, (mkey, mlabel) in enumerate(metric_defs):
            offset = (j - n_met / 2 + 0.5) * bar_w
            for i, k in enumerate(keys):
                val = data[k][tier].get(mkey, 0)
                bar = ax.bar(x[i] + offset, val, bar_w,
                             color=MODEL_STYLE[k][1], alpha=alphas[j],
                             edgecolor="white",
                             label=mlabel if i == 0 and ax_i == 0 else "")
                if val > 0:
                    ax.text(x[i] + offset, val + 0.8, f"{val:.1f}%",
                            ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_STYLE[k][0] for k in keys], fontsize=8, rotation=20)
        ax.set_title(f"{tier_labels_map.get(tier, tier)} (n={tier_n.get(tier, '?')})")
        ax.set_ylim(0, 105)
        if ax_i == 0:
            ax.set_ylabel("Accuracy (%)")

    handles, labels_leg = axes[0].get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels_leg) if l not in seen and not seen.add(l)]
    fig.legend([h for h, l in unique], [l for h, l in unique],
               loc="upper right", fontsize=9, bbox_to_anchor=(0.98, 0.98))

    fig.suptitle("GT-in-Pool / Top-10 / Top-1 by Difficulty Tier (FP, p0∪p1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "U3_metrics_by_tier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ U3_metrics_by_tier.png")


def plot_u4_pool_ssr(
    model_traces: Dict[str, List[dict]],
    model_metrics: Dict[str, dict],
    output_dir: Path,
):
    """U4: Pool size distribution + SSR."""
    _fig_style()

    keys = [k for k in MODEL_STYLE if k in model_traces]

    # U4a: Pool size box plot
    fig, ax = plt.subplots(figsize=(8, 5))
    pool_data = []
    labels = []
    colors = []
    for key in keys:
        pools = [t.get("final_pool_size", 0) for t in model_traces[key]]
        pool_data.append(pools)
        labels.append(MODEL_STYLE[key][0])
        colors.append(MODEL_STYLE[key][1])

    bp = ax.boxplot(pool_data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Pool Size (candidates)")
    ax.set_title("Candidate Pool Size Distribution (FP, p0∪p1)")

    # Add mean annotation
    for i, (key, pools) in enumerate(zip(keys, pool_data)):
        mean_v = np.mean(pools)
        ax.text(i + 1, mean_v, f"μ={mean_v:.0f}", ha="center", va="bottom",
                fontsize=8, color="red")

    fig.tight_layout()
    fig.savefig(output_dir / "U4_pool_size_distribution.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ U4_pool_size_distribution.png")


def plot_u5_fp_vs_mc(
    precomputed_dir: Path,
    cases: List[dict],
    output_dir: Path,
    model_traces_fp: Optional[Dict[str, List[dict]]] = None,
):
    """U5: FP vs MC modality comparison — field accuracy + end-to-end retrieval.

    Row 1: field accuracy (Storey, IFC Class, SR Rate) — from precomputed
    Row 2: end-to-end retrieval (GT-in-Pool, Top-1, MRR@10) — from traces
    Each model uses its own color; FP = solid bar, MC = hatched lighter bar.
    """
    _fig_style()

    keys = list(MODEL_STYLE.keys())  # ordered: lora5r32, lora5r16, lora2, gemini
    fp_accs = {}
    mc_accs = {}

    for key in keys:
        fp_path = precomputed_dir / PRECOMPUTED_MAP.get(key, "")
        mc_path = precomputed_dir / PRECOMPUTED_MAP_MC.get(key, "")
        if fp_path.exists() and cases:
            fp_accs[key] = compute_field_accuracy(
                load_precomputed(str(fp_path)), cases)
        if mc_path.exists() and cases:
            mc_accs[key] = compute_field_accuracy(
                load_precomputed(str(mc_path)), cases)

    if not fp_accs:
        print("  ⚠ U5: no precomputed files found, skipping")
        return

    avail_keys = [k for k in keys if k in fp_accs]

    # FP end-to-end metrics from traces
    fp_e2e = {}
    if model_traces_fp:
        for key in avail_keys:
            if key in model_traces_fp:
                fp_e2e[key] = compute_metrics(model_traces_fp[key])

    # MC traces — try to discover
    mc_e2e = {}
    mc_trace_dir = precomputed_dir / "strategy_ablation_v3"
    mc_files = sorted(mc_trace_dir.glob("traces_*_MC_*.jsonl")
                      ) if mc_trace_dir.exists() else []
    mc_full = [f for f in mc_files if f.stat().st_size > 100_000]
    if len(mc_full) >= 4:
        for trace_path in mc_full[-4:]:
            traces = load_traces(str(trace_path))
            if not traces:
                continue
            sr_count = sum(
                1 for t in traces
                if t.get("internals", {}).get("constraints", {})
                .get("spatial_relations"))
            sr_rate = sr_count / len(traces) if traces else 0
            if sr_rate < 0.01:
                mk = "lora2"
            elif sr_rate < 0.97:
                mk = "gemini"
            elif "lora5r32" not in mc_e2e:
                mk = "lora5r32"
            else:
                mk = "lora5r16"
            mc_e2e[mk] = compute_metrics(traces)

    has_e2e = bool(fp_e2e)
    n_rows = 2 if has_e2e else 1
    fig, all_axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        all_axes = [all_axes]  # make indexable by row

    # --- Row 1: Field accuracy ---
    field_defs = [
        ("storey_acc", "Storey Accuracy"),
        ("ifc_class_acc", "IFC Class Accuracy"),
        ("sr_rate", "SR Extraction Rate"),
    ]
    for ax, (field, flabel) in zip(all_axes[0], field_defs):
        x = np.arange(len(avail_keys))
        width = 0.35
        fp_vals = [fp_accs.get(k, {}).get(field, 0) for k in avail_keys]
        mc_vals = [mc_accs.get(k, {}).get(field, 0) for k in avail_keys]
        colors = [MODEL_STYLE[k][1] for k in avail_keys]

        bars_fp = ax.bar(x - width / 2, fp_vals, width,
                         color=colors, alpha=0.9, edgecolor="white")
        bars_mc = ax.bar(x + width / 2, mc_vals, width,
                         color=colors, alpha=0.4, edgecolor="white",
                         hatch="//")
        _bar_label(ax, bars_fp)
        _bar_label(ax, bars_mc)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_STYLE[k][0] for k in avail_keys],
                           rotation=15)
        ax.set_ylabel("%")
        ax.set_title(flabel)
        ax.set_ylim(0, 110)

    # --- Row 2: End-to-end retrieval ---
    if has_e2e:
        e2e_defs = [
            ("gt_in_pct", "GT-in-Pool (%)", "{:.1f}%"),
            ("top1_pct",  "Top-1 (%)",      "{:.1f}%"),
            ("mrr",       "MRR@10",         "{:.4f}"),
        ]
        for ax, (mkey, mlabel, fmt) in zip(all_axes[1], e2e_defs):
            x = np.arange(len(avail_keys))
            width = 0.35
            fp_vals = [fp_e2e.get(k, {}).get(mkey, 0) for k in avail_keys]
            mc_vals = [mc_e2e.get(k, {}).get(mkey, 0) for k in avail_keys]
            colors = [MODEL_STYLE[k][1] for k in avail_keys]

            bars_fp = ax.bar(x - width / 2, fp_vals, width,
                             color=colors, alpha=0.9, edgecolor="white")
            _bar_label(ax, bars_fp, fmt=fmt)

            if mc_e2e:
                bars_mc = ax.bar(x + width / 2, mc_vals, width,
                                 color=colors, alpha=0.4, edgecolor="white",
                                 hatch="//")
                _bar_label(ax, bars_mc, fmt=fmt)

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_STYLE[k][0] for k in avail_keys],
                               rotation=15)
            ax.set_ylabel(mlabel)
            ax.set_title(f"End-to-End: {mlabel}")

            if "pct" in mkey:
                all_v = fp_vals + (mc_vals if mc_e2e else [])
                ax.set_ylim(0, max(max(all_v) * 1.3, 5) if all_v else 100)
            else:
                all_v = fp_vals + (mc_vals if mc_e2e else [])
                max_v = max(all_v) if all_v else 0.1
                y_top = max(max_v * 1.5, 0.02)
                ax.set_ylim(0, y_top)
                ax.yaxis.set_major_formatter(
                    mticker.FormatStrFormatter("%.3f"))

        if not mc_e2e:
            fig.text(0.5, 0.01,
                     "Note: MC end-to-end traces not available — "
                     "row 2 shows FP only.",
                     ha="center", fontsize=8, color="gray",
                     style="italic")

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.9, label="FP (floor plan)"),
        Patch(facecolor="gray", alpha=0.4, hatch="//",
              label="MC (multi-crop)"),
    ]
    all_axes[0][0].legend(handles=legend_elements, loc="lower left",
                          fontsize=8)

    fig.suptitle("FP vs MC Modality Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "U5_modality_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ U5_modality_comparison.png")


def plot_u6_field_accuracy(
    precomputed_dir: Path,
    cases: List[dict],
    output_dir: Path,
):
    """U6: Per-field extraction accuracy heatmap."""
    _fig_style()

    rows = []
    row_labels = []
    for cond_label, pmap in [("FP", PRECOMPUTED_MAP), ("MC", PRECOMPUTED_MAP_MC)]:
        for key in MODEL_STYLE:
            path = precomputed_dir / pmap.get(key, "")
            if not path.exists():
                continue
            acc = compute_field_accuracy(load_precomputed(str(path)), cases)
            rows.append([acc["storey_acc"], acc["ifc_class_acc"], acc["sr_rate"]])
            row_labels.append(f"{MODEL_STYLE[key][0]} {cond_label}")

    if not rows:
        print("  ⚠ U6: no precomputed files found, skipping")
        return

    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(7, max(4, len(rows) * 0.5 + 1)))

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Storey Acc", "IFC Class Acc", "SR Rate"])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(len(rows)):
        for j in range(3):
            val = data[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_title("Per-Field Extraction Accuracy")
    fig.colorbar(im, ax=ax, shrink=0.8, label="%")
    fig.tight_layout()
    fig.savefig(output_dir / "U6_field_accuracy_heatmap.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ U6_field_accuracy_heatmap.png")


def plot_u7_spatial_relations(
    model_traces: Dict[str, List[dict]], output_dir: Path
):
    """U7: Spatial relation extraction — SR rate + stacked predicate dist per model."""
    _fig_style()

    keys = [k for k in MODEL_STYLE if k in model_traces]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: SR extraction rate per model
    sr_rates = []
    for key in keys:
        traces = model_traces[key]
        sr = sum(1 for t in traces
                 if t.get("internals", {}).get("constraints", {}).get("spatial_relations"))
        sr_rates.append(sr / len(traces) * 100 if traces else 0)

    bars = ax1.bar([MODEL_STYLE[k][0] for k in keys], sr_rates,
                   color=[MODEL_STYLE[k][1] for k in keys], width=0.5)
    _bar_label(ax1, bars)
    ax1.set_ylabel("SR Extraction Rate (%)")
    ax1.set_title("Spatial Relation Extraction Rate")
    ax1.set_ylim(0, 115)

    # Right: Stacked column chart — predicate distribution across ALL models
    # Collect predicate counts per model
    all_preds: set = set()
    model_pred_counts: Dict[str, Dict[str, int]] = {}
    for key in keys:
        counts: Dict[str, int] = {}
        for t in model_traces[key]:
            srs = t.get("internals", {}).get("constraints", {}).get("spatial_relations", [])
            for sr in srs:
                pred = sr.get("predicate", "UNKNOWN")
                counts[pred] = counts.get(pred, 0) + 1
                all_preds.add(pred)
        model_pred_counts[key] = counts

    if all_preds:
        preds = sorted(all_preds)
        pred_colors = plt.cm.Set2(np.linspace(0, 1, max(len(preds), 1)))

        x = np.arange(len(keys))
        bar_width = 0.5
        bottom = np.zeros(len(keys))

        for j, pred in enumerate(preds):
            vals = [model_pred_counts[k].get(pred, 0) for k in keys]
            ax2.bar(x, vals, bar_width, bottom=bottom, label=pred,
                    color=pred_colors[j], edgecolor="white")
            # Add count labels on each segment if > 0
            for i, v in enumerate(vals):
                if v > 2:
                    ax2.text(x[i], bottom[i] + v / 2, str(v),
                             ha="center", va="center", fontsize=8,
                             fontweight="bold", color="white")
            bottom += np.array(vals)

        # Total label on top
        for i, total in enumerate(bottom):
            if total > 0:
                ax2.text(x[i], total + 0.5, str(int(total)),
                         ha="center", va="bottom", fontsize=9)

        ax2.set_xticks(x)
        ax2.set_xticklabels([MODEL_STYLE[k][0] for k in keys])
        ax2.set_ylabel("Count")
        ax2.set_title("Predicate Distribution (All Models)")
        ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "U7_spatial_relation_analysis.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ U7_spatial_relation_analysis.png")


def plot_u8_strategy_distribution(
    model_traces: Dict[str, List[dict]], output_dir: Path
):
    """U8: Query planner strategy distribution per model."""
    _fig_style()

    keys = [k for k in MODEL_STYLE if k in model_traces]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Count strategy usage per model
    strategy_set = set()
    model_strats: Dict[str, Dict[str, int]] = {}
    for key in keys:
        strat_counts: Dict[str, int] = {}
        for t in model_traces[key]:
            rr_list = t.get("internals", {}).get("retrieval_results", [])
            if rr_list:
                strat = rr_list[0].get("strategy") or "fallback"
                strat_counts[strat] = strat_counts.get(strat, 0) + 1
                strategy_set.add(strat)
        model_strats[key] = strat_counts

    strategies = sorted(strategy_set)
    x = np.arange(len(keys))
    width = 0.8 / max(len(strategies), 1)
    strat_colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

    for i, strat in enumerate(strategies):
        vals = [model_strats.get(k, {}).get(strat, 0) for k in keys]
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=strat, color=strat_colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE[k][0] for k in keys])
    ax.set_ylabel("Cases")
    ax.set_title("Query Strategy Distribution (FP, p0∪p1)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "U8_strategy_distribution.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ U8_strategy_distribution.png")


def plot_u9_strategy_ablation(
    ablation_dir: Path, output_dir: Path
):
    """U9: Strategy ablation (p0∩p1, p0∪p1, p1_only, p0_only).

    Reads ALL strategy traces from the ablation directory.
    """
    _fig_style()

    strategies = ["p0_only", "p1_only", "p0_intersect_p1", "p0_union_p1"]
    strat_labels = {
        "p0_only": "P0 only",
        "p1_only": "P1 only",
        "p0_intersect_p1": "P0∩P1",
        "p0_union_p1": "P0∪P1",
    }

    # Collect traces per (model, strategy)
    # Try to find traces for all strategies — fall back to v2 if v3 only has p0∪p1
    model_strat_gip: Dict[str, Dict[str, float]] = {}

    for strat in strategies:
        pattern = f"traces_*_{strat}.jsonl"
        files = sorted(ablation_dir.glob(pattern))
        full = [f for f in files if f.stat().st_size > 100_000]
        if not full:
            # Try v2 directory
            v2_dir = ablation_dir.parent / "strategy_ablation_v2"
            if v2_dir.exists():
                files = sorted(v2_dir.glob(pattern))
                full = [f for f in files if f.stat().st_size > 100_000]

        if not full:
            continue

        # Take last 4 (one per model)
        for trace_path in full[-4:]:
            traces = load_traces(str(trace_path))
            if not traces:
                continue
            m = compute_metrics(traces)

            # Identify model (same heuristic as discover)
            sr_count = sum(
                1 for t in traces
                if t.get("internals", {}).get("constraints", {}).get("spatial_relations")
            )
            sr_rate = sr_count / len(traces) if traces else 0
            if sr_rate < 0.01:
                key = "lora2"
            elif sr_rate < 0.97:
                key = "gemini"
            else:
                existing = model_strat_gip.get("lora5r32", {})
                if strat not in existing:
                    key = "lora5r32"
                else:
                    key = "lora5r16"

            model_strat_gip.setdefault(key, {})[strat] = m["gt_in_pct"]

    if not model_strat_gip:
        print("  ⚠ U9: no strategy ablation traces found, skipping")
        return

    # Plot grouped bars: strategies on x-axis, models as groups
    avail_strats = [s for s in strategies
                    if any(s in d for d in model_strat_gip.values())]
    avail_models = [k for k in MODEL_STYLE if k in model_strat_gip]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(avail_strats))
    width = 0.8 / max(len(avail_models), 1)

    for i, key in enumerate(avail_models):
        vals = [model_strat_gip[key].get(s, 0) for s in avail_strats]
        offset = (i - len(avail_models) / 2 + 0.5) * width
        label, color, _ = MODEL_STYLE[key]
        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      edgecolor="white", alpha=0.85)
        _bar_label(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([strat_labels.get(s, s) for s in avail_strats])
    ax.set_ylabel("GT-in-Pool (%)")
    ax.set_title("Query Strategy Ablation (FP)")
    ax.set_ylim(0, max(
        v for d in model_strat_gip.values() for v in d.values()
    ) * 1.2 if model_strat_gip else 100)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "U9_strategy_ablation.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ U9_strategy_ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary text
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(
    model_metrics: Dict[str, dict],
    model_traces: Dict[str, List[dict]],
    field_accs: Dict[str, dict],
    output_dir: Path,
    tier_lookup: Dict[str, str] = None,
):
    """Write unified_eval_summary.txt with all key metrics."""
    lines = []

    # Section 1: Overall
    lines.append(f"{'System':<24} {'GT-in-Pool':>14} {'Top-1':>8} "
                 f"{'Top-10':>8} {'MRR@10':>8} {'AvgPool':>8}")
    lines.append("=" * 78)
    for key in MODEL_STYLE:
        if key not in model_metrics:
            continue
        m = model_metrics[key]
        label = MODEL_STYLE[key][0] + " FP"
        lines.append(
            f"{label:<24} {m['gt_in_pool']}/{m['n']} ({m['gt_in_pct']}%)"
            f"{'':>2} {m['top1_pct']:>6.1f}% {m['top10_pct']:>6.1f}% "
            f"{m['mrr']:>8.4f} {m['avg_pool']:>8.0f}"
        )

    # Section 2: Per IFC model
    lines.append("")
    ifc_models = ["AP", "BH", "DXA"]
    header_parts = [f"{'System':<24}"]
    for ifc_m in ifc_models:
        header_parts.append(f"{ifc_m:>16}")
    lines.append("".join(header_parts))
    lines.append("=" * 72)

    for key in MODEL_STYLE:
        if key not in model_traces:
            continue
        label = MODEL_STYLE[key][0] + " FP"
        parts = [f"{label:<24}"]
        for ifc_m in ifc_models:
            subset = [t for t in model_traces[key]
                      if classify_model(t["scenario_id"]) == ifc_m]
            sm = compute_metrics(subset)
            parts.append(f"{sm['gt_in_pool']:>3}/{len(subset)} ({sm['gt_in_pct']:>5.1f}%)")
        lines.append("".join(parts))

    # Section 3: Per tier
    lines.append("")
    tiers = ["T1", "T2", "T3"]
    tier_labels = {"T1": "T1 easy", "T2": "T2 med", "T3": "T3 hard"}
    header_parts = [f"{'System':<24}"]
    for t in tiers:
        header_parts.append(f"{tier_labels[t]:>16}")
    lines.append("".join(header_parts))
    lines.append("=" * 72)

    for key in MODEL_STYLE:
        if key not in model_traces:
            continue
        label = MODEL_STYLE[key][0] + " FP"
        parts = [f"{label:<24}"]
        for tier in tiers:
            subset = [t for t in model_traces[key]
                      if classify_tier(t, tier_lookup) == tier]
            sm = compute_metrics(subset)
            parts.append(f"{sm['gt_in_pool']:>3}/{len(subset)} ({sm['gt_in_pct']:>5.1f}%)")
        lines.append("".join(parts))

    # Section 4: Field accuracy
    if field_accs:
        lines.append("")
        lines.append("Per-Field Extraction Accuracy")
        lines.append("=" * 60)
        lines.append(f"{'System':<24} {'storey_acc':>11} {'ifc_class_acc':>14} {'SR_rate':>10}")
        for label, acc in field_accs.items():
            lines.append(
                f"{label:<24} {acc['storey_acc']:>10.1f}% "
                f"{acc['ifc_class_acc']:>13.1f}% {acc['sr_rate']:>9.1f}%"
            )

    lines.append("")
    lines.append("NOTE: GT-in-Pool = GT GUID in rr[0] candidates "
                 "(first retrieval strategy's full pool).")
    lines.append("Strategy: p0∪p1 (union of spatial + storey+type).")

    text = "\n".join(lines)
    (output_dir / "unified_eval_summary.txt").write_text(text)
    print("  ✓ unified_eval_summary.txt")
    print()
    print(text)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate U-series unified evaluation plots"
    )
    parser.add_argument(
        "--ablation-dir",
        default=str(PROJECT_ROOT / "logs" / "evaluation_output" / "unified"
                    / "strategy_ablation_v3"),
        help="Directory containing strategy ablation trace files",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(PROJECT_ROOT / "logs" / "evaluation_output" / "unified"
                    / "plots"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--traces", action="append", default=None,
        help="Explicit trace file (repeat for each model)",
    )
    parser.add_argument(
        "--label", action="append", default=None,
        help="Label for each --traces file (must match count)",
    )
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "evaluation" / "cases"
                    / "cases_unified_test.jsonl"),
        help="Path to test cases JSONL (for field accuracy)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir = Path(args.ablation_dir)

    print(f"Output: {output_dir}")
    print(f"Ablation dir: {ablation_dir}")
    print()

    # Load test cases (for field accuracy)
    cases = []
    if Path(args.cases).exists():
        cases = load_traces(args.cases)
        print(f"Loaded {len(cases)} test cases from {args.cases}")

    # Discover or load traces
    if args.traces:
        # Explicit mode
        labels = args.label or [f"model_{i}" for i in range(len(args.traces))]
        model_traces = {}
        for path, label in zip(args.traces, labels):
            # Map label to model key
            key = label.lower().replace("-", "").replace("_", "").replace(" ", "")
            for mk in MODEL_STYLE:
                if mk in key:
                    key = mk
                    break
            model_traces[key] = load_traces(path)
            print(f"  {key:<12} ← {path} (n={len(model_traces[key])})")
    else:
        # Auto-discover
        print("Auto-discovering v3 traces...")
        discovered = discover_v3_traces(ablation_dir)
        model_traces = {k: traces for k, (_, traces) in discovered.items()}

    if not model_traces:
        print("ERROR: No traces found. Use --traces or check --ablation-dir.")
        sys.exit(1)

    # Compute metrics
    print("\nComputing metrics...")
    model_metrics = {}
    for key, traces in model_traces.items():
        model_metrics[key] = compute_metrics(traces)
        m = model_metrics[key]
        print(f"  {MODEL_STYLE.get(key, (key,))[0]:<12}: "
              f"GT-in-Pool={m['gt_in_pool']}/{m['n']} ({m['gt_in_pct']}%)  "
              f"Top-1={m['top1_pct']}%  Avg Pool={m['avg_pool']}")

    # Build tier lookup from cases
    tier_lookup = build_tier_lookup(cases) if cases else {}
    if tier_lookup:
        from collections import Counter
        tier_dist = Counter(tier_lookup.values())
        print(f"Tier distribution: {dict(sorted(tier_dist.items()))}")

    # Compute field accuracy from precomputed constraints
    precomputed_dir = ablation_dir.parent
    field_accs: Dict[str, dict] = {}
    for cond_label, pmap in [("FP", PRECOMPUTED_MAP), ("MC", PRECOMPUTED_MAP_MC)]:
        for key in MODEL_STYLE:
            path = precomputed_dir / pmap.get(key, "")
            if path.exists() and cases:
                acc = compute_field_accuracy(load_precomputed(str(path)), cases)
                field_accs[f"{MODEL_STYLE[key][0]} {cond_label}"] = acc

    # Generate all plots
    print(f"\nGenerating plots → {output_dir}/")
    plot_u1_gt_in_pool(model_metrics, output_dir)
    plot_u2_gt_by_ifc_model(model_traces, output_dir)
    plot_u3_gt_by_tier(model_traces, output_dir, tier_lookup=tier_lookup)
    plot_u4_pool_ssr(model_traces, model_metrics, output_dir)
    plot_u5_fp_vs_mc(precomputed_dir, cases, output_dir,
                     model_traces_fp=model_traces)
    plot_u6_field_accuracy(precomputed_dir, cases, output_dir)
    plot_u7_spatial_relations(model_traces, output_dir)
    plot_u8_strategy_distribution(model_traces, output_dir)
    plot_u9_strategy_ablation(ablation_dir, output_dir)

    # Write summary
    print()
    write_summary(model_metrics, model_traces, field_accs, output_dir,
                  tier_lookup=tier_lookup)

    print(f"\nDone. {len(list(output_dir.glob('U*.png')))} plots generated.")


if __name__ == "__main__":
    main()
