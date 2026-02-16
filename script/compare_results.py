#!/usr/bin/env python3
"""
Compare Evaluation Results

Reads eval JSON files from logs/evaluations/ and prints a side-by-side
comparison table. Works with V1 (main_mcp.py) JSON outputs and V2
(script/run.py) CSV summaries.

Supports N-way trace comparison with chart generation for thesis figures.

Usage:
  # Auto-discover mode (scan logs/evaluations/)
  python script/compare_results.py                    # Show all results
  python script/compare_results.py --latest           # Show only the most recent batch
  python script/compare_results.py --latest 4         # Show the 4 most recent results
  python script/compare_results.py --dir logs/custom  # Custom directory
  python script/compare_results.py --csv results.csv  # Export to CSV

  # Traces comparison mode (N-way with charts)
  python script/compare_results.py \\
    --traces logs/evaluations/traces_A_v2_prompt.jsonl --label "V2 Prompt" \\
    --traces logs/evaluations/traces_B_v2_lora.jsonl  --label "V2 LoRA" \\
    --plots --output logs/comparisons/v03_full

  # Mix: auto-discover + charts
  python script/compare_results.py --latest 4 --plots --output logs/plots/
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# loading
# ─────────────────────────────────────────────────────────────────────────────

def load_v1_results(eval_dir: Path) -> list:
    """Load all V1 eval JSON files and extract key metrics."""
    results = []
    for f in sorted(eval_dir.glob("eval_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: skipping {f.name}: {e}", file=sys.stderr)
            continue

        experiment = data.get("experiment", {})
        summary = data.get("summary", {})
        retrieval = summary.get("retrieval", {})
        rq2 = summary.get("rq2_schema", {})
        dataset = data.get("dataset", {})

        results.append({
            "file": f.name,
            "timestamp": data.get("timestamp", ""),
            "pipeline": "v1",
            "mode": experiment.get("mode", "unknown"),
            "description": experiment.get("description", ""),
            "dataset": dataset.get("name", "unknown"),
            "cases": summary.get("total", 0),
            "top1": retrieval.get("top1_accuracy", 0),
            "top3": retrieval.get("top3_accuracy", 0),
            "top5": retrieval.get("top5_accuracy", 0),
            "precision_1": retrieval.get("precision_at_1", 0),
            "recall": retrieval.get("recall", 0),
            "f1": retrieval.get("f1_score", 0),
            "rq2_pass_rate": rq2.get("pass_rate", None),
            "rq2_fill_rate": rq2.get("avg_fill_rate", None),
        })

    return results


def load_v2_results(eval_dir: Path) -> list:
    """Load V2 summary CSV files and extract key metrics."""
    results = []
    for f in sorted(eval_dir.glob("summary_*.csv")):
        try:
            metrics = _parse_v2_csv(f)
        except OSError as e:
            print(f"  Warning: skipping {f.name}: {e}", file=sys.stderr)
            continue

        if not metrics:
            continue

        # Extract profile name from filename: summary_TIMESTAMP_PROFILE.csv
        parts = f.stem.split("_", 2)  # summary, timestamp, profile...
        profile = parts[2] if len(parts) > 2 else "unknown"
        timestamp = parts[1] if len(parts) > 1 else ""

        results.append({
            "file": f.name,
            "timestamp": timestamp,
            "pipeline": "v2",
            "mode": profile,
            "description": f"V2 profile: {profile}",
            "dataset": "cases_v2",
            "cases": metrics.get("total_scenarios", 0),
            "top1": metrics.get("top1_accuracy", 0),
            "top3": metrics.get("topk_accuracy", 0),
            "top5": None,
            "precision_1": None,
            "recall": None,
            "f1": None,
            "rq2_pass_rate": metrics.get("rq2_pass_rate", None),
            "rq2_fill_rate": metrics.get("rq2_fill_rate", None),
            # V2-specific
            "parse_rate": metrics.get("constraints_parse_rate", None),
            "rerank_gain": metrics.get("avg_rerank_gain", None),
            "search_space_reduction": metrics.get("avg_search_space_reduction", None),
        })

    return results


def _parse_v2_csv(path: Path) -> dict:
    """Parse a V2 summary CSV into a flat dict of metrics."""
    metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            key, val = row[0].strip(), row[1].strip()
            # Skip section headers
            if key.startswith("===") or key == "Metric":
                continue
            # Try to parse as number
            try:
                metrics[_normalize_key(key)] = float(val)
            except ValueError:
                if val != "N/A":
                    metrics[_normalize_key(key)] = val
    return metrics


def _normalize_key(key: str) -> str:
    """Normalize CSV metric name to snake_case."""
    return (
        key.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("avg_", "avg_")
    )


# ─────────────────────────────────────────────────────────────────────────────
# traces loading (N-way comparison)
# ─────────────────────────────────────────────────────────────────────────────

def load_traces(path: str) -> list:
    """Load evaluation traces from a JSONL file."""
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def _compute_ssr(traces: list) -> dict:
    """Compute search-space reduction split by validity.

    A reduction is **valid** only when the ground-truth element survives
    (guid_match=True).  Otherwise it is an **over-reduction** — the pipeline
    pruned away the correct answer.

    Returns dict with:
        valid_reductions: list[float]   — per-case (ini-fin)/ini where guid_match
        invalid_reductions: list[float] — per-case (ini-fin)/ini where NOT guid_match
        valid_ssr: float|None           — mean of valid_reductions
        invalid_ssr: float|None         — mean of invalid_reductions (over-reduction)
        over_reduction_rate: float|None — fraction of reduced cases that lost the answer
    """
    valid = []
    invalid = []
    for t in traces:
        ini = t.get("initial_pool_size", 0)
        fin = t.get("final_pool_size", 0)
        if ini <= 0:
            continue
        ratio = (ini - fin) / ini
        if t.get("guid_match", False):
            valid.append(ratio)
        else:
            invalid.append(ratio)

    n_reduced = len(valid) + len(invalid)
    return {
        "valid_reductions": valid,
        "invalid_reductions": invalid,
        "valid_ssr": (sum(valid) / len(valid)) if valid else None,
        "invalid_ssr": (sum(invalid) / len(invalid)) if invalid else None,
        "over_reduction_rate": (
            len(invalid) / n_reduced if n_reduced > 0 else None
        ),
    }


def traces_to_result(traces: list, label: str, file_path: str) -> dict:
    """Convert a list of traces to the same result dict format used by the table."""
    total = len(traces)
    if total == 0:
        return {"pipeline": "?", "mode": label, "cases": 0, "top1": 0}

    hits = sum(1 for t in traces if t.get("guid_match", False))
    name_hits = sum(1 for t in traces if t.get("name_match", False))
    storey_hits = sum(1 for t in traces if t.get("storey_match", False))
    successful = [t for t in traces if t.get("success", False)]

    # Parse rate (V2 traces only)
    parse_success = sum(1 for t in traces if t.get("constraints_parse_success", False))
    has_parse = any("constraints_parse_success" in t for t in traces)

    # Search space reduction (valid = GT retained, invalid = GT lost)
    ssr = _compute_ssr(traces)

    pipeline_type = traces[0].get("pipeline_type", "unknown") if traces else "unknown"

    return {
        "file": Path(file_path).name,
        "timestamp": traces[0].get("timestamp", "") if traces else "",
        "pipeline": pipeline_type,
        "mode": label,
        "description": label,
        "dataset": "synth_v0.3",
        "cases": total,
        "top1": hits / total if total > 0 else 0,
        "top3": None,
        "top5": None,
        "precision_1": None,
        "recall": None,
        "f1": None,
        "rq2_pass_rate": None,
        "rq2_fill_rate": None,
        "parse_rate": parse_success / total if has_parse and total > 0 else None,
        "search_space_reduction": ssr["valid_ssr"],
        "over_reduction_rate": ssr["over_reduction_rate"],
        "name_match_rate": name_hits / total if total > 0 else 0,
        "storey_match_rate": storey_hits / total if total > 0 else 0,
        "avg_latency_ms": (
            sum(t.get("total_latency_ms", 0) for t in successful) / len(successful)
            if successful else 0
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# chart generation (N-way comparison)
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_charts(
    experiments: dict,  # label -> list of traces
    output_dir: str,
    title: str = "Evaluation Comparison",
) -> None:
    """Generate N-way comparison charts from trace data.

    Reuses chart functions from src/eval/visualizations.py where possible,
    and adds summary-level charts specific to N-way comparison.

    Args:
        experiments: Dict mapping label -> list of traces
        output_dir: Directory to save PNG charts
        title: Descriptive title for charts
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Add project paths for importing visualizations
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))

    from src.eval.visualizations import (
        plot_condition_wise_comparison,
        plot_efficiency_comparison,
    )

    print(f"\n{'=' * 60}")
    print(f"Generating Comparison Charts ({len(experiments)} experiments, 6 charts)")
    print(f"{'=' * 60}\n")

    for label, traces in experiments.items():
        print(f"  {label}: {len(traces)} traces")

    # ── Chart 1: Overall metrics bar chart ──
    _plot_overall_metrics(experiments, f"{output_dir}/1_overall_metrics.png", title)

    # ── Chart 2: Condition-wise comparison (A1-C3) ──
    plot_condition_wise_comparison(
        experiments,
        f"{output_dir}/2_condition_comparison.png",
        title=f"{title} — Per Condition (A1-C3)",
    )

    # ── Chart 3: Search space reduction (valid vs over-reduced) ──
    _plot_search_space_box(experiments, f"{output_dir}/3_search_space_reduction.png")

    # ── Chart 4: Efficiency (latency, API calls, cost) ──
    plot_efficiency_comparison(
        experiments,
        f"{output_dir}/4_efficiency_comparison.png",
    )

    # ── Chart 5: Per-condition heatmap (accuracy matrix) ──
    _plot_accuracy_heatmap(experiments, f"{output_dir}/5_accuracy_heatmap.png", title)

    # ── Chart 6: Per-case detail (all 84 cases: hit + SSR) ──
    _plot_per_case_detail(experiments, f"{output_dir}/6_accuracy_heatmap_details.png", title)

    print(f"\n{'=' * 60}")
    print(f"Charts saved to: {output_dir}/")
    print(f"{'=' * 60}\n")


def _extract_condition(trace: dict) -> str:
    """Extract condition (A1-C3) from trace.

    Tries in order:
    1. trace.bench.condition (set by run.py since v0.3)
    2. scenario.bench.condition (legacy)
    3. run_id suffix
    """
    # Top-level bench field (set by run.py)
    cond = (trace.get("bench") or {}).get("condition")
    if cond:
        return cond

    # Legacy: nested inside scenario
    cond = (trace.get("scenario") or {}).get("bench", {}).get("condition")
    if cond:
        return cond

    run_id = trace.get("run_id", "")
    if run_id and "_" in run_id:
        parts = run_id.split("_")
        if len(parts) >= 4:
            c = parts[-1]
            if c in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]:
                return c
    return "Unknown"


def _plot_overall_metrics(
    experiments: dict, output_path: str, title: str = ""
) -> None:
    """Bar chart comparing Top-1, Name Match, Storey Match, Valid SSR, and Over-Reduction.

    SSR is only computed on cases where guid_match=True (GT survived the reduction).
    Over-Reduction shows the fraction of cases where the pipeline pruned the GT away.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(experiments.keys())
    n = len(labels)

    # Compute metrics per experiment
    metrics = {}
    for label, traces in experiments.items():
        total = len(traces)
        if total == 0:
            metrics[label] = {
                "Top-1": 0, "Name": 0, "Storey": 0, "Valid SSR": 0, "OverRed": 0,
            }
            continue
        guid_hits = sum(1 for t in traces if t.get("guid_match", False))
        name_hits = sum(1 for t in traces if t.get("name_match", False))
        storey_hits = sum(1 for t in traces if t.get("storey_match", False))
        ssr = _compute_ssr(traces)
        metrics[label] = {
            "Top-1": guid_hits / total * 100,
            "Name": name_hits / total * 100,
            "Storey": storey_hits / total * 100,
            "Valid SSR": (ssr["valid_ssr"] * 100) if ssr["valid_ssr"] is not None else 0,
            "OverRed": (ssr["over_reduction_rate"] * 100)
                       if ssr["over_reduction_rate"] is not None else 0,
        }

    metric_names = ["Top-1", "Name", "Storey", "Valid SSR", "OverRed"]
    x = np.arange(len(metric_names))
    width = 0.8 / n
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, label in enumerate(labels):
        vals = [metrics[label][m] for m in metric_names]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width, label=label,
            color=colors[i % len(colors)], alpha=0.85,
            edgecolor="black", linewidth=0.8,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold",
                )

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(f"{title} — Overall Metrics" if title else "Overall Metrics",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "Top-1\nAccuracy", "Name\nMatch", "Storey\nMatch",
        "Valid SSR\n(GT retained)", "Over-Reduction\n(GT lost)",
    ], fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, loc="upper right", frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _plot_search_space_box(experiments: dict, output_path: str) -> None:
    """Box plot of search space reduction: valid (GT retained) vs over-reduced (GT lost)."""
    import matplotlib.pyplot as plt
    import numpy as np

    exp_labels = list(experiments.keys())
    valid_data = []
    invalid_data = []
    tick_labels = []

    for label, traces in experiments.items():
        ssr = _compute_ssr(traces)
        v = [r * 100 for r in ssr["valid_reductions"]]
        iv = [r * 100 for r in ssr["invalid_reductions"]]
        if v or iv:
            valid_data.append(v if v else [0])
            invalid_data.append(iv if iv else [0])
            n_v, n_iv = len(ssr["valid_reductions"]), len(ssr["invalid_reductions"])
            tick_labels.append(f"{label}\n(valid={n_v}, lost={n_iv})")

    if not tick_labels:
        print("  No search space data for box plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(tick_labels)
    positions_v = np.arange(n) * 2
    positions_iv = positions_v + 0.6

    bp_v = ax.boxplot(valid_data, positions=positions_v, widths=0.5,
                      patch_artist=True, manage_ticks=False)
    bp_iv = ax.boxplot(invalid_data, positions=positions_iv, widths=0.5,
                       patch_artist=True, manage_ticks=False)

    for patch in bp_v["boxes"]:
        patch.set_facecolor("#2ecc71")
        patch.set_alpha(0.7)
    for patch in bp_iv["boxes"]:
        patch.set_facecolor("#e74c3c")
        patch.set_alpha(0.7)
    for median in bp_v["medians"] + bp_iv["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_xticks(positions_v + 0.3)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("Search Space Reduction (%)", fontsize=12)
    ax.set_title("Search Space Reduction: Valid (GT retained) vs Over-Reduced (GT lost)",
                 fontsize=13, fontweight="bold")
    ax.legend([bp_v["boxes"][0], bp_iv["boxes"][0]],
              ["Valid (GT in final)", "Over-Reduced (GT lost)"],
              fontsize=10, loc="lower left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _plot_accuracy_heatmap(
    experiments: dict, output_path: str, title: str = ""
) -> None:
    """Heatmap: experiments (rows) x conditions (cols) showing Top-1 accuracy."""
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    labels = list(experiments.keys())

    matrix = []
    for label in labels:
        traces = experiments[label]
        by_cond = {}
        for t in traces:
            c = _extract_condition(t)
            if c not in by_cond:
                by_cond[c] = {"hits": 0, "total": 0}
            by_cond[c]["total"] += 1
            if t.get("guid_match", False):
                by_cond[c]["hits"] += 1

        row = []
        for c in conditions:
            s = by_cond.get(c, {"hits": 0, "total": 0})
            row.append(s["hits"] / s["total"] * 100 if s["total"] > 0 else 0)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, max(3, len(labels) * 1.2)))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    # Add condition group separators
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.8, linewidth=2)
    ax.axvline(x=5.5, color="gray", linestyle="--", alpha=0.8, linewidth=2)

    # Group labels
    ax.text(1, -0.8, "Text Only", ha="center", fontsize=9, style="italic", color="gray")
    ax.text(4, -0.8, "Images+Text", ha="center", fontsize=9, style="italic", color="gray")
    ax.text(7, -0.8, "Full Multimodal", ha="center", fontsize=9, style="italic", color="gray")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Top-1 Accuracy (%)", fontsize=11)

    ax.set_title(
        f"{title} — Accuracy Heatmap" if title else "Accuracy Heatmap (Conditions x Experiments)",
        fontsize=14, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _plot_per_case_detail(
    experiments: dict, output_path: str, title: str = ""
) -> None:
    """Detailed per-case heatmap: rows = all 84 cases (grouped by condition),
    columns = Hit + SSR for each experiment.

    Each experiment contributes 2 columns: Hit (1/0) and SSR (%).
    Cases are sorted by condition (A1→C3), then by case_id within condition.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    exp_labels = list(experiments.keys())
    n_exp = len(exp_labels)

    # ── Build per-case lookup for each experiment ──
    # {case_id: {"hit": bool, "ssr": float}}
    exp_data = {}
    all_case_ids = set()
    for label, traces in experiments.items():
        lookup = {}
        for t in traces:
            cid = t.get("scenario_id", "")
            init = t.get("initial_pool_size") or 0
            final = t.get("final_pool_size") or 0
            ssr = (1.0 - final / init) * 100 if init > 0 else 0.0
            lookup[cid] = {
                "hit": t.get("guid_match", False),
                "ssr": ssr,
            }
            all_case_ids.add(cid)
        exp_data[label] = lookup

    # ── Sort cases by condition then case_id ──
    # Need condition info — extract from any experiment's traces
    case_cond = {}
    for label, traces in experiments.items():
        for t in traces:
            cid = t.get("scenario_id", "")
            if cid not in case_cond:
                case_cond[cid] = _extract_condition(t)

    cond_order = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "Unknown"]
    sorted_cases = sorted(
        all_case_ids,
        key=lambda cid: (
            cond_order.index(case_cond.get(cid, "Unknown"))
            if case_cond.get(cid, "Unknown") in cond_order
            else 99,
            cid,
        ),
    )
    n_cases = len(sorted_cases)

    # ── Build matrices: hit_matrix and ssr_matrix ──
    # Columns: [exp0_hit, exp0_ssr, exp1_hit, exp1_ssr, ...]
    hit_matrix = np.full((n_cases, n_exp), np.nan)
    ssr_matrix = np.full((n_cases, n_exp), np.nan)

    for j, label in enumerate(exp_labels):
        lookup = exp_data[label]
        for i, cid in enumerate(sorted_cases):
            d = lookup.get(cid)
            if d:
                hit_matrix[i, j] = 1.0 if d["hit"] else 0.0
                ssr_matrix[i, j] = d["ssr"]

    # ── Figure layout: two side-by-side heatmaps ──
    fig, (ax_hit, ax_ssr) = plt.subplots(
        1, 2,
        figsize=(4 + n_exp * 1.8, max(12, n_cases * 0.22)),
        gridspec_kw={"width_ratios": [n_exp, n_exp], "wspace": 0.15},
    )

    # -- Left panel: Hit/Miss --
    hit_cmap = mcolors.ListedColormap(["#f4cccc", "#b6d7a8"])  # red-ish miss, green-ish hit
    hit_cmap.set_bad(color="#eeeeee")
    ax_hit.imshow(hit_matrix, cmap=hit_cmap, aspect="auto", vmin=0, vmax=1,
                  interpolation="nearest")

    for i in range(n_cases):
        for j in range(n_exp):
            v = hit_matrix[i, j]
            if not np.isnan(v):
                ax_hit.text(j, i, "HIT" if v == 1 else "-",
                            ha="center", va="center", fontsize=6,
                            fontweight="bold" if v == 1 else "normal",
                            color="#274e13" if v == 1 else "#990000")

    ax_hit.set_xticks(range(n_exp))
    ax_hit.set_xticklabels(exp_labels, fontsize=8, rotation=30, ha="right")
    ax_hit.set_title("Top-1 Hit", fontsize=11, fontweight="bold")

    # -- Right panel: SSR --
    im_ssr = ax_ssr.imshow(ssr_matrix, cmap="YlOrRd", aspect="auto",
                           vmin=80, vmax=100, interpolation="nearest")

    for i in range(n_cases):
        for j in range(n_exp):
            v = ssr_matrix[i, j]
            if not np.isnan(v):
                ax_ssr.text(j, i, f"{v:.0f}",
                            ha="center", va="center", fontsize=5.5,
                            color="white" if v > 95 else "black")

    ax_ssr.set_xticks(range(n_exp))
    ax_ssr.set_xticklabels(exp_labels, fontsize=8, rotation=30, ha="right")
    ax_ssr.set_title("SSR %", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im_ssr, ax=ax_ssr, shrink=0.4, pad=0.02)
    cbar.set_label("Search-Space Reduction %", fontsize=8)

    # ── Shared y-axis: case labels with condition separators ──
    # Short labels: condition + case number
    y_labels = []
    for cid in sorted_cases:
        # e.g. "SYNTH_V3_001_SK_001" → "001"
        parts = cid.split("_")
        short = parts[2] if len(parts) >= 3 else cid[-3:]
        y_labels.append(f"{case_cond.get(cid, '?')} {short}")

    ax_hit.set_yticks(range(n_cases))
    ax_hit.set_yticklabels(y_labels, fontsize=5.5, family="monospace")
    ax_ssr.set_yticks(range(n_cases))
    ax_ssr.set_yticklabels([], fontsize=5.5)  # hide right panel y-labels (shared)

    # ── Condition group separators ──
    prev_cond = None
    for i, cid in enumerate(sorted_cases):
        c = case_cond.get(cid, "Unknown")
        if prev_cond is not None and c != prev_cond:
            for ax in (ax_hit, ax_ssr):
                ax.axhline(y=i - 0.5, color="black", linewidth=0.8, alpha=0.6)
        prev_cond = c

    fig.suptitle(
        f"{title} — Per-Case Detail (Hit + SSR)" if title else "Per-Case Detail",
        fontsize=13, fontweight="bold", y=1.0,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# display
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: list) -> None:
    """Print a formatted comparison table to stdout."""
    if not results:
        print("No evaluation results found.")
        return

    # Column definitions: (header, key, format, width)
    columns = [
        ("Pipeline", "pipeline", "s", 8),
        ("Mode", "mode", "s", 16),
        ("Cases", "cases", "d", 6),
        ("Top-1", "top1", ".3f", 7),
        ("Top-3", "top3", ".3f", 7),
        ("Top-5", "top5", ".3f", 7),
        ("P@1", "precision_1", ".3f", 7),
        ("Recall", "recall", ".3f", 7),
        ("F1", "f1", ".3f", 7),
    ]

    # Check if any result has RQ2 data
    has_rq2 = any(r.get("rq2_pass_rate") is not None for r in results)
    if has_rq2:
        columns.append(("RQ2 Pass", "rq2_pass_rate", ".3f", 9))
        columns.append(("RQ2 Fill", "rq2_fill_rate", ".3f", 9))

    # Check if any result has V2-specific data
    has_v2 = any(r.get("parse_rate") is not None for r in results)
    if has_v2:
        columns.append(("Parse%", "parse_rate", ".3f", 7))
        columns.append(("SSR", "search_space_reduction", ".3f", 7))

    # Check if any result has over-reduction data
    has_overred = any(r.get("over_reduction_rate") is not None for r in results)
    if has_overred:
        columns.append(("OverRed", "over_reduction_rate", ".3f", 7))

    # Print header
    header_parts = []
    sep_parts = []
    for header, _, _, width in columns:
        header_parts.append(f"{header:>{width}}")
        sep_parts.append("-" * width)

    print("  ".join(header_parts))
    print("  ".join(sep_parts))

    # Print rows
    for r in results:
        row_parts = []
        for _, key, fmt, width in columns:
            val = r.get(key)
            if val is None:
                cell = "-"
            elif fmt == "s":
                cell = str(val)
            elif fmt == "d":
                cell = str(int(val))
            else:
                cell = f"{val:{fmt}}"
            row_parts.append(f"{cell:>{width}}")
        print("  ".join(row_parts))

    # Print best results
    print()
    _print_best(results)


def _print_best(results: list) -> None:
    """Print which experiment scored best on key metrics."""
    if len(results) < 2:
        return

    metrics_to_check = [
        ("top1", "Top-1 Accuracy", True),
        ("f1", "F1 Score", True),
    ]

    for key, label, higher_is_better in metrics_to_check:
        valid = [(r, r.get(key)) for r in results if r.get(key) is not None]
        if not valid:
            continue
        if higher_is_better:
            best_r, best_v = max(valid, key=lambda x: x[1])
        else:
            best_r, best_v = min(valid, key=lambda x: x[1])
        print(f"  Best {label}: {best_r['mode']} ({best_v:.3f})")


def export_csv(results: list, path: Path) -> None:
    """Export comparison to CSV."""
    if not results:
        return

    fieldnames = [
        "pipeline", "mode", "dataset", "cases",
        "top1", "top3", "top5", "precision_1", "recall", "f1",
        "rq2_pass_rate", "rq2_fill_rate",
        "parse_rate", "search_space_reduction", "rerank_gain",
        "file", "timestamp",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Exported to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results across experiments",
    )
    parser.add_argument(
        "--dir", default="logs/evaluations",
        help="Directory containing eval results (default: logs/evaluations)",
    )
    parser.add_argument(
        "--latest", nargs="?", const=4, type=int, default=None,
        help="Show only the N most recent results (default: 4)",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Export comparison to CSV file",
    )
    parser.add_argument(
        "--v1-only", action="store_true",
        help="Show only V1 results",
    )
    parser.add_argument(
        "--v2-only", action="store_true",
        help="Show only V2 results",
    )

    # ── Traces comparison mode ──
    parser.add_argument(
        "--traces", action="append", default=None,
        help="Path to traces JSONL file (repeat for N-way comparison)",
    )
    parser.add_argument(
        "--label", action="append", default=None,
        help="Label for each --traces file (must match count)",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate comparison charts (requires --traces or --dir)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for charts (default: logs/comparisons/latest)",
    )
    parser.add_argument(
        "--title", default=None,
        help="Chart title (default: auto-generated)",
    )
    parser.add_argument(
        "--cases", default=None,
        help="Path to cases JSONL file (to enrich older traces with bench/condition info)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # ── Mode 1: Traces comparison (N-way) ──
    if args.traces:
        labels = args.label or []
        # Auto-generate labels for any missing ones
        for i in range(len(labels), len(args.traces)):
            stem = Path(args.traces[i]).stem
            # e.g. traces_20260214_210555_v2_prompt -> v2_prompt
            parts = stem.split("_")
            if len(parts) >= 4:
                labels.append("_".join(parts[3:]))
            else:
                labels.append(stem)

        # Load all traces
        experiments = {}
        results = []
        for trace_path, label in zip(args.traces, labels):
            resolved = Path(trace_path)
            if not resolved.is_absolute():
                resolved = project_root / trace_path
            if not resolved.exists():
                print(f"WARNING: traces file not found: {resolved}")
                continue
            traces = load_traces(str(resolved))
            experiments[label] = traces
            results.append(traces_to_result(traces, label, str(resolved)))
            print(f"  Loaded {len(traces)} traces: {label} ({resolved.name})")

        if not results:
            print("No traces loaded. Check file paths.")
            return

        # Enrich traces with bench info from cases file (for older traces
        # that don't have the bench field)
        if args.cases:
            cases_path = Path(args.cases)
            if not cases_path.is_absolute():
                cases_path = project_root / cases_path
            if cases_path.exists():
                bench_map = {}
                with open(cases_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            c = json.loads(line)
                            bench_map[c["case_id"]] = c.get("bench")
                enriched = 0
                for traces in experiments.values():
                    for t in traces:
                        if not t.get("bench") and t.get("scenario_id") in bench_map:
                            t["bench"] = bench_map[t["scenario_id"]]
                            enriched += 1
                if enriched:
                    print(f"  Enriched {enriched} traces with bench/condition from {cases_path.name}")
            else:
                print(f"  WARNING: cases file not found: {cases_path}")

        print()
        print_comparison_table(results)

        if args.plots:
            output_dir = args.output or "logs/comparisons/latest"
            if not Path(output_dir).is_absolute():
                output_dir = str(project_root / output_dir)
            title = args.title or f"Evaluation ({len(results)} experiments)"
            generate_comparison_charts(experiments, output_dir, title)

        if args.csv:
            export_csv(results, Path(args.csv))

        return

    # ── Mode 2: Auto-discover from directory ──
    eval_dir = project_root / args.dir

    if not eval_dir.exists():
        print(f"No results directory found: {eval_dir}")
        print("Run experiments first, then compare.")
        return

    # Load results
    results = []
    if not args.v2_only:
        results.extend(load_v1_results(eval_dir))
    if not args.v1_only:
        results.extend(load_v2_results(eval_dir))

    if not results:
        print(f"No evaluation results found in {eval_dir}")
        print("Run experiments first:")
        print("  ./run_mcp.sh --all           # V1 experiments")
        print("  ./run_mcp.sh --all --v2      # V1 + V2 experiments")
        return

    # Sort by timestamp descending (most recent first)
    results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    # Filter to latest N
    if args.latest is not None:
        results = results[:args.latest]

    # Sort for display: v1 first, then v2, within each group by mode name
    results.sort(key=lambda r: (r["pipeline"], r["mode"]))

    print(f"Found {len(results)} evaluation results in {eval_dir}")
    print()

    print_comparison_table(results)

    if args.csv:
        export_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
