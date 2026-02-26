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
    paired_ablation: bool = False,
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
    n_charts = 13 if paired_ablation else 10
    print(f"Generating Comparison Charts ({len(experiments)} experiments, {n_charts} charts)")
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

    # ── Chart 6b: Full condition heatmap (case × profile × condition grid) ──
    _plot_full_condition_heatmap(
        experiments, f"{output_dir}/6b_full_condition_heatmap.png", title
    )

    # ── Chart 7: Modality gain (A vs B vs C per difficulty level) ──
    _plot_modality_gain(experiments, f"{output_dir}/7_modality_gain.png", title, paired_ablation)

    # ── Chart 8: Difficulty degradation (X1 → X2 → X3 per system) ──
    _plot_difficulty_degradation(experiments, f"{output_dir}/8_difficulty_degradation.png", title, paired_ablation)

    # ── Chart 9: Candidate density vs accuracy (scatter) ──
    _plot_density_vs_accuracy(experiments, f"{output_dir}/9_density_vs_accuracy.png", title)

    # ── Charts 10-12: Paired modality ablation (MA/MB/MC) ──
    if paired_ablation:
        print("\n  --- Paired Modality Ablation Charts ---")
        _plot_paired_modality_accuracy(experiments, f"{output_dir}/10_paired_modality.png", title)
        _plot_modality_delta(experiments, f"{output_dir}/11_modality_delta.png", title)
        _plot_modality_x_difficulty(experiments, f"{output_dir}/12_modality_x_difficulty.png", title)

    # ── Chart 13: Query plan strategy distribution ──
    _plot_query_plan_distribution(experiments, f"{output_dir}/13_query_plan_distribution.png", title)

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

    # Detect conditions present in traces (support both A1-C3 and MA/MB/MC)
    all_conds_found: set = set()
    for traces in experiments.values():
        for t in traces:
            c = _extract_condition(t)
            if c:
                all_conds_found.add(c)
    std_order = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    abl_order = ["MA", "MB", "MC"]
    if all_conds_found <= set(abl_order):
        conditions = [c for c in abl_order if c in all_conds_found]
    elif all_conds_found <= set(std_order):
        conditions = [c for c in std_order if c in all_conds_found]
    else:
        conditions = sorted(all_conds_found)
    labels = list(experiments.keys())

    matrix = []
    for label in labels:
        traces = experiments[label]
        by_cond: dict = {}
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

    # Add condition group separators and labels (only for full A1-C3 layout)
    if conditions == ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]:
        ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.8, linewidth=2)
        ax.axvline(x=5.5, color="gray", linestyle="--", alpha=0.8, linewidth=2)
        ax.text(1, -0.8, "Text Only", ha="center", fontsize=9, style="italic", color="gray")
        ax.text(4, -0.8, "Images+Text", ha="center", fontsize=9, style="italic", color="gray")
        ax.text(7, -0.8, "Full Multimodal", ha="center", fontsize=9, style="italic", color="gray")
    elif conditions == ["MA", "MB", "MC"]:
        ax.text(0, -0.8, "Text Only", ha="center", fontsize=9, style="italic", color="gray")
        ax.text(1, -0.8, "Img+Text", ha="center", fontsize=9, style="italic", color="gray")
        ax.text(2, -0.8, "Full Multimodal", ha="center", fontsize=9, style="italic", color="gray")

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


def _plot_full_condition_heatmap(
    experiments: dict, output_path: str, title: str = ""
) -> None:
    """Chart 6b: Full condition × profile heatmap.

    Rows  = individual cases (sorted by building then case ID).
    Cols  = (profile × condition) in fixed order:
              LoRA-MA  LoRA-MB  LoRA-MC  |  LoRA-MA-  LoRA-MB-  LoRA-MC-
              Prompt-MA Prompt-MB Prompt-MC | Prompt-MA- Prompt-MB- Prompt-MC-
    Cell  = HIT (green) / MISS (red) / N/A (light gray).

    Also shows a per-row summary column (% conditions hit) and per-column
    accuracy bar below the grid.

    Works with the merged-experiment format produced by run_update_plots():
      experiments = {"V2 LoRA (MA/MB/MC × 4D±)": [300 traces],
                     "V2 Prompt (MA/MB/MC × 4D±)": [300 traces]}
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # ── Detect profiles from experiment labels ────────────────────────────────
    # Map experiment label → profile key ("LoRA" or "Prompt")
    _PROFILE_ALIASES = {
        "lora": "LoRA",
        "LoRA": "LoRA",
        "prompt": "Prompt",
        "Prompt": "Prompt",
    }

    def _label_to_profile(label: str) -> str:
        low = label.lower()
        if "lora" in low:
            return "LoRA"
        if "prompt" in low:
            return "Prompt"
        return label

    # ── Collect all traces, tagged with (profile, condition) ─────────────────
    # Build:  lookup[(case_id, profile, cond)] = hit (bool)
    lookup: dict = {}
    case_meta: dict = {}   # case_id → {"building": str, "tier": str}

    CONDS_ON  = ["MA",  "MB",  "MC" ]
    CONDS_OFF = ["MA-", "MB-", "MC-"]
    ALL_CONDS = CONDS_ON + CONDS_OFF

    PROFILES_ORDER = ["LoRA", "Prompt"]

    for label, traces in experiments.items():
        profile = _label_to_profile(label)
        for t in traces:
            cid  = t.get("scenario_id", "")
            cond = (t.get("bench") or {}).get("condition", "")
            if not cid or cond not in ALL_CONDS:
                continue
            lookup[(cid, profile, cond)] = t.get("guid_match", False)

            # Case metadata
            if cid not in case_meta:
                import re as _re
                m = _re.search(r"_([A-Z]+)_SK_", cid)
                bld = m.group(1) if m else "?"
                tier = (t.get("difficulty_tags") or {}).get("tier", "")
                case_meta[cid] = {"building": bld, "tier": tier}

    if not lookup:
        print("  Skipped: 6b_full_condition_heatmap.png (no MA/MB/MC traces found)")
        return

    # ── Sort cases: building → case_id ────────────────────────────────────────
    BLD_ORDER = {"AP": 0, "BH": 1, "DXA": 2}
    all_case_ids = sorted(
        {cid for (cid, _, _) in lookup},
        key=lambda c: (BLD_ORDER.get(case_meta[c]["building"], 9), c),
    )
    n_cases = len(all_case_ids)

    # ── Build 12-column matrix ─────────────────────────────────────────────────
    # Col order: LoRA-MA, LoRA-MB, LoRA-MC, LoRA-MA-, LoRA-MB-, LoRA-MC-,
    #            Prompt-MA, Prompt-MB, Prompt-MC, Prompt-MA-, Prompt-MB-, Prompt-MC-
    col_defs = []
    for prof in PROFILES_ORDER:
        for cond in ALL_CONDS:
            col_defs.append((prof, cond))

    n_cols = len(col_defs)
    matrix = np.full((n_cases, n_cols), np.nan)  # nan = N/A

    for row_i, cid in enumerate(all_case_ids):
        for col_j, (prof, cond) in enumerate(col_defs):
            val = lookup.get((cid, prof, cond))
            if val is not None:
                matrix[row_i, col_j] = 1.0 if val else 0.0

    # ── Summary stats ──────────────────────────────────────────────────────────
    # Per-row: hit rate across available columns
    row_hit_rate = np.nanmean(matrix, axis=1)   # 0–1
    # Per-column accuracy
    col_acc = np.nanmean(matrix, axis=0)         # 0–1

    # ── Figure ────────────────────────────────────────────────────────────────
    # Main grid + right summary bar + bottom accuracy row
    row_h = max(0.22, 10 / n_cases)
    fig_h  = n_cases * row_h + 2.5
    fig_w  = n_cols * 0.82 + 3.5

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: main heatmap | summary bar; plus a bottom accuracy strip
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[n_cases, 2],
        width_ratios=[n_cols, 1.6],
        hspace=0.04,
        wspace=0.04,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_sum  = fig.add_subplot(gs[0, 1])
    ax_acc  = fig.add_subplot(gs[1, 0])
    ax_acc_sum = fig.add_subplot(gs[1, 1])
    ax_acc_sum.axis("off")

    # ── Colours ───────────────────────────────────────────────────────────────
    import matplotlib.colors as mcolors
    HIT_COLOR  = "#b6d7a8"   # soft green
    MISS_COLOR = "#f4cccc"   # soft red
    NA_COLOR   = "#e8e8e8"   # light grey
    cmap_hit  = mcolors.ListedColormap([MISS_COLOR, HIT_COLOR])
    cmap_hit.set_bad(color=NA_COLOR)

    # ── Draw main heatmap ─────────────────────────────────────────────────────
    masked = np.ma.masked_invalid(matrix)
    ax_main.imshow(masked, cmap=cmap_hit, aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")

    # Cell text: "✓" / "·" (no huge HIT/- text — too small at 50×12)
    for row_i in range(n_cases):
        for col_j in range(n_cols):
            v = matrix[row_i, col_j]
            if np.isnan(v):
                continue
            if v == 1:
                ax_main.text(col_j, row_i, "✓", ha="center", va="center",
                             fontsize=6.5, color="#274e13", fontweight="bold")
            else:
                ax_main.text(col_j, row_i, "·", ha="center", va="center",
                             fontsize=9, color="#990000")

    # ── Vertical dividers: LoRA | Prompt  and  4D-ON | 4D-OFF ─────────────────
    # Between LoRA and Prompt halves
    ax_main.axvline(5.5,  color="black", linewidth=1.8, alpha=0.7)
    # Within LoRA: 4D-ON vs 4D-OFF
    ax_main.axvline(2.5,  color="#666666", linewidth=0.9, alpha=0.5, linestyle="--")
    # Within Prompt: 4D-ON vs 4D-OFF
    ax_main.axvline(8.5,  color="#666666", linewidth=0.9, alpha=0.5, linestyle="--")

    # ── Horizontal dividers between buildings ─────────────────────────────────
    prev_bld = None
    for row_i, cid in enumerate(all_case_ids):
        bld = case_meta[cid]["building"]
        if prev_bld is not None and bld != prev_bld:
            ax_main.axhline(row_i - 0.5, color="black", linewidth=1.2, alpha=0.7)
        prev_bld = bld

    # ── X-axis labels ─────────────────────────────────────────────────────────
    SHORT_COND = {"MA": "MA", "MB": "MB", "MC": "MC",
                  "MA-": "MA⁻", "MB-": "MB⁻", "MC-": "MC⁻"}
    PROF_COLORS = {"LoRA": "#c06000", "Prompt": "#1a56b0"}

    col_labels = [SHORT_COND[c] for _, c in col_defs]
    ax_main.set_xticks(range(n_cols))
    ax_main.set_xticklabels(col_labels, fontsize=7.5, fontweight="bold", rotation=0)
    for tick, (prof, _) in zip(ax_main.get_xticklabels(), col_defs):
        tick.set_color(PROF_COLORS[prof])

    # Profile group labels: placed in the bottom accuracy bar
    lora_mid   = (0 + 5) / 2     # centre of LoRA columns 0-5
    prompt_mid = (6 + 11) / 2    # centre of Prompt columns 6-11
    top_y = ax_acc.get_ylim()[1] if ax_acc.get_ylim()[1] else 60
    ax_acc.text(lora_mid,   top_y * 0.92, "LoRA",   ha="center", va="top",
                fontsize=8, fontweight="bold", color=PROF_COLORS["LoRA"])
    ax_acc.text(prompt_mid, top_y * 0.92, "Prompt", ha="center", va="top",
                fontsize=8, fontweight="bold", color=PROF_COLORS["Prompt"])

    # ── Y-axis labels ─────────────────────────────────────────────────────────
    # Short label: building + number, e.g. "AP 084"
    y_labels = []
    for cid in all_case_ids:
        parts = cid.split("_")
        num  = parts[2] if len(parts) >= 3 else cid[-3:]
        bld  = case_meta[cid]["building"]
        tier = case_meta[cid]["tier"]
        y_labels.append(f"{bld} {num}" + (f" {tier}" if tier else ""))

    ax_main.set_yticks(range(n_cases))
    ax_main.set_yticklabels(y_labels, fontsize=5.5, family="monospace")

    # ── Right panel: per-row hit-rate bar ─────────────────────────────────────
    bar_colors = [
        "#b6d7a8" if r >= 0.5 else "#f4cccc" for r in row_hit_rate
    ]
    ax_sum.barh(range(n_cases), row_hit_rate * 100,
                color=bar_colors, height=0.75, edgecolor="none")
    ax_sum.set_xlim(0, 105)
    ax_sum.set_yticks([])
    ax_sum.set_xlabel("Hit %\n(all cond.)", fontsize=7)
    ax_sum.axvline(50, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_sum.tick_params(axis="x", labelsize=6)

    # Add building group markers
    prev_bld = None
    for row_i, cid in enumerate(all_case_ids):
        bld = case_meta[cid]["building"]
        if prev_bld is not None and bld != prev_bld:
            ax_sum.axhline(row_i - 0.5, color="black", linewidth=1.2, alpha=0.7)
        prev_bld = bld

    # ── Bottom panel: per-column accuracy bar ─────────────────────────────────
    bar_col_colors = [PROF_COLORS[p] for p, _ in col_defs]
    # 4D-OFF bars lighter
    bar_col_alpha = [0.9 if c in CONDS_ON else 0.45 for _, c in col_defs]
    col_acc_pct = col_acc * 100
    bars = ax_acc.bar(range(n_cols), col_acc_pct, color=bar_col_colors,
                      alpha=1.0, edgecolor="none")
    for bar, alpha in zip(bars, bar_col_alpha):
        bar.set_alpha(alpha)

    for col_j, (acc, (prof, cond)) in enumerate(zip(col_acc_pct, col_defs)):
        if not np.isnan(acc):
            ax_acc.text(col_j, acc + 0.5, f"{acc:.0f}%",
                        ha="center", va="bottom", fontsize=6,
                        color=PROF_COLORS[prof], fontweight="bold")

    ax_acc.set_xlim(-0.5, n_cols - 0.5)
    ax_acc.set_ylim(0, max(col_acc_pct[~np.isnan(col_acc_pct)]) * 1.28 if n_cols else 100)
    ax_acc.set_xticks([])
    ax_acc.set_ylabel("Acc %", fontsize=7)
    ax_acc.tick_params(axis="y", labelsize=6)
    ax_acc.axvline(5.5,  color="black", linewidth=1.8, alpha=0.7)
    ax_acc.axvline(2.5,  color="#666666", linewidth=0.9, alpha=0.5, linestyle="--")
    ax_acc.axvline(8.5,  color="#666666", linewidth=0.9, alpha=0.5, linestyle="--")
    ax_acc.grid(axis="y", alpha=0.25, linewidth=0.5, linestyle="--")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=HIT_COLOR,  label="Hit  (✓)"),
        mpatches.Patch(color=MISS_COLOR, label="Miss (·)"),
        mpatches.Patch(color=NA_COLOR,   label="N/A"),
        mpatches.Patch(color=PROF_COLORS["LoRA"],   label="LoRA"),
        mpatches.Patch(color=PROF_COLORS["Prompt"], label="Prompt"),
    ]
    fig.legend(handles=legend_handles, fontsize=7.5, loc="upper right",
               ncol=5, bbox_to_anchor=(0.98, 1.0), framealpha=0.9)

    # ── Title ─────────────────────────────────────────────────────────────────
    subtitle = (
        "Solid = 4D ON (MA/MB/MC)  ·  Faded = 4D OFF (MA⁻/MB⁻/MC⁻)  ·  "
        "Sorted by building (AP → BH → DXA)"
    )
    fig.suptitle(
        (f"{title}\n" if title else "") +
        f"Per-Case × All Conditions  ({n_cases} cases × 12 conditions)\n{subtitle}",
        fontsize=10, fontweight="bold", y=1.01,
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _plot_modality_gain(
    experiments: dict, output_path: str, title: str = "", paired_ablation: bool = False
) -> None:
    """Chart 7: Modality gain — grouped bar chart.

    Standard mode: For each difficulty level (T1/T2/T3), compare accuracy
    across modalities (A/B/C).
    Paired ablation mode: For each difficulty tier (T1/T2/T3), compare
    accuracy across MA/MB/MC conditions per extractor.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    exp_labels = list(experiments.keys())
    n_exp = len(exp_labels)
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c",
              "#e67e22", "#1abc9c", "#8e44ad"]

    if paired_ablation:
        # Paired mode: facets = T1/T2/T3, x-axis = MA/MB/MC, colors = extractor
        modalities_pa = _split_by_modality(experiments)
        mod_order = [m for m in ("MA", "MB", "MC") if m in modalities_pa]
        mod_display_pa = {"MA": "Text Only\n(MA)", "MB": "Img+Text\n(MB)", "MC": "Full\n(MC)"}
        tiers = [("T1", "T1 (Easy)"), ("T2", "T2 (Medium)"), ("T3", "T3 (Hard)")]
        extractors = sorted({k for md in modalities_pa.values() for k in md})
        n_ext = len(extractors)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for ti, (tier_key, tier_label) in enumerate(tiers):
            ax = axes[ti]
            x = np.arange(len(mod_order))
            width = 0.8 / max(n_ext, 1)

            for ei, ext in enumerate(extractors):
                vals = []
                for mod in mod_order:
                    traces = modalities_pa.get(mod, {}).get(ext, [])
                    subset = [t for t in traces
                              if (t.get("difficulty_tags") or {}).get("tier") == tier_key]
                    if subset:
                        acc = sum(1 for t in subset if t.get("guid_match", False)) / len(subset) * 100
                    else:
                        acc = 0
                    vals.append(acc)

                offset = (ei - n_ext / 2 + 0.5) * width
                bars = ax.bar(
                    x + offset, vals, width, label=ext if ti == 0 else None,
                    color=colors[ei % len(colors)], alpha=0.85,
                    edgecolor="black", linewidth=0.5,
                )
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, h + 1,
                            f"{h:.0f}", ha="center", va="bottom", fontsize=7,
                            fontweight="bold",
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([mod_display_pa[m] for m in mod_order], fontsize=9)
            ax.set_title(tier_label, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 110)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
    else:
        # Auto-detect condition style: MA/MB/MC vs A1/B1/C1
        _sample_conds = set()
        for _tr in experiments.values():
            for _t in _tr[:20]:
                _c = _extract_condition(_t)
                if _c:
                    _sample_conds.add(_c)
        use_ma_style = bool(_sample_conds & {"MA", "MB", "MC"}) and not bool(_sample_conds & {"A1", "B1", "C1"})

        if use_ma_style:
            # MA/MB/MC mode: facets = T1/T2/T3, x-axis = modality (4D-ON only)
            ma_mods = ["MA", "MB", "MC"]
            mod_display_ma = {"MA": "Text Only\n(MA)", "MB": "Img + Text\n(MB)", "MC": "Full Multimodal\n(MC)"}
            tiers = [("T1", "T1 (Easy)"), ("T2", "T2 (Medium)"), ("T3", "T3 (Hard)")]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

            for di, (tier_key, tier_label) in enumerate(tiers):
                ax = axes[di]
                x = np.arange(len(ma_mods))
                width = 0.8 / n_exp

                for ei, label in enumerate(exp_labels):
                    traces = experiments[label]
                    vals = []
                    for mod in ma_mods:
                        subset = [t for t in traces
                                  if _extract_condition(t) == mod
                                  and (t.get("difficulty_tags") or {}).get("tier") == tier_key]
                        acc = sum(1 for t in subset if t.get("guid_match", False)) / len(subset) * 100 \
                              if subset else 0
                        vals.append(acc)

                    offset = (ei - n_exp / 2 + 0.5) * width
                    bars = ax.bar(
                        x + offset, vals, width, label=label if di == 0 else None,
                        color=colors[ei % len(colors)], alpha=0.85,
                        edgecolor="black", linewidth=0.5,
                    )
                    for bar in bars:
                        h = bar.get_height()
                        if h > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2, h + 1,
                                f"{h:.0f}", ha="center", va="bottom", fontsize=7,
                                fontweight="bold",
                            )

                ax.set_xticks(x)
                ax.set_xticklabels([mod_display_ma[m] for m in ma_mods], fontsize=9)
                ax.set_title(tier_label, fontsize=12, fontweight="bold")
                ax.set_ylim(0, 110)
                ax.grid(axis="y", alpha=0.3, linestyle="--")
        else:
            modalities = ["A", "B", "C"]
            mod_display = {"A": "Text Only", "B": "Img + Text", "C": "Full Multimodal"}
            difficulty_levels = ["1", "2", "3"]
            diff_display = {"1": "T1 (Easy)", "2": "T2 (Medium)", "3": "T3 (Hard)"}

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

            for di, diff in enumerate(difficulty_levels):
                ax = axes[di]
                x = np.arange(len(modalities))
                width = 0.8 / n_exp

                for ei, label in enumerate(exp_labels):
                    traces = experiments[label]
                    vals = []
                    for mod in modalities:
                        cond = f"{mod}{diff}"
                        subset = [t for t in traces if _extract_condition(t) == cond]
                        if subset:
                            acc = sum(1 for t in subset if t.get("guid_match", False)) / len(subset) * 100
                        else:
                            acc = 0
                        vals.append(acc)

                    offset = (ei - n_exp / 2 + 0.5) * width
                    bars = ax.bar(
                        x + offset, vals, width, label=label if di == 0 else None,
                        color=colors[ei % len(colors)], alpha=0.85,
                        edgecolor="black", linewidth=0.5,
                    )
                    for bar in bars:
                        h = bar.get_height()
                        if h > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2, h + 1,
                                f"{h:.0f}", ha="center", va="bottom", fontsize=7,
                                fontweight="bold",
                            )

                ax.set_xticks(x)
                ax.set_xticklabels([mod_display[m] for m in modalities], fontsize=9)
                ax.set_title(diff_display[diff], fontsize=12, fontweight="bold")
                ax.set_ylim(0, 110)
                ax.grid(axis="y", alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    fig.legend(
        loc="upper center", ncol=n_exp, fontsize=10,
        bbox_to_anchor=(0.5, 1.02), frameon=True, shadow=True,
    )
    fig.suptitle(
        f"{title} — Modality Gain" if title else "Modality Gain (A → B → C)",
        fontsize=14, fontweight="bold", y=1.08,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _plot_difficulty_degradation(
    experiments: dict, output_path: str, title: str = "", paired_ablation: bool = False
) -> None:
    """Chart 8: Difficulty degradation — line chart.

    Standard mode: For each modality group (A/B/C), plot accuracy as
    difficulty increases (T1 → T2 → T3).
    Paired ablation mode: For each MA/MB/MC condition, plot accuracy
    across difficulty tiers (T1 → T2 → T3) per extractor.
    """
    import matplotlib.pyplot as plt

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c",
              "#e67e22", "#16a085", "#8e44ad"]
    markers = ["o", "s", "D", "^", "v", "P", "X", "h", "*"]

    if paired_ablation:
        modalities_pa = _split_by_modality(experiments)
        mod_order = [m for m in ("MA", "MB", "MC") if m in modalities_pa]
        mod_display_pa = {"MA": "Text Only (MA)", "MB": "Img+Text (MB)", "MC": "Full Multimodal (MC)"}
        tiers = [("T1", "T1\n(Easy)"), ("T2", "T2\n(Med)"), ("T3", "T3\n(Hard)")]
        extractors = sorted({k for md in modalities_pa.values() for k in md})

        fig, axes = plt.subplots(1, len(mod_order), figsize=(18, 6), sharey=True)
        if len(mod_order) == 1:
            axes = [axes]

        for mi, mod in enumerate(mod_order):
            ax = axes[mi]
            for ei, ext in enumerate(extractors):
                traces = modalities_pa.get(mod, {}).get(ext, [])
                accs = []
                for tier_key, _ in tiers:
                    subset = [t for t in traces
                              if (t.get("difficulty_tags") or {}).get("tier") == tier_key]
                    acc = sum(1 for t in subset if t.get("guid_match", False)) / len(subset) * 100 \
                          if subset else 0
                    accs.append(acc)

                ax.plot(
                    range(len(tiers)), accs,
                    marker=markers[ei % len(markers)], markersize=10,
                    color=colors[ei % len(colors)], linewidth=2.5,
                    label=ext if mi == 0 else None, alpha=0.85,
                )
                for xi, acc in enumerate(accs):
                    ax.annotate(
                        f"{acc:.0f}%", (xi, acc),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=8, fontweight="bold",
                        color=colors[ei % len(colors)],
                    )

            ax.set_xticks(range(len(tiers)))
            ax.set_xticklabels([lbl for _, lbl in tiers], fontsize=10)
            ax.set_title(mod_display_pa[mod], fontsize=12, fontweight="bold")
            ax.set_ylim(-5, 115)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.grid(axis="x", alpha=0.15, linestyle=":")
    else:
        exp_labels = list(experiments.keys())

        # Auto-detect condition style: MA/MB/MC vs A1/B1/C1
        _sample_conds = set()
        for _tr in experiments.values():
            for _t in _tr[:20]:
                _c = _extract_condition(_t)
                if _c:
                    _sample_conds.add(_c)
        use_ma_style = bool(_sample_conds & {"MA", "MB", "MC"}) and not bool(_sample_conds & {"A1", "B1", "C1"})

        if use_ma_style:
            # MA/MB/MC mode: facets = MA/MB/MC (4D-ON only), x-axis = T1/T2/T3 from difficulty_tags.tier
            ma_mods = ["MA", "MB", "MC"]
            mod_display_ma = {"MA": "Text Only (MA)", "MB": "Img + Text (MB)", "MC": "Full Multimodal (MC)"}
            tiers = [("T1", "T1\n(Easy)"), ("T2", "T2\n(Medium)"), ("T3", "T3\n(Hard)")]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

            for mi, mod in enumerate(ma_mods):
                ax = axes[mi]
                for ei, label in enumerate(exp_labels):
                    traces = experiments[label]
                    accs = []
                    for tier_key, _ in tiers:
                        subset = [t for t in traces
                                  if _extract_condition(t) == mod
                                  and (t.get("difficulty_tags") or {}).get("tier") == tier_key]
                        acc = sum(1 for t in subset if t.get("guid_match", False)) / len(subset) * 100 \
                              if subset else 0
                        accs.append(acc)

                    ax.plot(
                        range(len(tiers)), accs,
                        marker=markers[ei % len(markers)], markersize=10,
                        color=colors[ei % len(colors)], linewidth=2.5,
                        label=label if mi == 0 else None, alpha=0.85,
                    )
                    for xi, acc in enumerate(accs):
                        ax.annotate(
                            f"{acc:.0f}%", (xi, acc),
                            textcoords="offset points", xytext=(0, 12),
                            ha="center", fontsize=8, fontweight="bold",
                            color=colors[ei % len(colors)],
                        )

                ax.set_xticks(range(len(tiers)))
                ax.set_xticklabels([lbl for _, lbl in tiers], fontsize=10)
                ax.set_title(mod_display_ma[mod], fontsize=12, fontweight="bold")
                ax.set_ylim(-5, 115)
                ax.grid(axis="y", alpha=0.3, linestyle="--")
                ax.grid(axis="x", alpha=0.15, linestyle=":")
        else:
            modalities = ["A", "B", "C"]
            mod_display = {"A": "Text Only (A)", "B": "Img + Text (B)", "C": "Full Multimodal (C)"}
            difficulty_levels = ["1", "2", "3"]
            diff_labels = ["T1\n(Easy)", "T2\n(Medium)", "T3\n(Hard)"]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

            for mi, mod in enumerate(modalities):
                ax = axes[mi]
                for ei, label in enumerate(exp_labels):
                    traces = experiments[label]
                    accs = []
                    for diff in difficulty_levels:
                        cond = f"{mod}{diff}"
                        subset = [t for t in traces if _extract_condition(t) == cond]
                        if subset:
                            acc = sum(1 for t in subset if t.get("guid_match", False)) / len(subset) * 100
                        else:
                            acc = 0
                        accs.append(acc)

                    ax.plot(
                        range(3), accs,
                        marker=markers[ei % len(markers)], markersize=10,
                        color=colors[ei % len(colors)], linewidth=2.5,
                        label=label if mi == 0 else None, alpha=0.85,
                    )
                    for xi, acc in enumerate(accs):
                        ax.annotate(
                            f"{acc:.0f}%", (xi, acc),
                            textcoords="offset points", xytext=(0, 12),
                            ha="center", fontsize=8, fontweight="bold",
                            color=colors[ei % len(colors)],
                        )

                ax.set_xticks(range(3))
                ax.set_xticklabels(diff_labels, fontsize=10)
                ax.set_title(mod_display[mod], fontsize=12, fontweight="bold")
                ax.set_ylim(-5, 115)
                ax.grid(axis="y", alpha=0.3, linestyle="--")
                ax.grid(axis="x", alpha=0.15, linestyle=":")

    axes[0].set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    legend_ncol = len(extractors) if paired_ablation else len(experiments)
    fig.legend(
        loc="upper center", ncol=legend_ncol, fontsize=10,
        bbox_to_anchor=(0.5, 1.02), frameon=True, shadow=True,
    )
    fig.suptitle(
        f"{title} — Difficulty Degradation" if title else "Difficulty Degradation (T1 → T2 → T3)",
        fontsize=14, fontweight="bold", y=1.08,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _plot_density_vs_accuracy(
    experiments: dict, output_path: str, title: str = ""
) -> None:
    """Chart 9: Candidate density (k) vs accuracy — scatter plot.

    Shows how system performance degrades as the candidate pool grows.
    Each dot is a case; x = candidate_density_k, y = hit (jittered).
    Also plots a binned trend line (moving average).
    Requires traces enriched with difficulty_tags from --cases.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    exp_labels = list(experiments.keys())
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    # Check if any traces have difficulty_tags
    has_tags = False
    for traces in experiments.values():
        if any(t.get("difficulty_tags") for t in traces):
            has_tags = True
            break
    if not has_tags:
        print("  Skipped: 9_density_vs_accuracy.png (no difficulty_tags; use --cases to enrich)")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    for ei, label in enumerate(exp_labels):
        traces = experiments[label]
        ks = []
        hits = []
        for t in traces:
            dt = t.get("difficulty_tags") or {}
            k = dt.get("candidate_density_k")
            if k is not None:
                ks.append(k)
                hits.append(1 if t.get("guid_match", False) else 0)

        if not ks:
            continue

        ks = np.array(ks, dtype=float)
        hits = np.array(hits, dtype=float)

        # Jitter y for visibility
        jitter = (np.random.RandomState(42 + ei).rand(len(hits)) - 0.5) * 0.08
        ax.scatter(
            ks, hits + jitter,
            color=colors[ei % len(colors)], alpha=0.35, s=40,
            edgecolors="none",
        )

        # Binned trend line
        bin_edges = [0, 2, 10, 20, 50, 100, 200]
        bin_centers = []
        bin_accs = []
        for b0, b1 in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (ks >= b0) & (ks < b1)
            if mask.sum() >= 2:
                bin_centers.append((b0 + b1) / 2)
                bin_accs.append(hits[mask].mean() * 100)

        if bin_centers:
            ax.plot(
                bin_centers, [a / 100 for a in bin_accs],
                color=colors[ei % len(colors)], linewidth=2.5,
                marker="o", markersize=8, label=f"{label} (trend)",
                alpha=0.9,
            )

    ax.set_xlabel("Candidate Density (k = same-type elements on storey)", fontsize=12)
    ax.set_ylabel("Top-1 Hit (1 = correct, 0 = wrong)", fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Miss", "Hit"], fontsize=11)
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xlim(-0.5, 200)
    ax.legend(fontsize=10, loc="center right", frameon=True, shadow=True)
    ax.grid(axis="both", alpha=0.3, linestyle="--")

    ax.set_title(
        f"{title} — Candidate Density vs Accuracy" if title else "Candidate Density vs Accuracy",
        fontsize=14, fontweight="bold",
    )

    # Annotate density zones
    ax.axvspan(-0.5, 2, alpha=0.06, color="green", label="_S1 zone")
    ax.axvspan(20, 200, alpha=0.06, color="red", label="_S2 zone")
    ax.text(1, -0.15, "S1\nSingleton", ha="center", fontsize=8, color="green", style="italic")
    ax.text(80, -0.15, "S2\nDense Cluster", ha="center", fontsize=8, color="red", style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Charts 10-12: Paired Modality Ablation
# ─────────────────────────────────────────────────────────────────────────────


def _split_by_modality(experiments: dict) -> dict:
    """Group experiments by modality condition (MA/MB/MC).

    Returns:
        Dict mapping modality -> {extractor_label: [traces]}
        e.g. {"MA": {"V2 Prompt": [...]}, "MB": {"V2 Prompt": [...]}}
    """
    modalities = {}
    for label, traces in experiments.items():
        # Detect modality from label (e.g., "v2_prompt-MA" → "MA")
        # or from trace bench.condition
        for mod in ("MA", "MB", "MC"):
            if mod in label:
                base_label = label.replace(f"-{mod}", "").replace(f"_{mod}", "")
                if not base_label:
                    base_label = label
                modalities.setdefault(mod, {})[base_label] = traces
                break
        else:
            # Try from trace bench.condition
            sample_conds = set()
            for t in traces[:5]:
                cond = _extract_condition(t)
                if cond in ("MA", "MB", "MC"):
                    sample_conds.add(cond)
            if len(sample_conds) == 1:
                mod = sample_conds.pop()
                modalities.setdefault(mod, {})[label] = traces

    return modalities


def _paired_case_map(modalities: dict) -> dict:
    """Build case_id -> {modality: trace} mapping for paired analysis.

    Returns:
        Dict[case_id, Dict[modality, trace]]
    """
    paired = {}
    for mod, exp_dict in modalities.items():
        for _label, traces in exp_dict.items():
            for t in traces:
                cid = t.get("scenario_id", "")
                paired.setdefault(cid, {})[mod] = t
    return paired


def _plot_paired_modality_accuracy(
    experiments: dict, output_path: str, title: str = ""
) -> None:
    """Chart 10: Paired modality accuracy — grouped bar chart.

    Same cases under MA (text-only), MB (img+text), MC (full multimodal).
    If multiple extractors (Prompt, LoRA), show side-by-side.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    modalities = _split_by_modality(experiments)
    if len(modalities) < 2:
        print("  Skipped: 10_paired_modality.png (need at least 2 modality conditions)")
        return

    mod_order = [m for m in ("MA", "MB", "MC") if m in modalities]
    mod_display = {"MA": "Text Only\n(MA)", "MB": "Img + Text\n(MB)", "MC": "Full Multimodal\n(MC)"}

    # Get all extractor labels (union across modalities)
    all_extractors = set()
    for mod_exps in modalities.values():
        all_extractors.update(mod_exps.keys())
    extractor_labels = sorted(all_extractors)

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    n_ext = len(extractor_labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mod_order))
    width = 0.7 / max(n_ext, 1)

    for ei, ext_label in enumerate(extractor_labels):
        accs = []
        counts = []
        for mod in mod_order:
            traces = modalities.get(mod, {}).get(ext_label, [])
            if traces:
                hits = sum(1 for t in traces if t.get("guid_match", False))
                accs.append(hits / len(traces) * 100)
                counts.append(len(traces))
            else:
                accs.append(0)
                counts.append(0)

        offset = (ei - n_ext / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, accs, width,
            label=ext_label, color=colors[ei % len(colors)],
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        for bar, acc, cnt in zip(bars, accs, counts):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{acc:.1f}%\n(n={cnt})", ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([mod_display.get(m, m) for m in mod_order], fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_ylim(0, max(50, ax.get_ylim()[1] + 15))
    ax.legend(fontsize=10, frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax.set_title(
        f"{title} — Paired Modality Ablation" if title else "Paired Modality Ablation",
        fontsize=14, fontweight="bold",
    )

    # Add arrow annotations
    ax.annotate(
        "", xy=(len(mod_order) - 0.7, ax.get_ylim()[1] * 0.9),
        xytext=(0.3, ax.get_ylim()[1] * 0.9),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.5),
    )
    ax.text(
        len(mod_order) / 2 - 0.2, ax.get_ylim()[1] * 0.93,
        "More evidence →", ha="center", fontsize=9, color="grey", style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _plot_modality_delta(
    experiments: dict, output_path: str, title: str = ""
) -> None:
    """Chart 11: Per-case modality delta — histogram.

    For each case, compute: did adding images (MB vs MA) help or hurt?
    Shows distribution of per-case deltas.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    modalities = _split_by_modality(experiments)
    if "MA" not in modalities or "MB" not in modalities:
        print("  Skipped: 11_modality_delta.png (need both MA and MB)")
        return

    fig, axes_list = plt.subplots(1, 2, figsize=(16, 6))

    comparisons = [
        ("MA", "MB", "+Images (MB − MA)", axes_list[0]),
        ("MB", "MC", "+Floorplan (MC − MB)", axes_list[1]) if "MC" in modalities else None,
    ]
    comparisons = [c for c in comparisons if c is not None]

    # If only MA vs MB, use single plot
    if len(comparisons) == 1:
        plt.close(fig)
        fig, ax_single = plt.subplots(figsize=(10, 6))
        comparisons = [("MA", "MB", "+Images (MB − MA)", ax_single)]

    colors_bar = {"helped": "#2ecc71", "hurt": "#e74c3c", "same": "#95a5a6"}

    for mod_base, mod_new, comp_title, ax in comparisons:
        # Get all extractor labels for this pair
        base_exps = modalities.get(mod_base, {})
        new_exps = modalities.get(mod_new, {})
        common_extractors = set(base_exps.keys()) & set(new_exps.keys())

        for ext_label in sorted(common_extractors):
            base_traces = {t.get("scenario_id"): t for t in base_exps[ext_label]}
            new_traces = {t.get("scenario_id"): t for t in new_exps[ext_label]}

            common_ids = set(base_traces.keys()) & set(new_traces.keys())
            helped = 0
            hurt = 0
            same = 0

            for cid in common_ids:
                base_hit = base_traces[cid].get("guid_match", False)
                new_hit = new_traces[cid].get("guid_match", False)
                if new_hit and not base_hit:
                    helped += 1
                elif base_hit and not new_hit:
                    hurt += 1
                else:
                    same += 1

            total = len(common_ids)
            categories = ["Helped\n(0→1)", "Hurt\n(1→0)", "No Change"]
            values = [helped, hurt, same]
            bar_colors = [colors_bar["helped"], colors_bar["hurt"], colors_bar["same"]]

            bars = ax.bar(categories, values, color=bar_colors, alpha=0.85,
                         edgecolor="black", linewidth=0.5)

            for bar, val in zip(bars, values):
                if val > 0:
                    pct = val / total * 100 if total > 0 else 0
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val} ({pct:.0f}%)", ha="center", va="bottom",
                        fontsize=10, fontweight="bold",
                    )

            ax.set_ylabel("Number of Cases", fontsize=12)
            ax.set_title(f"{comp_title}\n({ext_label}, n={total})",
                        fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            # Net effect annotation
            net = helped - hurt
            net_text = f"Net: {'+' if net > 0 else ''}{net} cases"
            net_color = "#2ecc71" if net > 0 else "#e74c3c" if net < 0 else "#95a5a6"
            ax.text(
                0.95, 0.95, net_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=12, fontweight="bold",
                color=net_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=net_color, alpha=0.8),
            )

    fig.suptitle(
        f"{title} — Per-Case Modality Delta" if title else "Per-Case Modality Delta",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _plot_modality_x_difficulty(
    experiments: dict, output_path: str, title: str = ""
) -> None:
    """Chart 12: Modality × Difficulty heatmap.

    Rows = modality (MA/MB/MC), Columns = difficulty tier or H-tag.
    Cell value = accuracy. Shows where images help most.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    modalities = _split_by_modality(experiments)
    if len(modalities) < 2:
        print("  Skipped: 12_modality_x_difficulty.png (need at least 2 modalities)")
        return

    mod_order = [m for m in ("MA", "MB", "MC") if m in modalities]
    mod_display = {"MA": "Text Only (MA)", "MB": "Img+Text (MB)", "MC": "Full (MC)"}

    # Difficulty categories from difficulty_tags
    diff_categories = [
        ("Tier 1", lambda t: (t.get("difficulty_tags") or {}).get("tier") in ("T1", "Tier 1")),
        ("Tier 2", lambda t: (t.get("difficulty_tags") or {}).get("tier") in ("T2", "Tier 2")),
        ("Tier 3", lambda t: (t.get("difficulty_tags") or {}).get("tier") in ("T3", "Tier 3")),
        ("H1 (k≥20)", lambda t: (t.get("difficulty_tags") or {}).get("candidate_density_k", 0) >= 20),
        ("H2 (relational)", lambda t: (t.get("difficulty_tags") or {}).get("requires_relation", False)),
        ("H3 (conflict)", lambda t: (t.get("difficulty_tags") or {}).get("conflict_injected", False)),
    ]

    # Check if any traces have difficulty_tags
    has_tags = False
    for mod_exps in modalities.values():
        for traces in mod_exps.values():
            if any(t.get("difficulty_tags") for t in traces):
                has_tags = True
                break

    if not has_tags:
        # Fall back to tier from bench condition (less precise)
        diff_categories = [
            ("Overall", lambda t: True),
        ]
        print("  Note: No difficulty_tags found; showing overall accuracy only")

    # Build accuracy matrix: rows = modalities, cols = difficulty categories
    # Use first extractor found for simplicity (or average across extractors)
    all_extractors = set()
    for mod_exps in modalities.values():
        all_extractors.update(mod_exps.keys())
    extractor_labels = sorted(all_extractors)

    n_rows = len(mod_order) * len(extractor_labels)
    n_cols = len(diff_categories)
    matrix = np.full((n_rows, n_cols), np.nan)
    count_matrix = np.zeros((n_rows, n_cols), dtype=int)
    row_labels = []

    for mi, mod in enumerate(mod_order):
        for ei, ext_label in enumerate(extractor_labels):
            row_idx = mi * len(extractor_labels) + ei
            row_labels.append(f"{mod_display.get(mod, mod)}\n{ext_label}")
            traces = modalities.get(mod, {}).get(ext_label, [])
            for ci, (cat_name, cat_fn) in enumerate(diff_categories):
                subset = [t for t in traces if cat_fn(t)]
                if subset:
                    hits = sum(1 for t in subset if t.get("guid_match", False))
                    matrix[row_idx, ci] = hits / len(subset) * 100
                    count_matrix[row_idx, ci] = len(subset)

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 2), max(5, n_rows * 0.8)))

    # Heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=50)

    # Text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            cnt = count_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 15 or val > 85 else "black"
                ax.text(
                    j, i, f"{val:.0f}%\n(n={cnt})",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=text_color,
                )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c[0] for c in diff_categories], fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Modality group separators
    for mi in range(1, len(mod_order)):
        sep_y = mi * len(extractor_labels) - 0.5
        ax.axhline(y=sep_y, color="black", linewidth=1.5)

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)

    ax.set_title(
        f"{title} — Modality × Difficulty" if title else "Modality × Difficulty",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _plot_query_plan_distribution(
    experiments: dict, output_path: str, title: str = ""
) -> None:
    """Chart 13: Query plan strategy distribution by modality condition and model.

    For each (profile × condition) bucket, shows:
      - Stacked bars: which strategy (P0–P8) produced the first non-empty pool
      - White dashed line: Top-1 accuracy overlay
      - Annotation above each bar: avg cascade depth (how many plans were tried)

    Condition columns: MA / MB / MC / MA⁻ / MB⁻ / MC⁻  (one panel per profile).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    STRATEGY_ORDER = [
        "spatial_triplet",
        "space_and_type",
        "name_keyword",
        "neighbor_type",
        "storey_and_type",
        "storey_only",
        "type_only",
        "keyword",
        "fallback",
    ]
    STRATEGY_LABELS = {
        "spatial_triplet": "P0: Spatial",
        "space_and_type":  "P1: Space+Type",
        "name_keyword":    "P2: Name KW",
        "neighbor_type":   "P3: Neighbor",
        "storey_and_type": "P4: Storey+Type",
        "storey_only":     "P5: Storey",
        "type_only":       "P6: Type",
        "keyword":         "P7: Keyword",
        "fallback":        "P8: Fallback",
    }
    STRATEGY_COLORS = {
        "spatial_triplet": "#7c3aed",
        "space_and_type":  "#2563eb",
        "name_keyword":    "#0891b2",
        "neighbor_type":   "#059669",
        "storey_and_type": "#d97706",
        "storey_only":     "#ea580c",
        "type_only":       "#dc2626",
        "keyword":         "#db2777",
        "fallback":        "#64748b",
    }

    def _label_to_profile(label: str) -> str:
        low = label.lower()
        if "lora" in low:
            return "LoRA"
        if "prompt" in low:
            return "Prompt"
        return label

    ALL_CONDS   = ["MA", "MB", "MC", "MA-", "MB-", "MC-"]
    COND_LABELS = ["MA", "MB", "MC", "MA⁻", "MB⁻", "MC⁻"]

    # Collect per (profile, condition): strategy counts, cascade depths, accuracy
    counts: dict = {}   # (profile, cond) → {strategy: int}
    depths: dict = {}   # (profile, cond) → [int, ...]
    accs:   dict = {}   # (profile, cond) → [hits, total]

    for label, traces in experiments.items():
        profile = _label_to_profile(label)
        for trace in traces:
            cond = _extract_condition(trace)
            if not cond:
                continue
            key = (profile, cond)
            if key not in counts:
                counts[key] = {}
                depths[key] = []
                accs[key]   = [0, 0]

            # Successful strategy = first retrieval_result with pool_size > 0
            rrs = (trace.get("internals") or {}).get("retrieval_results") or []
            successful_strategy = None
            depth = 0
            for rr in rrs:
                pool_size = rr.get("pool_size", 0) or 0
                depth += 1
                if pool_size > 0:
                    qpu = rr.get("query_plan_used") or {}
                    successful_strategy = qpu.get("strategy", "fallback")
                    break
            if successful_strategy is None:
                successful_strategy = "fallback"

            counts[key][successful_strategy] = counts[key].get(successful_strategy, 0) + 1
            depths[key].append(depth)
            accs[key][1] += 1
            if trace.get("guid_match", False):
                accs[key][0] += 1

    if not counts:
        print("  Skipped: 13_query_plan_distribution.png (no internals.retrieval_results data)")
        return

    profiles = sorted({k[0] for k in counts}, key=lambda p: (p != "LoRA", p))
    n_profiles = len(profiles)

    fig, axes = plt.subplots(n_profiles, 1, figsize=(16, 5 * n_profiles), squeeze=False)

    for pi, profile in enumerate(profiles):
        ax = axes[pi][0]
        x      = np.arange(len(ALL_CONDS))
        width  = 0.55
        bottom = np.zeros(len(ALL_CONDS))

        # ── Stacked bars ────────────────────────────────────────────────────
        for strat in STRATEGY_ORDER:
            vals = []
            for cond in ALL_CONDS:
                key = (profile, cond)
                total = accs.get(key, [0, 0])[1]
                cnt   = counts.get(key, {}).get(strat, 0)
                vals.append(cnt / total * 100 if total > 0 else 0)
            if any(v > 0 for v in vals):
                ax.bar(
                    x, vals, width, bottom=bottom,
                    color=STRATEGY_COLORS[strat],
                    label=STRATEGY_LABELS[strat],
                )
                bottom += np.array(vals)

        # ── Avg cascade depth annotation ─────────────────────────────────
        for ci, cond in enumerate(ALL_CONDS):
            key  = (profile, cond)
            dep  = depths.get(key, [])
            total_c = accs.get(key, [0, 0])[1]
            if dep and total_c:
                avg_d = sum(dep) / len(dep)
                ax.text(
                    ci, bottom[ci] + 1.0, f"↓{avg_d:.1f}",
                    ha="center", va="bottom", fontsize=8, color="#94a3b8",
                )

        # ── Top-1 accuracy overlay ───────────────────────────────────────
        ax2 = ax.twinx()
        acc_vals = []
        for cond in ALL_CONDS:
            hits, total = accs.get((profile, cond), [0, 0])
            acc_vals.append(hits / total * 100 if total > 0 else float("nan"))

        ax2.plot(
            x, acc_vals, "o--", color="#111827", linewidth=2,
            markersize=7, markeredgecolor="#6b7280", label="Top-1 Acc %",
        )
        for xi, av in enumerate(acc_vals):
            if not (av != av):  # skip NaN
                ax2.text(xi, av + 2, f"{av:.0f}%", ha="center",
                         va="bottom", fontsize=8, color="#111827")
        ax2.set_ylim(0, 120)
        ax2.set_ylabel("Top-1 Accuracy (%)", fontsize=9, color="#94a3b8")
        ax2.tick_params(axis="y", labelcolor="#94a3b8")
        ax2.set_yticks([0, 20, 40, 60, 80, 100])

        # ── Axes decoration ──────────────────────────────────────────────
        ax.set_xlim(-0.5, len(ALL_CONDS) - 0.5)
        ax.set_ylim(0, 115)
        ax.set_xticks(x)
        ax.set_xticklabels(COND_LABELS, fontsize=11, fontweight="bold")
        ax.set_ylabel("Cases (%)", fontsize=10)
        ax.set_title(
            f"{profile} — Query Plan Strategy (first successful, P0–P8)",
            fontsize=12, fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        # Divider between 4D-ON and 4D-OFF columns
        ax.axvline(2.5, color="#475569", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.text(1.0, 108, "4D ON",  ha="center", fontsize=9, color="#64748b")
        ax.text(4.0, 108, "4D OFF", ha="center", fontsize=9, color="#64748b")

        ax.legend(loc="upper right", fontsize=8, ncol=3,
                  framealpha=0.9, title="Strategy (priority ↓)")

    suptitle = (
        f"{title} — Query Plan Strategy Distribution" if title
        else "Query Plan Strategy Distribution"
    )
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
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
    parser.add_argument(
        "--paired-ablation", action="store_true",
        dest="paired_ablation",
        help="Generate paired modality ablation charts (Charts 10-12). "
             "Expects traces with MA/MB/MC conditions.",
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
                difficulty_map = {}
                with open(cases_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            c = json.loads(line)
                            bench_map[c["case_id"]] = c.get("bench")
                            difficulty_map[c["case_id"]] = c.get("difficulty_tags")
                enriched = 0
                for traces in experiments.values():
                    for t in traces:
                        sid = t.get("scenario_id")
                        if not t.get("bench") and sid in bench_map:
                            t["bench"] = bench_map[sid]
                            enriched += 1
                        if not t.get("difficulty_tags") and sid in difficulty_map:
                            t["difficulty_tags"] = difficulty_map[sid]
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
            generate_comparison_charts(
                experiments, output_dir, title,
                paired_ablation=args.paired_ablation,
            )

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
