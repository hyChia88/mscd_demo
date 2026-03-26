"""
Evaluation Visualizations for MSCD Demo

Generates publication-ready charts for thesis:
1. Top-1 Accuracy across conditions
2. Search Space Reduction (funnel chart)
3. Constraints Parse Rate
4. Image Parse Timing
5. V1 vs V2 Robustness (ambiguity tolerance)
6. Vision Model Impact (Prompt vs VLM)
7. Compliance & Safety (hallucination analysis)
8. Efficiency Analysis (latency & token cost)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_condition_from_trace(trace: Dict[str, Any]) -> str:
    """
    Extract experimental condition (A1-C3) from trace.

    Tries multiple sources in order:
    1. trace.bench.condition (set by run.py since v0.3)
    2. scenario.bench.condition (legacy format from synthetic dataset)
    3. run_id suffix (e.g., "20260211_011004_v2_prompt_A1" -> "A1")

    Args:
        trace: Evaluation trace dictionary

    Returns:
        Condition string (A1-C3) or "Unknown"
    """
    # Top-level bench field (set by run.py)
    cond = (trace.get("bench") or {}).get("condition")
    if cond:
        return cond

    # Legacy: nested inside scenario
    cond = (trace.get("scenario") or {}).get("bench", {}).get("condition")
    if cond:
        return cond

    # Extract from run_id (format: YYYYMMDD_HHMMSS_profile_CONDITION)
    run_id = trace.get("run_id", "")
    if run_id and "_" in run_id:
        parts = run_id.split("_")
        if len(parts) >= 4:
            # Last part should be condition (A1, A2, A3, B1, B2, B3, C1, C2, C3)
            potential_cond = parts[-1]
            if potential_cond in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]:
                return potential_cond

    return "Unknown"


def load_traces_from_jsonl(traces_path) -> List[Dict[str, Any]]:
    """
    Load evaluation traces from JSONL file(s).

    Args:
        traces_path: Either a single file path (str) or a list of file paths

    Returns:
        List of all traces merged from all files
    """
    traces = []

    # Handle both single file and multiple files
    file_paths = [traces_path] if isinstance(traces_path, str) else traces_path

    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))

    return traces


def load_summary_from_csv(summary_path: str) -> pd.DataFrame:
    """Load summary metrics from CSV."""
    return pd.read_csv(summary_path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Top-1 Accuracy Across Conditions
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_by_condition(
    traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Top-1 Accuracy by Experimental Condition",
    before_traces: Optional[List[Dict[str, Any]]] = None,
    main_label: str = "Main",
    before_label: str = "Baseline",
):
    """
    Bar chart showing accuracy for each condition (A1-C3).

    When *before_traces* is provided, draws grouped bars (V1 vs V2).
    Otherwise falls back to single-series mode.

    Args:
        traces: List of evaluation trace dicts (shown as "main")
        output_path: Where to save the plot (PNG)
        title: Plot title
        before_traces: Optional second set of traces for side-by-side comparison
        main_label: Legend label for *traces*
        before_label: Legend label for *before_traces*
    """

    def _condition_accuracy(trace_list):
        stats = {}
        for trace in trace_list:
            cond = extract_condition_from_trace(trace)
            if cond not in stats:
                stats[cond] = {"total": 0, "hits": 0}
            stats[cond]["total"] += 1
            if trace.get("guid_match", False):
                stats[cond]["hits"] += 1
        return {c: s["hits"] / s["total"] if s["total"] > 0 else 0
                for c, s in stats.items()}

    main_acc = _condition_accuracy(traces)
    before_acc = _condition_accuracy(before_traces) if before_traces else None

    all_conds = sorted(set(list(main_acc.keys()) + (list(before_acc.keys()) if before_acc else [])))

    fig, ax = plt.subplots(figsize=(12, 6))

    if before_acc is not None:
        # ── Grouped bar mode ──
        x = np.arange(len(all_conds))
        width = 0.35

        vals_before = [before_acc.get(c, 0) for c in all_conds]
        vals_main   = [main_acc.get(c, 0) for c in all_conds]

        bars1 = ax.bar(x - width / 2, vals_before, width, label=before_label,
                        color='steelblue', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width / 2, vals_main,   width, label=main_label,
                        color='mediumseagreen', edgecolor='black', linewidth=0.5)

        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f'{h:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f'{h:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(all_conds)
        ax.legend(fontsize=10)
    else:
        # ── Single bar mode ──
        vals = [main_acc.get(c, 0) for c in all_conds]
        bars = ax.bar(all_conds, vals, color='steelblue', edgecolor='black')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    f'{h:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Experimental Condition", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Search Space Reduction (Funnel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_search_space_reduction(
    traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Search Space Reduction Across Pipeline Stages",
    before_traces: Optional[List[Dict[str, Any]]] = None,
    main_label: str = "Main",
    before_label: str = "Baseline",
):
    """
    Funnel chart showing candidate count at each stage.

    When *before_traces* is provided, draws side-by-side grouped bars
    comparing two pipelines (e.g. V1 vs V2).

    Stages: Initial Pool → After Retrieval → Final Candidates
    """

    def _pool_stats(trace_list):
        initials, finals = [], []
        for t in trace_list:
            ini = t.get("initial_pool_size", 0)
            fin = t.get("final_pool_size", 0)
            if ini > 0:
                initials.append(ini)
                finals.append(fin)
        if not initials:
            return None
        avg_i = np.mean(initials)
        avg_f = np.mean(finals)
        return {
            "stages": ["Initial Pool", "After Retrieval\n(Constraints + Query)", "Final Candidates"],
            "counts": [avg_i, (avg_i + avg_f) / 2, avg_f],
            "reduction": (avg_i - avg_f) / avg_i if avg_i > 0 else 0,
        }

    main_stats = _pool_stats(traces)
    before_stats = _pool_stats(before_traces) if before_traces else None

    if main_stats is None and before_stats is None:
        print("⚠️  No pool size data available")
        return

    stages = ["Initial Pool", "After Retrieval\n(Constraints + Query)", "Final Candidates"]

    if before_stats is not None and main_stats is not None:
        # ── Side-by-side grouped horizontal bars ──
        fig, ax = plt.subplots(figsize=(12, 6))
        y = np.arange(len(stages))
        height = 0.35

        bars1 = ax.barh(y - height / 2, before_stats["counts"], height,
                         label=f'{before_label} (reduction: {before_stats["reduction"]:.1%})',
                         color='steelblue', edgecolor='black', linewidth=0.5)
        bars2 = ax.barh(y + height / 2, main_stats["counts"], height,
                         label=f'{main_label} (reduction: {main_stats["reduction"]:.1%})',
                         color='mediumseagreen', edgecolor='black', linewidth=0.5)

        max_val = max(max(before_stats["counts"]), max(main_stats["counts"]))
        for bar in list(bars1) + list(bars2):
            w = bar.get_width()
            ax.text(w + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{w:.0f}', va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(stages)
        ax.set_xlabel("Average Candidate Count (Log Scale)", fontsize=12)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
    else:
        # ── Single pipeline mode ──
        stats = main_stats or before_stats
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        ax.barh(stages, stats["counts"], color=colors, edgecolor='black', height=0.6)

        for i, (stage, count) in enumerate(zip(stages, stats["counts"])):
            ax.text(count + max(stats["counts"]) * 0.02, i, f'{count:.0f}',
                    va='center', fontsize=11, fontweight='bold')

        ax.set_xlabel("Average Candidate Count (Log Scale)", fontsize=12)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        ax.text(0.95, 0.05, f'Reduction: {stats["reduction"]:.1%}',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Constraints Parse Rate
# ─────────────────────────────────────────────────────────────────────────────

def plot_constraints_parse_rate(
    v2_traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Constraints Extraction Success Rate"
):
    """
    Pie chart showing constraints parse success vs. failure.
    """
    success_count = 0
    fail_count = 0

    for trace in v2_traces:
        if trace.get("constraints_parse_success", False):
            success_count += 1
        else:
            fail_count += 1

    if success_count + fail_count == 0:
        print("⚠️  No V2 trace data available")
        return

    # Plot pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = ['Success', 'Failed']
    sizes = [success_count, fail_count]
    colors = ['#2ca02c', '#d62728']
    explode = (0.1, 0)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'}
    )

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add count annotation
    ax.text(0, -1.3, f'Total: {success_count + fail_count} cases',
            ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Image Parsing Timing Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_image_parse_timing(
    v2_traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Image Parsing Latency Distribution (VLM)"
):
    """
    Histogram + box plot showing image parse timing.
    """
    parse_times = []
    for trace in v2_traces:
        img_parse_ms = trace.get("image_parse_ms", 0)
        if img_parse_ms > 0:
            parse_times.append(img_parse_ms)

    if not parse_times:
        print("⚠️  No image parse timing data available")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(parse_times, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(parse_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(parse_times):.0f} ms')
    ax1.axvline(np.median(parse_times), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(parse_times):.0f} ms')
    ax1.set_xlabel("Parse Time (ms)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distribution of Image Parse Latency", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Box plot
    ax2.boxplot(parse_times, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    ax2.set_ylabel("Parse Time (ms)", fontsize=12)
    ax2.set_title("Box Plot (Outlier Detection)", fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5: Vision Impact (Prompt-Only vs VLM)
# ─────────────────────────────────────────────────────────────────────────────

def plot_vision_impact(
    before_traces: List[Dict[str, Any]],
    after_traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Vision Model Impact on Accuracy (B/C Conditions)"
):
    """
    Grouped bar chart comparing accuracy before/after VLM integration.

    Args:
        before_traces: Traces from OLD pipeline (prompt-only, no VLM)
        after_traces: Traces from NEW pipeline (with VLM)
    """
    def compute_accuracy(traces):
        if not traces:
            return 0.0
        hits = sum(1 for t in traces if t.get("guid_match", False))
        return hits / len(traces)

    before_acc = compute_accuracy(before_traces)
    after_acc = compute_accuracy(after_traces)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ['Prompt-Only\n(Before)', 'Prompt + VLM\n(After)']
    accuracies = [before_acc, after_acc]
    colors = ['#ff7f0e', '#2ca02c']

    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', width=0.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel("Top-1 Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add improvement annotation
    improvement = (after_acc - before_acc) / before_acc if before_acc > 0 else float('inf')
    if improvement < float('inf'):
        ax.text(0.5, 0.9, f'Improvement: +{improvement:.0%}',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 6: Per-Case Success/Failure Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_case_heatmap(
    traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Per-Case Retrieval Success Heatmap",
    before_traces: Optional[List[Dict[str, Any]]] = None,
    main_label: str = "Main",
    before_label: str = "Baseline",
):
    """
    Heatmap showing which cases succeeded/failed.

    When *before_traces* is provided, shows two side-by-side heatmaps
    for V1 vs V2 comparison.

    Rows: Case IDs (grouped by condition)
    Columns: Metrics (GUID Match, Name Match, Storey Match)
    """

    def _build_heatmap_df(trace_list, limit=50):
        rows = []
        for trace in trace_list[:limit]:
            cond = extract_condition_from_trace(trace)
            case_id = trace.get("scenario_id", "Unknown")[:20]
            rows.append({
                "label": f"[{cond}] {case_id}",
                "GUID": 1 if trace.get("guid_match", False) else 0,
                "Name": 1 if trace.get("name_match", False) else 0,
                "Storey": 1 if trace.get("storey_match", False) else 0,
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df = df.set_index("label")
        return df

    if before_traces:
        # ── Side-by-side heatmaps ──
        df_before = _build_heatmap_df(before_traces)
        df_main = _build_heatmap_df(traces)

        if df_before is None and df_main is None:
            print("⚠️  No case data available")
            return

        n_rows = max(len(df_before) if df_before is not None else 0,
                     len(df_main) if df_main is not None else 0)
        fig_height = max(10, n_rows * 0.3)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, fig_height))

        if df_before is not None:
            sns.heatmap(df_before, cmap='RdYlGn', linewidths=0.5, linecolor='gray',
                        ax=ax1, vmin=0, vmax=1, annot=False, cbar=False)
            ax1.set_title(before_label, fontsize=12, fontweight='bold')
            ax1.set_ylabel("Case ID", fontsize=10)
        else:
            ax1.set_visible(False)

        if df_main is not None:
            sns.heatmap(df_main, cmap='RdYlGn', linewidths=0.5, linecolor='gray',
                        ax=ax2, vmin=0, vmax=1, annot=False,
                        cbar_kws={'label': 'Match (0=No, 1=Yes)'})
            ax2.set_title(main_label, fontsize=12, fontweight='bold')
            ax2.set_ylabel("")
        else:
            ax2.set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    else:
        # ── Single heatmap ──
        df = _build_heatmap_df(traces)
        if df is None:
            print("⚠️  No case data available")
            return

        fig, ax = plt.subplots(figsize=(8, max(10, len(df) * 0.3)))
        sns.heatmap(df, cmap='RdYlGn', cbar_kws={'label': 'Match (0=No, 1=Yes)'},
                    linewidths=0.5, linecolor='gray', ax=ax,
                    vmin=0, vmax=1, annot=False)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Match Type", fontsize=12)
        ax.set_ylabel("Case ID", fontsize=12)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 7: Condition-Wise Performance Matrix (A1-C3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_condition_wise_comparison(
    experiments: Dict[str, List[Dict[str, Any]]],
    output_path: str,
    title: str = "Performance Across Experimental Conditions (A1-C3)"
):
    """
    **KEY THESIS FIGURE**: General condition-wise comparison across experiments.

    Compares any number of experiments (v1, v2, v3, baseline, etc.) across
    all 9 experimental conditions showing modality impact.

    Args:
        experiments: Dict mapping experiment_name → list of traces
        output_path: Where to save the plot
        title: Chart title
    """
    print(f"\n→ Generating Condition-Wise Comparison...")

    exp_names = list(experiments.keys())

    # Detect conditions dynamically from traces
    all_conds_found: set = set()
    for traces in experiments.values():
        for t in traces:
            c = extract_condition_from_trace(t)
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

    # Calculate accuracy for each experiment × condition
    data = []
    for exp_name in exp_names:
        traces = experiments[exp_name]

        # Group by condition
        by_condition: dict = {}
        for trace in traces:
            cond = extract_condition_from_trace(trace)
            if cond not in by_condition:
                by_condition[cond] = []
            by_condition[cond].append(trace)

        # Calculate accuracy per condition
        for cond in conditions:
            hits = sum(1 for t in by_condition.get(cond, []) if t.get("guid_match", False))
            total = len(by_condition.get(cond, []))
            acc = (hits / total * 100) if total > 0 else 0
            data.append({
                'Experiment': exp_name,
                'Condition': cond,
                'Accuracy': acc,
                'Count': total
            })

    if not data:
        print("⚠️  No data available for comparison")
        return

    df = pd.DataFrame(data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 9))

    # Use different colors for each experiment
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    x = np.arange(len(conditions))
    n_exp = len(exp_names)
    width = 0.8 / n_exp  # Dynamic bar width

    for i, exp_name in enumerate(exp_names):
        exp_data = df[df['Experiment'] == exp_name]
        accuracies = [exp_data[exp_data['Condition'] == c]['Accuracy'].values[0]
                     if len(exp_data[exp_data['Condition'] == c]) > 0 else 0
                     for c in conditions]

        offset = (i - n_exp/2 + 0.5) * width
        bars = ax.bar(x + offset, accuracies, width,
                     label=exp_name,
                     color=colors[i % len(colors)],
                     alpha=0.85,
                     edgecolor='black',
                     linewidth=1.2)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.0f}%',
                       ha='center', va='bottom',
                       fontsize=8, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Experimental Condition', fontsize=14, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=True, ncol=min(3, n_exp))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add condition group separators and labels (layout depends on condition set)
    if conditions == ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]:
        ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.6, linewidth=2)
        ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.6, linewidth=2)
        ax.text(1, 105, 'Text Only\n(No Images)', ha='center', fontsize=10,
                style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax.text(4, 105, 'Images + Text\n(No Floorplan)', ha='center', fontsize=10,
                style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax.text(7.5, 105, 'Full Multimodal\n(Images + Floorplan)', ha='center', fontsize=10,
                style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        condition_info = (
            "Condition Details:\n"
            "A1: -Img -Plan -4D ~Blur | A2: -Img -Plan +4D Clear | A3: -Img -Plan +4D+ Clear\n"
            "B1: +Img -Plan -4D ~Blur | B2: +Img -Plan +4D Clear | B3: +Img -Plan -4D Clear\n"
            "C1: +Img +Plan -4D Clear | C2: +Img +Plan +4D Clear | C3: +Img +Plan +4D+ Clear+CLIP"
        )
        fig.text(0.5, 0.01, condition_info, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    elif conditions == ["MA", "MB", "MC"]:
        ax.text(0, 105, 'Text Only\n(MA)', ha='center', fontsize=10,
                style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax.text(1, 105, 'Img + Text\n(MB)', ha='center', fontsize=10,
                style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax.text(2, 105, 'Full Multimodal\n(MC)', ha='center', fontsize=10,
                style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        condition_info = (
            "Paired Modality Ablation (same 84 cases under all 3 conditions)\n"
            "MA: Text-only  |  MB: Site photos + Text  |  MC: Site photos + Floorplan + Text"
        )
        fig.text(0.5, 0.01, condition_info, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Efficiency Comparison (Latency, API Calls, Cost)
# ─────────────────────────────────────────────────────────────────────────────

def plot_efficiency_comparison(
    experiments: Dict[str, List[Dict[str, Any]]],
    output_path: Optional[str] = None,
):
    """
    Compare V1 vs V2 efficiency: latency, API calls, and estimated cost.

    Args:
        experiments: Dict mapping experiment name -> list of traces
        output_path: Where to save the plot
    """
    if not experiments:
        print("  ⚠ No experiments to compare efficiency. Skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    exp_names = list(experiments.keys())
    colors = sns.color_palette("Set2", len(exp_names))

    # Collect per-experiment aggregates
    latencies = {}
    api_calls = {}
    api_costs = {}

    for exp_name, traces in experiments.items():
        valid = [t for t in traces if t.get("success", False)]
        # Filter out negative/zero latencies (clock skew outliers)
        latencies[exp_name] = [t.get("total_latency_ms", 0) for t in valid
                               if t.get("total_latency_ms", 0) > 0]
        api_calls[exp_name] = [t.get("api_calls_count", 0) for t in valid]
        api_costs[exp_name] = [t.get("api_cost_estimate", 0) for t in valid]

    # ── Panel 1: Latency (ms) → convert to seconds ──
    ax = axes[0]
    positions = range(len(exp_names))
    lat_secs = {n: [v / 1000 for v in latencies[n]] for n in exp_names}
    means = [np.mean(lat_secs[n]) if lat_secs[n] else 0 for n in exp_names]
    stds = [np.std(lat_secs[n]) if lat_secs[n] else 0 for n in exp_names]
    bars = ax.bar(positions, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(exp_names, fontsize=10)
    ax.set_ylabel("Latency (seconds)", fontsize=11)
    ax.set_title("Avg Latency per Case", fontsize=13, fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{m:.1f}s", ha="center", va="bottom", fontsize=9)

    # ── Panel 2: API Calls ──
    ax = axes[1]
    means = [np.mean(api_calls[n]) if api_calls[n] else 0 for n in exp_names]
    stds = [np.std(api_calls[n]) if api_calls[n] else 0 for n in exp_names]
    bars = ax.bar(positions, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(exp_names, fontsize=10)
    ax.set_ylabel("API Calls", fontsize=11)
    ax.set_title("Avg API Calls per Case", fontsize=13, fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    # ── Panel 3: Estimated Cost ──
    ax = axes[2]
    total_costs = [sum(api_costs[n]) for n in exp_names]
    bars = ax.bar(positions, total_costs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(exp_names, fontsize=10)
    ax.set_ylabel("Estimated Cost (USD)", fontsize=11)
    ax.set_title("Total Estimated API Cost", fontsize=13, fontweight="bold")
    for bar, c in zip(bars, total_costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"${c:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Pipeline Efficiency Comparison (V1 Agent vs V2 Constraints)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Map profile short names to descriptive labels
PROFILE_LABELS = {
    "baseline": "V1 Agent-Driven\n(MCP Tool-Calling)",
    "prompt":   "V2 Constraints-Driven\n(Structured Extraction)",
}


def _is_pipeline_comparison(
    before_traces: List[Dict[str, Any]],
    after_traces: List[Dict[str, Any]],
) -> bool:
    """Return True when *before* and *after* come from different pipeline types (v1 vs v2).

    This distinguishes a v1-vs-v2 comparison (where chart 5 / VLM ablation
    is meaningless) from a VLM ablation comparison (same pipeline, with/without VLM).
    """
    before_types = {t.get("pipeline_type", "unknown") for t in before_traces[:20]}
    after_types = {t.get("pipeline_type", "unknown") for t in after_traces[:20]}
    # If the two sets are disjoint (e.g. {"v1"} vs {"v2"}), it's a pipeline comparison
    return bool(before_types and after_types and before_types.isdisjoint(after_types))


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    traces_path: str,
    v2_traces_path: Optional[str] = None,
    before_traces_path: Optional[str] = None,
    output_dir: str = "logs/plots"
):
    """
    Generate all evaluation plots from trace files.

    Args:
        traces_path: Path to main traces JSONL (V1 or V2)
        v2_traces_path: Path to V2-specific traces (optional, for V2 metrics)
        before_traces_path: Path to "before VLM" traces (for comparison)
        output_dir: Where to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Generating Evaluation Visualizations")
    print(f"{'='*60}\n")

    # Load traces
    traces = load_traces_from_jsonl(traces_path)
    path_desc = f"{len(traces_path)} files" if isinstance(traces_path, list) else traces_path
    print(f"Loaded {len(traces)} traces from {path_desc}")

    v2_traces = []
    if v2_traces_path:
        v2_traces = load_traces_from_jsonl(v2_traces_path)
        path_desc = f"{len(v2_traces_path)} files" if isinstance(v2_traces_path, list) else v2_traces_path
        print(f"Loaded {len(v2_traces)} V2 traces from {path_desc}")

    before_traces = []
    if before_traces_path:
        before_traces = load_traces_from_jsonl(before_traces_path)
        path_desc = f"{len(before_traces_path)} files" if isinstance(before_traces_path, list) else before_traces_path
        print(f"Loaded {len(before_traces)} 'before' traces from {path_desc}")

    print()

    # ── Resolve descriptive labels for comparison mode ──
    main_exp_label = "Main"
    before_exp_label = "Baseline"
    if before_traces:
        first_main_path = traces_path[0] if isinstance(traces_path, list) else traces_path
        first_before_path = before_traces_path[0] if isinstance(before_traces_path, list) else before_traces_path

        main_exp_label = Path(first_main_path).stem.replace("traces_", "").split("_")[-1] if traces_path else "Main"
        before_exp_label = Path(first_before_path).stem.replace("traces_", "").split("_")[-1] if before_traces_path else "Baseline"

        main_exp_label = PROFILE_LABELS.get(main_exp_label, main_exp_label)
        before_exp_label = PROFILE_LABELS.get(before_exp_label, before_exp_label)

    # ── Generate plots ──

    # Chart 1: Skip in comparison mode (chart 7 covers V1 vs V2 accuracy).
    if not before_traces:
        plot_accuracy_by_condition(traces, f"{output_dir}/1_accuracy_by_condition.png")

    # Chart 2: In comparison mode, show V1 vs V2 side-by-side search space reduction
    if before_traces:
        plot_search_space_reduction(
            traces, f"{output_dir}/2_search_space_reduction.png",
            before_traces=before_traces,
            main_label=main_exp_label, before_label=before_exp_label,
        )
    else:
        plot_search_space_reduction(traces, f"{output_dir}/2_search_space_reduction.png")

    # Charts 3-4: V2-specific diagnostics
    if v2_traces:
        plot_constraints_parse_rate(v2_traces, f"{output_dir}/3_constraints_parse_rate.png")
        plot_image_parse_timing(v2_traces, f"{output_dir}/4_image_parse_timing.png")

    # Chart 5: VLM ablation only — skip in v1-vs-v2 pipeline comparison mode
    if before_traces and traces and not _is_pipeline_comparison(before_traces, traces):
        plot_vision_impact(before_traces, traces, f"{output_dir}/5_vision_impact.png")

    # Chart 6: In comparison mode, show V1 vs V2 side-by-side heatmap
    if before_traces:
        plot_per_case_heatmap(
            traces, f"{output_dir}/6_per_case_heatmap.png",
            before_traces=before_traces,
            main_label=main_exp_label, before_label=before_exp_label,
        )
    else:
        plot_per_case_heatmap(traces, f"{output_dir}/6_per_case_heatmap.png")

    # Charts 7-8: Multi-experiment comparison
    if before_traces and traces:
        experiments = {
            before_exp_label: before_traces,
            main_exp_label: traces,
        }
        plot_condition_wise_comparison(experiments, f"{output_dir}/7_condition_comparison.png")
        plot_efficiency_comparison(experiments, f"{output_dir}/8_efficiency_comparison.png")

    print(f"\n{'='*60}")
    print(f"✓ All plots saved to: {output_dir}/")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Modality Analysis — Charts 9-11
# Analyzes the contribution of each input modality and 4D project context.
# ─────────────────────────────────────────────────────────────────────────────

# What each condition makes available to the model at inference time.
# Source of truth: profiles.yaml → conditions section.
_CONDITION_META: Dict[str, Dict] = {
    # Paired modality ablation — 4D context ON
    "MA":  {"images": False, "floorplan": False, "context_4d": True},
    "MB":  {"images": True,  "floorplan": False, "context_4d": True},
    "MC":  {"images": True,  "floorplan": True,  "context_4d": True},
    # 4D ablation variants — same modalities as MA/MB/MC but 4D context OFF
    "MA-": {"images": False, "floorplan": False, "context_4d": False},
    "MB-": {"images": True,  "floorplan": False, "context_4d": False},
    "MC-": {"images": True,  "floorplan": True,  "context_4d": False},
    # Full A1-C3 grid
    "A1": {"images": False, "floorplan": False, "context_4d": True},
    "A2": {"images": False, "floorplan": False, "context_4d": True},
    "A3": {"images": False, "floorplan": False, "context_4d": True},
    "B1": {"images": True,  "floorplan": False, "context_4d": False},
    "B2": {"images": True,  "floorplan": False, "context_4d": False},
    "B3": {"images": True,  "floorplan": False, "context_4d": False},
    "C1": {"images": False, "floorplan": True,  "context_4d": False},
    "C2": {"images": True,  "floorplan": True,  "context_4d": False},
    "C3": {"images": True,  "floorplan": True,  "context_4d": True},
}

_PALETTE = {
    "text_4d":   "#3b82f6",  # blue  — text + 4D
    "photo":     "#f59e0b",  # amber — site photos added
    "floorplan": "#10b981",  # green — floorplan added
    "no_4d":     "#ef4444",  # red   — 4D masked
}
_DARK_BG  = "white"
_GRID_COL = "#94a3b8"   # slate-400 — visible on white, used for spines/grid/separators


def _load_traces_from_files(
    roots: Optional[List[str]] = None,
    run_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load individual *.trace.json files from evaluation log directories.

    Args:
        roots: Override list of root directories (default: output/synth_v*/traces)
        run_filter: Only load from run dirs containing this substring (e.g. "lora")
    """
    import re as _re
    repo_root = Path(__file__).parent.parent.parent
    default_roots = [
        repo_root / "logs" / "evaluations" / "synth_v03" / "traces",
        repo_root / "logs" / "evaluations" / "synth_v04" / "traces",
    ]
    search_roots = [Path(r) for r in roots] if roots else default_roots

    traces: List[Dict[str, Any]] = []
    for root in search_roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_filter and run_filter not in run_dir.name:
                continue
            for tf in sorted(run_dir.glob("*.trace.json")):
                try:
                    t = json.loads(tf.read_text(encoding="utf-8"))
                    cond = (t.get("bench") or {}).get("condition", "")
                    if cond and cond != "Unknown":
                        t["_profile"] = (
                            "LoRA" if "lora" in run_dir.name
                            else "Prompt" if "prompt" in run_dir.name
                            else "Other"
                        )
                        m = _re.search(r"_([A-Z]+)_SK_", t.get("scenario_id", ""))
                        t["_building"] = m.group(1) if m else "?"
                        traces.append(t)
                except Exception:
                    pass
    return traces


def _load_traces_from_jsonl(
    paths: List[str],
) -> List[Dict[str, Any]]:
    """
    Load traces from JSONL files, tagging _profile and _building from filename.

    Profile is inferred from filename:  'lora' → 'LoRA', 'prompt' → 'Prompt'.
    Building is extracted from scenario_id using regex _([A-Z]+)_SK_.
    Condition is read from bench.condition (set by --condition-override at run time).
    """
    import re as _re

    traces: List[Dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        name = p.stem.lower()
        if "lora" in name:
            profile = "LoRA"
        elif "prompt" in name:
            profile = "Prompt"
        else:
            profile = "Other"

        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                cond = (t.get("bench") or {}).get("condition", "")
                if cond and cond != "Unknown":
                    t["_profile"] = profile
                    m = _re.search(r"_([A-Z]+)_SK_", t.get("scenario_id", ""))
                    t["_building"] = m.group(1) if m else "?"
                    traces.append(t)
            except Exception:
                pass
    return traces


def _wilson_ci(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval. Returns (lo, hi) as fractions."""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _style_dark(fig: Any, ax: Any) -> None:
    """Apply the publication theme (white background, black text)."""
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    ax.tick_params(colors="black")
    ax.spines[:].set_color(_GRID_COL)
    ax.grid(axis="y", alpha=0.3, color=_GRID_COL, linestyle="--")
    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.title.set_color("black")


# ── Chart 9: Visual Modality Stack (MA / MB / MC) ────────────────────────────

def plot_modality_stack(
    traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Visual Modality Contribution (Paired Ablation)",
) -> None:
    """
    Grouped bar chart: MA vs MB vs MC × LoRA vs Prompt.

    Shows the incremental accuracy gain when site photos (MB) and
    floorplan (MC) are added on top of 4D text context (MA).
    All three conditions have 4D project context active.
    """
    conds    = ["MA", "MB", "MC"]
    profiles = ["LoRA", "Prompt"]
    colors   = {"LoRA": _PALETTE["floorplan"], "Prompt": _PALETTE["text_4d"]}

    by_cp: Dict[Tuple, List] = {(c, p): [] for c in conds for p in profiles}
    for t in traces:
        c = (t.get("bench") or {}).get("condition", "")
        if c in conds:
            p = t.get("_profile", "Other")
            if p in profiles:
                by_cp[(c, p)].append(t)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _style_dark(fig, ax)

    x = np.arange(len(conds))
    bar_w = 0.32
    offsets = [-bar_w / 2, bar_w / 2]

    for pi, prof in enumerate(profiles):
        accs, lo_errs, hi_errs, ns = [], [], [], []
        for c in conds:
            grp = by_cp[(c, prof)]
            hits = sum(1 for t in grp if t.get("guid_match", False))
            n    = len(grp)
            acc  = hits / n if n else 0.0
            lo, hi = _wilson_ci(hits, n)
            accs.append(acc * 100)
            lo_errs.append((acc - lo) * 100)
            hi_errs.append((hi - acc) * 100)
            ns.append(n)

        bars = ax.bar(
            x + offsets[pi], accs, bar_w,
            yerr=[lo_errs, hi_errs],
            label=prof, color=colors[prof], alpha=0.88,
            edgecolor="none", linewidth=0.6,
            capsize=4, error_kw={"ecolor": "black", "linewidth": 1.0},
        )
        for bar, acc, n in zip(bars, accs, ns):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2.0,
                    f"{acc:.0f}%\n(n={n})",
                    ha="center", va="bottom", fontsize=8.5,
                    color="black", fontweight="bold",
                )

        # Delta annotations between consecutive conditions
        for ci in range(len(conds) - 1):
            a0 = accs[ci]; a1 = accs[ci + 1]
            delta = a1 - a0
            if abs(delta) > 0.5:
                ax.annotate(
                    f"{delta:+.0f}%",
                    xy=(ci + 1 + offsets[pi], a1 + 3),
                    xytext=((ci + ci + 1) / 2 + offsets[pi], max(a0, a1) + 16),
                    fontsize=8, color="#b45309", ha="center",
                    arrowprops=dict(arrowstyle="-|>", color="#b45309", lw=0.9),
                )

    # Background bands showing what each condition adds
    band_info = [
        (0, "Text only\n+ 4D context"),
        (1, "+ Site photos"),
        (2, "+ Floorplan"),
    ]
    for xi, label in band_info:
        ax.axvspan(xi - 0.45, xi + 0.45, alpha=0.06, color="#e2e8f0", zorder=0)
        ax.text(xi, -9, label, ha="center", va="top",
                fontsize=8.5, color="#94a3b8", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(["MA", "MB", "MC"], fontsize=13, color="black", fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.legend(fontsize=10, facecolor=_DARK_BG, edgecolor=_GRID_COL,
              labelcolor="black", loc="upper left")
    ax.set_title(
        f"{title}\n"
        "MA = Text + 4D context  ·  MB = + Site Photos  ·  MC = + Floorplan",
        fontsize=12, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved: {output_path}")
    plt.close()


# ── Chart 10: 4D Context Impact (A1-C3 grid) ─────────────────────────────────

def plot_4d_context_impact(
    traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "4D Project Context Impact (A1–C3 Conditions)",
) -> None:
    """
    Bar chart of all nine A1-C3 conditions, colour-coded by 4D context availability.

    Blue bars = 4D project context active (task_status available to model).
    Red bars  = 4D masked to N/A.
    Dashed reference lines show the average accuracy for each group.
    """
    grid_conds_order = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    present = {(t.get("bench") or {}).get("condition", "") for t in traces}
    grid_conds = [c for c in grid_conds_order if c in present]

    if not grid_conds:
        print("⚠  No A1-C3 traces — skipping Chart 10 (4D context impact)")
        return

    by_cond: Dict[str, List] = {c: [] for c in grid_conds}
    for t in traces:
        c = (t.get("bench") or {}).get("condition", "")
        if c in grid_conds:
            by_cond[c].append(t)

    fig, ax = plt.subplots(figsize=(13, 6))
    _style_dark(fig, ax)

    x = np.arange(len(grid_conds))
    bar_w = 0.6
    accs: List[float] = []
    lo_errs: List[float] = []
    hi_errs: List[float] = []
    bar_colors: List[str] = []
    counts: List[int] = []

    for c in grid_conds:
        grp  = by_cond[c]
        hits = sum(1 for t in grp if t.get("guid_match", False))
        n    = len(grp)
        acc  = hits / n if n else 0.0
        lo, hi = _wilson_ci(hits, n)
        accs.append(acc * 100)
        lo_errs.append((acc - lo) * 100)
        hi_errs.append((hi - acc) * 100)
        counts.append(n)
        has_4d = _CONDITION_META.get(c, {}).get("context_4d", False)
        bar_colors.append(_PALETTE["text_4d"] if has_4d else _PALETTE["no_4d"])

    ax.bar(
        x, accs, bar_w,
        yerr=[lo_errs, hi_errs],
        color=bar_colors, alpha=0.88,
        edgecolor="none", linewidth=0.8,
        capsize=4, error_kw={"ecolor": "black", "linewidth": 1.2},
    )

    for xi, (acc, n, c) in enumerate(zip(accs, counts, grid_conds)):
        ax.text(xi, acc + 2.2, f"{acc:.0f}%\n(n={n})",
                ha="center", va="bottom", fontsize=8.5,
                color="black", fontweight="bold")
        # Modality icon row below x-axis
        meta = _CONDITION_META.get(c, {})
        icons = []
        if meta.get("images"):     icons.append("img")
        if meta.get("floorplan"):  icons.append("plan")
        if meta.get("context_4d"): icons.append("4D")
        ax.text(xi, -9, " ".join(icons) or "-",
                ha="center", va="top", fontsize=7.5, color="#94a3b8")

    # Group separator verticals
    ax.axvline(2.5, color="#475569", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.axvline(5.5, color="#475569", linewidth=1.5, linestyle="--", alpha=0.6)
    for gx, glabel in [(1, "Group A\nText-only"), (4, "Group B\nSite Photos"),
                       (7.5, "Group C\nFloorplan")]:
        ax.text(gx, 103, glabel, ha="center", fontsize=9,
                color="#94a3b8", style="italic")

    # Average reference lines for 4D ON vs OFF
    with_4d = [a for a, c in zip(accs, grid_conds)
               if _CONDITION_META.get(c, {}).get("context_4d")]
    no_4d   = [a for a, c in zip(accs, grid_conds)
               if not _CONDITION_META.get(c, {}).get("context_4d")]
    handles = [
        mpatches.Patch(color=_PALETTE["text_4d"], label="4D context ON"),
        mpatches.Patch(color=_PALETTE["no_4d"],   label="4D context OFF"),
    ]
    if with_4d:
        ax.axhline(np.mean(with_4d), color=_PALETTE["text_4d"],
                   linewidth=1.5, linestyle=":", alpha=0.8)
        handles.append(Line2D([0], [0], color=_PALETTE["text_4d"], linewidth=1.5,
                               linestyle=":", label=f"Avg 4D ON: {np.mean(with_4d):.0f}%"))
    if no_4d:
        ax.axhline(np.mean(no_4d), color=_PALETTE["no_4d"],
                   linewidth=1.5, linestyle=":", alpha=0.8)
        handles.append(Line2D([0], [0], color=_PALETTE["no_4d"], linewidth=1.5,
                               linestyle=":", label=f"Avg 4D OFF: {np.mean(no_4d):.0f}%"))

    ax.legend(handles=handles, fontsize=9, facecolor=_DARK_BG,
              edgecolor=_GRID_COL, labelcolor="black", loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(grid_conds, fontsize=11, color="black", fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"{title}\n"
        "Blue = 4D task status available to model  ·  Red = 4D masked to N/A",
        fontsize=12, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved: {output_path}")
    plt.close()


# ── Chart 11: Modality × Building heatmap ────────────────────────────────────

def plot_modality_x_building(
    traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Accuracy by Building × Modality Condition",
) -> None:
    """
    Heatmap: rows = building (AP, BH, DXA), columns = condition (MA/MB/MC).
    Shows whether modality importance is consistent across building types.
    Yellow delta annotations show per-row incremental gain.
    """
    conds     = ["MA", "MB", "MC"]
    buildings = ["AP", "BH", "DXA"]
    cond_labels = {
        "MA": "MA\n(Text + 4D)",
        "MB": "MB\n(Photos + 4D)",
        "MC": "MC\n(Floorplan\n+ Photos + 4D)",
    }
    bld_labels = {
        "AP":  "AP\nAdvancedProject\n(10-storey office)",
        "BH":  "BH  BasicHouse\n(2-storey residential)",
        "DXA": "DXA  Duplex_A\n(split-level duplex)",
    }

    paired = [t for t in traces
              if (t.get("bench") or {}).get("condition", "") in conds]
    if not paired:
        print("⚠  No MA/MB/MC traces — skipping Chart 11 (modality × building)")
        return

    data   = np.full((len(buildings), len(conds)), np.nan)
    counts = np.zeros_like(data, dtype=int)

    for bi, bld in enumerate(buildings):
        for ci, cond in enumerate(conds):
            grp = [t for t in paired
                   if t.get("_building") == bld
                   and (t.get("bench") or {}).get("condition") == cond]
            if grp:
                hits = sum(1 for t in grp if t.get("guid_match", False))
                data[bi, ci]   = hits / len(grp) * 100
                counts[bi, ci] = len(grp)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style_dark(fig, ax)

    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    # Cell annotations
    for bi in range(len(buildings)):
        for ci in range(len(conds)):
            if not np.isnan(data[bi, ci]):
                val = data[bi, ci]
                n   = counts[bi, ci]
                txt_col = "black" if 25 < val < 78 else "white"
                ax.text(ci, bi, f"{val:.0f}%\n(n={n})",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color=txt_col)
                # Delta from previous column
                if ci > 0 and not np.isnan(data[bi, ci - 1]):
                    delta = data[bi, ci] - data[bi, ci - 1]
                    ax.text(ci, bi + 0.38, f"{delta:+.0f}%",
                            ha="center", va="center",
                            fontsize=7.5, color="#b45309", style="italic")
            else:
                ax.text(ci, bi, "n/a", ha="center", va="center",
                        fontsize=10, color="#64748b")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors="black", labelsize=9)
    cbar.set_label("Top-1 Accuracy (%)", color="black", fontsize=10)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([cond_labels[c] for c in conds], fontsize=9, color="black")
    ax.set_yticks(range(len(buildings)))
    ax.set_yticklabels([bld_labels[b] for b in buildings], fontsize=9, color="black")
    ax.tick_params(colors="black", length=0)
    ax.set_title(
        f"{title}\n"
        "Yellow = Δ accuracy vs previous condition",
        fontsize=12, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved: {output_path}")
    plt.close()


# ── Chart 12: Paired 4D Ablation (MA vs MA-, MB vs MB-, MC vs MC-) ───────────

def plot_4d_paired_ablation(
    traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "4D Project Context Impact — Paired Modality Ablation",
) -> None:
    """
    Grouped bar chart showing the isolated impact of 4D project context.

    Each group is one modality level (Text-only / Photos / Full).
    Within each group: two bars — 4D ON (blue) vs 4D OFF (red).
    The delta annotation between bars shows the pure 4D contribution
    at each modality level, controlling for visual inputs.

    Requires runs under MA/MB/MC (4D ON) AND MA-/MB-/MC- (4D OFF).
    """
    pairs = [
        ("MA",  "MA-",  "Text-only\n(no images)"),
        ("MB",  "MB-",  "+ Site Photos"),
        ("MC",  "MC-",  "+ Floorplan"),
    ]
    color_on  = _PALETTE["text_4d"]   # blue  — 4D ON
    color_off = _PALETTE["no_4d"]     # red   — 4D OFF

    # Check we have any data at all
    all_conds = {(t.get("bench") or {}).get("condition", "") for t in traces}
    needed = {c for pair in pairs for c in (pair[0], pair[1])}
    present = needed & all_conds
    if not present:
        print("⚠  No MA/MB/MC or MA-/MB-/MC- traces — skipping Chart 12 (4D paired ablation)")
        return

    # Bucket traces per condition
    by_cond: Dict[str, List] = {c: [] for pair in pairs for c in (pair[0], pair[1])}
    for t in traces:
        c = (t.get("bench") or {}).get("condition", "")
        if c in by_cond:
            by_cond[c].append(t)

    n_groups = len(pairs)
    x = np.arange(n_groups)
    bar_w = 0.30
    offsets = [-bar_w / 2, bar_w / 2]

    fig, ax = plt.subplots(figsize=(11, 6))
    _style_dark(fig, ax)

    for label, color, side, cond_idx in [
        ("4D ON",  color_on,  0, 0),
        ("4D OFF", color_off, 1, 1),
    ]:
        accs, lo_errs, hi_errs, ns = [], [], [], []
        for with_cond, without_cond, _ in pairs:
            cond = with_cond if cond_idx == 0 else without_cond
            grp  = by_cond[cond]
            hits = sum(1 for t in grp if t.get("guid_match", False))
            n    = len(grp)
            acc  = hits / n if n else 0.0
            lo, hi = _wilson_ci(hits, n)
            accs.append(acc * 100)
            lo_errs.append((acc - lo) * 100)
            hi_errs.append((hi - acc) * 100)
            ns.append(n)

        bars = ax.bar(
            x + offsets[side], accs, bar_w,
            yerr=[lo_errs, hi_errs],
            label=label, color=color, alpha=0.88,
            edgecolor="none", linewidth=0.6,
            capsize=4, error_kw={"ecolor": "black", "linewidth": 1.0},
        )
        for bar, acc, n in zip(bars, accs, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc:.0f}%\n(n={n})" if n else "n/a",
                ha="center", va="bottom", fontsize=8.5,
                color="black", fontweight="bold",
            )

    # Delta annotation: 4D ON − 4D OFF for each group
    for gi, (with_cond, without_cond, _) in enumerate(pairs):
        grp_on  = by_cond[with_cond]
        grp_off = by_cond[without_cond]
        if not grp_on or not grp_off:
            continue
        acc_on  = sum(1 for t in grp_on  if t.get("guid_match")) / len(grp_on)  * 100
        acc_off = sum(1 for t in grp_off if t.get("guid_match")) / len(grp_off) * 100
        delta = acc_on - acc_off
        top = max(acc_on, acc_off) + 8
        ax.annotate(
            f"4D = {delta:+.0f}%",
            xy=(gi, top),
            ha="center", va="bottom",
            fontsize=9, color="#b45309", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=_DARK_BG, ec="#b45309", lw=0.8),
        )

    # Band backgrounds + x-axis group labels
    for gi, (_, _, group_label) in enumerate(pairs):
        ax.axvspan(gi - 0.45, gi + 0.45, alpha=0.06, color="#e2e8f0", zorder=0)
        ax.text(gi, -10, group_label, ha="center", va="top",
                fontsize=8.5, color="#94a3b8", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{w}\nvs {wo}" for w, wo, _ in pairs],
        fontsize=11, color="black", fontweight="bold",
    )
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.legend(
        fontsize=10, facecolor=_DARK_BG, edgecolor=_GRID_COL,
        labelcolor="black", loc="upper left",
    )
    ax.set_title(
        f"{title}\n"
        "Each group: same modality inputs — only 4D project context differs",
        fontsize=12, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved: {output_path}")
    plt.close()


# ── Chart 13: Full 6-condition × 2-profile dual view ─────────────────────────
#
# Two sub-plots in one figure:
#   Left  — Line chart: modality progression for 4 series
#              LoRA+4D (solid orange), LoRA−4D (dashed orange)
#              Prompt+4D (solid blue),  Prompt−4D (dashed blue)
#   Right — Grouped bar: 3 groups (Text / Images / Img+Plan)
#              4 bars per group: LoRA-ON | Prompt-ON | LoRA-OFF | Prompt-OFF
#              Shading: darker = 4D ON, lighter+hatched = 4D OFF

# Color palettes: warm (LoRA) and cool (Prompt)
_LORA_ON    = "#f97316"   # orange-500  — LoRA + 4D
_LORA_OFF   = "#fdba74"   # orange-300  — LoRA − 4D
_PROMPT_ON  = "#3b82f6"   # blue-500    — Prompt + 4D
_PROMPT_OFF = "#93c5fd"   # blue-300    — Prompt − 4D

_MODALITY_LEVELS = [
    ("MA",  "MA-", "Text only"),
    ("MB",  "MB-", "Images"),
    ("MC",  "MC-", "Img + Plan"),
]


def _acc_ci(traces_list):
    """Return (accuracy_pct, lo_err, hi_err, n) for a list of traces."""
    hits = sum(1 for t in traces_list if t.get("guid_match", False))
    n    = len(traces_list)
    acc  = hits / n if n else 0.0
    lo, hi = _wilson_ci(hits, n)
    return acc * 100, (acc - lo) * 100, (hi - acc) * 100, n


def plot_modality_dual_profile(
    traces: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Modality Ablation — LoRA vs Prompt × 4D Context",
) -> None:
    """
    Chart 13: Side-by-side line + grouped-bar figure showing all 12 runs.

    Left subplot  — Line chart with 4 series (profile × 4D):
      Shows modality progression from Text → Images → Img+Plan.

    Right subplot — Grouped bar chart (3 modality levels × 4 bars):
      LoRA-4D-ON | Prompt-4D-ON | LoRA-4D-OFF | Prompt-4D-OFF
      Warm (orange) = LoRA, Cool (blue) = Prompt.
      Solid = 4D ON, hatched lighter = 4D OFF.
    """
    # ── Bucket traces ────────────────────────────────────────────────────────
    all_conds = {(t.get("bench") or {}).get("condition", "") for t in traces}
    needed = {c for on, off, _ in _MODALITY_LEVELS for c in (on, off)}
    if not (needed & all_conds):
        print("⚠  No MA/MB/MC/MA-/MB-/MC- traces — skipping Chart 13")
        return

    by_cp: Dict[Tuple, List] = {}
    for t in traces:
        c = (t.get("bench") or {}).get("condition", "")
        p = t.get("_profile", "Other")
        if c in needed and p in ("LoRA", "Prompt"):
            by_cp.setdefault((c, p), []).append(t)

    def _get(cond, profile):
        return by_cp.get((cond, profile), [])

    # ── Figure layout ────────────────────────────────────────────────────────
    fig, (ax_line, ax_bar) = plt.subplots(
        1, 2, figsize=(15, 6),
        gridspec_kw={"width_ratios": [1, 1.3]},
    )
    for ax in (ax_line, ax_bar):
        _style_dark(fig, ax)

    x_line = np.arange(len(_MODALITY_LEVELS))
    xlabels = [lbl for _, _, lbl in _MODALITY_LEVELS]

    # ── Left: Line chart ─────────────────────────────────────────────────────
    series = [
        ("LoRA",   True,  _LORA_ON,   "LoRA + 4D",    "o",  "-"),
        ("LoRA",   False, _LORA_OFF,  "LoRA − 4D",    "o",  "--"),
        ("Prompt", True,  _PROMPT_ON, "Prompt + 4D",  "s",  "-"),
        ("Prompt", False, _PROMPT_OFF,"Prompt − 4D",  "s",  "--"),
    ]
    for profile, is_4d_on, color, label, marker, ls in series:
        accs, lo_errs, hi_errs = [], [], []
        for on_cond, off_cond, _ in _MODALITY_LEVELS:
            cond = on_cond if is_4d_on else off_cond
            grp  = _get(cond, profile)
            a, lo, hi, _ = _acc_ci(grp)
            accs.append(a); lo_errs.append(lo); hi_errs.append(hi)

        has_data = any(len(_get(on if is_4d_on else off, profile)) > 0
                       for on, off, _ in _MODALITY_LEVELS)
        if not has_data:
            continue

        ax_line.plot(x_line, accs, marker=marker, ls=ls, color=color,
                     linewidth=2.0, markersize=7, label=label, alpha=0.92)
        ax_line.fill_between(
            x_line,
            [a - lo for a, lo in zip(accs, lo_errs)],
            [a + hi for a, hi in zip(accs, hi_errs)],
            color=color, alpha=0.12,
        )

    # Shade between LoRA ON/OFF and Prompt ON/OFF to show 4D gap
    for profile, color_on, color_off in [("LoRA", _LORA_ON, _LORA_OFF),
                                          ("Prompt", _PROMPT_ON, _PROMPT_OFF)]:
        on_vals  = [_acc_ci(_get(on,  profile))[0] for on,  off, _ in _MODALITY_LEVELS]
        off_vals = [_acc_ci(_get(off, profile))[0] for on,  off, _ in _MODALITY_LEVELS]
        if any(v > 0 for v in on_vals) and any(v > 0 for v in off_vals):
            ax_line.fill_between(x_line, off_vals, on_vals,
                                 color=color_on, alpha=0.08, zorder=0)

    ax_line.set_xticks(x_line)
    ax_line.set_xticklabels(xlabels, fontsize=11, color="black")
    ax_line.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax_line.set_ylim(0, 110)
    ax_line.set_title("Modality Progression", fontsize=12, color="black", pad=8)
    ax_line.legend(fontsize=9, facecolor=_DARK_BG, edgecolor=_GRID_COL,
                   labelcolor="black", loc="lower right")

    # ── Right: Grouped bar chart ─────────────────────────────────────────────
    n_groups = len(_MODALITY_LEVELS)
    x_bar    = np.arange(n_groups)
    bar_w    = 0.18
    # 4 bars per group: LoRA-ON | Prompt-ON || LoRA-OFF | Prompt-OFF
    bar_defs = [
        ("LoRA",   True,  _LORA_ON,    "",     -1.5),
        ("Prompt", True,  _PROMPT_ON,  "",     -0.5),
        ("LoRA",   False, _LORA_OFF,   "////",  0.5),
        ("Prompt", False, _PROMPT_OFF, "////",  1.5),
    ]
    for profile, is_4d_on, color, hatch, offset in bar_defs:
        accs, lo_errs, hi_errs, ns = [], [], [], []
        for on_cond, off_cond, _ in _MODALITY_LEVELS:
            cond = on_cond if is_4d_on else off_cond
            grp  = _get(cond, profile)
            a, lo, hi, n = _acc_ci(grp)
            accs.append(a); lo_errs.append(lo); hi_errs.append(hi); ns.append(n)

        has_data = any(n > 0 for n in ns)
        if not has_data:
            continue

        label = f"{'LoRA' if profile == 'LoRA' else 'Prompt'} {'+ 4D' if is_4d_on else '− 4D'}"
        bars = ax_bar.bar(
            x_bar + offset * bar_w, accs, bar_w,
            yerr=[lo_errs, hi_errs],
            label=label, color=color, hatch=hatch,
            alpha=0.88 if is_4d_on else 0.70,
            edgecolor="none", linewidth=0.5,
            capsize=3, error_kw={"ecolor": "black", "linewidth": 0.8},
        )
        for bar, a, n in zip(bars, accs, ns):
            if n > 0:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{a:.0f}%",
                    ha="center", va="bottom", fontsize=7.5,
                    color="black", fontweight="bold",
                )

    # Vertical dividers between modality groups
    for xi in x_bar[1:]:
        ax_bar.axvline(xi - 0.5, color=_GRID_COL, linewidth=0.8, alpha=0.5)

    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(xlabels, fontsize=11, color="black")
    ax_bar.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax_bar.set_ylim(0, 120)
    ax_bar.set_title("Per-Condition Detail", fontsize=12, color="black", pad=8)
    ax_bar.legend(fontsize=8.5, facecolor=_DARK_BG, edgecolor=_GRID_COL,
                  labelcolor="black", loc="upper left", ncol=2)

    # ── Shared title ─────────────────────────────────────────────────────────
    fig.suptitle(
        f"{title}\n"
        "Orange = LoRA  ·  Blue = Prompt  ·  Solid = 4D ON  ·  Hatched = 4D OFF",
        fontsize=12, fontweight="bold", color="black", y=1.02,
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✓ Saved: {output_path}")
    plt.close()


# ── Modality analysis entry point ────────────────────────────────────────────

def generate_modality_plots(
    output_dir: str = "plots/archive/modality",
    roots: Optional[List[str]] = None,
    run_filter: Optional[str] = None,
    traces_jsonl: Optional[List[str]] = None,
) -> None:
    """
    Generate all modality-importance charts and save to *output_dir*.

    Two loading modes:
      - traces_jsonl: list of JSONL paths (profile inferred from filename)
      - roots/run_filter: load from *.trace.json subdirectory trees (legacy)

    Args:
        output_dir:    Destination folder (created if absent).
        roots:         Override trace root directories (legacy mode).
        run_filter:    Only include runs whose name contains this substring (legacy).
        traces_jsonl:  List of JSONL file paths to load (new mode).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Modality Importance Analysis")
    print(f"{'='*60}")

    if traces_jsonl:
        traces = _load_traces_from_jsonl(traces_jsonl)
    else:
        traces = _load_traces_from_files(roots=roots, run_filter=run_filter)
    if not traces:
        print("⚠  No traces loaded — check TRACE_ROOTS paths")
        return

    from collections import Counter
    cond_counts = Counter(
        (t.get("bench") or {}).get("condition", "?") for t in traces
    )
    print(f"Loaded {len(traces)} traces | conditions: {dict(sorted(cond_counts.items()))}\n")

    plot_modality_stack(
        traces,
        output_path=f"{output_dir}/9_modality_stack_MA_MB_MC.png",
    )
    plot_4d_context_impact(
        traces,
        output_path=f"{output_dir}/10_4d_context_impact_A1_C3.png",
    )
    plot_modality_x_building(
        traces,
        output_path=f"{output_dir}/11_modality_x_building.png",
    )
    plot_4d_paired_ablation(
        traces,
        output_path=f"{output_dir}/12_4d_paired_ablation.png",
    )
    plot_modality_dual_profile(
        traces,
        output_path=f"{output_dir}/13_modality_dual_profile.png",
    )

    print(f"\n✓ Modality plots saved to: {output_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualizations.py <traces_jsonl> [v2_traces_jsonl] [before_traces_jsonl] [output_dir]")
        print("       Use script/generate_plots.py for a full CLI interface (including --modality).")
        sys.exit(1)

    traces_path = sys.argv[1]
    v2_traces_path = sys.argv[2] if len(sys.argv) > 2 else None
    before_traces_path = sys.argv[3] if len(sys.argv) > 3 else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "logs/plots"

    generate_all_plots(traces_path, v2_traces_path, before_traces_path, output_dir)
