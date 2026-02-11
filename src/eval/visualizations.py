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
    1. scenario.bench.condition (legacy format from synthetic dataset)
    2. run_id suffix (e.g., "20260211_011004_v2_prompt_A1" -> "A1")

    Args:
        trace: Evaluation trace dictionary

    Returns:
        Condition string (A1-C3) or "Unknown"
    """
    # Try legacy format first
    cond = trace.get("scenario", {}).get("bench", {}).get("condition")
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

    conditions = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    exp_names = list(experiments.keys())

    # Calculate accuracy for each experiment × condition
    data = []
    for exp_name in exp_names:
        traces = experiments[exp_name]

        # Group by condition
        by_condition = {}
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

    # Add condition group separators
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.6, linewidth=2)
    ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.6, linewidth=2)

    # Group labels
    ax.text(1, 105, 'Text Only\n(No Images)', ha='center', fontsize=10,
            style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(4, 105, 'Images + Text\n(No Floorplan)', ha='center', fontsize=10,
            style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(7.5, 105, 'Full Multimodal\n(Images + Floorplan)', ha='center', fontsize=10,
            style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    # Add condition details legend
    condition_info = (
        "Condition Details:\n"
        "A1: -Img -Plan -4D ~Blur | A2: -Img -Plan +4D Clear | A3: -Img -Plan +4D+ Clear\n"
        "B1: +Img -Plan -4D ~Blur | B2: +Img -Plan +4D Clear | B3: +Img -Plan -4D Clear\n"
        "C1: +Img +Plan -4D Clear | C2: +Img +Plan +4D Clear | C3: +Img +Plan +4D+ Clear+CLIP"
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualizations.py <traces_jsonl> [v2_traces_jsonl] [before_traces_jsonl] [output_dir]")
        sys.exit(1)

    traces_path = sys.argv[1]
    v2_traces_path = sys.argv[2] if len(sys.argv) > 2 else None
    before_traces_path = sys.argv[3] if len(sys.argv) > 3 else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "logs/plots"

    generate_all_plots(traces_path, v2_traces_path, before_traces_path, output_dir)
