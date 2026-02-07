#!/usr/bin/env python3
"""
Compare Evaluation Results

Reads eval JSON files from logs/evaluations/ and prints a side-by-side
comparison table. Works with V1 (main_mcp.py) JSON outputs and V2
(script/run.py) CSV summaries.

Usage:
  python script/compare_results.py                    # Show all results
  python script/compare_results.py --latest           # Show only the most recent batch
  python script/compare_results.py --latest 4         # Show the 4 most recent results
  python script/compare_results.py --dir logs/custom  # Custom directory
  python script/compare_results.py --csv results.csv  # Export to CSV
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
    args = parser.parse_args()

    # Resolve directory relative to project root
    project_root = Path(__file__).resolve().parent.parent
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
