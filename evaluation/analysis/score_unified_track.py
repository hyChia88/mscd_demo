#!/usr/bin/env python3
"""Score Track B unified end-to-end outputs and write comparable summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mscd")

from evaluation.analysis.compare_results import (  # type: ignore
    build_tier_lookup,
    classify_model,
    classify_tier,
    compute_field_accuracy,
    compute_metrics,
    load_precomputed,
    load_traces,
)
from evaluation.track_registry import GROUP_DISPLAY, METRICS_DIR, TRACK_B2_ORDER, TRACK_B_ORDER


DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_unified_test.jsonl"


def _parse_kv_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {value}")
    key, raw = value.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key or not raw:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {value}")
    return key, raw


def _load_cases(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _subset_metrics(traces: List[dict]) -> dict:
    return compute_metrics(traces)


def _pool_stats(traces: List[dict]) -> dict:
    initials = [t.get("initial_pool_size") for t in traces if t.get("initial_pool_size") is not None]
    finals = [t.get("final_pool_size") for t in traces if t.get("final_pool_size") is not None]
    reductions = []
    for trace in traces:
        initial = trace.get("initial_pool_size")
        final = trace.get("final_pool_size")
        if initial is None or final is None or initial == 0:
            continue
        reductions.append(1.0 - (final / initial))
    return {
        "avg_initial_pool": round(sum(initials) / len(initials), 1) if initials else None,
        "median_initial_pool": round(statistics.median(initials), 1) if initials else None,
        "avg_final_pool": round(sum(finals) / len(finals), 1) if finals else None,
        "median_final_pool": round(statistics.median(finals), 1) if finals else None,
        "avg_search_space_reduction": round(sum(reductions) / len(reductions), 4) if reductions else None,
    }


def score_group(
    *,
    group_key: str,
    traces_path: Path,
    cases: List[dict],
    tier_lookup: Dict[str, str],
    precomputed_path: Path | None = None,
) -> dict:
    traces = load_traces(str(traces_path))
    overall = _subset_metrics(traces)

    by_ifc: Dict[str, dict] = {}
    for ifc_model in sorted({classify_model(t.get("scenario_id", t.get("scenario", {}).get("id", ""))) for t in traces}):
        subset = [
            t for t in traces
            if classify_model(t.get("scenario_id", t.get("scenario", {}).get("id", ""))) == ifc_model
        ]
        by_ifc[ifc_model] = _subset_metrics(subset)

    by_tier: Dict[str, dict] = {}
    for tier in ("T1", "T2", "T3"):
        subset = [t for t in traces if classify_tier(t, tier_lookup) == tier]
        by_tier[tier] = _subset_metrics(subset)

    field_accuracy = None
    if precomputed_path is not None:
        field_accuracy = compute_field_accuracy(load_precomputed(str(precomputed_path)), cases)

    return {
        "group": group_key,
        "display_name": GROUP_DISPLAY.get(group_key, group_key),
        "traces_path": str(traces_path),
        "precomputed_path": str(precomputed_path) if precomputed_path else None,
        "overall": overall,
        "pool_stats": _pool_stats(traces),
        "field_accuracy": field_accuracy,
        "by_ifc_model": by_ifc,
        "by_tier": by_tier,
    }


def _summary_rows(metrics_by_group: Dict[str, dict], group_order: List[str]) -> List[dict]:
    rows = []
    order = group_order + [g for g in metrics_by_group if g not in group_order]
    seen = set()
    for group in order:
        if group in seen or group not in metrics_by_group:
            continue
        seen.add(group)
        metrics = metrics_by_group[group]
        overall = metrics["overall"]
        field = metrics.get("field_accuracy") or {}
        pool = metrics.get("pool_stats") or {}
        rows.append(
            {
                "group": group,
                "display_name": metrics["display_name"],
                "gt_in_pct": overall.get("gt_in_pct", 0),
                "top10_pct": overall.get("top10_pct", 0),
                "top1_pct": overall.get("top1_pct", 0),
                "mrr": overall.get("mrr", 0),
                "avg_pool": overall.get("avg_pool", 0),
                "median_final_pool": pool.get("median_final_pool"),
                "avg_search_space_reduction": pool.get("avg_search_space_reduction"),
                "storey_acc": field.get("storey_acc"),
                "ifc_class_acc": field.get("ifc_class_acc"),
                "sr_rate": field.get("sr_rate"),
            }
        )
    return rows


def write_outputs(
    metrics_by_group: Dict[str, dict],
    out_dir: Path,
    *,
    summary_prefix: str,
    metric_suffix: str,
    title: str,
    group_order: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for group, metrics in metrics_by_group.items():
        _json_dump(out_dir / f"{group}__{metric_suffix}.json", metrics)

    rows = _summary_rows(metrics_by_group, group_order)
    csv_path = out_dir / f"{summary_prefix}_summary.csv"
    md_path = out_dir / f"{summary_prefix}_summary.md"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "display_name",
                "gt_in_pct",
                "top10_pct",
                "top1_pct",
                "mrr",
                "avg_pool",
                "median_final_pool",
                "avg_search_space_reduction",
                "storey_acc",
                "ifc_class_acc",
                "sr_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {title}",
        "",
        "| Group | GT-in-Pool | Top-10 | Top-1 | MRR@10 | Avg Pool | Med Pool | Reduction | Storey Acc | IFC Acc | SR Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def fmt_opt(value):
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.1f}" if value > 1 else f"{value:.1%}"
            return str(value)

        lines.append(
            f"| {row['display_name']} | "
            f"{row['gt_in_pct']:.1f}% | "
            f"{row['top10_pct']:.1f}% | "
            f"{row['top1_pct']:.1f}% | "
            f"{row['mrr']:.4f} | "
            f"{row['avg_pool']:.1f} | "
            f"{fmt_opt(row['median_final_pool'])} | "
            f"{fmt_opt(row['avg_search_space_reduction'])} | "
            f"{fmt_opt(row['storey_acc'])} | "
            f"{fmt_opt(row['ifc_class_acc'])} | "
            f"{fmt_opt(row['sr_rate'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--trace", action="append", default=[], metavar="GROUP=PATH")
    parser.add_argument("--precomputed", action="append", default=[], metavar="GROUP=PATH")
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--summary-prefix", default="track_b")
    parser.add_argument("--metric-suffix", default="unified_metrics")
    parser.add_argument("--title", default="Track B-1 — Unified End-to-End Evaluation")
    parser.add_argument(
        "--order",
        choices=("track_b", "track_b2"),
        default="track_b",
        help="Display ordering preset for summary tables.",
    )
    args = parser.parse_args()

    trace_map = {k: Path(v) for k, v in (_parse_kv_arg(item) for item in args.trace)}
    precomputed_map = {k: Path(v) for k, v in (_parse_kv_arg(item) for item in args.precomputed)}
    if not trace_map:
        raise SystemExit("No trace files provided. Use --trace GROUP=PATH.")

    cases = _load_cases(args.cases)
    tier_lookup = build_tier_lookup(cases)

    metrics_by_group: Dict[str, dict] = {}
    for group, trace_path in trace_map.items():
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        precomputed_path = precomputed_map.get(group)
        if precomputed_path is not None and not precomputed_path.exists():
            raise FileNotFoundError(f"Precomputed file not found: {precomputed_path}")
        metrics = score_group(
            group_key=group,
            traces_path=trace_path,
            cases=cases,
            tier_lookup=tier_lookup,
            precomputed_path=precomputed_path,
        )
        metrics_by_group[group] = metrics
        overall = metrics["overall"]
        print(
            f"[{group}] GT-in-Pool={overall['gt_in_pct']:.1f}% "
            f"Top-10={overall['top10_pct']:.1f}% "
            f"Top-1={overall['top1_pct']:.1f}% "
            f"AvgPool={overall['avg_pool']:.1f}"
        )

    group_order = TRACK_B_ORDER if args.order == "track_b" else TRACK_B2_ORDER
    write_outputs(
        metrics_by_group,
        args.out_dir,
        summary_prefix=args.summary_prefix,
        metric_suffix=args.metric_suffix,
        title=args.title,
        group_order=group_order,
    )
    print("\nTrack B scoring complete.")
    print(f"  Metrics dir: {args.out_dir}")


if __name__ == "__main__":
    main()
