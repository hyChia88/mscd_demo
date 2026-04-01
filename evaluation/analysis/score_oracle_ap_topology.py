#!/usr/bin/env python3
"""Summarize AP-held-out topology-faithful oracle runs.

This script reads already-generated Track B-2 oracle traces and produces:
1. overall strategy comparison for the current implementation
2. P1-only vs full-topology delta
3. per-slice metrics by relation multiplicity and U1~U6 universe taxonomy
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mscd")

from evaluation.analysis.compare_results import compute_metrics, load_traces  # type: ignore


DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "oracle_ap_heldout"
DEFAULT_TRACES = {
    "p0_only": DEFAULT_OUT_DIR / "strategy_p0_only" / "traces_20260401_020952_v2_lora_p0_only.jsonl",
    "p1_only_strategy": DEFAULT_OUT_DIR / "strategy_p1_only" / "traces_20260401_021220_v2_lora_p1_only.jsonl",
    "p0_intersect_p1": DEFAULT_OUT_DIR / "strategy_p0_intersect_p1" / "traces_20260401_021515_v2_lora_p0_intersect_p1.jsonl",
    "p0_union_p1": DEFAULT_OUT_DIR / "strategy_p0_union_p1" / "traces_20260401_021932_v2_lora_p0_union_p1.jsonl",
    "p1_only_upper_bound": DEFAULT_OUT_DIR / "p1_only_upper_bound" / "traces_20260401_022455_v2_lora_p1_only.jsonl",
    "full_topology_union": DEFAULT_OUT_DIR / "all_cases" / "traces_20260401_014252_v2_lora_p0_union_p1.jsonl",
}

U_ORDER = ["U1", "U2", "U3", "U4", "U5", "U6"]
MULT_ORDER = ["1-rel", "2-rel", "3-rel"]


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pool_stats(traces: List[dict]) -> dict:
    finals = [t.get("final_pool_size") for t in traces if t.get("final_pool_size") is not None]
    initials = [t.get("initial_pool_size") for t in traces if t.get("initial_pool_size") is not None]
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


def _topology_family(rels: List[dict]) -> str:
    preds = [str(r.get("predicate", "?")) for r in rels]
    pred_hist = Counter(preds)
    n = len(rels)
    if n == 1:
        return f"singleton:{preds[0]}"
    if n == 2:
        if pred_hist == Counter({"FILLS": 1, "NEXT_TO": 1}):
            return "paired:FILLS+NEXT_TO"
        return "paired:other"
    if n == 3:
        if pred_hist == Counter({"FILLS": 1, "NEXT_TO": 2}):
            objs = [str(r.get("object_type", "?")) for r in rels if r.get("predicate") == "NEXT_TO"]
            if len(set(objs)) > 1:
                return "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)"
            return "triad:FILLS+NEXT_TO+NEXT_TO"
        return "triad:other"
    return f"{n}-rel:other"


def _universe_key(rels: List[dict]) -> str:
    family = _topology_family(rels)
    if family == "singleton:CONNECTS_TO":
        return "U1"
    if family == "singleton:ADJACENT_TO":
        return "U2"
    if family == "paired:FILLS+NEXT_TO":
        return "U3"
    if family == "triad:FILLS+NEXT_TO+NEXT_TO":
        return "U4"
    if family == "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)":
        return "U5"
    return "U6"


def _universe_label(key: str) -> str:
    return {
        "U1": "U1 Wall-Connectivity",
        "U2": "U2 Adjacency-Singleton",
        "U3": "U3 Opening-Paired",
        "U4": "U4 Symmetric-Triad",
        "U5": "U5 Mixed-Triad",
        "U6": "U6 Rare/Edge",
    }[key]


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def _format_num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if value < 1 else f"{value:.1f}"


def _slice_cases(cases: List[dict]) -> Dict[str, dict]:
    case_meta: Dict[str, dict] = {}
    for case in cases:
        rels = list(case.get("labels", {}).get("constraints", {}).get("spatial_relations", []) or [])
        case_id = case["case_id"]
        case_meta[case_id] = {
            "multiplicity": f"{len(rels)}-rel",
            "family": _topology_family(rels),
            "universe": _universe_key(rels),
        }
    return case_meta


def _trace_subset(traces: List[dict], case_ids: Iterable[str]) -> List[dict]:
    keep = set(case_ids)
    subset = []
    for trace in traces:
        cid = trace.get("scenario_id", trace.get("scenario", {}).get("id", ""))
        if cid in keep:
            subset.append(trace)
    return subset


def _score_trace_set(traces: List[dict]) -> dict:
    overall = compute_metrics(traces)
    return {
        "overall": overall,
        "pool_stats": _pool_stats(traces),
    }


def _default_trace_map() -> Dict[str, Path]:
    return {k: v for k, v in DEFAULT_TRACES.items() if v.exists()}


def _parse_kv_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected KEY=PATH, got {value}")
    k, raw = value.split("=", 1)
    return k.strip(), Path(raw.strip())


def _build_rows(metrics_by_group: Dict[str, dict], order: List[str]) -> List[dict]:
    rows = []
    for key in order:
        if key not in metrics_by_group:
            continue
        item = metrics_by_group[key]
        overall = item["overall"]
        pool = item["pool_stats"]
        rows.append(
            {
                "group": key,
                "top10_pct": overall.get("top10_pct"),
                "top1_pct": overall.get("top1_pct"),
                "mrr": overall.get("mrr"),
                "avg_pool": overall.get("avg_pool"),
                "median_final_pool": pool.get("median_final_pool"),
                "avg_search_space_reduction": pool.get("avg_search_space_reduction"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--trace", action="append", default=[], metavar="GROUP=PATH")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    trace_map = _default_trace_map()
    for item in args.trace:
        key, path = _parse_kv_arg(item)
        trace_map[key] = path
    if not trace_map:
        raise SystemExit("No trace files found or provided.")

    cases = _load_jsonl(args.cases)
    case_meta = _slice_cases(cases)

    metrics_by_group: Dict[str, dict] = {}
    for group, path in trace_map.items():
        if not path.exists():
            raise FileNotFoundError(path)
        traces = load_traces(str(path))
        metrics_by_group[group] = _score_trace_set(traces)

    slice_case_ids = {
        "multiplicity": {key: [cid for cid, meta in case_meta.items() if meta["multiplicity"] == key] for key in MULT_ORDER},
        "universe": {key: [cid for cid, meta in case_meta.items() if meta["universe"] == key] for key in U_ORDER},
    }

    sliced: Dict[str, Dict[str, dict]] = {"multiplicity": {}, "universe": {}}
    for slice_kind, buckets in slice_case_ids.items():
        for bucket_key, case_ids in buckets.items():
            bucket_results = {}
            for group, path in trace_map.items():
                traces = _trace_subset(load_traces(str(path)), case_ids)
                bucket_results[group] = _score_trace_set(traces)
            sliced[slice_kind][bucket_key] = bucket_results

    phase2a_order = ["p0_only", "p1_only_strategy", "p0_intersect_p1", "p0_union_p1"]
    phase2b_order = ["p1_only_upper_bound", "full_topology_union"]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "overall": metrics_by_group,
        "sliced": sliced,
    }
    (out_dir / "oracle_topology_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# AP Held-out Oracle Topology Summary",
        "",
        "## Phase 2A — Current-System Strategy Search",
        "",
        "| Strategy | Top-10 | Top-1 | MRR@10 | Avg Pool | Med Pool | Reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _build_rows(metrics_by_group, phase2a_order):
        lines.append(
            f"| {row['group']} | "
            f"{_format_pct(row['top10_pct'])} | "
            f"{_format_pct(row['top1_pct'])} | "
            f"{_format_num(row['mrr'])} | "
            f"{_format_num(row['avg_pool'])} | "
            f"{_format_num(row['median_final_pool'])} | "
            f"{_format_pct((row['avg_search_space_reduction'] or 0) * 100)} |"
        )

    lines.extend(
        [
            "",
            "## Phase 2B — Topology-Faithful Oracle",
            "",
            "| Oracle Setting | Top-10 | Top-1 | MRR@10 | Avg Pool | Med Pool | Reduction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _build_rows(metrics_by_group, phase2b_order):
        lines.append(
            f"| {row['group']} | "
            f"{_format_pct(row['top10_pct'])} | "
            f"{_format_pct(row['top1_pct'])} | "
            f"{_format_num(row['mrr'])} | "
            f"{_format_num(row['avg_pool'])} | "
            f"{_format_num(row['median_final_pool'])} | "
            f"{_format_pct((row['avg_search_space_reduction'] or 0) * 100)} |"
        )

    if "p1_only_upper_bound" in metrics_by_group and "full_topology_union" in metrics_by_group:
        p1 = metrics_by_group["p1_only_upper_bound"]
        full = metrics_by_group["full_topology_union"]
        lines.extend(
            [
                "",
                "## Full-Topology Minus P1-only",
                "",
                f"- Top-10 delta: {(full['overall']['top10_pct'] - p1['overall']['top10_pct']):.1f} pts",
                f"- Top-1 delta: {(full['overall']['top1_pct'] - p1['overall']['top1_pct']):.1f} pts",
                f"- MRR delta: {(full['overall']['mrr'] - p1['overall']['mrr']):.4f}",
                f"- Median pool delta: {(full['pool_stats']['median_final_pool'] - p1['pool_stats']['median_final_pool']):.1f}",
                f"- Reduction delta: {((full['pool_stats']['avg_search_space_reduction'] - p1['pool_stats']['avg_search_space_reduction']) * 100):.1f} pts",
            ]
        )

    for slice_kind, keys in [("multiplicity", MULT_ORDER), ("universe", U_ORDER)]:
        lines.extend(
            [
                "",
                f"## By {slice_kind.title()}",
                "",
            ]
        )
        for key in keys:
            bucket = sliced[slice_kind].get(key)
            if not bucket:
                continue
            label = key if slice_kind == "multiplicity" else _universe_label(key)
            lines.extend(
                [
                    f"### {label}",
                    "",
                    "| Run | Top-10 | Top-1 | MRR@10 | Med Pool | Reduction |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            bucket_order = [k for k in phase2b_order if k in bucket] or list(bucket.keys())
            for group in bucket_order:
                row = bucket[group]
                overall = row["overall"]
                pool = row["pool_stats"]
                lines.append(
                    f"| {group} | "
                    f"{_format_pct(overall.get('top10_pct'))} | "
                    f"{_format_pct(overall.get('top1_pct'))} | "
                    f"{_format_num(overall.get('mrr'))} | "
                    f"{_format_num(pool.get('median_final_pool'))} | "
                    f"{_format_pct((pool.get('avg_search_space_reduction') or 0) * 100)} |"
                )
            lines.append("")

    (out_dir / "oracle_topology_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'oracle_topology_summary.md'}")
    print(f"Wrote {out_dir / 'oracle_topology_metrics.json'}")


if __name__ == "__main__":
    main()
