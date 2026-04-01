#!/usr/bin/env python3
"""AP held-out oracle waterfall and gap analysis.

This is the strict oracle companion for Track B-2.

It answers:
1. Under perfect extraction, how small can the candidate pool become at
   P1 -> 1-hop -> 2-hop?
2. How far are current G-series / Gemini Track B-2 runs from that oracle?
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mscd")

from evaluation.analysis.compare_results import compute_metrics  # type: ignore
from evaluation.oracle_waterfall import (  # type: ignore
    discover_gt_edges,
    discover_hop2_edges,
    get_total_elements,
    load_config,
    lookup_gt_element,
    run_1hop_query,
    run_2hop_query,
    run_p1_query,
)
from evaluation.track_registry import GROUP_DISPLAY


DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
DEFAULT_TRACE_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "ap_e2e"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "oracle_ap_heldout"
DEFAULT_GROUPS = [
    "g0_canonical",
    "g1_fullaug",
    "g2_fullaug_lowlr",
    "g3_fullaug_r32",
    "gemini_ap",
]


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _median(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.median(values) if values else None


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def _find_latest_trace(trace_root: Path, group: str) -> Optional[Path]:
    group_dir = trace_root / group
    if not group_dir.exists():
        return None
    traces = sorted(group_dir.glob("traces_*.jsonl"))
    return traces[-1] if traces else None


def _pool_stats(records: List[dict], key: str) -> dict:
    vals = [r[key] for r in records if r.get(key) is not None]
    return {
        "avg": round(_mean(vals), 1) if vals else None,
        "median": round(_median(vals), 1) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def _fmt_num(value: Optional[float], pct: bool = False) -> str:
    if value is None:
        return "-"
    if pct:
        return f"{value:.1%}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value)}"
    return f"{value:.1f}"


def _actual_metrics_from_trace(trace_path: Path) -> dict:
    traces = _load_jsonl(trace_path)
    overall = compute_metrics(traces)
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
        "overall": overall,
        "pool_stats": {
            "avg_initial_pool": round(_mean(initials), 1) if initials else None,
            "median_initial_pool": round(_median(initials), 1) if initials else None,
            "avg_final_pool": round(_mean(finals), 1) if finals else None,
            "median_final_pool": round(_median(finals), 1) if finals else None,
            "avg_search_space_reduction": round(_mean(reductions), 4) if reductions else None,
        },
    }


def run_oracle(cases: List[dict]) -> List[dict]:
    cfg = load_config()
    neo4j_cfg = cfg.get("neo4j", {})

    from py2neo import Graph

    neo4j_conn = Graph(
        neo4j_cfg.get("uri", "bolt://localhost:7687"),
        auth=("neo4j", neo4j_cfg.get("password", "password")),
    )

    results: List[dict] = []
    for case in cases:
        case_id = case["case_id"]
        gt = case.get("ground_truth", {})
        gt_guid = gt.get("target_guid")
        model_key = "AP"

        gt_node = lookup_gt_element(neo4j_conn, gt_guid, model_key)
        if not gt_node:
            results.append(
                {
                    "case_id": case_id,
                    "error": "gt_not_found_in_neo4j",
                }
            )
            continue

        gt_type = gt_node["ifc_type"]
        gt_storey = gt_node["storey"] or ""
        total = get_total_elements(neo4j_conn, model_key)

        p1_cands = run_p1_query(neo4j_conn, gt_type, gt_storey, model_key)
        p1_pool = len(p1_cands)
        p1_gt_in = gt_guid in {c["guid"] for c in p1_cands}

        gt_edges = discover_gt_edges(neo4j_conn, gt_guid, model_key)
        all_1hop = {}
        best_1hop = None

        for edge_type, neighbors in gt_edges.items():
            if edge_type == "CONTINUOUS":
                cands = run_1hop_query(
                    neo4j_conn, gt_type, gt_storey, model_key, "CONTINUOUS", "", ""
                )
            else:
                obj_type = neighbors[0]["ifc_type"]
                cands = run_1hop_query(
                    neo4j_conn, gt_type, gt_storey, model_key, edge_type, obj_type, ""
                )

            pool = len(cands)
            gt_in = gt_guid in {c["guid"] for c in cands}
            all_1hop[edge_type] = {
                "pool": pool,
                "gt_in": gt_in,
                "obj_type": neighbors[0].get("ifc_type", "") if edge_type != "CONTINUOUS" else "",
            }
            if gt_in and (best_1hop is None or pool < best_1hop["pool"]):
                best_1hop = {
                    "edge_type": edge_type,
                    "pool": pool,
                    "neighbors": neighbors,
                }

        best_2hop = None
        if best_1hop and best_1hop["edge_type"] != "CONTINUOUS":
            ref_guids = [n["guid"] for n in best_1hop["neighbors"]]
            hop2_edges = discover_hop2_edges(neo4j_conn, ref_guids, model_key)
            for edge_type2, neighbors2 in hop2_edges.items():
                obj_type2 = neighbors2[0]["ifc_type"]
                cands2 = run_2hop_query(
                    neo4j_conn,
                    gt_type,
                    gt_storey,
                    model_key,
                    best_1hop["edge_type"],
                    best_1hop["neighbors"][0].get("ifc_type", ""),
                    "",
                    edge_type2,
                    obj_type2,
                    "",
                )
                pool2 = len(cands2)
                gt_in2 = gt_guid in {c["guid"] for c in cands2}
                if gt_in2 and (best_2hop is None or pool2 < best_2hop["pool"]):
                    best_2hop = {
                        "edge_type": f"{best_1hop['edge_type']}→{edge_type2}",
                        "pool": pool2,
                    }

        results.append(
            {
                "case_id": case_id,
                "gt_guid": gt_guid,
                "gt_type": gt_type,
                "gt_storey": gt_storey,
                "total_pool": total,
                "p1_pool": p1_pool,
                "p1_gt_in": p1_gt_in,
                "has_edges": bool(gt_edges),
                "edge_types_found": list(gt_edges.keys()),
                "best_1hop_edge": best_1hop["edge_type"] if best_1hop else None,
                "best_1hop_pool": best_1hop["pool"] if best_1hop else None,
                "best_2hop_edge": best_2hop["edge_type"] if best_2hop else None,
                "best_2hop_pool": best_2hop["pool"] if best_2hop else None,
                "all_1hop": all_1hop,
            }
        )

    return results


def build_summary(records: List[dict]) -> dict:
    valid = [r for r in records if "error" not in r]
    with_edges = [r for r in valid if r.get("has_edges")]
    with_1hop = [r for r in valid if r.get("best_1hop_pool") is not None]
    with_2hop = [r for r in valid if r.get("best_2hop_pool") is not None]

    return {
        "n": len(valid),
        "with_edges": len(with_edges),
        "with_1hop": len(with_1hop),
        "with_2hop": len(with_2hop),
        "total_pool": _pool_stats(valid, "total_pool"),
        "p1_pool": _pool_stats(valid, "p1_pool"),
        "best_1hop_pool": _pool_stats(with_1hop, "best_1hop_pool"),
        "best_2hop_pool": _pool_stats(with_2hop, "best_2hop_pool"),
        "p1_gt_in_rate": round(sum(1 for r in valid if r.get("p1_gt_in")) / len(valid), 4) if valid else None,
    }


def compare_to_actual(records: List[dict], trace_root: Path, groups: List[str]) -> List[dict]:
    summary = build_summary(records)
    oracle_h1_med = summary["best_1hop_pool"]["median"]
    oracle_h2_med = summary["best_2hop_pool"]["median"]
    comparisons = []
    for group in groups:
        trace_path = _find_latest_trace(trace_root, group)
        if trace_path is None:
            continue
        actual = _actual_metrics_from_trace(trace_path)
        pool = actual["pool_stats"]
        overall = actual["overall"]
        final_med = pool.get("median_final_pool")
        h1_gap = (final_med - oracle_h1_med) if final_med is not None and oracle_h1_med is not None else None
        h2_gap = (final_med - oracle_h2_med) if final_med is not None and oracle_h2_med is not None else None
        comparisons.append(
            {
                "group": group,
                "display_name": GROUP_DISPLAY.get(group, group),
                "trace_path": str(trace_path),
                "gt_in_pct": overall.get("gt_in_pct"),
                "top10_pct": overall.get("top10_pct"),
                "top1_pct": overall.get("top1_pct"),
                "mrr": overall.get("mrr"),
                "avg_final_pool": pool.get("avg_final_pool"),
                "median_final_pool": final_med,
                "avg_search_space_reduction": pool.get("avg_search_space_reduction"),
                "oracle_h1_median_gap": h1_gap,
                "oracle_h2_median_gap": h2_gap,
            }
        )
    return comparisons


def write_outputs(out_dir: Path, records: List[dict], summary: dict, comparisons: List[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "oracle_ap_heldout_records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "oracle_ap_heldout_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "oracle_ap_heldout_vs_models.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# AP Held-out Oracle Waterfall",
        "",
        f"- Cases: {summary['n']}",
        f"- With edges: {summary['with_edges']}",
        f"- With 1-hop oracle: {summary['with_1hop']}",
        f"- With 2-hop oracle: {summary['with_2hop']}",
        "",
        "## Oracle Waterfall",
        "",
        f"- Total pool median: {_fmt_num(summary['total_pool']['median'])}",
        f"- P1 pool median: {_fmt_num(summary['p1_pool']['median'])}",
        f"- Best 1-hop pool median: {_fmt_num(summary['best_1hop_pool']['median'])}",
        f"- Best 2-hop pool median: {_fmt_num(summary['best_2hop_pool']['median'])}",
        f"- P1 GT-in rate: {_fmt_num(summary['p1_gt_in_rate'], pct=True)}",
        "",
        "## Oracle vs Track B-2 Models",
        "",
        "| Group | GT-in-Pool | Top-10 | Top-1 | MRR@10 | Med Pool | Reduction | Gap vs Oracle 1-hop | Gap vs Oracle 2-hop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in comparisons:
        lines.append(
            f"| {row['display_name']} | "
            f"{row['gt_in_pct']:.1f}% | "
            f"{row['top10_pct']:.1f}% | "
            f"{row['top1_pct']:.1f}% | "
            f"{row['mrr']:.4f} | "
            f"{_fmt_num(row['median_final_pool'])} | "
            f"{_fmt_num(row['avg_search_space_reduction'], pct=True)} | "
            f"{_fmt_num(row['oracle_h1_median_gap'])} | "
            f"{_fmt_num(row['oracle_h2_median_gap'])} |"
        )

    (out_dir / "oracle_ap_heldout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--groups", nargs="*", default=DEFAULT_GROUPS)
    args = parser.parse_args()

    cases = _load_jsonl(args.cases)
    records = run_oracle(cases)
    summary = build_summary(records)
    comparisons = compare_to_actual(records, args.trace_root, args.groups)
    write_outputs(args.out_dir, records, summary, comparisons)

    print(f"Wrote oracle outputs to {args.out_dir}")
    print(
        "Oracle medians: "
        f"total={_fmt_num(summary['total_pool']['median'])}, "
        f"P1={_fmt_num(summary['p1_pool']['median'])}, "
        f"1-hop={_fmt_num(summary['best_1hop_pool']['median'])}, "
        f"2-hop={_fmt_num(summary['best_2hop_pool']['median'])}"
    )


if __name__ == "__main__":
    main()
