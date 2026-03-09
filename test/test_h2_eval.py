"""
H2 Hard-Negative Evaluation — Priority-0 Retrieval on Topology Test Cases

Measures the benefit of Priority-0 (topology) queries over attribute baseline:

  Attribute baseline : Top-1 = 1/N (e.g., 2.2% for 46 identical IfcWindows)
  Neo4j Priority-0   : returns a small pool (1-10 candidates) via edge traversal

Metrics reported per case:
  pre_pool_size  — attribute pool before retrieval (N identical elements)
  post_pool_size — candidates returned by retrieval system
  gt_in_pool     — True if target_guid is in returned candidates
  fallback       — True if Priority-0 edge traversal failed, system degraded
  ssr            — Search Space Reduction = (pre - post) / pre

Summary metrics:
  GT-in-pool rate  — fraction of H2 cases where system finds the right element
  Fallback rate    — fraction where Priority-0 failed (should be < 20%)
  Mean SSR         — average search space reduction (higher = better)

Run with:
    conda run -n mscd_demo python test/test_h2_eval.py
    conda run -n mscd_demo python test/test_h2_eval.py --max-cases 20
    conda run -n mscd_demo python test/test_h2_eval.py --pattern FILLS_RELATION
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from v2.constraints_to_query import QueryPlanner
from v2.retrieval_backend import RetrievalBackend
from v2.types import Constraints, QueryPlan, SpatialTriplet
from ifc_engine import IFCEngine

IFC_PATH = "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"
H2_PATH = "../data_curation/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl"

# Predicate → strategy mapping (for display)
STRATEGY_MAP = {
    "ADJACENT_TO": "spatial_triplet",
    "FILLS":       "spatial_triplet",
    "CONTINUOUS":  "continuous_span",
}


def build_constraints(h2: dict) -> Constraints:
    predicate = h2["predicate"]
    triplet = SpatialTriplet(
        subject_type=h2["subject_type"],
        predicate=predicate,
        object_type=h2["ref_type"] or h2["subject_type"],
    )
    # For CONTINUOUS: storey_name = top_constraint (the upper bound that discriminates).
    # Use the actual top_constraint stored per-case — different walls span to different tops.
    storey = h2["storey_name"]
    if predicate == "CONTINUOUS":
        storey = h2.get("top_constraint") or h2["storey_name"]
    return Constraints(
        ifc_class=h2["subject_type"],
        storey_name=storey,
        spatial_relations=[triplet],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2", default=H2_PATH, help="H2 eval set JSONL")
    parser.add_argument("--max-cases", type=int, default=0, help="Cap (0 = all)")
    parser.add_argument("--pattern", default="", help="Filter by pattern name")
    parser.add_argument("--ifc", default=IFC_PATH)
    args = parser.parse_args()

    # ── Load H2 cases ─────────────────────────────────────────────────────────
    with open(args.h2) as f:
        h2_cases = [json.loads(l) for l in f if l.strip()]

    if args.pattern:
        h2_cases = [c for c in h2_cases if args.pattern in c["pattern"]]
    if args.max_cases > 0:
        h2_cases = h2_cases[:args.max_cases]

    print(f"H2 cases loaded: {len(h2_cases)}")
    if args.pattern:
        print(f"  (filtered to pattern containing '{args.pattern}')")

    # ── Set up engine + backend ───────────────────────────────────────────────
    print(f"\nLoading IFC model + Neo4j...")
    try:
        from py2neo import Graph
        g = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
        g.run("RETURN 1")
        engine = IFCEngine(args.ifc, neo4j_conn=g)
        backend = RetrievalBackend(engine=engine, retrieval_mode="neo4j")
        print("  Neo4j connected.")
    except Exception as e:
        print(f"  Neo4j unavailable: {e} — falling back to memory mode")
        engine = IFCEngine(args.ifc)
        backend = RetrievalBackend(engine=engine, retrieval_mode="memory")

    planner = QueryPlanner()

    # ── Run evaluation ────────────────────────────────────────────────────────
    print()
    hdr = (f"{'H2-ID':<8} {'Pattern':<22} {'Pre-Pool':>8} "
           f"{'Post-Pool':>9} {'SSR':>6} {'GT':>4} {'Fallback'}")
    print(hdr)
    print("-" * len(hdr))

    results = []
    for h2 in h2_cases:
        c = build_constraints(h2)
        plans = planner.plan(c)
        plan = plans[0]

        result = asyncio.run(backend.execute_plan(plan))

        returned_guids = {cand.get("guid") for cand in result.candidates}
        gt_in_pool = h2["target_guid"] in returned_guids

        pre_pool = h2["pool_size"]
        post_pool = len(result.candidates)
        ssr = (pre_pool - post_pool) / pre_pool if pre_pool > 0 else 0.0
        fallback = result.fallback_triggered

        gt_mark = "✅" if gt_in_pool else "❌"
        fb_mark = "YES" if fallback else "no"

        print(f"{h2['h2_id']:<8} {h2['pattern']:<22} {pre_pool:>8} "
              f"{post_pool:>9} {ssr:>5.0%}  {gt_mark}  {fb_mark}")

        results.append({
            "h2_id": h2["h2_id"],
            "pattern": h2["pattern"],
            "predicate": h2["predicate"],
            "pre_pool_size": pre_pool,
            "post_pool_size": post_pool,
            "ssr": ssr,
            "gt_in_pool": gt_in_pool,
            "fallback": fallback,
            "strategy_asked": plan.strategy,
            "strategy_used": result.strategy_actually_used or plan.strategy,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("No results — check H2 path and pattern filter.")
        return

    gt_count = sum(1 for r in results if r["gt_in_pool"])
    fb_count = sum(1 for r in results if r["fallback"])
    mean_ssr = sum(r["ssr"] for r in results) / n
    mean_pre = sum(r["pre_pool_size"] for r in results) / n
    mean_post = sum(r["post_pool_size"] for r in results) / n
    attr_baseline = sum(100.0 / r["pre_pool_size"] for r in results) / n

    print()
    print(f"{'='*60}")
    print(f"  H2 Evaluation Summary ({n} cases)")
    print(f"{'='*60}")
    print(f"  GT-in-pool rate  : {gt_count}/{n}  ({100*gt_count/n:.0f}%)")
    print(f"  Fallback rate    : {fb_count}/{n}  ({100*fb_count/n:.0f}%)")
    print(f"  Mean SSR         : {mean_ssr:.0%}  "
          f"({mean_pre:.0f} → {mean_post:.0f} candidates)")
    print(f"  Attr baseline    : {attr_baseline:.1f}%  (random guess from pool)")
    print()

    # Per-pattern breakdown
    from collections import Counter
    by_pat = {}
    for r in results:
        by_pat.setdefault(r["pattern"], []).append(r)
    print(f"  Per-pattern:")
    for pat in sorted(by_pat):
        pr = by_pat[pat]
        m = len(pr)
        gc = sum(1 for x in pr if x["gt_in_pool"])
        fc = sum(1 for x in pr if x["fallback"])
        ms = sum(x["ssr"] for x in pr) / m
        mp = sum(x["post_pool_size"] for x in pr) / m
        bl = sum(100.0/x["pre_pool_size"] for x in pr) / m
        print(f"    {pat:<25}  {gc}/{m} GT  {fc}/{m} fallback  "
              f"SSR={ms:.0%}  post_pool={mp:.1f}  attr_bl={bl:.1f}%")

    # Pass/fail: majority GT-in-pool
    print()
    if gt_count / n >= 0.7:
        print(f"✅ PASS: {gt_count}/{n} ({100*gt_count/n:.0f}%) H2 cases find GT — "
              f"topology retrieval works at scale.")
    else:
        print(f"❌ FAIL: Only {gt_count}/{n} ({100*gt_count/n:.0f}%) H2 cases find GT — "
              f"check Priority-0 execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
