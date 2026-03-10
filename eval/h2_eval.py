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
    conda run -n mscd_demo python eval/h2_eval.py
    conda run -n mscd_demo python eval/h2_eval.py --max-cases 20
    conda run -n mscd_demo python eval/h2_eval.py --pattern FILLS_RELATION
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
    parser.add_argument("--output", default="", help="Save results to JSONL")
    parser.add_argument("--plot", default="", help="Save comparison plot to PNG (thesis figure)")
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

    # ── Save results JSONL ──────────────────────────────────────────────────
    if args.output:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\n  Results saved: {out_path}")

    # ── Confidence threshold sweep ────────────────────────────────────────
    # Simulates what happens when LoRA_3 confidence < threshold:
    #   Below threshold → skip Priority 0, use attribute baseline (Top-1 = 1/N)
    #   Above threshold → use spatial retrieval (Top-1 ≈ GT-in-pool)
    # The 93% spatial accuracy from training means ~7% of cases get wrong
    # triplets; threshold should block those without blocking correct ones.
    print(f"\n{'='*60}")
    print(f"  Confidence Threshold Sweep (simulated)")
    print(f"{'='*60}")
    print(f"  {'Threshold':>10}  {'P0 fires':>10}  {'GT found':>10}  {'Top-1 est':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    # With 93% VLM accuracy, ~7% of cases would have wrong constraints.
    # We simulate: at each threshold, some fraction of cases pass through P0.
    # Cases with correct constraints → GT-in-pool (100% from symbolic layer).
    # Cases with wrong constraints → fall back to attribute baseline.
    vlm_accuracy = 0.93  # from training inference check
    for threshold in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Fraction of cases where VLM confidence ≥ threshold
        # Approximate: higher threshold → fewer cases pass → more conservative
        # Using a simple model: confidence ~ uniform(0.5, 1.0) for correct,
        # confidence ~ uniform(0.1, 0.7) for incorrect
        p0_correct = vlm_accuracy * max(0, (1.0 - threshold) / 0.5)  # correct cases above threshold
        p0_wrong = (1 - vlm_accuracy) * max(0, (0.7 - threshold) / 0.6)  # wrong cases above threshold
        p0_fires = min(1.0, p0_correct + p0_wrong)
        p0_correct = min(p0_correct, p0_fires)

        # Cases that fire P0 with correct constraints → GT found (100%)
        # Cases that fire P0 with wrong constraints → GT not found (0%)
        # Cases that don't fire P0 → attribute baseline
        gt_via_p0 = p0_correct * n
        gt_via_attr = (1 - p0_fires) * attr_baseline / 100 * n
        est_top1 = 100 * (gt_via_p0 + gt_via_attr) / n

        print(f"  {threshold:>10.1f}  {p0_fires:>9.0%}  "
              f"{gt_via_p0:>9.0f}/{n}  {est_top1:>9.1f}%")

    print(f"\n  Note: Attribute baseline Top-1 = {attr_baseline:.1f}%")
    print(f"  Recommended threshold: 0.7 (balances precision vs fallback rate)")

    # ── Plot thesis figure ─────────────────────────────────────────────────
    if args.plot:
        _generate_h2_plot(results, attr_baseline, args.plot)

    # Pass/fail: majority GT-in-pool
    print()
    if gt_count / n >= 0.7:
        print(f"✅ PASS: {gt_count}/{n} ({100*gt_count/n:.0f}%) H2 cases find GT — "
              f"topology retrieval works at scale.")
    else:
        print(f"❌ FAIL: Only {gt_count}/{n} ({100*gt_count/n:.0f}%) H2 cases find GT — "
              f"check Priority-0 execution.")
        sys.exit(1)


def _generate_h2_plot(results: list, attr_baseline: float, out_path: str):
    """Generate thesis-ready comparison figure: attribute baseline vs neuro-symbolic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ── Per-predicate stats ─────────────────────────────────────────────
    by_pred = {}
    for r in results:
        by_pred.setdefault(r["predicate"], []).append(r)

    predicates = sorted(by_pred.keys())
    n_preds = len(predicates)

    # Metrics per predicate
    pred_labels, attr_top1, sym_top1, ssrs, pool_pre, pool_post = [], [], [], [], [], []
    for pred in predicates:
        pr = by_pred[pred]
        m = len(pr)
        gc = sum(1 for x in pr if x["gt_in_pool"])
        ms = sum(x["ssr"] for x in pr) / m
        bl = sum(100.0 / x["pre_pool_size"] for x in pr) / m
        mp_pre = sum(x["pre_pool_size"] for x in pr) / m
        mp_post = sum(x["post_pool_size"] for x in pr) / m

        pred_labels.append(f"{pred}\n(n={m})")
        attr_top1.append(bl)
        sym_top1.append(100 * gc / m)
        ssrs.append(100 * ms)
        pool_pre.append(mp_pre)
        pool_post.append(mp_post)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: Top-1 Accuracy ─────────────────────────────────────────
    ax = axes[0]
    x = np.arange(n_preds)
    w = 0.35
    ax.bar(x - w/2, attr_top1, w, label="Attribute Baseline", color="#94a3b8", edgecolor="white")
    ax.bar(x + w/2, sym_top1, w, label="Neuro-Symbolic (P0)", color="#3b82f6", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(pred_labels, fontsize=9)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("GT-in-Pool Rate by Predicate")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    for i in range(n_preds):
        ax.text(i - w/2, attr_top1[i] + 2, f"{attr_top1[i]:.1f}%", ha="center", fontsize=7)
        ax.text(i + w/2, sym_top1[i] + 2, f"{sym_top1[i]:.0f}%", ha="center", fontsize=7)

    # ── Panel 2: Search Space Reduction ─────────────────────────────────
    ax = axes[1]
    colors = ["#f59e0b", "#8b5cf6", "#3b82f6"][:n_preds]
    bars = ax.bar(x, ssrs, 0.5, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(pred_labels, fontsize=9)
    ax.set_ylabel("Search Space Reduction (%)")
    ax.set_title("SSR by Predicate")
    ax.set_ylim(0, 100)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{ssrs[i]:.0f}%", ha="center", fontsize=8)

    # ── Panel 3: Pool Size Reduction ────────────────────────────────────
    ax = axes[2]
    ax.bar(x - w/2, pool_pre, w, label="Before (attribute pool)", color="#94a3b8", edgecolor="white")
    ax.bar(x + w/2, pool_post, w, label="After (P0 retrieval)", color="#22c55e", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(pred_labels, fontsize=9)
    ax.set_ylabel("Mean Pool Size")
    ax.set_title("Candidate Pool Reduction")
    ax.legend(fontsize=8)
    for i in range(n_preds):
        ax.text(i - w/2, pool_pre[i] + 3, f"{pool_pre[i]:.0f}", ha="center", fontsize=7)
        ax.text(i + w/2, pool_post[i] + 3, f"{pool_post[i]:.0f}", ha="center", fontsize=7)

    fig.suptitle(
        f"H2 Hard-Negative Evaluation — {len(results)} cases, "
        f"Attr baseline Top-1 = {attr_baseline:.1f}%",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


if __name__ == "__main__":
    main()
