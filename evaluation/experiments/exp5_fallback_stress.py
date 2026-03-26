"""
Experiment 5: Fallback Stress Test — OPTIONAL MATCH Robustness

Proves that OPTIONAL MATCH provides industrial robustness. When the VLM
hallucinates hop-2, the system degrades gracefully instead of catastrophically.

For each H2 FILLS case, runs 3 conditions:
  1. CORRECT:   Oracle 2-hop constraints -> GT in pool (baseline)
  2. CORRUPTED + OPTIONAL MATCH: Flip hop-2 object_type to wrong class
     -> GT STILL in pool (graceful degradation via hop-1)
  3. CORRUPTED + HARD MATCH: Same corruption but with hard MATCH on hop-2
     -> GT LOST (0 results or GT missing — catastrophic failure)

Produces:
  - Console comparison table
  - Bar chart (thesis figure)

Requires: Neo4j running with AdvancedProject.ifc loaded.

Usage:
    conda run -n mscd_demo python evaluation/experiments/exp5_fallback_stress.py
    conda run -n mscd_demo python evaluation/experiments/exp5_fallback_stress.py --plot plots/archive/exp5_fallback.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

IFC_PATH = "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"
H2_PATH = "../../data_curation/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl"

# Wrong types to inject for hop-2 corruption
CORRUPT_TYPES = ["IfcSlab", "IfcStair", "IfcRailing", "IfcColumn"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2", default=H2_PATH, help="H2 eval set JSONL")
    parser.add_argument("--plot", default="", help="Save comparison plot to PNG")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    # ── Load FILLS cases ─────────────────────────────────────────────────
    with open(args.h2) as f:
        all_cases = [json.loads(l) for l in f if l.strip()]

    fills_cases = [c for c in all_cases if c["predicate"] == "FILLS"]
    if args.max_cases > 0:
        fills_cases = fills_cases[:args.max_cases]

    print(f"H2 FILLS cases: {len(fills_cases)}")

    # ── Connect to Neo4j ─────────────────────────────────────────────────
    try:
        from py2neo import Graph
        g = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
        g.run("RETURN 1")
        print("Neo4j connected.\n")
    except Exception as e:
        print(f"Neo4j unavailable: {e}")
        sys.exit(1)

    from ifc_engine import IFCEngine
    engine = IFCEngine(IFC_PATH, neo4j_conn=g)

    def resolve_storey(storey_name):
        if not storey_name:
            return []
        try:
            siblings = engine._resolve_storey_query(storey_name)
            return [s.lower() for s in siblings] if siblings else []
        except Exception:
            return [storey_name.lower()]

    # ── Run 3 conditions per case ────────────────────────────────────────
    print(f"{'H2-ID':<8} {'Correct':>8} {'Corrupt+OPT':>12} {'Corrupt+HARD':>13} "
          f"{'GT_correct':>11} {'GT_opt':>7} {'GT_hard':>8}")
    print("-" * 75)

    results = []
    import random
    rng = random.Random(42)

    for case in fills_cases:
        subject_type = case["subject_type"]
        ref_type = case["ref_type"] or "IfcWallStandardCase"
        storey = case["storey_name"]
        target_guid = case["target_guid"]
        storey_list = resolve_storey(storey)

        # Pick a random wrong type for corruption
        corrupt_type = rng.choice([t for t in CORRUPT_TYPES if t != ref_type])

        # ── Condition 1: Correct 2-hop OPTIONAL MATCH ────────────────────
        cypher_correct = """
            MATCH (t:IFCElement)-[:FILLS]->(ref:IFCElement)
            WHERE (t.ifc_type = $type OR t.ifc_type STARTS WITH $type)
              AND (ref.ifc_type = $ref_type OR ref.ifc_type STARTS WITH $ref_type)
              AND (size($storey_list) = 0
                   OR ANY(s IN $storey_list WHERE toLower(t.storey) CONTAINS s))
            OPTIONAL MATCH (ref)-[:CONNECTS_TO]->(ref2:IFCElement)
            WHERE ref2.ifc_type = $ref_type OR ref2.ifc_type STARTS WITH $ref_type
            RETURN DISTINCT t.guid AS guid, (ref2 IS NOT NULL) AS has_hop2
            ORDER BY has_hop2 DESC
        """
        r_correct = list(g.run(cypher_correct, type=subject_type,
                               ref_type=ref_type, storey_list=storey_list))
        pool_correct = len(r_correct)
        gt_correct = any(row["guid"] == target_guid for row in r_correct)

        # ── Condition 2: Corrupted hop-2 + OPTIONAL MATCH (safe) ─────────
        cypher_corrupt_opt = """
            MATCH (t:IFCElement)-[:FILLS]->(ref:IFCElement)
            WHERE (t.ifc_type = $type OR t.ifc_type STARTS WITH $type)
              AND (ref.ifc_type = $ref_type OR ref.ifc_type STARTS WITH $ref_type)
              AND (size($storey_list) = 0
                   OR ANY(s IN $storey_list WHERE toLower(t.storey) CONTAINS s))
            OPTIONAL MATCH (ref)-[:CONNECTS_TO]->(ref2:IFCElement)
            WHERE ref2.ifc_type = $corrupt_type OR ref2.ifc_type STARTS WITH $corrupt_type
            RETURN DISTINCT t.guid AS guid, (ref2 IS NOT NULL) AS has_hop2
            ORDER BY has_hop2 DESC
        """
        r_corrupt_opt = list(g.run(cypher_corrupt_opt, type=subject_type,
                                   ref_type=ref_type, storey_list=storey_list,
                                   corrupt_type=corrupt_type))
        pool_corrupt_opt = len(r_corrupt_opt)
        gt_corrupt_opt = any(row["guid"] == target_guid for row in r_corrupt_opt)

        # ── Condition 3: Corrupted hop-2 + HARD MATCH (dangerous) ────────
        cypher_corrupt_hard = """
            MATCH (t:IFCElement)-[:FILLS]->(ref:IFCElement)
            WHERE (t.ifc_type = $type OR t.ifc_type STARTS WITH $type)
              AND (ref.ifc_type = $ref_type OR ref.ifc_type STARTS WITH $ref_type)
              AND (size($storey_list) = 0
                   OR ANY(s IN $storey_list WHERE toLower(t.storey) CONTAINS s))
            MATCH (ref)-[:CONNECTS_TO]->(ref2:IFCElement)
            WHERE ref2.ifc_type = $corrupt_type OR ref2.ifc_type STARTS WITH $corrupt_type
            RETURN DISTINCT t.guid AS guid
        """
        r_corrupt_hard = list(g.run(cypher_corrupt_hard, type=subject_type,
                                    ref_type=ref_type, storey_list=storey_list,
                                    corrupt_type=corrupt_type))
        pool_corrupt_hard = len(r_corrupt_hard)
        gt_corrupt_hard = any(row["guid"] == target_guid for row in r_corrupt_hard)

        print(f"{case['h2_id']:<8} {pool_correct:>8} {pool_corrupt_opt:>12} "
              f"{pool_corrupt_hard:>13} "
              f"{'Y' if gt_correct else 'N':>11} "
              f"{'Y' if gt_corrupt_opt else 'N':>7} "
              f"{'Y' if gt_corrupt_hard else 'N':>8}")

        results.append({
            "h2_id": case["h2_id"],
            "subject_type": subject_type,
            "corrupt_type": corrupt_type,
            "pool_correct": pool_correct,
            "pool_corrupt_opt": pool_corrupt_opt,
            "pool_corrupt_hard": pool_corrupt_hard,
            "gt_correct": gt_correct,
            "gt_corrupt_opt": gt_corrupt_opt,
            "gt_corrupt_hard": gt_corrupt_hard,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("No results.")
        return

    gt_c = sum(1 for r in results if r["gt_correct"])
    gt_o = sum(1 for r in results if r["gt_corrupt_opt"])
    gt_h = sum(1 for r in results if r["gt_corrupt_hard"])

    avg_pool_c = sum(r["pool_correct"] for r in results) / n
    avg_pool_o = sum(r["pool_corrupt_opt"] for r in results) / n
    avg_pool_h = sum(r["pool_corrupt_hard"] for r in results) / n

    over_red_c = sum(1 for r in results if not r["gt_correct"]) / n
    over_red_o = sum(1 for r in results if not r["gt_corrupt_opt"]) / n
    over_red_h = sum(1 for r in results if not r["gt_corrupt_hard"]) / n

    print()
    print(f"{'='*65}")
    print(f"  Experiment 5: Fallback Stress Test ({n} FILLS cases)")
    print(f"{'='*65}")
    print(f"  {'Condition':<25} {'GT-in-pool':>11} {'Avg Pool':>9} {'Over-red':>9}")
    print(f"  {'-'*25} {'-'*11} {'-'*9} {'-'*9}")
    print(f"  {'Correct (oracle)':.<25} {gt_c:>3}/{n} ({100*gt_c/n:.0f}%) "
          f"{avg_pool_c:>8.1f} {over_red_c:>8.0%}")
    print(f"  {'Corrupted + OPTIONAL':.<25} {gt_o:>3}/{n} ({100*gt_o/n:.0f}%) "
          f"{avg_pool_o:>8.1f} {over_red_o:>8.0%}")
    print(f"  {'Corrupted + HARD MATCH':.<25} {gt_h:>3}/{n} ({100*gt_h/n:.0f}%) "
          f"{avg_pool_h:>8.1f} {over_red_h:>8.0%}")
    print()

    if gt_o == n and gt_h < n:
        print(f"  PASS: OPTIONAL MATCH preserves GT in {gt_o}/{n} corrupted cases")
        print(f"        HARD MATCH loses GT in {n - gt_h}/{n} cases")
        print(f"        -> OPTIONAL MATCH is the correct industrial choice")
    elif gt_o < n:
        print(f"  NOTE: OPTIONAL MATCH lost GT in {n - gt_o} cases")
        print(f"        (investigate: hop-1 should always preserve GT)")

    if args.plot:
        _generate_plot(results, args.plot)


def _generate_plot(results, out_path):
    """Generate thesis-ready fallback comparison chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(results)

    # ── Metrics ──────────────────────────────────────────────────────────
    conditions = ["Correct\n(oracle)", "Corrupted\n+ OPTIONAL", "Corrupted\n+ HARD"]
    gt_rates = [
        100 * sum(1 for r in results if r["gt_correct"]) / n,
        100 * sum(1 for r in results if r["gt_corrupt_opt"]) / n,
        100 * sum(1 for r in results if r["gt_corrupt_hard"]) / n,
    ]
    avg_pools = [
        sum(r["pool_correct"] for r in results) / n,
        sum(r["pool_corrupt_opt"] for r in results) / n,
        sum(r["pool_corrupt_hard"] for r in results) / n,
    ]
    over_red = [100 - r for r in gt_rates]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#22c55e", "#f59e0b", "#ef4444"]

    # ── Panel 1: GT-in-pool rate ─────────────────────────────────────────
    bars1 = ax1.bar(conditions, gt_rates, color=colors, edgecolor="white",
                    width=0.6)
    for bar, rate in zip(bars1, gt_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{rate:.0f}%", ha="center", fontsize=12, fontweight="bold")
    ax1.set_ylabel("GT-in-Pool Rate (%)", fontsize=12)
    ax1.set_title("Ground Truth Retention\nUnder Hop-2 Hallucination", fontsize=13,
                  fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.axhline(y=100, color="#22c55e", linestyle="--", alpha=0.3)

    # ── Panel 2: Pool size comparison ────────────────────────────────────
    bars2 = ax2.bar(conditions, avg_pools, color=colors, edgecolor="white",
                    width=0.6)
    for bar, pool in zip(bars2, avg_pools):
        label = f"{pool:.1f}" if pool > 0 else "0"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 max(bar.get_height(), 0.5) + 0.3,
                 label, ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Pool Size", fontsize=12)
    ax2.set_title("Candidate Pool Size\n(0 = catastrophic failure)", fontsize=13,
                  fontweight="bold")
    ax2.set_ylim(0, max(avg_pools) * 1.2 + 1)

    fig.suptitle(
        f"Experiment 5: Fallback Stress Test ({n} FILLS cases)\n"
        f"OPTIONAL MATCH = safety net | HARD MATCH = brittle",
        fontsize=13, fontweight="bold", y=1.05,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Fallback stress plot saved: {out_path}")


if __name__ == "__main__":
    main()
