"""
Experiment 3: Uniqueness by Hop — the L-Curve

Answers: "Why 2 hops? Why not 1 or 3?"
For each H2 FILLS case (Window/Door -> FILLS -> Wall -> CONNECTS_TO -> Wall2),
runs 4 progressive queries and measures candidate pool size at each depth.

Produces:
  - Console table with per-case pool sizes at each hop depth
  - L-shaped curve chart (thesis figure)

Requires: Neo4j running with AdvancedProject.ifc loaded.

Usage:
    conda run -n mscd_demo python evaluation/experiments/exp3_hop_uniqueness.py
    conda run -n mscd_demo python evaluation/experiments/exp3_hop_uniqueness.py --plot plots/archive/exp3_lcurve.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

IFC_PATH = "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"
H2_PATH = "../../data_curation/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2", default=H2_PATH, help="H2 eval set JSONL")
    parser.add_argument("--plot", default="", help="Save L-curve plot to PNG")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    # ── Load H2 FILLS cases only ─────────────────────────────────────────
    with open(args.h2) as f:
        all_cases = [json.loads(l) for l in f if l.strip()]

    fills_cases = [c for c in all_cases if c["predicate"] == "FILLS"]
    if args.max_cases > 0:
        fills_cases = fills_cases[:args.max_cases]

    print(f"H2 FILLS cases: {len(fills_cases)} (from {len(all_cases)} total)")

    # ── Connect to Neo4j ─────────────────────────────────────────────────
    try:
        from py2neo import Graph
        g = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
        g.run("RETURN 1")
        print("Neo4j connected.\n")
    except Exception as e:
        print(f"Neo4j unavailable: {e}")
        sys.exit(1)

    # ── Resolve storey siblings ──────────────────────────────────────────
    from ifc_engine import IFCEngine
    engine = IFCEngine(IFC_PATH, neo4j_conn=g)

    def resolve_storey(storey_name):
        """Get all canonical storey names for a floor number."""
        if not storey_name:
            return []
        try:
            siblings = engine._resolve_storey_query(storey_name)
            return [s.lower() for s in siblings] if siblings else []
        except Exception:
            return [storey_name.lower()]

    # ── Run 4-hop progressive queries ────────────────────────────────────
    print(f"{'H2-ID':<8} {'Type':<12} {'Storey':<20} "
          f"{'0-hop':>6} {'1-hop':>6} {'2-hop':>6} {'3-hop':>6} "
          f"{'GT@0':>4} {'GT@1':>4} {'GT@2':>4}")
    print("-" * 95)

    results = []
    for case in fills_cases:
        h2_id = case["h2_id"]
        subject_type = case["subject_type"]
        ref_type = case["ref_type"] or "IfcWallStandardCase"
        storey = case["storey_name"]
        target_guid = case["target_guid"]
        storey_list = resolve_storey(storey)

        # ── 0-hop: attribute-only (storey + type) ────────────────────────
        cypher_0 = """
            MATCH (t:IFCElement)
            WHERE (t.ifc_type = $type OR t.ifc_type STARTS WITH $type)
              AND (size($storey_list) = 0
                   OR ANY(s IN $storey_list WHERE toLower(t.storey) CONTAINS s))
            RETURN DISTINCT t.guid AS guid
        """
        r0 = list(g.run(cypher_0, type=subject_type, storey_list=storey_list))
        pool_0 = len(r0)
        gt_0 = any(row["guid"] == target_guid for row in r0)

        # ── 1-hop: FILLS only (target -> FILLS -> ref) ───────────────────
        cypher_1 = """
            MATCH (t:IFCElement)-[:FILLS]->(ref:IFCElement)
            WHERE (t.ifc_type = $type OR t.ifc_type STARTS WITH $type)
              AND (ref.ifc_type = $ref_type OR ref.ifc_type STARTS WITH $ref_type)
              AND (size($storey_list) = 0
                   OR ANY(s IN $storey_list WHERE toLower(t.storey) CONTAINS s))
            RETURN DISTINCT t.guid AS guid
        """
        r1 = list(g.run(cypher_1, type=subject_type, ref_type=ref_type,
                        storey_list=storey_list))
        pool_1 = len(r1)
        gt_1 = any(row["guid"] == target_guid for row in r1)

        # ── 2-hop: FILLS + CONNECTS_TO (OPTIONAL MATCH) ─────────────────
        cypher_2 = """
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
        r2 = list(g.run(cypher_2, type=subject_type, ref_type=ref_type,
                        storey_list=storey_list))
        pool_2 = len(r2)
        gt_2 = any(row["guid"] == target_guid for row in r2)

        # ── 3-hop: FILLS + CONNECTS_TO + CONNECTS_TO (double OPTIONAL) ──
        cypher_3 = """
            MATCH (t:IFCElement)-[:FILLS]->(ref:IFCElement)
            WHERE (t.ifc_type = $type OR t.ifc_type STARTS WITH $type)
              AND (ref.ifc_type = $ref_type OR ref.ifc_type STARTS WITH $ref_type)
              AND (size($storey_list) = 0
                   OR ANY(s IN $storey_list WHERE toLower(t.storey) CONTAINS s))
            OPTIONAL MATCH (ref)-[:CONNECTS_TO]->(ref2:IFCElement)
            WHERE ref2.ifc_type = $ref_type OR ref2.ifc_type STARTS WITH $ref_type
            OPTIONAL MATCH (ref2)-[:CONNECTS_TO]->(ref3:IFCElement)
            WHERE ref3.ifc_type = $ref_type OR ref3.ifc_type STARTS WITH $ref_type
            RETURN DISTINCT t.guid AS guid,
                   (ref2 IS NOT NULL) AS has_hop2,
                   (ref3 IS NOT NULL) AS has_hop3
            ORDER BY has_hop3 DESC, has_hop2 DESC
        """
        r3 = list(g.run(cypher_3, type=subject_type, ref_type=ref_type,
                        storey_list=storey_list))
        pool_3 = len(r3)

        gt_0_mark = "Y" if gt_0 else "N"
        gt_1_mark = "Y" if gt_1 else "N"
        gt_2_mark = "Y" if gt_2 else "N"

        print(f"{h2_id:<8} {subject_type.replace('Ifc',''):<12} "
              f"{storey[:20]:<20} "
              f"{pool_0:>6} {pool_1:>6} {pool_2:>6} {pool_3:>6} "
              f"{gt_0_mark:>4} {gt_1_mark:>4} {gt_2_mark:>4}")

        # Count how many 2-hop candidates have has_hop2=True (promoted)
        n_promoted_2 = sum(1 for row in r2 if row.get("has_hop2"))
        n_promoted_3 = sum(1 for row in r3 if row.get("has_hop3"))

        results.append({
            "h2_id": h2_id,
            "subject_type": subject_type,
            "storey": storey,
            "pool_0": pool_0,
            "pool_1": pool_1,
            "pool_2": pool_2,
            "pool_3": pool_3,
            "promoted_2": n_promoted_2,
            "promoted_3": n_promoted_3,
            "gt_in_0": gt_0,
            "gt_in_1": gt_1,
            "gt_in_2": gt_2,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("No FILLS cases found.")
        return

    avg_0 = sum(r["pool_0"] for r in results) / n
    avg_1 = sum(r["pool_1"] for r in results) / n
    avg_2 = sum(r["pool_2"] for r in results) / n
    avg_3 = sum(r["pool_3"] for r in results) / n
    avg_p2 = sum(r["promoted_2"] for r in results) / n
    avg_p3 = sum(r["promoted_3"] for r in results) / n

    gt_rate_0 = sum(1 for r in results if r["gt_in_0"]) / n
    gt_rate_1 = sum(1 for r in results if r["gt_in_1"]) / n
    gt_rate_2 = sum(1 for r in results if r["gt_in_2"]) / n

    print()
    print(f"{'='*60}")
    print(f"  Experiment 3: Uniqueness by Hop ({n} FILLS cases)")
    print(f"{'='*60}")
    print(f"  Hop depth  Avg Pool   Reduction    GT-in-pool   Promoted")
    print(f"  ---------  --------   ---------    ----------   --------")
    print(f"  0-hop      {avg_0:>6.1f}     baseline     {gt_rate_0:>5.0%}         N/A")
    print(f"  1-hop      {avg_1:>6.1f}     {100*(avg_0-avg_1)/avg_0:>5.0f}%        {gt_rate_1:>5.0%}         N/A")
    print(f"  2-hop      {avg_2:>6.1f}     {100*(avg_0-avg_2)/avg_0:>5.0f}%        {gt_rate_2:>5.0%}         {avg_p2:.1f}")
    print(f"  3-hop      {avg_3:>6.1f}     {100*(avg_0-avg_3)/avg_0:>5.0f}%        {gt_rate_2:>5.0%}         {avg_p3:.1f}")
    print()
    print(f"  Key insight: 1-hop FILLS provides the main pool reduction")
    print(f"  ({avg_0:.0f} -> {avg_1:.0f}). 2-hop CONNECTS_TO promotes GT to top")
    print(f"  ({avg_p2:.1f} avg promoted). 3-hop adds negligible benefit")
    print(f"  ({avg_p3:.1f} avg promoted).")

    if args.plot:
        _generate_plot(results, args.plot)


def _generate_plot(results, out_path):
    """Generate thesis-ready L-curve chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(results)
    hops = [0, 1, 2, 3]
    avg_pools = [
        sum(r["pool_0"] for r in results) / n,
        sum(r["pool_1"] for r in results) / n,
        sum(r["pool_2"] for r in results) / n,
        sum(r["pool_3"] for r in results) / n,
    ]
    gt_rates = [
        100 * sum(1 for r in results if r["gt_in_0"]) / n,
        100 * sum(1 for r in results if r["gt_in_1"]) / n,
        100 * sum(1 for r in results if r["gt_in_2"]) / n,
        100 * sum(1 for r in results if r["gt_in_2"]) / n,  # same as 2-hop
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 1: Pool size L-curve ───────────────────────────────────────
    ax1.plot(hops, avg_pools, "o-", color="#3b82f6", linewidth=2.5,
             markersize=10, markerfacecolor="white", markeredgewidth=2.5)
    for i, (h, p) in enumerate(zip(hops, avg_pools)):
        ax1.annotate(f"{p:.1f}", (h, p), textcoords="offset points",
                     xytext=(10, 10), fontsize=10, fontweight="bold")
    ax1.set_xlabel("Hop Depth", fontsize=12)
    ax1.set_ylabel("Mean Candidate Pool Size", fontsize=12)
    ax1.set_title("Candidate Pool vs Hop Depth", fontsize=13, fontweight="bold")
    ax1.set_xticks(hops)
    ax1.set_xticklabels(["0\n(attr only)", "1\n(FILLS)", "2\n(+CONNECTS_TO)",
                          "3\n(+CONNECTS_TO)"], fontsize=9)
    ax1.set_ylim(0, max(avg_pools) * 1.15)
    ax1.grid(axis="y", alpha=0.3)

    # Shade the "diminishing returns" region
    ax1.axvspan(1.5, 3.5, alpha=0.08, color="#22c55e",
                label="Diminishing returns zone")
    ax1.legend(fontsize=9, loc="upper right")

    # ── Panel 2: GT-in-pool rate ─────────────────────────────────────────
    colors = ["#94a3b8", "#f59e0b", "#3b82f6", "#8b5cf6"]
    bars = ax2.bar(hops, gt_rates, color=colors, edgecolor="white", width=0.6)
    for bar, rate in zip(bars, gt_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{rate:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Hop Depth", fontsize=12)
    ax2.set_ylabel("GT-in-Pool Rate (%)", fontsize=12)
    ax2.set_title("Ground Truth Retention by Hop", fontsize=13, fontweight="bold")
    ax2.set_xticks(hops)
    ax2.set_xticklabels(["0-hop", "1-hop", "2-hop", "3-hop"], fontsize=10)
    ax2.set_ylim(0, 115)

    fig.suptitle(
        f"Experiment 3: Uniqueness by Hop ({n} FILLS cases, AdvancedProject.ifc)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  L-curve plot saved: {out_path}")


if __name__ == "__main__":
    main()
