"""
Experiment 1: Attribute Entropy Quantification

Quantifies the homogeneity crisis in industrialized construction IFC data.
Counts elements per (storey, ifc_type) bucket and calculates attribute-only
Top-1 probability = 1/bucket_size.

Produces:
  - Console table of per-bucket counts
  - "Attribute Entropy by Element Type and Storey" bar chart (thesis figure)

Requires: Neo4j running with AdvancedProject.ifc loaded.

Usage:
    conda run -n mscd_demo python evaluation/experiments/exp1_attribute_entropy.py
    conda run -n mscd_demo python evaluation/experiments/exp1_attribute_entropy.py --plot docs/plots/exp1_entropy.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", default="", help="Save bar chart to PNG")
    parser.add_argument("--h2", default="",
                        help="H2 JSONL to extract pool_size stats (optional)")
    args = parser.parse_args()

    # ── Connect to Neo4j ─────────────────────────────────────────────────
    try:
        from py2neo import Graph
        g = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
        g.run("RETURN 1")
        print("Neo4j connected.\n")
    except Exception as e:
        print(f"Neo4j unavailable: {e}")
        print("This experiment requires Neo4j with AdvancedProject.ifc loaded.")
        sys.exit(1)

    # ── Query: count elements per (storey, ifc_type) ─────────────────────
    query = """
    MATCH (e:Element)
    WHERE e.storey IS NOT NULL AND e.ifc_type IS NOT NULL
    WITH e.storey AS storey, e.ifc_type AS ifc_type, count(e) AS n
    WHERE n > 1
    RETURN storey, ifc_type, n
    ORDER BY n DESC
    """
    rows = list(g.run(query))
    total_elements = g.evaluate("MATCH (e:Element) RETURN count(e)")

    print(f"{'='*70}")
    print(f"  Experiment 1: Attribute Entropy Quantification")
    print(f"{'='*70}")
    print(f"  Total elements in graph: {total_elements}")
    print(f"  Buckets with >1 identical element: {len(rows)}")
    print()

    # ── Print table ──────────────────────────────────────────────────────
    print(f"  {'Storey':<25} {'IFC Type':<28} {'Count':>6} {'Top-1 %':>8}")
    print(f"  {'-'*25} {'-'*28} {'-'*6} {'-'*8}")

    buckets = []
    for row in rows:
        storey = row["storey"]
        ifc_type = row["ifc_type"]
        n = row["n"]
        top1 = 100.0 / n
        print(f"  {storey:<25} {ifc_type:<28} {n:>6} {top1:>7.1f}%")
        buckets.append({
            "storey": storey,
            "ifc_type": ifc_type,
            "count": n,
            "top1_pct": round(top1, 2),
        })

    # ── Summary stats ────────────────────────────────────────────────────
    total_in_buckets = sum(b["count"] for b in buckets)
    weighted_top1 = sum(b["top1_pct"] * b["count"] for b in buckets) / total_in_buckets
    max_bucket = max(buckets, key=lambda b: b["count"])

    print()
    print(f"  Summary:")
    print(f"    Elements in duplicate buckets: {total_in_buckets}/{total_elements} "
          f"({100*total_in_buckets/total_elements:.0f}%)")
    print(f"    Weighted avg Top-1 (random): {weighted_top1:.1f}%")
    print(f"    Worst bucket: {max_bucket['count']}x {max_bucket['ifc_type']} "
          f"on {max_bucket['storey']} (Top-1 = {max_bucket['top1_pct']:.1f}%)")

    # ── Per-type summary ─────────────────────────────────────────────────
    type_agg = {}
    for b in buckets:
        t = b["ifc_type"]
        if t not in type_agg:
            type_agg[t] = {"total": 0, "max_bucket": 0, "n_buckets": 0}
        type_agg[t]["total"] += b["count"]
        type_agg[t]["max_bucket"] = max(type_agg[t]["max_bucket"], b["count"])
        type_agg[t]["n_buckets"] += 1

    print()
    print(f"  Per-type breakdown:")
    for t in sorted(type_agg, key=lambda x: type_agg[x]["total"], reverse=True):
        info = type_agg[t]
        print(f"    {t:<30} {info['total']:>5} elements, "
              f"{info['n_buckets']} buckets, worst={info['max_bucket']}")

    # ── H2 pool sizes (if provided) ─────────────────────────────────────
    if args.h2 and os.path.exists(args.h2):
        print()
        print(f"  H2 hard-negative pool sizes:")
        with open(args.h2) as f:
            h2_cases = [json.loads(l) for l in f if l.strip()]
        pools = [c["pool_size"] for c in h2_cases]
        baseline_top1 = [100.0 / p for p in pools]
        print(f"    Cases: {len(h2_cases)}")
        print(f"    Avg pool size: {sum(pools)/len(pools):.1f}")
        print(f"    Avg attribute-only Top-1: {sum(baseline_top1)/len(baseline_top1):.1f}%")
        print(f"    Min pool: {min(pools)}, Max pool: {max(pools)}")

    # ── Plot ─────────────────────────────────────────────────────────────
    if args.plot:
        _generate_plot(buckets, total_elements, weighted_top1, args.plot)


def _generate_plot(buckets, total_elements, weighted_top1, out_path):
    """Generate thesis-ready attribute entropy bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by ifc_type, show top 6 types
    type_buckets = {}
    for b in buckets:
        t = b["ifc_type"].replace("IfcWallStandardCase", "IfcWall*")
        type_buckets.setdefault(t, []).append(b)

    top_types = sorted(type_buckets, key=lambda t: sum(b["count"] for b in type_buckets[t]),
                       reverse=True)[:6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel 1: Bucket sizes by type (stacked by storey) ────────────────
    x = np.arange(len(top_types))
    counts = [sum(b["count"] for b in type_buckets[t]) for t in top_types]
    max_counts = [max(b["count"] for b in type_buckets[t]) for t in top_types]
    n_storeys = [len(type_buckets[t]) for t in top_types]

    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"]
    bars = ax1.bar(x, counts, color=colors, edgecolor="white", linewidth=0.5)
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{counts[i]}", ha="center", fontsize=9, fontweight="bold")
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f"worst: {max_counts[i]}/floor", ha="center", fontsize=7,
                 color="white", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace("Ifc", "") for t in top_types],
                         fontsize=9, rotation=15, ha="right")
    ax1.set_ylabel("Total Duplicate Elements")
    ax1.set_title("Duplicate Elements by IFC Type\n(attribute-identical within same storey)")

    # ── Panel 2: Top-1 accuracy by worst bucket ─────────────────────────
    worst_per_type = []
    for t in top_types:
        worst = max(type_buckets[t], key=lambda b: b["count"])
        worst_per_type.append(worst)

    labels = [f"{w['ifc_type'].replace('Ifc', '').replace('StandardCase', '*')}\n"
              f"({w['storey'][:15]})" for w in worst_per_type]
    top1s = [w["top1_pct"] for w in worst_per_type]

    bars2 = ax2.bar(range(len(worst_per_type)), top1s, color="#94a3b8",
                    edgecolor="white", linewidth=0.5)
    ax2.axhline(y=weighted_top1, color="#ef4444", linestyle="--", linewidth=1.5,
                label=f"Weighted avg = {weighted_top1:.1f}%")
    for i, bar in enumerate(bars2):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{top1s[i]:.1f}%", ha="center", fontsize=8)
    ax2.set_xticks(range(len(worst_per_type)))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Random Chance Top-1 (%)")
    ax2.set_title("Attribute-Only Top-1 Accuracy\n(worst bucket per type)")
    ax2.set_ylim(0, max(top1s) * 1.3)
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"Experiment 1: Attribute Entropy in AdvancedProject.ifc ({total_elements} elements)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


if __name__ == "__main__":
    main()
