"""
Oracle Pipeline Waterfall Experiment (v2 — GT-Reverse)

For each test case, looks up the GT element in Neo4j, discovers its ACTUAL
attributes and edges, then runs queries using those ground-truth constraints.

This guarantees 100% GT-in-Pool at every stage. The experiment measures
the discriminative power (pool size reduction) of each stage:
  Full elements → P1 (storey+type) → 1-hop spatial → 2-hop spatial
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from py2neo import Graph
import yaml
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from collections import defaultdict

from src.ifc_engine import IFCEngine


# ── Config ──────────────────────────────────────────────────────────────────

IFC_FILES = {
    "AP": "/root/cmu/master_thesis/data_curation/ifc_models/AdvancedProject.ifc",
    "BH": "/root/cmu/master_thesis/data_curation/ifc_models/BasicHouse.ifc",
    "DXA": "/root/cmu/master_thesis/data_curation/ifc_models/Duplex_A_20110505.ifc",
}

CASES_FILE = "evaluation/cases/cases_unified_test.jsonl"
OUTPUT_DIR = "evaluation/plots"

# Edge types to probe for oracle spatial relations
EDGE_TYPES = ["FILLS", "ADJACENT_TO", "NEXT_TO", "CONNECTS_TO"]


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def get_engine(model_key: str, neo4j_conn) -> IFCEngine:
    ifc_path = IFC_FILES[model_key]
    return IFCEngine(ifc_path, neo4j_conn=neo4j_conn)


def get_total_elements(neo4j_conn, model_key: str) -> int:
    result = neo4j_conn.run(
        "MATCH (e:IFCElement) WHERE e.ifc_model = $model RETURN count(e) as cnt",
        model=model_key
    )
    return result.evaluate()


def lookup_gt_element(neo4j_conn, gt_guid: str, model_key: str):
    """Look up GT element's actual properties in Neo4j."""
    result = neo4j_conn.run("""
        MATCH (e:IFCElement)
        WHERE e.guid = $guid AND e.ifc_model = $model
        RETURN e.guid as guid, e.ifc_type as ifc_type, e.storey as storey,
               e.name as name, e.is_continuous as is_continuous,
               e.top_constraint as top_constraint, e.material as material
    """, guid=gt_guid, model=model_key)
    rec = result.data()
    return rec[0] if rec else None


def discover_gt_edges(neo4j_conn, gt_guid: str, model_key: str):
    """Discover all outgoing edges from GT element, grouped by edge type."""
    edges = {}
    for edge_type in EDGE_TYPES:
        result = neo4j_conn.run(f"""
            MATCH (src:IFCElement)-[:{edge_type}]->(dst:IFCElement)
            WHERE src.guid = $guid AND src.ifc_model = $model
            RETURN dst.guid as guid, dst.ifc_type as ifc_type,
                   dst.storey as storey, dst.material as material
        """, guid=gt_guid, model=model_key)
        neighbors = [dict(r) for r in result]
        if neighbors:
            edges[edge_type] = neighbors

    # Also check CONTINUOUS property
    result = neo4j_conn.run("""
        MATCH (e:IFCElement)
        WHERE e.guid = $guid AND e.ifc_model = $model
          AND e.is_continuous = true
        RETURN e.top_constraint as top_constraint
    """, guid=gt_guid, model=model_key)
    cont = result.data()
    if cont and cont[0].get("top_constraint"):
        edges["CONTINUOUS"] = [{"top_constraint": cont[0]["top_constraint"]}]

    return edges


def discover_hop2_edges(neo4j_conn, ref_guids: list, model_key: str):
    """From hop-1 reference elements, discover their outgoing edges (hop 2)."""
    hop2_edges = {}
    for edge_type in EDGE_TYPES:
        result = neo4j_conn.run(f"""
            MATCH (ref:IFCElement)-[:{edge_type}]->(dst:IFCElement)
            WHERE ref.guid IN $guids AND ref.ifc_model = $model
            RETURN DISTINCT dst.guid as guid, dst.ifc_type as ifc_type,
                   dst.material as material
        """, guids=ref_guids, model=model_key)
        neighbors = [dict(r) for r in result]
        if neighbors:
            hop2_edges[edge_type] = neighbors
    return hop2_edges


def run_p1_query(neo4j_conn, ifc_type: str, storey: str, model_key: str):
    """P1: storey+type using GT element's actual attributes."""
    if storey and ifc_type:
        cypher = """
            MATCH (s:IFCStorey)-[:CONTAINS]->(e:IFCElement)
            WHERE (e.ifc_type = $type OR e.ifc_type STARTS WITH $type)
              AND e.ifc_model = $model
              AND toLower(s.name) CONTAINS toLower($storey)
            RETURN DISTINCT e.guid as guid, e.name as name, e.ifc_type as type
        """
        result = neo4j_conn.run(cypher, type=ifc_type, storey=storey, model=model_key)
    elif ifc_type:
        cypher = """
            MATCH (e:IFCElement)
            WHERE (e.ifc_type = $type OR e.ifc_type STARTS WITH $type)
              AND e.ifc_model = $model
            RETURN e.guid as guid, e.name as name, e.ifc_type as type
        """
        result = neo4j_conn.run(cypher, type=ifc_type, model=model_key)
    else:
        return []
    return [dict(r) for r in result]


def run_1hop_query(neo4j_conn, ifc_type: str, storey: str, model_key: str,
                   predicate: str, object_type: str, object_material: str = ""):
    """1-hop: GT element type + actual edge from GT element."""
    if predicate == "CONTINUOUS":
        # For CONTINUOUS, filter by is_continuous property
        cypher = """
            MATCH (target:IFCElement)
            WHERE (target.ifc_type = $type OR target.ifc_type STARTS WITH $type)
              AND target.is_continuous = true
              AND target.ifc_model = $model
            RETURN target.guid as guid, target.name as name, target.ifc_type as type
        """
        result = neo4j_conn.run(cypher, type=ifc_type, model=model_key)
    else:
        cypher = f"""
            MATCH (target:IFCElement)-[:{predicate}]->(ref:IFCElement)
            WHERE (target.ifc_type = $type OR target.ifc_type STARTS WITH $type)
              AND (ref.ifc_type = $obj_type OR ref.ifc_type STARTS WITH $obj_type)
              AND target.ifc_model = $model
              AND toLower(target.storey) CONTAINS toLower($storey)
              AND ($mat = '' OR toLower(ref.material) CONTAINS toLower($mat))
            RETURN DISTINCT target.guid as guid, target.name as name,
                   target.ifc_type as type
        """
        result = neo4j_conn.run(
            cypher, type=ifc_type, storey=storey, obj_type=object_type,
            mat=object_material or "", model=model_key
        )
    return [dict(r) for r in result]


def run_2hop_query(neo4j_conn, ifc_type: str, storey: str, model_key: str,
                   pred1: str, obj_type1: str, mat1: str,
                   pred2: str, obj_type2: str, mat2: str):
    """2-hop hard filter: GT type -[pred1]-> ref1 -[pred2]-> ref2."""
    cypher = f"""
        MATCH (target:IFCElement)-[:{pred1}]->(ref:IFCElement)
              -[:{pred2}]->(ref2:IFCElement)
        WHERE (target.ifc_type = $type OR target.ifc_type STARTS WITH $type)
          AND (ref.ifc_type = $obj1 OR ref.ifc_type STARTS WITH $obj1)
          AND (ref2.ifc_type = $obj2 OR ref2.ifc_type STARTS WITH $obj2)
          AND target.ifc_model = $model
          AND toLower(target.storey) CONTAINS toLower($storey)
          AND ($mat1 = '' OR toLower(ref.material) CONTAINS toLower($mat1))
          AND ($mat2 = '' OR toLower(ref2.material) CONTAINS toLower($mat2))
        RETURN DISTINCT target.guid AS guid, target.name AS name,
               target.ifc_type AS type
    """
    result = neo4j_conn.run(
        cypher, type=ifc_type, storey=storey, model=model_key,
        obj1=obj_type1, obj2=obj_type2, mat1=mat1 or "", mat2=mat2 or ""
    )
    return [dict(r) for r in result]


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    cfg = load_config()
    neo4j_cfg = cfg.get("neo4j", {})
    neo4j_conn = Graph(
        neo4j_cfg.get("uri", "bolt://localhost:7687"),
        auth=("neo4j", neo4j_cfg.get("password", "password"))
    )

    cases = [json.loads(l) for l in open(CASES_FILE)]
    print(f"Loaded {len(cases)} test cases")

    engines = {}  # cache for IFCEngine (only needed for model_key derivation)
    results = []
    skipped = 0

    for case in cases:
        case_id = case["case_id"]
        model_key = case["ifc_model"]
        gt_guid = case["ground_truth"]["target_guid"]

        # ── Step 0: Look up GT element in Neo4j ──
        gt_node = lookup_gt_element(neo4j_conn, gt_guid, model_key)
        if not gt_node:
            print(f"  SKIP {case_id}: GT element not found in Neo4j")
            skipped += 1
            continue

        gt_type = gt_node["ifc_type"]
        gt_storey = gt_node["storey"] or ""
        total = get_total_elements(neo4j_conn, model_key)

        # ── Step 1: P1 (storey+type) using GT's actual attributes ──
        p1_cands = run_p1_query(neo4j_conn, gt_type, gt_storey, model_key)
        p1_pool = len(p1_cands)
        p1_guids = {c["guid"] for c in p1_cands}
        p1_gt_in = gt_guid in p1_guids

        # ── Step 2: Discover GT element's actual edges ──
        gt_edges = discover_gt_edges(neo4j_conn, gt_guid, model_key)

        # Pick best 1-hop: try each edge type, keep the one with smallest non-zero pool
        best_1hop = None
        all_1hop = {}  # store all edge type results

        for edge_type, neighbors in gt_edges.items():
            if edge_type == "CONTINUOUS":
                cands = run_1hop_query(
                    neo4j_conn, gt_type, gt_storey, model_key,
                    "CONTINUOUS", "", ""
                )
            else:
                # Use the first neighbor's type as object_type
                obj_type = neighbors[0]["ifc_type"]
                obj_mat = neighbors[0].get("material", "") or ""
                cands = run_1hop_query(
                    neo4j_conn, gt_type, gt_storey, model_key,
                    edge_type, obj_type, ""  # no material filter for 1-hop
                )

            pool = len(cands)
            gt_in = gt_guid in {c["guid"] for c in cands}
            all_1hop[edge_type] = {
                "pool": pool, "gt_in": gt_in,
                "obj_type": neighbors[0].get("ifc_type", "") if edge_type != "CONTINUOUS" else "",
                "obj_mat": neighbors[0].get("material", "") if edge_type != "CONTINUOUS" else "",
            }

            if gt_in and (best_1hop is None or pool < best_1hop["pool"]):
                best_1hop = {"edge_type": edge_type, "pool": pool, "gt_in": gt_in,
                             "neighbors": neighbors}

        # ── Step 3: 2-hop from best 1-hop's neighbors ──
        best_2hop = None
        if best_1hop and best_1hop["edge_type"] != "CONTINUOUS":
            ref_guids = [n["guid"] for n in best_1hop["neighbors"]]
            hop2_edges = discover_hop2_edges(neo4j_conn, ref_guids, model_key)

            for edge_type2, neighbors2 in hop2_edges.items():
                obj_type2 = neighbors2[0]["ifc_type"]
                obj_mat2 = neighbors2[0].get("material", "") or ""

                cands2 = run_2hop_query(
                    neo4j_conn, gt_type, gt_storey, model_key,
                    best_1hop["edge_type"],
                    best_1hop["neighbors"][0].get("ifc_type", ""),
                    "",  # no mat filter hop1
                    edge_type2, obj_type2, ""  # no mat filter hop2
                )
                pool2 = len(cands2)
                gt_in2 = gt_guid in {c["guid"] for c in cands2}

                if gt_in2 and (best_2hop is None or pool2 < best_2hop["pool"]):
                    best_2hop = {
                        "edge_type": f"{best_1hop['edge_type']}→{edge_type2}",
                        "pool": pool2, "gt_in": gt_in2,
                        "obj_type2": obj_type2,
                    }

        # Also try 2-hop with material filter for tighter pool
        best_2hop_mat = None
        if best_1hop and best_1hop["edge_type"] != "CONTINUOUS":
            ref_guids = [n["guid"] for n in best_1hop["neighbors"]]
            hop2_edges = discover_hop2_edges(neo4j_conn, ref_guids, model_key)

            for edge_type2, neighbors2 in hop2_edges.items():
                obj_type2 = neighbors2[0]["ifc_type"]
                obj_mat2 = neighbors2[0].get("material", "") or ""
                if not obj_mat2:
                    continue

                cands2m = run_2hop_query(
                    neo4j_conn, gt_type, gt_storey, model_key,
                    best_1hop["edge_type"],
                    best_1hop["neighbors"][0].get("ifc_type", ""),
                    "",
                    edge_type2, obj_type2, obj_mat2
                )
                pool2m = len(cands2m)
                gt_in2m = gt_guid in {c["guid"] for c in cands2m}

                if gt_in2m and (best_2hop_mat is None or pool2m < best_2hop_mat["pool"]):
                    best_2hop_mat = {
                        "edge_type": f"{best_1hop['edge_type']}→{edge_type2}+mat",
                        "pool": pool2m, "gt_in": gt_in2m,
                    }

        rec = {
            "case_id": case_id,
            "model": model_key,
            "gt_guid": gt_guid,
            "gt_type": gt_type,
            "gt_storey": gt_storey,
            "total": total,
            "p1_pool": p1_pool,
            "p1_gt_in": p1_gt_in,
            "has_edges": len(gt_edges) > 0,
            "edge_types_found": list(gt_edges.keys()),
            "all_1hop": all_1hop,
            "best_1hop_edge": best_1hop["edge_type"] if best_1hop else None,
            "best_1hop_pool": best_1hop["pool"] if best_1hop else None,
            "best_1hop_gt_in": best_1hop["gt_in"] if best_1hop else None,
            "best_2hop_edge": best_2hop["edge_type"] if best_2hop else None,
            "best_2hop_pool": best_2hop["pool"] if best_2hop else None,
            "best_2hop_gt_in": best_2hop["gt_in"] if best_2hop else None,
            "best_2hop_mat_edge": best_2hop_mat["edge_type"] if best_2hop_mat else None,
            "best_2hop_mat_pool": best_2hop_mat["pool"] if best_2hop_mat else None,
            "best_2hop_mat_gt_in": best_2hop_mat["gt_in"] if best_2hop_mat else None,
        }
        results.append(rec)

        # Print progress
        edges_str = ",".join(gt_edges.keys()) if gt_edges else "none"
        h1 = f"→ 1hop={best_1hop['pool']}({best_1hop['edge_type']})" if best_1hop else ""
        h2 = f"→ 2hop={best_2hop['pool']}({best_2hop['edge_type']})" if best_2hop else ""
        h2m = f"→ 2hop+mat={best_2hop_mat['pool']}" if best_2hop_mat else ""
        print(f"  {case_id:40s} [{model_key}] {gt_type:25s} total={total} P1={p1_pool} {h1} {h2} {h2m}  edges=[{edges_str}]")

    # ── Analysis ────────────────────────────────────────────────────────
    print(f"\nSkipped: {skipped}")
    print("\n" + "=" * 80)
    print("ORACLE WATERFALL SUMMARY (GT-Reverse)")
    print("=" * 80)

    has_edge = [r for r in results if r["has_edges"]]
    has_1hop = [r for r in results if r["best_1hop_pool"] is not None]
    has_2hop = [r for r in results if r["best_2hop_pool"] is not None]
    has_2hop_mat = [r for r in results if r["best_2hop_mat_pool"] is not None]
    no_edge = [r for r in results if not r["has_edges"]]

    for label, subset in [
        (f"All cases (n={len(results)})", results),
        (f"Cases with edges (n={len(has_edge)})", has_edge),
        (f"Cases with 1-hop result (n={len(has_1hop)})", has_1hop),
        (f"Cases with 2-hop result (n={len(has_2hop)})", has_2hop),
        (f"Cases with 2-hop+mat result (n={len(has_2hop_mat)})", has_2hop_mat),
        (f"No edges (attr-only) (n={len(no_edge)})", no_edge),
    ]:
        if not subset:
            continue
        n = len(subset)
        avg_total = np.mean([r["total"] for r in subset])
        avg_p1 = np.mean([r["p1_pool"] for r in subset])
        p1_gt = sum(1 for r in subset if r["p1_gt_in"])

        print(f"\n--- {label} ---")
        print(f"  Total (avg):   {avg_total:.0f}")
        print(f"  P1 (avg):      {avg_p1:.1f}   GT-in: {p1_gt}/{n} ({100*p1_gt/n:.1f}%)")

        h1_sub = [r for r in subset if r["best_1hop_pool"] is not None]
        if h1_sub:
            avg_h1 = np.mean([r["best_1hop_pool"] for r in h1_sub])
            h1_gt = sum(1 for r in h1_sub if r["best_1hop_gt_in"])
            print(f"  1-hop (avg):   {avg_h1:.1f}   GT-in: {h1_gt}/{len(h1_sub)} ({100*h1_gt/len(h1_sub):.1f}%)")

        h2_sub = [r for r in subset if r["best_2hop_pool"] is not None]
        if h2_sub:
            avg_h2 = np.mean([r["best_2hop_pool"] for r in h2_sub])
            h2_gt = sum(1 for r in h2_sub if r["best_2hop_gt_in"])
            print(f"  2-hop (avg):   {avg_h2:.1f}   GT-in: {h2_gt}/{len(h2_sub)} ({100*h2_gt/len(h2_sub):.1f}%)")

        h2m_sub = [r for r in subset if r["best_2hop_mat_pool"] is not None]
        if h2m_sub:
            avg_h2m = np.mean([r["best_2hop_mat_pool"] for r in h2m_sub])
            h2m_gt = sum(1 for r in h2m_sub if r["best_2hop_mat_gt_in"])
            print(f"  2hop+mat (avg):{avg_h2m:.1f}   GT-in: {h2m_gt}/{len(h2m_sub)} ({100*h2m_gt/len(h2m_sub):.1f}%)")

    # Per-edge-type breakdown
    print(f"\n--- Per-Edge-Type (best 1-hop) ---")
    edge_groups = defaultdict(list)
    for r in results:
        if r["best_1hop_edge"]:
            edge_groups[r["best_1hop_edge"]].append(r)

    for edge, group in sorted(edge_groups.items(), key=lambda x: np.mean([r["best_1hop_pool"] for r in x[1]])):
        n = len(group)
        avg_p1 = np.mean([r["p1_pool"] for r in group])
        avg_h1 = np.mean([r["best_1hop_pool"] for r in group])
        reduction = (1 - avg_h1 / avg_p1) * 100 if avg_p1 > 0 else 0
        p1_gt = sum(1 for r in group if r["p1_gt_in"])
        h1_gt = sum(1 for r in group if r["best_1hop_gt_in"])
        print(f"  {edge:15s} (n={n:3d}): P1={avg_p1:6.1f} → 1hop={avg_h1:5.1f} ({reduction:+.0f}%)  "
              f"GT: P1={p1_gt}/{n}, 1hop={h1_gt}/{n}")

    # ── Waterfall Plot ──────────────────────────────────────────────────
    # Only cases with at least 1-hop
    plot_cases = has_1hop
    n_plot = len(plot_cases)

    stages = ["Full\nElements", "P1\n(storey+type)", "Best 1-Hop\nSpatial"]
    pools = [
        np.mean([r["total"] for r in plot_cases]),
        np.mean([r["p1_pool"] for r in plot_cases]),
        np.mean([r["best_1hop_pool"] for r in plot_cases]),
    ]
    gt_rates = [
        1.0,
        sum(1 for r in plot_cases if r["p1_gt_in"]) / n_plot,
        sum(1 for r in plot_cases if r["best_1hop_gt_in"]) / n_plot,
    ]

    if has_2hop:
        stages.append("Best 2-Hop\nSpatial")
        pools.append(np.mean([r["best_2hop_pool"] for r in has_2hop]))
        gt_rates.append(sum(1 for r in has_2hop if r["best_2hop_gt_in"]) / len(has_2hop))

    if has_2hop_mat:
        stages.append("2-Hop\n+Material")
        pools.append(np.mean([r["best_2hop_mat_pool"] for r in has_2hop_mat]))
        gt_rates.append(sum(1 for r in has_2hop_mat if r["best_2hop_mat_gt_in"]) / len(has_2hop_mat))

    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(stages))
    width = 0.5
    colors = ["#4472C4", "#5B9BD5", "#ED7D31", "#FFC000", "#70AD47"][:len(stages)]

    bars = ax1.bar(x, pools, width, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Average Pool Size (log scale)", fontsize=12)
    ax1.set_ylim(0.5, max(pools) * 5)

    for bar, val in zip(bars, pools):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    for i in range(len(pools) - 1):
        reduction = (1 - pools[i + 1] / pools[i]) * 100
        mid_x = (x[i] + x[i + 1]) / 2
        mid_y = (pools[i] * pools[i + 1]) ** 0.5
        ax1.annotate(f"−{reduction:.0f}%",
                     xy=(mid_x, mid_y), fontsize=11, ha="center", va="center",
                     color="red", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.9))

    ax2 = ax1.twinx()
    ax2.plot(x, [g * 100 for g in gt_rates], "D-", color="green", markersize=8,
             linewidth=2, label="GT-in-Pool %")
    for xi, gt in zip(x, gt_rates):
        ax2.text(xi + 0.15, gt * 100 + 2, f"{gt*100:.0f}%", fontsize=10,
                 color="green", fontweight="bold")
    ax2.set_ylabel("GT-in-Pool (%)", fontsize=12, color="green")
    ax2.set_ylim(0, 115)
    ax2.tick_params(axis="y", labelcolor="green")

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=11)
    n_2h = len(has_2hop)
    n_2hm = len(has_2hop_mat)
    ax1.set_title(
        f"Oracle Pipeline Waterfall — GT-Reverse (Perfect Extraction)\n"
        f"n={n_plot} with 1-hop, {n_2h} with 2-hop, {n_2hm} with 2-hop+material",
        fontsize=13
    )
    ax1.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "oracle_waterfall_v2.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nWaterfall plot saved: {out_path}")

    # ── Per-Edge-Type Plot ──────────────────────────────────────────────
    if edge_groups:
        n_types = len(edge_groups)
        fig2, axes = plt.subplots(1, n_types, figsize=(4.5 * n_types, 5), sharey=True)
        if n_types == 1:
            axes = [axes]

        colors_map = {
            "FILLS": "#ED7D31", "ADJACENT_TO": "#5B9BD5",
            "CONTINUOUS": "#70AD47", "NEXT_TO": "#FFC000", "CONNECTS_TO": "#9E480E"
        }

        for ax, (edge, group) in zip(axes, sorted(edge_groups.items())):
            n = len(group)
            vals = [
                np.mean([r["total"] for r in group]),
                np.mean([r["p1_pool"] for r in group]),
                np.mean([r["best_1hop_pool"] for r in group]),
            ]
            gt_in = [
                n,
                sum(1 for r in group if r["p1_gt_in"]),
                sum(1 for r in group if r["best_1hop_gt_in"]),
            ]
            labels = ["Full", "P1", "1-Hop"]
            color = colors_map.get(edge, "#888888")

            bars = ax.bar(range(len(vals)), vals, color=color, edgecolor="black", linewidth=0.5)
            ax.set_yscale("log")
            ax.set_ylim(0.5, max(vals) * 8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_title(f"{edge}\n(n={n})", fontsize=11, fontweight="bold")

            for bar, val, gt in zip(bars, vals, gt_in):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2,
                        f"{val:.0f}\nGT:{gt}/{n}", ha="center", va="bottom", fontsize=9)

        axes[0].set_ylabel("Pool Size (log)", fontsize=11)
        fig2.suptitle("Oracle Waterfall by Best 1-Hop Edge Type (GT-Reverse)", fontsize=13, fontweight="bold")
        fig2.tight_layout()

        out_path2 = os.path.join(OUTPUT_DIR, "oracle_waterfall_by_predicate.png")
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
        print(f"Per-predicate plot saved: {out_path2}")

    # Save raw data
    out_json = os.path.join(OUTPUT_DIR, "oracle_waterfall_data.jsonl")
    with open(out_json, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Raw data saved: {out_json}")


if __name__ == "__main__":
    main()
