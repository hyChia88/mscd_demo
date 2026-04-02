#!/usr/bin/env python3
"""
add_topology_edges.py — Add topology edges + properties to Neo4j after IFC export.

Adds:
  1. ADJACENT_TO edges  — centroid distance < threshold (default 1500mm), cross-type pairs
  2. top_constraint     — property on IFCElement nodes for CONTINUOUS query support

Run after ifc_to_neo4j.py (IFCElement nodes must already exist).

Usage (from mscd_demo/):
    conda run -n mscd_demo python legacy/script/add_topology_edges.py \
        --index ../data_curation/references/element_index.jsonl \
        --threshold 1500.0
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from py2neo import Graph


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def connect(uri: str, user: str, password: str) -> Graph:
    g = Graph(uri, auth=(user, password))
    g.run("RETURN 1")
    print(f"Connected to Neo4j at {uri}")
    return g


# ---------------------------------------------------------------------------
# ADJACENT_TO edges
# ---------------------------------------------------------------------------

def add_adjacent_to_edges(
    g: Graph,
    elements: List[Dict],
    threshold: float = 1500.0,
    min_dist: float = 100.0,
) -> int:
    """
    Create (a)-[:ADJACENT_TO]->(b) bidirectional edges for cross-type element
    pairs whose centroids are within threshold (exclusive min_dist floor filters
    degenerate shared-origin cases).

    Uses MERGE so re-runs are safe.
    """
    # Group by storey
    with_storey: Dict[str, List[Dict]] = defaultdict(list)
    no_storey: List[Dict] = []

    for el in elements:
        if not el.get("centroid"):
            continue
        storey = el.get("storey_name", "")
        (with_storey[storey] if storey else no_storey).append(el)

    # Z-band fallback for elements without storey_name (e.g. aggregated railings)
    if no_storey and with_storey:
        z_meds: Dict[str, float] = {}
        for st, grp in with_storey.items():
            zs = sorted(e["centroid"]["z"] for e in grp)
            z_meds[st] = zs[len(zs) // 2]
        for el in no_storey:
            best = min(z_meds, key=lambda s: abs(z_meds[s] - el["centroid"]["z"]))
            with_storey[best].append(el)

    # Dedup by guid within each storey (IfcWallStandardCase double-counted in index)
    for storey in list(with_storey):
        seen: Dict[str, Dict] = {}
        for el in with_storey[storey]:
            seen[el["global_id"]] = el
        with_storey[storey] = list(seen.values())

    pairs: List[Tuple[str, str]] = []
    for storey, group in with_storey.items():
        for i in range(len(group)):
            a = group[i]
            ca = a["centroid"]
            for j in range(i + 1, len(group)):
                b = group[j]
                if a["ifc_class"] == b["ifc_class"]:
                    continue  # Same type — skip
                cb = b["centroid"]
                dx = ca["x"] - cb["x"]
                dy = ca["y"] - cb["y"]
                dz = ca["z"] - cb["z"]
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                if min_dist < dist <= threshold:
                    pairs.append((a["global_id"], b["global_id"]))

    if not pairs:
        print("  No ADJACENT_TO pairs found within threshold.")
        return 0

    # Batch write — both directions in one pass
    cypher = """
    UNWIND $pairs AS p
    MATCH (a:IFCElement {guid: p[0]})
    MATCH (b:IFCElement {guid: p[1]})
    MERGE (a)-[:ADJACENT_TO]->(b)
    MERGE (b)-[:ADJACENT_TO]->(a)
    """
    g.run(cypher, pairs=[[a, b] for a, b in pairs])
    return len(pairs) * 2  # bidirectional


# ---------------------------------------------------------------------------
# CONTINUOUS property
# ---------------------------------------------------------------------------

def add_continuous_property(g: Graph, elements: List[Dict]) -> int:
    """
    Set top_constraint + is_continuous properties on IFCElement nodes for
    elements that span multiple storeys (Top Constraint ≠ Base Constraint in
    the Revit Constraints pset exported to IFC).

    These properties enable the CONTINUOUS Cypher template to use WHERE:
        MATCH (target:IFCElement)
        WHERE target.ifc_type = $subject_type
          AND target.top_constraint CONTAINS $storey
    """
    updates: List[Dict] = []
    seen: set = set()

    for el in elements:
        if el["global_id"] in seen:
            continue
        c = el.get("psets", {}).get("Constraints", {})
        base = str(c.get("Base Constraint", "")).strip()
        top = str(c.get("Top Constraint", "")).strip()

        if not base or not top or "Unconnected" in top:
            continue
        base_norm = base.removeprefix("Level: ").strip()
        top_norm = top.removeprefix("Level: ").strip()
        if base_norm == top_norm:
            continue

        seen.add(el["global_id"])
        updates.append({
            "guid": el["global_id"],
            "top_constraint": top_norm,
            "base_constraint": base_norm,
            "is_continuous": True,
        })

    if not updates:
        print("  No CONTINUOUS elements found.")
        return 0

    cypher = """
    UNWIND $updates AS u
    MATCH (n:IFCElement {guid: u.guid})
    SET n.top_constraint   = u.top_constraint,
        n.base_constraint  = u.base_constraint,
        n.is_continuous    = u.is_continuous
    """
    g.run(cypher, updates=updates)
    return len(updates)


# ---------------------------------------------------------------------------
# CONNECTS_TO edges (wall-to-wall path connections)
# ---------------------------------------------------------------------------

def add_connects_to_edges(g: Graph, ifc_path: str) -> int:
    """
    Create (wall)-[:CONNECTS_TO]->(wall) edges from IfcRelConnectsPathElements.

    These encode wall-to-wall topology — which walls meet at corners/T-junctions.
    Enables the CONNECTS_TO predicate for 2-hop queries:
        Window -[:FILLS]-> Wall -[:CONNECTS_TO]-> adjacent Wall
    """
    import ifcopenshell

    ifc = ifcopenshell.open(ifc_path)
    rels = ifc.by_type("IfcRelConnectsPathElements")

    if not rels:
        print("  No IfcRelConnectsPathElements found.")
        return 0

    pairs = []
    for rel in rels:
        relating = rel.RelatingElement
        related = rel.RelatedElement
        if relating and related:
            pairs.append([relating.GlobalId, related.GlobalId])

    if not pairs:
        return 0

    # Batch write — bidirectional edges
    cypher = """
    UNWIND $pairs AS p
    MATCH (a:IFCElement {guid: p[0]})
    MATCH (b:IFCElement {guid: p[1]})
    MERGE (a)-[:CONNECTS_TO]->(b)
    MERGE (b)-[:CONNECTS_TO]->(a)
    """
    g.run(cypher, pairs=pairs)
    return len(pairs) * 2  # bidirectional


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(g: Graph):
    fills = g.run("MATCH ()-[:FILLS]->()         RETURN count(*) AS c").data()[0]["c"]
    adj   = g.run("MATCH ()-[:ADJACENT_TO]->()   RETURN count(*) AS c").data()[0]["c"]
    conn  = g.run("MATCH ()-[:CONNECTS_TO]->()   RETURN count(*) AS c").data()[0]["c"]
    cont  = g.run(
        "MATCH (n:IFCElement) WHERE n.is_continuous = true RETURN count(*) AS c"
    ).data()[0]["c"]
    print(f"\n  Graph state after topology enrichment:")
    print(f"    FILLS edges          : {fills}")
    print(f"    ADJACENT_TO edges    : {adj}  (bidirectional)")
    print(f"    CONNECTS_TO edges    : {conn}  (bidirectional)")
    print(f"    is_continuous nodes  : {cont}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enrich Neo4j with ADJACENT_TO edges and CONTINUOUS properties"
    )
    parser.add_argument(
        "--index",
        default="../data_curation/references/element_index.jsonl",
        help="Path to element_index.jsonl (must have centroid + psets fields)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1500.0,
        help="ADJACENT_TO centroid distance threshold in same units as centroids"
             " (default: 1500.0 for mm-unit IFC files)",
    )
    parser.add_argument("--ifc",
                        default="../data_curation/ifc_models/AdvancedProject.ifc",
                        help="IFC file path (for CONNECTS_TO edges from IfcRelConnectsPathElements)")
    parser.add_argument("--uri",      default="bolt://localhost:7687")
    parser.add_argument("--user",     default="neo4j")
    parser.add_argument("--password", default="password")
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: index not found at {index_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading element index: {index_path}")
    elements: List[Dict] = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                elements.append(json.loads(line))
    print(f"  {len(elements)} elements loaded")

    g = connect(args.uri, args.user, args.password)

    print(f"\nAdding ADJACENT_TO edges (threshold={args.threshold}mm) ...")
    adj = add_adjacent_to_edges(g, elements, threshold=args.threshold)
    print(f"  Created {adj} ADJACENT_TO edges")

    print("\nAdding top_constraint / is_continuous properties ...")
    cont = add_continuous_property(g, elements)
    print(f"  Updated {cont} nodes")

    ifc_path = Path(args.ifc)
    if ifc_path.exists():
        print(f"\nAdding CONNECTS_TO edges from {ifc_path} ...")
        conn = add_connects_to_edges(g, str(ifc_path))
        print(f"  Created {conn} CONNECTS_TO edges")
    else:
        print(f"\nSkipping CONNECTS_TO — IFC file not found: {ifc_path}")

    verify(g)
    print("\nDone.")


if __name__ == "__main__":
    main()
