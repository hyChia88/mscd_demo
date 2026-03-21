#!/usr/bin/env python3
"""
neo4j_prepare_views.py

Prepare a Neo4j database for thesis screenshots in either:
  - base mode      : IFC-centric graph with core containment + FILLS only
  - enriched mode  : retrieval-oriented graph with all current enrichments

This script is intended for local use before opening Neo4j Desktop Explore
or Bloom-style graph visualization.

Examples:
    conda run -n mscd_demo python mscd_demo/script/neo4j_prepare_views.py --mode base
    conda run -n mscd_demo python mscd_demo/script/neo4j_prepare_views.py --mode enriched

If Neo4j is already running and populated, you can skip the start step:
    conda run -n mscd_demo python mscd_demo/script/neo4j_prepare_views.py --mode enriched --skip-start
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from py2neo import Graph


ROOT = Path(__file__).resolve().parents[2]
MSCD_DIR = ROOT / "mscd_demo"
NEO4J_INIT = MSCD_DIR / "script" / "neo4j_init.sh"
IFC_PATH_DEFAULT = ROOT / "data_curation" / "ifc_models" / "AdvancedProject.ifc"
INDEX_PATH_DEFAULT = ROOT / "data_curation" / "references" / "element_index.jsonl"


def info(msg: str) -> None:
    print(f"[neo4j_prepare_views] {msg}")


def fail(msg: str, code: int = 1) -> None:
    print(f"[neo4j_prepare_views] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    info("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def connect_graph(uri: str, user: str, password: str) -> Graph:
    try:
        g = Graph(uri, auth=(user, password))
        g.run("RETURN 1")
        return g
    except Exception as exc:
        fail(
            "Failed to connect to Neo4j at "
            f"{uri}. Make sure Neo4j is running and credentials are correct.\n{exc}"
        )


def export_base_graph(ifc_path: Path, g: Graph) -> None:
    sys.path.insert(0, str(MSCD_DIR / "src"))
    from ifc_engine import IFCEngine  # imported lazily so script stays lightweight

    info(f"Exporting IFC graph from: {ifc_path}")
    engine = IFCEngine(str(ifc_path), neo4j_conn=g)
    stats = engine.export_to_neo4j(clear_existing=True)
    info(f"Base export complete: {stats}")


def prune_to_base_view(g: Graph) -> None:
    info("Pruning enriched edges/properties to create the base view")

    queries = [
        "MATCH ()-[r:NEXT_TO]->() DELETE r",
        "MATCH ()-[r:CONNECTS_TO]->() DELETE r",
        "MATCH ()-[r:ADJACENT_TO]->() DELETE r",
        """
        MATCH (n:IFCElement)
        REMOVE n.wall_position_index,
               n.wall_child_total,
               n.wall_child_count,
               n.top_constraint,
               n.base_constraint,
               n.is_continuous,
               n.material
        """,
    ]
    for q in queries:
        g.run(q)


def apply_full_enrichment(ifc_path: Path, index_path: Path, uri: str, user: str, password: str) -> None:
    info("Applying full enrichment layer")
    run(
        [
            "conda",
            "run",
            "-n",
            "mscd_demo",
            "python",
            "legacy/script/add_topology_edges.py",
            "--index",
            str(index_path),
            "--ifc",
            str(ifc_path),
            "--uri",
            uri,
            "--user",
            user,
            "--password",
            password,
        ],
        cwd=MSCD_DIR,
    )


def print_stats(g: Graph) -> None:
    queries = {
        "IFCElement nodes": "MATCH (n:IFCElement) RETURN count(n) AS c",
        "IFCStorey nodes": "MATCH (n:IFCStorey) RETURN count(n) AS c",
        "IFCSpace nodes": "MATCH (n:IFCSpace) RETURN count(n) AS c",
        "CONTAINS edges": "MATCH ()-[r:CONTAINS]->() RETURN count(r) AS c",
        "FILLS edges": "MATCH ()-[r:FILLS]->() RETURN count(r) AS c",
        "NEXT_TO edges": "MATCH ()-[r:NEXT_TO]->() RETURN count(r) AS c",
        "CONNECTS_TO edges": "MATCH ()-[r:CONNECTS_TO]->() RETURN count(r) AS c",
        "ADJACENT_TO edges": "MATCH ()-[r:ADJACENT_TO]->() RETURN count(r) AS c",
        "continuous nodes": "MATCH (n:IFCElement) WHERE n.is_continuous = true RETURN count(n) AS c",
        "material-tagged nodes": "MATCH (n:IFCElement) WHERE n.material IS NOT NULL RETURN count(n) AS c",
    }

    print("")
    print("=== Graph State ===")
    for label, q in queries.items():
        try:
            c = g.run(q).data()[0]["c"]
        except Exception:
            c = "ERR"
        print(f"{label:22}: {c}")
    print("")


def print_visualization_tips(mode: str) -> None:
    print("=== Suggested Explore / Bloom Views ===")
    if mode == "base":
        print("1. Search for an IFCStorey node such as '1 - First Floor'")
        print("2. Expand outgoing CONTAINS edges")
        print("3. Search for one wall node and expand its FILLS edges")
        print("4. Capture an overview screenshot emphasizing the IFC-native skeleton")
    else:
        print("1. Search for a dense wall node with nearby fillers")
        print("2. Expand CONNECTS_TO, NEXT_TO, FILLS, and ADJACENT_TO")
        print("3. Show node properties such as material and is_continuous")
        print("4. Capture an overview screenshot emphasizing retrieval-oriented enrichment")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare base or enriched Neo4j view for thesis screenshots.")
    parser.add_argument("--mode", choices=["base", "enriched"], required=True)
    parser.add_argument("--ifc", type=Path, default=IFC_PATH_DEFAULT)
    parser.add_argument("--index", type=Path, default=INDEX_PATH_DEFAULT)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--skip-start", action="store_true", help="Skip neo4j_init.sh --start-only")
    args = parser.parse_args()

    if not args.ifc.exists():
        fail(f"IFC file not found: {args.ifc}")
    if not args.index.exists():
        fail(f"Index file not found: {args.index}")

    if not args.skip_start:
        if not NEO4J_INIT.exists():
            fail(f"neo4j_init.sh not found: {NEO4J_INIT}")
        run(["bash", str(NEO4J_INIT), "--start-only"], cwd=MSCD_DIR)

    g = connect_graph(args.uri, args.user, args.password)

    export_base_graph(args.ifc, g)

    if args.mode == "base":
        prune_to_base_view(g)
    else:
        apply_full_enrichment(args.ifc, args.index, args.uri, args.user, args.password)

    print_stats(g)
    print_visualization_tips(args.mode)
    info(f"Done. Neo4j is now prepared in '{args.mode}' mode.")


if __name__ == "__main__":
    main()
