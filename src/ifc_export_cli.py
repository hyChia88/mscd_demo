#!/usr/bin/env python3
"""CLI wrapper for IFC → Neo4j export. Used by neo4j_init.sh."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from py2neo import Graph
from ifc_engine import IFCEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", required=True, help="Path to IFC file")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    args = parser.parse_args()

    if not os.path.exists(args.ifc):
        print(f"Error: IFC file not found: {args.ifc}", file=sys.stderr)
        sys.exit(1)

    g = Graph(args.uri, auth=(args.user, args.password))
    engine = IFCEngine(args.ifc, neo4j_conn=g)
    stats = engine.export_to_neo4j(clear_existing=True)

    nodes = g.run("MATCH (n:IFCElement) RETURN count(n) AS c").data()[0]["c"]
    fills = g.run("MATCH ()-[:FILLS]->() RETURN count(*) AS c").data()[0]["c"]
    print(f"Exported: {nodes} nodes, {fills} FILLS edges")

    if nodes == 0:
        print("Error: export produced 0 nodes", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
