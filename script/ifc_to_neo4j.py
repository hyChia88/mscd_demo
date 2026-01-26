#!/usr/bin/env python3
"""
IFC to Neo4j Graph Export Script

Exports IFC semantic model to Neo4j graph database for advanced reasoning queries.
Based on Zhu et al. (2023) IFC-Graph methodology.

Usage:
    python script/ifc_to_neo4j.py [--clear]

Options:
    --clear     Clear existing IFC nodes before import

Prerequisites:
    - Neo4j server running (default: bolt://localhost:7687)
    - py2neo package installed: pip install py2neo
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ifc_engine import IFCEngine


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        return {
            "ifc": {"model_path": "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"},
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password"
            }
        }

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Export IFC model to Neo4j graph database")
    parser.add_argument("--clear", action="store_true", help="Clear existing IFC nodes before import")
    parser.add_argument("--ifc", type=str, help="Override IFC file path")
    args = parser.parse_args()

    # Load config
    config = load_config()
    base_dir = Path(__file__).parent.parent

    # Get IFC path
    ifc_path = args.ifc or str(base_dir / config.get("ifc", {}).get("model_path"))

    # Get Neo4j config
    neo4j_config = config.get("neo4j", {})
    neo4j_uri = neo4j_config.get("uri", "bolt://localhost:7687")
    neo4j_user = neo4j_config.get("user", "neo4j")
    neo4j_password = neo4j_config.get("password", "password")

    print("=" * 60)
    print("🔗 IFC to Neo4j Graph Export")
    print("=" * 60)
    print(f"   IFC File: {ifc_path}")
    print(f"   Neo4j URI: {neo4j_uri}")
    print("=" * 60)

    # Connect to Neo4j
    try:
        from py2neo import Graph
        graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print("✅ Connected to Neo4j")
    except ImportError:
        print("❌ py2neo not installed. Run: pip install py2neo")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("   Make sure Neo4j is running and credentials are correct.")
        sys.exit(1)

    # Initialize IFC Engine with Neo4j connection
    try:
        engine = IFCEngine(ifc_path, neo4j_conn=graph)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Export to Neo4j
    print("\n🚀 Starting export...")
    stats = engine.export_to_neo4j(clear_existing=args.clear)

    if "error" not in stats:
        print("\n" + "=" * 60)
        print("📊 Export Summary")
        print("=" * 60)
        print(f"   Nodes created: {stats['nodes']}")
        print(f"   Relationships created: {stats['relationships']}")
        print("=" * 60)

        # Show sample queries
        print("\n💡 Sample Cypher queries to try:")
        print("   - MATCH (n:IFCElement) RETURN n LIMIT 10")
        print("   - MATCH (s:IFCStorey)-[:CONTAINS]->(e) RETURN s.name, count(e)")
        print("   - MATCH (e:IFCElement) WHERE e.FireRating IS NOT NULL RETURN e")
    else:
        print(f"❌ Export failed: {stats['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
