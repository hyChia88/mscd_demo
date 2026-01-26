#!/usr/bin/env python3
"""
IFC Engine Test & Visualization Script

Verifies IFCEngine functionality and displays model structure.

Usage:
    python script/test_ifc_engine.py
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ifc_engine import IFCEngine


def load_config():
    """Load IFC path from config"""
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config.get("ifc", {}).get("model_path")
    return None


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    print_section("IFC Engine Test & Visualization")

    # Load IFC path from config
    base_dir = Path(__file__).parent.parent
    ifc_path = load_config()

    if ifc_path:
        full_path = base_dir / ifc_path
    else:
        full_path = base_dir / "data" / "ifc" / "AdvancedProject" / "IFC" / "AdvancedProject.ifc"

    print(f"IFC File: {full_path}")

    # Initialize engine
    try:
        engine = IFCEngine(str(full_path))
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # =========================================================================
    # 1. Spatial Index Overview
    # =========================================================================
    print_section("1. Spatial Index (Rooms/Storeys)")

    print(f"Total groups indexed: {len(engine.spatial_index)}")
    print("\nAvailable spaces/groups:")

    for space_name, elements in sorted(engine.spatial_index.items()):
        print(f"  • {space_name}: {len(elements)} elements")

    # =========================================================================
    # 2. Element Type Statistics
    # =========================================================================
    print_section("2. Element Type Statistics")

    type_counts = {}
    for space_name, elements in engine.spatial_index.items():
        for el in elements:
            el_type = el.get("type", "Unknown")
            type_counts[el_type] = type_counts.get(el_type, 0) + 1

    print(f"{'Type':<30} {'Count':>10}")
    print("-" * 42)
    for el_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"{el_type:<30} {count:>10}")

    # =========================================================================
    # 3. Sample Elements
    # =========================================================================
    print_section("3. Sample Elements (First 5)")

    sample_count = 0
    for space_name, elements in engine.spatial_index.items():
        for el in elements[:2]:  # 2 per space
            if sample_count >= 5:
                break
            print(f"\n  [{sample_count+1}] {el.get('name', 'Unnamed')}")
            print(f"      GUID: {el.get('guid')}")
            print(f"      Type: {el.get('type')}")
            print(f"      Space: {space_name}")
            sample_count += 1
        if sample_count >= 5:
            break

    # =========================================================================
    # 4. Test Property Extraction
    # =========================================================================
    print_section("4. Property Extraction Test")

    # Get first element with a GUID
    test_guid = None
    for space_name, elements in engine.spatial_index.items():
        if elements:
            test_guid = elements[0].get("guid")
            test_name = elements[0].get("name")
            break

    if test_guid:
        print(f"Testing element: {test_name}")
        print(f"GUID: {test_guid}")

        props = engine.get_element_properties(test_guid)

        if "error" not in props:
            print("\nBasic Properties:")
            for key in ["GlobalId", "Name", "Type", "ObjectType"]:
                if key in props:
                    print(f"  {key}: {props[key]}")

            if "PropertySets" in props:
                print("\nProperty Sets:")
                for pset_name, pset_props in props["PropertySets"].items():
                    print(f"  [{pset_name}]")
                    for prop_name, prop_value in list(pset_props.items())[:5]:
                        print(f"    • {prop_name}: {prop_value}")
                    if len(pset_props) > 5:
                        print(f"    ... and {len(pset_props)-5} more properties")
        else:
            print(f"  ❌ {props['error']}")

    # =========================================================================
    # 5. Test Spatial Query
    # =========================================================================
    print_section("5. Spatial Query Test")

    # Try to find elements in a space
    test_queries = ["floor", "level", "storey", "room", "01", "ground"]

    for query in test_queries:
        results = engine.find_elements_in_space(query)
        if results:
            print(f"\n  Query: '{query}' → Found {len(results)} elements")
            for el in results[:3]:
                print(f"    • {el.get('name')} ({el.get('type')})")
            if len(results) > 3:
                print(f"    ... and {len(results)-3} more")
            break
    else:
        print("  No elements found with test queries")

    # =========================================================================
    # 6. Neo4j Status
    # =========================================================================
    print_section("6. Neo4j Integration Status")

    if engine.neo4j_conn:
        print("  ✅ Neo4j connection active")
    else:
        print("  ⚪ Neo4j not connected (optional)")
        print("  To enable: python script/ifc_to_neo4j.py")

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Summary")

    total_elements = sum(len(els) for els in engine.spatial_index.values())
    print(f"  ✅ IFC Engine loaded successfully")
    print(f"  📊 {len(engine.spatial_index)} spatial groups")
    print(f"  🧱 {total_elements} total elements indexed")
    print(f"  📋 {len(type_counts)} unique element types")


if __name__ == "__main__":
    main()
