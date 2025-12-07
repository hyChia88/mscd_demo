#!/usr/bin/env python3
"""
Complete Visual Matching Workflow - IFC Integration

This demonstrates the full workflow:
1. Load IFC model
2. Query elements by location
3. Use visual matching to identify specific element
4. Get detailed properties

Usage:
    python examples/visual_matching_workflow.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ifc_engine import IFCEngine
from src.visual_matcher import VisualAligner


def workflow_damage_report():
    """Simulated workflow: Inspector reports damage → System identifies element"""

    print("\n" + "="*70)
    print(" COMPLETE VISUAL MATCHING WORKFLOW")
    print(" Site Damage Report → BIM Element Identification")
    print("="*70)

    # ========================================================================
    # STEP 1: Initialize Systems
    # ========================================================================
    print("\n📦 STEP 1: Initializing Systems")
    print("─"*70)

    print("  Loading IFC model...")
    ifc_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "BasicHouse.ifc"
    )

    if not os.path.exists(ifc_path):
        print(f"\n❌ Error: IFC file not found at {ifc_path}")
        print("   Please ensure BasicHouse.ifc exists in the data/ folder")
        return 1

    engine = IFCEngine(ifc_path)

    print("  Loading visual matcher...")
    aligner = VisualAligner()

    print("✅ Systems ready\n")

    # ========================================================================
    # STEP 2: Site Report Input
    # ========================================================================
    print("📋 STEP 2: Site Damage Report")
    print("─"*70)

    location = "floor 0"
    observation = "Water damage on brown wooden base cabinet near sink area"

    print(f"  Location: {location}")
    print(f"  Observation: {observation}")
    print()

    # ========================================================================
    # STEP 3: Query BIM Model
    # ========================================================================
    print("🔍 STEP 3: Querying BIM Model")
    print("─"*70)

    print(f"  Searching for elements in '{location}'...")
    all_elements = engine.find_elements_in_space(location)

    if not all_elements:
        print(f"\n❌ No elements found in '{location}'")
        print("   Try running: python src/inspect_ifc.py")
        print("   to see available spaces in your IFC file")
        return 1

    print(f"  ✅ Found {len(all_elements)} total elements")

    # Filter for relevant element type (cabinets)
    print(f"  Filtering for cabinets...")
    cabinets = [
        el for el in all_elements
        if "cabinet" in el['name'].lower() or el['type'] == 'IfcFurniture'
    ]

    print(f"  ✅ Found {len(cabinets)} cabinet/furniture elements\n")

    if not cabinets:
        print("  ⚠️  No cabinets found. Using all elements for demo...")
        cabinets = all_elements[:10]  # Limit to first 10

    # ========================================================================
    # STEP 4: Prepare for Visual Matching
    # ========================================================================
    print("🎯 STEP 4: Preparing Visual Matching")
    print("─"*70)

    # Limit to first 10 for performance
    candidates = cabinets[:10]

    # Create rich descriptions from BIM data
    candidate_descriptions = []
    candidate_guids = []
    candidate_info = []

    print("  Candidate elements:")
    for i, item in enumerate(candidates, 1):
        # Create description
        desc = f"{item['name']} ({item['type']})"
        if item.get('description'):
            desc += f" - {item['description']}"

        candidate_descriptions.append(desc)
        candidate_guids.append(item['guid'])
        candidate_info.append(item)

        print(f"    {i}. {desc}")

    print()

    # ========================================================================
    # STEP 5: Visual Matching
    # ========================================================================
    print("🔬 STEP 5: Running Visual Similarity Analysis")
    print("─"*70)

    print(f"  Comparing observation to {len(candidate_descriptions)} candidates...")

    idx, score, match = aligner.find_best_match(
        observation,
        candidate_descriptions
    )

    print(f"\n  ✅ Match found!\n")

    # ========================================================================
    # STEP 6: Results
    # ========================================================================
    print("📊 STEP 6: Match Results")
    print("="*70)

    matched_element = candidate_info[idx]
    matched_guid = candidate_guids[idx]

    print(f"\n🎯 IDENTIFIED ELEMENT:")
    print(f"  Name:        {matched_element['name']}")
    print(f"  Type:        {matched_element['type']}")
    print(f"  GUID:        {matched_guid}")
    print(f"  Confidence:  {score:.2%}")

    if matched_element.get('description'):
        print(f"  Description: {matched_element['description']}")

    # ========================================================================
    # STEP 7: Get Detailed Properties
    # ========================================================================
    print(f"\n📄 STEP 7: Fetching Detailed Properties")
    print("─"*70)

    details = engine.get_element_properties(matched_guid)
    print(f"\n{details}\n")

    # ========================================================================
    # STEP 8: Summary & Next Actions
    # ========================================================================
    print("📝 STEP 8: Summary & Recommended Actions")
    print("="*70)

    print(f"\n✅ Successfully identified damaged element:")
    print(f"   Element: {matched_element['name']}")
    print(f"   GUID: {matched_guid}")
    print(f"   Match confidence: {score:.2%}")

    print(f"\n📋 Recommended next actions:")
    print(f"   1. Create work order for GUID: {matched_guid}")
    print(f"   2. Schedule inspection of {matched_element['name']}")
    print(f"   3. Document damage with photos")
    print(f"   4. Generate 3D view for verification")

    print("\n" + "="*70)
    print("✅ Workflow completed successfully!")
    print("="*70 + "\n")

    return 0


def workflow_comparison():
    """Demo: Compare multiple observations"""

    print("\n" + "="*70)
    print(" MULTI-OBSERVATION WORKFLOW")
    print(" Processing Multiple Damage Reports")
    print("="*70)

    # Initialize
    print("\n📦 Initializing...")
    ifc_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "BasicHouse.ifc"
    )

    if not os.path.exists(ifc_path):
        print(f"\n❌ Error: IFC file not found")
        return 1

    engine = IFCEngine(ifc_path)
    aligner = VisualAligner()

    # Multiple damage reports
    reports = [
        ("floor 0", "Cracked grey concrete wall"),
        ("floor 0", "Damaged wooden brown cabinet"),
        ("floor 0", "Broken window frame, aluminum"),
    ]

    print(f"\n📋 Processing {len(reports)} damage reports...\n")

    results = []

    for location, observation in reports:
        print(f"{'─'*70}")
        print(f"Location: {location}")
        print(f"Observation: {observation}")

        # Query BIM
        elements = engine.find_elements_in_space(location)
        if not elements:
            print("  ❌ No elements found")
            continue

        # Prepare candidates
        candidates = elements[:20]  # Limit for performance
        descriptions = [f"{el['name']} ({el['type']})" for el in candidates]
        guids = [el['guid'] for el in candidates]

        # Visual matching
        idx, score, match = aligner.find_best_match(observation, descriptions)

        results.append({
            'observation': observation,
            'match': match,
            'guid': guids[idx],
            'confidence': score
        })

        print(f"  → Match: {match}")
        print(f"  → GUID: {guids[idx]}")
        print(f"  → Confidence: {score:.2%}\n")

    # Summary
    print("="*70)
    print("📊 SUMMARY")
    print("="*70)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['observation']}")
        print(f"   └─ {result['match']} ({result['confidence']:.0%})")

    print("\n" + "="*70 + "\n")

    return 0


def main():
    """Run workflows"""
    print("\n" + "="*70)
    print(" VISUAL MATCHING WORKFLOW DEMONSTRATIONS")
    print("="*70)

    print("\nSelect workflow:")
    print("  1. Single damage report (detailed)")
    print("  2. Multiple damage reports (batch)")
    print("  3. Run both")

    try:
        # For automated demo, run both
        choice = "3"

        if choice in ["1", "3"]:
            result = workflow_damage_report()
            if result != 0:
                return result

        if choice in ["2", "3"]:
            result = workflow_comparison()
            if result != 0:
                return result

        print("\n✅ All workflows completed!\n")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
