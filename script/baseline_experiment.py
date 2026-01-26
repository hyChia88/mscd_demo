#!/usr/bin/env python3
"""
Baseline Experiment: Quantifying Redundancy Without 4D Context

This experiment demonstrates the disambiguation challenge in BIM element queries.
Based on GT_007: When querying for a window without 4D task context,
how many redundant results are returned?

Research Question: RQ3 (Abductive Reasoning)
Hypothesis: Without semantic context (4D task, role, phase), queries return
            N redundant elements where N = number of floors with identical geometry.

Usage:
    python script/baseline_experiment.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ifc_engine import IFCEngine


def load_config():
    """Load configuration from config.yaml"""
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {"ifc": {"model_path": "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"}}


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def run_baseline_experiment():
    """
    Baseline Experiment: Query without 4D context

    Scenario (GT_007):
    - User sends image of window sealant issue
    - User asks: "Is the sealant leaking here?"
    - Ground Truth: Window on Level 6
    - Without context: System returns ALL windows matching visual description
    """

    print_section("BASELINE EXPERIMENT: Redundancy Without 4D Context")
    print("Based on GT_007_RQ3_HighRise_Abductive")
    print("\nHypothesis: Without semantic context, queries return redundant elements")

    # Load IFC model
    config = load_config()
    base_dir = Path(__file__).parent.parent
    ifc_path = str(base_dir / config.get("ifc", {}).get("model_path"))

    print(f"\nLoading IFC: {ifc_path}")
    engine = IFCEngine(ifc_path)

    # =========================================================================
    # Experiment 1: Query "window" without any context
    # =========================================================================
    print_section("Experiment 1: Raw Query - 'window'")

    all_windows = engine.file.by_type("IfcWindow")
    print(f"Total IfcWindow elements in model: {len(all_windows)}")

    # Group by name pattern (to find identical windows across floors)
    window_by_name = defaultdict(list)
    for w in all_windows:
        # Extract base name (remove floor-specific suffix if present)
        name = w.Name if w.Name else "Unnamed"
        window_by_name[name].append({
            "guid": w.GlobalId,
            "name": name,
            "type": w.is_a()
        })

    print(f"\nUnique window types by name: {len(window_by_name)}")
    print("\nWindow distribution:")
    for name, windows in sorted(window_by_name.items(), key=lambda x: -len(x[1])):
        print(f"  • {name}: {len(windows)} instances")
        if len(windows) <= 5:
            for w in windows:
                print(f"      GUID: {w['guid']}")

    # =========================================================================
    # Experiment 2: Query "BALANS" window (GT_007 target type)
    # =========================================================================
    print_section("Experiment 2: Query - 'BALANS Fixed Single Window'")

    balans_windows = [w for w in all_windows if w.Name and "BALANS" in w.Name.upper()]
    print(f"Windows matching 'BALANS': {len(balans_windows)}")

    if balans_windows:
        print("\n⚠️  REDUNDANCY PROBLEM:")
        print(f"   Without 4D context, system returns {len(balans_windows)} candidates")
        print(f"   Ground Truth expects: 1 specific window (Level 6)")
        print(f"   Redundancy factor: {len(balans_windows)}x\n")

        for i, w in enumerate(balans_windows, 1):
            props = engine.get_element_properties(w.GlobalId)
            psets = props.get("PropertySets", {})

            # Try to find level/storey info
            level_info = "Unknown"
            for pset_name, pset_props in psets.items():
                if "Level" in str(pset_props) or "Storey" in str(pset_props):
                    level_info = str(pset_props)[:50]

            print(f"  [{i}] {w.Name}")
            print(f"      GUID: {w.GlobalId}")
            print(f"      Level hint: {level_info}")

    # =========================================================================
    # Experiment 3: Spatial Index Query
    # =========================================================================
    print_section("Experiment 3: Spatial Index - Query by Level")

    print("Available spatial groups in model:")
    for space_name in sorted(engine.spatial_index.keys()):
        element_count = len(engine.spatial_index[space_name])
        window_count = sum(1 for e in engine.spatial_index[space_name] if "Window" in e.get("type", ""))
        if window_count > 0:
            print(f"  • {space_name}: {element_count} elements ({window_count} windows)")

    # Query "6 - Sixth Floor" specifically (simulating WITH context)
    # Note: Actual storey name in IFC is "6 - Sixth Floor", not "Level 6"
    level_6_elements = engine.find_elements_in_space("sixth")  # partial match for "6 - Sixth Floor"
    level_6_windows = [e for e in level_6_elements if "Window" in e.get("type", "")]

    print(f"\n✅ WITH Context ('sixth' filter for '6 - Sixth Floor'):")
    print(f"   Elements on 6 - Sixth Floor: {len(level_6_elements)}")
    print(f"   Windows on 6 - Sixth Floor: {len(level_6_windows)}")

    if level_6_windows:
        print("\n   Level 6 Windows:")
        for w in level_6_windows:
            print(f"      • {w.get('name')} (GUID: {w.get('guid')})")

    # =========================================================================
    # Summary: Quantified Redundancy
    # =========================================================================
    print_section("EXPERIMENT SUMMARY")

    total_windows = len(all_windows)
    balans_count = len(balans_windows)
    level_6_count = len(level_6_windows)

    print("Scenario: GT_007 - User asks about window sealant on Level 6")
    print("")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  Query Mode                    │  Results  │  Precision    │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  No context (all windows)      │  {total_windows:>5}    │  {1/total_windows*100:>6.2f}%     │")
    print(f"│  Partial (name match 'BALANS') │  {balans_count:>5}    │  {1/max(balans_count,1)*100:>6.2f}%     │")
    print(f"│  Full context (4D Task: L6)    │  {level_6_count:>5}    │  {1/max(level_6_count,1)*100:>6.2f}%     │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("")
    print(f"📊 Redundancy Reduction with 4D Context:")
    print(f"   • From {total_windows} → {level_6_count} (↓ {(1-level_6_count/total_windows)*100:.1f}%)")
    print(f"   • Precision improvement: {1/total_windows*100:.2f}% → {1/max(level_6_count,1)*100:.2f}%")

    # Save results to JSON
    results = {
        "experiment": "baseline_redundancy",
        "scenario": "GT_007_RQ3_HighRise_Abductive",
        "metrics": {
            "total_windows": total_windows,
            "balans_windows": balans_count,
            "level_6_windows": level_6_count,
            "redundancy_without_context": total_windows,
            "redundancy_with_partial": balans_count,
            "redundancy_with_full_context": level_6_count,
            "precision_no_context": 1/total_windows if total_windows > 0 else 0,
            "precision_with_context": 1/level_6_count if level_6_count > 0 else 0
        },
        "conclusion": f"4D context reduces search space by {(1-level_6_count/total_windows)*100:.1f}%"
    }

    results_dir = base_dir / "logs" / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "baseline_gt007_results.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Results saved to: {results_file}")


if __name__ == "__main__":
    run_baseline_experiment()
