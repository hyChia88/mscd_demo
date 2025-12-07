#!/usr/bin/env python3
"""
Visual Matching Demo - Standalone Example

This demonstrates how to use the VisualAligner to match site observations
to BIM element descriptions using CLIP embeddings.

Usage:
    python examples/test_visual_matching.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visual_matcher import VisualAligner


def demo_basic_matching():
    """Demo 1: Basic text-to-text matching"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Visual Matching")
    print("="*70)

    # Initialize the aligner
    print("\nInitializing Visual Aligner (this may take a moment on first run)...")
    aligner = VisualAligner()

    # Scenario: Inspector found damage
    site_observation = "Cracked grey concrete slab with exposed rebar"

    # BIM elements in that area (from get_elements_by_room)
    candidate_descriptions = [
        "Wooden kitchen cabinet with white finish",
        "Grey concrete structural slab, 200mm thick",
        "White painted drywall partition",
        "Ceramic tile flooring"
    ]

    print(f"\n📋 Site Observation:")
    print(f"   '{site_observation}'")

    print(f"\n🔍 Searching among {len(candidate_descriptions)} BIM elements:")
    for i, desc in enumerate(candidate_descriptions, 1):
        print(f"   {i}. {desc}")

    # Find best match
    idx, score, match = aligner.find_best_match(
        site_observation,
        candidate_descriptions
    )

    # Display results
    print(f"\n✅ RESULT:")
    print(f"   Best Match: {match}")
    print(f"   Match Index: {idx + 1}")
    print(f"   Confidence: {score:.2%}")
    print("="*70)

    return aligner  # Return for reuse in next demo


def demo_multiple_observations(aligner):
    """Demo 2: Multiple site observations"""
    print("\n" + "="*70)
    print("DEMO 2: Multiple Site Observations")
    print("="*70)

    # Multiple damage reports
    observations = [
        ("Water damage on wooden cabinet", "Cabinet"),
        ("Crack in concrete wall", "Wall"),
        ("Broken window frame", "Window"),
        ("Damaged floor tiles", "Floor")
    ]

    # All possible BIM elements
    all_elements = [
        "M_Base Cabinet - Wooden, oak finish",
        "M_Wall Cabinet - White painted wood",
        "Yttervägg - Concrete exterior wall 200mm",
        "Innerdörr - Interior drywall partition",
        "IfcWindow - Double glazed, aluminum frame",
        "IfcDoor - Wooden interior door",
        "Floor Slab - Concrete, 150mm thick",
        "Ceramic floor tiles - 30x30cm"
    ]

    print(f"\n📋 Processing {len(observations)} site reports...")

    for obs_text, obs_type in observations:
        print(f"\n{'─'*70}")
        print(f"Observation: '{obs_text}' (Type: {obs_type})")

        # Find best match
        idx, score, match = aligner.find_best_match(obs_text, all_elements)

        print(f"  → Match: {match}")
        print(f"  → Confidence: {score:.2%}")

    print("="*70)


def demo_scoring_comparison(aligner):
    """Demo 3: Show similarity scores for all candidates"""
    print("\n" + "="*70)
    print("DEMO 3: Detailed Similarity Scoring")
    print("="*70)

    observation = "Damaged brown wooden kitchen cabinet"

    candidates = [
        "M_Base Cabinet - Kitchen, oak wood, brown finish",
        "M_Wall Cabinet - Kitchen, white painted",
        "M_Vanity - Bathroom, white laminate",
        "Yttervägg - Concrete wall, grey",
        "IfcDoor - Wooden door, brown"
    ]

    print(f"\n📋 Observation: '{observation}'")
    print(f"\n🔍 Computing similarity scores for all candidates:\n")

    # Get query embedding
    query_emb = aligner.get_text_embedding(observation)

    # Score each candidate
    results = []
    for candidate in candidates:
        cand_emb = aligner.get_text_embedding(candidate)
        score = (query_emb @ cand_emb.T).item()
        results.append((candidate, score))

    # Sort by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # Display ranked results
    print(f"{'Rank':<6} {'Score':<10} {'Element Description'}")
    print("─"*70)
    for rank, (desc, score) in enumerate(results, 1):
        marker = "✅" if rank == 1 else "  "
        print(f"{marker} {rank:<4} {score:>6.2%}    {desc}")

    print("\n" + "="*70)


def demo_semantic_understanding(aligner):
    """Demo 4: Test semantic understanding with different phrasings"""
    print("\n" + "="*70)
    print("DEMO 4: Semantic Understanding Test")
    print("="*70)

    # Same concept, different phrasings
    phrasings = [
        "cracked concrete",
        "concrete with crack",
        "damaged concrete surface",
        "concrete structural defect",
        "fissure in concrete"
    ]

    candidates = [
        "Concrete structural slab",
        "Wooden cabinet",
        "Glass window"
    ]

    print(f"\n🧪 Testing semantic understanding:")
    print(f"   All phrasings should match 'Concrete structural slab'\n")

    for phrasing in phrasings:
        idx, score, match = aligner.find_best_match(phrasing, candidates)
        status = "✅" if "Concrete" in match else "❌"
        print(f"{status} '{phrasing:35s}' → {match} ({score:.2%})")

    print("\n" + "="*70)


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print(" VISUAL MATCHING DEMONSTRATION")
    print(" Using CLIP Model for Semantic Similarity")
    print("="*70)

    try:
        # Demo 1: Basic matching
        aligner = demo_basic_matching()

        # Demo 2: Multiple observations
        demo_multiple_observations(aligner)

        # Demo 3: Detailed scoring
        demo_scoring_comparison(aligner)

        # Demo 4: Semantic understanding
        demo_semantic_understanding(aligner)

        print("\n✅ All demos completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have installed the required packages:")
        print("  pip install transformers torch numpy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
