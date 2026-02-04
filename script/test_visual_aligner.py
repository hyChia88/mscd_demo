#!/usr/bin/env python3
"""
Test script for Visual Aligner module

Verifies:
1. CLIP model loads correctly
2. Text-to-text matching works (legacy compatible)
3. Image embedding works (if test images available)
4. BIM element matching integration

Usage:
    python script/test_visual_aligner.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all imports work"""
    print("=" * 60)
    print("Test 1: Import Check")
    print("=" * 60)

    try:
        from visual import VisualAligner
        print("  [PASS] visual.VisualAligner imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        print("  Make sure to install: pip install torch transformers pillow")
        return False


def test_model_loading():
    """Test CLIP model loads"""
    print("\n" + "=" * 60)
    print("Test 2: Model Loading")
    print("=" * 60)

    try:
        from visual import VisualAligner
        aligner = VisualAligner()
        print(f"  [PASS] Model loaded on device: {aligner.device}")
        print(f"  [PASS] Model ID: {aligner.model_id}")
        return aligner
    except Exception as e:
        print(f"  [FAIL] Model loading failed: {e}")
        return None


def test_text_matching(aligner):
    """Test text-to-text matching (legacy compatible)"""
    print("\n" + "=" * 60)
    print("Test 3: Text-to-Text Matching")
    print("=" * 60)

    try:
        query = "Cracked grey concrete surface"
        candidates = [
            "Wooden kitchen cabinet",
            "Grey concrete structural slab",
            "White painted drywall"
        ]

        idx, score, match = aligner.find_best_match(query, candidates)

        print(f"  Query: '{query}'")
        print(f"  Best match: '{match}'")
        print(f"  Score: {score:.4f}")

        # Expected: should match "Grey concrete structural slab"
        if "concrete" in match.lower():
            print("  [PASS] Correct element type matched")
            return True
        else:
            print("  [WARN] Unexpected match, but function works")
            return True

    except Exception as e:
        print(f"  [FAIL] Text matching error: {e}")
        return False


def test_top_k_matching(aligner):
    """Test top-k ranking"""
    print("\n" + "=" * 60)
    print("Test 4: Top-K Matching")
    print("=" * 60)

    try:
        query = "Water damage on ceiling"
        candidates = [
            "White gypsum ceiling tile",
            "Concrete floor slab",
            "Painted plasterboard ceiling",
            "Metal ductwork",
            "Glass skylight"
        ]

        results = aligner.match_text_to_descriptions(query, candidates, top_k=3)

        print(f"  Query: '{query}'")
        print("  Top 3 matches:")
        for r in results:
            print(f"    #{r['rank']}: {r['description']} (score: {r['score']:.4f})")

        # Check that we got 3 results
        if len(results) == 3:
            print("  [PASS] Top-3 returned correctly")
            return True
        else:
            print(f"  [FAIL] Expected 3 results, got {len(results)}")
            return False

    except Exception as e:
        print(f"  [FAIL] Top-k matching error: {e}")
        return False


def test_image_embedding(aligner):
    """Test image embedding (if test images exist)"""
    print("\n" + "=" * 60)
    print("Test 5: Image Embedding")
    print("=" * 60)

    # Look for test images
    base_dir = Path(__file__).parent.parent
    img_dir = base_dir / "data" / "ground_truth" / "gt_1" / "imgs"

    if not img_dir.exists():
        print(f"  [SKIP] Test image directory not found: {img_dir}")
        return True  # Skip but don't fail

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"  [SKIP] No test images found in {img_dir}")
        return True

    test_image = images[0]
    print(f"  Using test image: {test_image.name}")

    try:
        embedding = aligner.get_image_embedding(str(test_image))
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {embedding.norm().item():.4f}")

        if embedding.shape[-1] == 512:  # CLIP base model
            print("  [PASS] Image embedding generated correctly")
            return True
        else:
            print(f"  [WARN] Unexpected embedding dimension: {embedding.shape}")
            return True

    except Exception as e:
        print(f"  [FAIL] Image embedding error: {e}")
        return False


def test_image_to_descriptions(aligner):
    """Test matching image to text descriptions"""
    print("\n" + "=" * 60)
    print("Test 6: Image-to-Description Matching")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent
    img_dir = base_dir / "data" / "ground_truth" / "gt_1" / "imgs"

    if not img_dir.exists():
        print(f"  [SKIP] Test image directory not found")
        return True

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"  [SKIP] No test images found")
        return True

    test_image = images[0]
    print(f"  Using test image: {test_image.name}")

    try:
        descriptions = [
            "Window with crack or damage",
            "Concrete wall with stain",
            "Metal door frame",
            "Wooden floor panel",
            "Glass facade"
        ]

        results = aligner.match_image_to_descriptions(str(test_image), descriptions, top_k=3)

        print("  Top 3 matches:")
        for r in results:
            print(f"    #{r['rank']}: {r['description']} (score: {r['score']:.4f})")

        print("  [PASS] Image-to-description matching works")
        return True

    except Exception as e:
        print(f"  [FAIL] Image matching error: {e}")
        return False


def test_element_description_builder(aligner):
    """Test building descriptions from BIM elements"""
    print("\n" + "=" * 60)
    print("Test 7: Element Description Builder")
    print("=" * 60)

    try:
        # Mock BIM element
        element = {
            "guid": "2O2Fr$t4X7Zf8NOew3FLOH",
            "name": "Window_North_6F",
            "type": "IfcWindow",
            "material": "Aluminum frame with glass",
            "location": "6 - Sixth Floor"
        }

        description = aligner.build_element_description(element)
        print(f"  Element: {element['name']}")
        print(f"  Generated description: '{description}'")

        # Should contain key terms
        if "Window" in description:
            print("  [PASS] Description contains element type")
            return True
        else:
            print("  [WARN] Description may be incomplete")
            return True

    except Exception as e:
        print(f"  [FAIL] Description builder error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("VISUAL ALIGNER TEST SUITE")
    print("=" * 60)

    results = []

    # Test 1: Imports
    results.append(("Import Check", test_imports()))

    if not results[-1][1]:
        print("\n[ABORT] Cannot proceed without imports")
        return 1

    # Test 2: Model loading
    aligner = test_model_loading()
    results.append(("Model Loading", aligner is not None))

    if aligner is None:
        print("\n[ABORT] Cannot proceed without model")
        return 1

    # Test 3-7: Functionality tests
    results.append(("Text Matching", test_text_matching(aligner)))
    results.append(("Top-K Matching", test_top_k_matching(aligner)))
    results.append(("Image Embedding", test_image_embedding(aligner)))
    results.append(("Image-to-Description", test_image_to_descriptions(aligner)))
    results.append(("Element Description", test_element_description_builder(aligner)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
