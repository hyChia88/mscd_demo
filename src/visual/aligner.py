"""
Visual Aligner - CLIP-based Multimodal Embedding for BIM Inspection

Uses OpenAI's CLIP model to create a shared embedding space between:
- Site evidence images (photos of defects, installations, etc.)
- BIM element descriptions (textual properties from IFC)

This enables semantic matching between what inspectors see on-site
and what's defined in the Building Information Model.

Reference: src/legacy/visual_matcher.py (text-only prototype)
"""

import sys
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  transformers not installed. Run: pip install transformers", file=sys.stderr)


class VisualAligner:
    """
    CLIP-based Visual Aligner for BIM element matching.

    Provides multimodal embedding capabilities:
    - Image → Vector (site photos, defect images)
    - Text → Vector (BIM element descriptions, defect descriptions)
    - Image ↔ Text similarity (match photos to BIM elements)
    - Image ↔ Image similarity (compare two site photos)

    Usage:
        aligner = VisualAligner()

        # Match a site photo to BIM element descriptions
        results = aligner.match_image_to_descriptions(
            image_path="site_photo.jpg",
            descriptions=["Grey concrete slab", "White painted wall", "Glass window"]
        )

        # Compare two images
        similarity = aligner.compare_images("photo1.jpg", "photo2.jpg")
    """

    _instance: Optional["VisualAligner"] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - CLIP model is expensive to load."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the Visual Aligner with CLIP model.

        Args:
            model_id: HuggingFace model ID for CLIP variant.
                      Default uses base model for faster loading.
                      For better accuracy, use "openai/clip-vit-large-patch14"
        """
        if self._initialized:
            return

        if not CLIP_AVAILABLE:
            raise ImportError("transformers package required. Run: pip install transformers torch")

        print(f"👁️  [VisualAligner] Initializing CLIP Model ({model_id})...", file=sys.stderr)

        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"✅ [VisualAligner] Model loaded on {self.device}", file=sys.stderr)
        self._initialized = True

    # ─────────────────────────────────────────────────────────────────────────
    # Core Embedding Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Encode text into CLIP embedding space.

        Args:
            text: Description text (e.g., "Cracked grey concrete surface")

        Returns:
            Normalized embedding tensor (1, 512) for base model
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # L2 normalize for cosine similarity
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def get_image_embedding(self, image_path: str) -> torch.Tensor:
        """
        Encode image into CLIP embedding space.

        Args:
            image_path: Path to image file (jpg, png, etc.)

        Returns:
            Normalized embedding tensor (1, 512) for base model

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file is not a valid image
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # L2 normalize for cosine similarity
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def get_image_embedding_from_pil(self, image: Image.Image) -> torch.Tensor:
        """
        Encode PIL Image into CLIP embedding space.

        Args:
            image: PIL Image object

        Returns:
            Normalized embedding tensor
        """
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return image_features / image_features.norm(dim=-1, keepdim=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Matching Methods
    # ─────────────────────────────────────────────────────────────────────────

    def match_image_to_descriptions(
        self,
        image_path: str,
        descriptions: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Match a site image to a list of BIM element descriptions.

        This is the core function for visual grounding - connecting what's
        seen in a photo to what's defined in the BIM model.

        Args:
            image_path: Path to site photo
            descriptions: List of textual descriptions (e.g., from IFC elements)
            top_k: Number of top matches to return

        Returns:
            List of matches sorted by similarity score:
            [
                {"rank": 1, "description": "...", "score": 0.85, "index": 2},
                {"rank": 2, "description": "...", "score": 0.72, "index": 0},
                ...
            ]
        """
        print(f"🔍 [VisualAligner] Matching image to {len(descriptions)} descriptions...", file=sys.stderr)

        # Get image embedding
        image_emb = self.get_image_embedding(image_path)

        # Get text embeddings for all descriptions
        scores = []
        for desc in descriptions:
            text_emb = self.get_text_embedding(desc)
            # Cosine similarity (embeddings are already normalized)
            score = (image_emb @ text_emb.T).item()
            scores.append(score)

        # Sort by score descending
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            results.append({
                "rank": rank,
                "description": descriptions[idx],
                "score": float(scores[idx]),
                "index": int(idx)
            })

        return results

    def match_text_to_descriptions(
        self,
        query_text: str,
        candidate_descriptions: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Match a text query to candidate descriptions (text-to-text).

        Useful for matching user's verbal description to BIM elements
        when no image is available.

        Args:
            query_text: User's description (e.g., "cracked concrete near window")
            candidate_descriptions: List of BIM element descriptions
            top_k: Number of top matches to return

        Returns:
            List of matches sorted by similarity score
        """
        print(f"🔍 [VisualAligner] Text matching: '{query_text[:50]}...'", file=sys.stderr)

        query_emb = self.get_text_embedding(query_text)

        scores = []
        for desc in candidate_descriptions:
            cand_emb = self.get_text_embedding(desc)
            score = (query_emb @ cand_emb.T).item()
            scores.append(score)

        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            results.append({
                "rank": rank,
                "description": candidate_descriptions[idx],
                "score": float(scores[idx]),
                "index": int(idx)
            })

        return results

    def compare_images(self, image_path1: str, image_path2: str) -> float:
        """
        Compare two images for semantic similarity.

        Useful for checking if two site photos show the same defect
        or the same area from different angles.

        Args:
            image_path1: Path to first image
            image_path2: Path to second image

        Returns:
            Similarity score between 0 and 1 (higher = more similar)
        """
        emb1 = self.get_image_embedding(image_path1)
        emb2 = self.get_image_embedding(image_path2)

        # Cosine similarity
        similarity = (emb1 @ emb2.T).item()
        return float(similarity)

    def find_best_match(
        self,
        query_text: str,
        candidate_descriptions: List[str]
    ) -> Tuple[int, float, str]:
        """
        Legacy-compatible method: Find the single best matching description.

        Matches the interface of src/legacy/visual_matcher.py

        Args:
            query_text: Query description
            candidate_descriptions: List of candidate descriptions

        Returns:
            Tuple of (best_index, score, best_description)
        """
        results = self.match_text_to_descriptions(
            query_text,
            candidate_descriptions,
            top_k=1
        )

        if results:
            best = results[0]
            return best["index"], best["score"], best["description"]

        return 0, 0.0, candidate_descriptions[0] if candidate_descriptions else ""

    # ─────────────────────────────────────────────────────────────────────────
    # BIM Integration Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def build_element_description(self, element: Dict[str, Any]) -> str:
        """
        Build a textual description from BIM element properties.

        Converts IFC element data into a description string suitable
        for CLIP embedding.

        Args:
            element: Dict with keys like 'name', 'type', 'material', 'location'

        Returns:
            Description string for embedding
        """
        parts = []

        # IFC type (e.g., "IfcWall" → "Wall")
        ifc_type = element.get("type", "")
        if ifc_type.startswith("Ifc"):
            ifc_type = ifc_type[3:]  # Remove "Ifc" prefix
        if ifc_type:
            parts.append(ifc_type)

        # Name
        name = element.get("name", "")
        if name:
            parts.append(name)

        # Material
        material = element.get("material", "")
        if material:
            parts.append(material)

        # Location context
        location = element.get("location", "") or element.get("storey", "")
        if location:
            parts.append(f"located in {location}")

        return " ".join(parts) if parts else "Unknown building element"

    def match_image_to_elements(
        self,
        image_path: str,
        elements: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Match a site image directly to BIM elements.

        Convenience method that builds descriptions from element dicts
        and performs matching in one call.

        Args:
            image_path: Path to site photo
            elements: List of element dicts with 'guid', 'name', 'type', etc.
            top_k: Number of top matches to return

        Returns:
            List of matches with element info and scores:
            [
                {
                    "rank": 1,
                    "guid": "2O2Fr$...",
                    "name": "Wall_Kitchen",
                    "type": "IfcWall",
                    "score": 0.85,
                    "description": "Wall Wall_Kitchen concrete"
                },
                ...
            ]
        """
        if not elements:
            return []

        # Build descriptions for all elements
        descriptions = [self.build_element_description(el) for el in elements]

        # Match image to descriptions
        matches = self.match_image_to_descriptions(image_path, descriptions, top_k)

        # Enrich results with element info
        results = []
        for match in matches:
            idx = match["index"]
            element = elements[idx]
            results.append({
                "rank": match["rank"],
                "guid": element.get("guid", ""),
                "name": element.get("name", ""),
                "type": element.get("type", ""),
                "score": match["score"],
                "description": match["description"]
            })

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Visual Aligner Test")
    print("=" * 60)

    aligner = VisualAligner()

    # Test 1: Text-to-text matching (legacy compatible)
    print("\n[Test 1] Text-to-Text Matching")
    site_observation = "Cracked grey concrete surface"
    bim_elements = [
        "Wooden kitchen cabinet",
        "Grey concrete structural slab",
        "White painted drywall"
    ]
    idx, score, match = aligner.find_best_match(site_observation, bim_elements)
    print(f"  Query: {site_observation}")
    print(f"  Best Match: {match} (score: {score:.4f})")

    # Test 2: Top-k matching
    print("\n[Test 2] Top-K Matching")
    results = aligner.match_text_to_descriptions(
        "Water damage on ceiling",
        [
            "White gypsum ceiling tile",
            "Concrete floor slab",
            "Painted plasterboard ceiling",
            "Metal ductwork",
            "Glass skylight"
        ],
        top_k=3
    )
    for r in results:
        print(f"  #{r['rank']}: {r['description']} (score: {r['score']:.4f})")

    # Test 3: Image matching (if test image exists)
    print("\n[Test 3] Image Matching")
    test_image = Path(__file__).parent.parent.parent / "data" / "ground_truth" / "gt_1" / "imgs"
    if test_image.exists():
        images = list(test_image.glob("*.jpg")) + list(test_image.glob("*.png"))
        if images:
            print(f"  Found test image: {images[0].name}")
            try:
                results = aligner.match_image_to_descriptions(
                    str(images[0]),
                    ["Window with crack", "Concrete wall", "Metal door frame"],
                    top_k=3
                )
                for r in results:
                    print(f"  #{r['rank']}: {r['description']} (score: {r['score']:.4f})")
            except Exception as e:
                print(f"  Image test failed: {e}")
        else:
            print("  No test images found")
    else:
        print("  Test image directory not found")

    print("\n" + "=" * 60)
    print("All tests completed!")
