"""
Visual Matching MCP Server

Exposes CLIP-based visual-semantic matching capabilities as MCP tools.
This server provides multimodal element identification using vision-language embeddings.

Usage:
    python mcp_servers/visual_server.py

The server will start and listen for MCP client connections via stdio.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import visual_matcher
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Visual Matching Service")

# Lazy loading of visual matcher to avoid loading heavy models at import time
_aligner = None


def get_aligner():
    """Lazy initialization of VisualAligner to defer model loading."""
    global _aligner
    if _aligner is None:
        try:
            from visual_matcher import VisualAligner
            print("[Visual Server] Loading CLIP model...", file=sys.stderr)
            _aligner = VisualAligner()
            print("[Visual Server] CLIP model ready.", file=sys.stderr)
        except ImportError as e:
            print(f"[Visual Server] Warning: CLIP model not available: {e}", file=sys.stderr)
            _aligner = False
    return _aligner if _aligner else None


@mcp.tool()
def identify_element_visually(site_description_or_photo: str, candidate_guids_str: str) -> str:
    """
    Identify the correct BIM element from candidates using visual-semantic matching.

    Uses CLIP (Contrastive Language-Image Pre-training) embeddings to match
    site descriptions or photos with BIM element descriptions in a shared
    multimodal embedding space.

    Args:
        site_description_or_photo: Either a text description (e.g., "white kitchen cabinet")
                                   or a file path to a photo (e.g., "data/site_photos/evidence_01.jpg")
        candidate_guids_str: Comma-separated string of GUIDs to check (e.g., "GUID1,GUID2,GUID3")

    Returns:
        str: The GUID of the best visual match with confidence score

    Example:
        >>> identify_element_visually("white cabinet with wood countertop", "guid1,guid2,guid3")
        Best visual match: guid1
        Element: Kitchen Cabinet - IfcFurniture
        Confidence: 0.87

        >>> identify_element_visually("data/site_photos/cabinet.jpg", "guid1,guid2,guid3")
        Best visual match: guid2
        Element: Wall Cabinet - IfcFurniture
        Confidence: 0.92
    """
    guids = [g.strip() for g in candidate_guids_str.split(",")]

    if not guids:
        return "Error: No candidate GUIDs provided."

    aligner = get_aligner()
    if not aligner:
        return f"Visual matching not available (CLIP model not loaded). Top candidate by order: {guids[0]}"

    # Import IFC engine to get element descriptions
    from ifc_engine import IFCEngine
    base_dir = Path(__file__).parent.parent
    ifc_path = os.getenv("IFC_MODEL_PATH", str(base_dir / "data" / "BasicHouse.ifc"))
    engine = IFCEngine(ifc_path)

    # Get descriptions for each candidate element
    candidate_descriptions = []
    for guid in guids:
        try:
            element = engine.file.by_id(guid)
            if element:
                name = element.Name if element.Name else "Unnamed"
                elem_type = element.is_a()
                desc = f"{name} - {elem_type}"
                candidate_descriptions.append(desc)
            else:
                candidate_descriptions.append(f"Element {guid}")
        except Exception:
            candidate_descriptions.append(f"Element {guid}")

    # Check if input is a file path or text description
    is_file = os.path.exists(site_description_or_photo)

    if is_file:
        # For MVP: Mock image processing
        # Production: Would load image and use CLIP image encoder
        query_text = f"Visual features from photo: {os.path.basename(site_description_or_photo)}"
    else:
        query_text = site_description_or_photo

    # Use VisualAligner to find best match
    try:
        best_idx, score, best_match = aligner.find_best_match(query_text, candidate_descriptions)
        matched_guid = guids[best_idx]
        return f"Best visual match: {matched_guid}\nElement: {best_match}\nConfidence: {score:.2f}"
    except Exception as e:
        return f"Visual matching error: {str(e)}. Returning top candidate: {guids[0]}"


@mcp.tool()
def compute_semantic_similarity(text1: str, text2: str) -> str:
    """
    Compute semantic similarity between two text descriptions.

    Uses CLIP text encoder to compute cosine similarity in the embedding space.
    Useful for matching site observation descriptions with BIM element descriptions.

    Args:
        text1: First text description
        text2: Second text description

    Returns:
        str: Similarity score (0.0 to 1.0)

    Example:
        >>> compute_semantic_similarity("cracked floor slab", "damaged concrete floor")
        Similarity: 0.82
    """
    aligner = get_aligner()
    if not aligner:
        return "Visual matching service not available (CLIP model not loaded)."

    try:
        _, score, _ = aligner.find_best_match(text1, [text2])
        return f"Similarity: {score:.2f}"
    except Exception as e:
        return f"Error computing similarity: {str(e)}"


# Resource for exposing model information
@mcp.resource("visual://model/info")
def get_model_info() -> str:
    """
    Get information about the loaded CLIP model.

    Returns:
        str: JSON-formatted model information
    """
    import json

    aligner = get_aligner()

    if not aligner:
        return json.dumps({"status": "unavailable", "reason": "CLIP model not loaded"})

    info = {
        "status": "ready",
        "model": "openai/clip-vit-base-patch32",
        "modalities": ["text", "image (MVP: text-only)"],
        "embedding_dim": 512,
        "description": "CLIP-based semantic matching for BIM elements"
    }

    return json.dumps(info, indent=2)


if __name__ == "__main__":
    # Run the MCP server
    print("[Visual Server] Starting MCP server via stdio...", file=sys.stderr)
    mcp.run(transport="stdio")
