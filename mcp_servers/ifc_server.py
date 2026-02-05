"""
IFC Query MCP Server

Exposes IFC/BIM query capabilities as MCP tools for the LangChain Agent.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │              MCP Server (this file)                  │
    │  ┌─────────────────────────────────────────────┐    │
    │  │         IFCEngine (Internal Singleton)      │    │
    │  │  - Loaded once at startup                   │    │
    │  │  - Maintains spatial index in memory        │    │
    │  └─────────────────────────────────────────────┘    │
    │                       │                              │
    │  ┌─────────────────────────────────────────────┐    │
    │  │           Exposed MCP Tools                  │    │
    │  │  - list_available_spaces()                  │    │
    │  │  - get_elements_by_room()                   │    │
    │  │  - get_elements_by_storey()     [NEW]       │    │
    │  │  - get_element_details()                    │    │
    │  │  - search_elements_by_type()    [NEW]       │    │
    │  └─────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘

Usage:
    python mcp_servers/ifc_server.py

The server will start and listen for MCP client connections via stdio.
"""

import os
import sys
import json
import yaml
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path to import ifc_engine
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastmcp import FastMCP
from ifc_engine import IFCEngine

# Lazy load VisualAligner to avoid heavy CLIP model load at startup
_visual_aligner = None
_visual_attempted = False  # Track if we already tried to load

def is_visual_enabled():
    """Check if visual analysis is enabled via environment variable."""
    return os.getenv("VISUAL_ENABLED", "true").lower() == "true"

def get_visual_aligner():
    """Lazy-load the VisualAligner singleton (CLIP model is heavy)."""
    global _visual_aligner, _visual_attempted

    # Check if visual is disabled via env var
    if not is_visual_enabled():
        if not _visual_attempted:
            print(f"[IFC Server] Visual analysis DISABLED (VISUAL_ENABLED=false)", file=sys.stderr)
            _visual_attempted = True
        return None

    if _visual_aligner is None and not _visual_attempted:
        _visual_attempted = True
        try:
            from visual.aligner import VisualAligner
            _visual_aligner = VisualAligner()
            print(f"[IFC Server] Visual analysis ENABLED (CLIP loaded)", file=sys.stderr)
        except ImportError as e:
            print(f"[IFC Server] VisualAligner not available: {e}", file=sys.stderr)
            return None

    return _visual_aligner

# Initialize MCP server
mcp = FastMCP("IFC Query Service")

# Load IFC engine (singleton pattern - loaded once, reused for all requests)
BASE_DIR = Path(__file__).parent.parent

# Global state for query mode
QUERY_MODE = "memory"  # "memory" or "neo4j"
neo4j_graph = None


def load_config():
    """Load full config from config.yaml"""
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_ifc_path():
    """Load IFC path from config.yaml or environment variable"""
    # First check environment variable (highest priority)
    if os.getenv("IFC_MODEL_PATH"):
        return os.getenv("IFC_MODEL_PATH")

    config = load_config()
    ifc_path = config.get("ifc", {}).get("model_path")
    if ifc_path:
        return str(BASE_DIR / ifc_path)

    # Fallback to default
    return str(BASE_DIR / "data" / "ifc" / "AdvancedProject" / "IFC" / "AdvancedProject.ifc")


def init_neo4j_connection():
    """Initialize Neo4j connection if enabled in config or env"""
    global neo4j_graph, QUERY_MODE

    # Check environment variable first (set by main_mcp.py for experiments)
    env_mode = os.getenv("QUERY_MODE", "").lower()
    if env_mode == "neo4j":
        force_neo4j = True
    elif env_mode == "memory":
        force_neo4j = False
    else:
        # Fall back to config
        config = load_config()
        force_neo4j = config.get("neo4j", {}).get("enabled", False)

    if not force_neo4j:
        QUERY_MODE = "memory"
        print(f"[IFC Server] Query mode: MEMORY (in-memory spatial index)", file=sys.stderr)
        return None

    # Try to connect to Neo4j
    try:
        from py2neo import Graph
        config = load_config()
        neo4j_config = config.get("neo4j", {})

        graph = Graph(
            neo4j_config.get("uri", "bolt://localhost:7687"),
            auth=(neo4j_config.get("user", "neo4j"), neo4j_config.get("password", "password"))
        )
        # Test connection
        graph.run("RETURN 1")

        neo4j_graph = graph
        QUERY_MODE = "neo4j"
        print(f"[IFC Server] Query mode: NEO4J (graph database)", file=sys.stderr)
        return graph

    except ImportError:
        print(f"[IFC Server] py2neo not installed, falling back to memory mode", file=sys.stderr)
        QUERY_MODE = "memory"
        return None
    except Exception as e:
        print(f"[IFC Server] Neo4j connection failed ({e}), falling back to memory mode", file=sys.stderr)
        QUERY_MODE = "memory"
        return None


# Initialize
IFC_PATH = load_ifc_path()
print(f"[IFC Server] Initializing with model: {IFC_PATH}", file=sys.stderr)

# Initialize Neo4j connection (will set QUERY_MODE)
neo4j_graph = init_neo4j_connection()

# Initialize IFC engine with Neo4j connection if available
engine = IFCEngine(IFC_PATH, neo4j_conn=neo4j_graph)
print(f"[IFC Server] Engine ready. Spatial index contains {len(engine.spatial_index)} groups.", file=sys.stderr)

# Log available storeys for debugging
storeys = [key for key in engine.spatial_index.keys() if 'floor' in key.lower() or 'level' in key.lower()]
print(f"[IFC Server] Available storeys: {storeys}", file=sys.stderr)


@mcp.tool()
def list_available_spaces() -> str:
    """
    Discover what rooms, floors, and spaces are available in the IFC model.

    Returns a list of space names with element counts that can be used
    with get_elements_by_room.

    Returns:
        str: Formatted list of available spaces with element counts

    Example:
        >>> list_available_spaces()
        Available spaces:
          - 'Kitchen' (12 elements)
          - 'Living Room' (8 elements)
    """
    available_spaces = list(engine.spatial_index.keys())

    if not available_spaces:
        return "No spaces found in the IFC model."

    details = []
    for space in available_spaces:
        count = len(engine.spatial_index[space])
        details.append(f"'{space}' ({count} elements)")

    return "Available spaces:\n" + "\n".join(f"  - {d}" for d in details)


@mcp.tool()
def get_elements_by_room(room_name: str) -> str:
    """
    Find BIM elements located in a specific room or space.

    Queries the IFC model's spatial hierarchy to find all building elements
    (walls, slabs, windows, doors, etc.) contained within the specified room.

    Args:
        room_name: Name of the room/space (e.g., 'Kitchen', 'Living Room', 'Bathroom')

    Returns:
        str: List of elements with their names, types, and GUIDs (limited to 30 elements)

    Example:
        >>> get_elements_by_room("Kitchen")
        [
            {'name': 'Wall_123', 'type': 'IfcWall', 'guid': '2O2Fr$t4X7Zf8NOew3FLOH'},
            {'name': 'Slab_045', 'type': 'IfcSlab', 'guid': '2O2Fr$t4X7Zf8NOew3FL0I'}
        ]
    """
    results = engine.find_elements_in_space(room_name)

    if not results:
        return f"No elements found in a room matching '{room_name}'."

    # Limit to prevent token overflow
    return str(results[:30])


@mcp.tool()
def get_elements_by_storey(storey_name: str) -> str:
    """
    Find all BIM elements on a specific building storey/floor.

    Use this tool when you know which floor/level the defect is on.
    The storey name comes from 4D task context (e.g., "6 - Sixth Floor").

    Available storeys in this model:
    - "1 - First Floor" (46 windows, 736 total elements)
    - "2 - Second Floor" (46 windows)
    - "3 - Third Floor" (46 windows)
    - "4 - Fourth Floor" (46 windows)
    - "5 - Fifth Floor" (46 windows)
    - "6 - Sixth Floor" (3 windows, 54 total elements)
    - "Level 1" (30 windows, 506 total elements)
    - "-1 - Garage" (basement)

    Args:
        storey_name: Name or partial name of the storey (e.g., "sixth", "Level 1", "first floor")

    Returns:
        str: JSON list of elements with name, type, and GUID

    Example:
        >>> get_elements_by_storey("sixth")
        Returns all 54 elements on "6 - Sixth Floor" including 3 windows
    """
    results = engine.find_elements_in_space(storey_name.lower())

    if not results:
        # List available options
        available = [k for k in engine.spatial_index.keys() if 'floor' in k or 'level' in k]
        return f"No elements found for storey '{storey_name}'. Available storeys: {available}"

    # Group by type for better readability
    by_type = {}
    for el in results:
        el_type = el.get("type", "Unknown")
        if el_type not in by_type:
            by_type[el_type] = []
        by_type[el_type].append(el)

    summary = f"Found {len(results)} elements on storey matching '{storey_name}':\n"
    for el_type, elements in sorted(by_type.items()):
        summary += f"\n  {el_type}: {len(elements)} elements"
        # Show first 3 of each type
        for el in elements[:3]:
            summary += f"\n    - {el.get('name')} (GUID: {el.get('guid')})"
        if len(elements) > 3:
            summary += f"\n    ... and {len(elements) - 3} more"

    return summary


@mcp.tool()
def get_element_details(guid: str) -> str:
    """
    Get detailed technical properties (Property Sets) of a specific element.

    Retrieves all property sets (Psets) associated with an IFC element,
    useful for compliance checking (material, fire rating, load capacity, etc.).

    Args:
        guid: Global Unique Identifier of the element

    Returns:
        str: Detailed property sets and values for the element

    Example:
        >>> get_element_details("2O2Fr$t4X7Zf8NOew3FLOH")
        Element: Wall_Kitchen_North
        Type: IfcWall
        Properties:
          - Pset_WallCommon:
              - FireRating: REI 120
              - LoadBearing: True
              - IsExternal: False
    """
    props = engine.get_element_properties(guid)
    return json.dumps(props, indent=2, default=str)


@mcp.tool()
def search_elements_by_type(element_type: str, storey_filter: Optional[str] = None) -> str:
    """
    Search for BIM elements by their IFC type, optionally filtered by storey.

    Use this tool to find all elements of a specific type (windows, doors, walls, etc.)
    across the entire model or within a specific floor.

    Common element types:
    - "IfcWindow" - Windows
    - "IfcDoor" - Doors
    - "IfcWall" or "IfcWallStandardCase" - Walls
    - "IfcSlab" - Floor slabs
    - "IfcColumn" - Columns
    - "IfcBeam" - Beams
    - "IfcFurnishingElement" - Furniture

    Args:
        element_type: IFC element type (e.g., "IfcWindow", "IfcDoor", "Wall")
        storey_filter: Optional storey name to filter results (e.g., "sixth", "Level 1")

    Returns:
        str: List of matching elements with their locations

    Example:
        >>> search_elements_by_type("IfcWindow", "sixth")
        Returns all windows on "6 - Sixth Floor" (3 windows)
    """
    results = []

    # Normalize search term
    search_type = element_type.lower()
    if not search_type.startswith("ifc"):
        search_type = f"ifc{search_type}"

    # Search across all spatial groups or filtered by storey
    if storey_filter:
        spaces_to_search = {k: v for k, v in engine.spatial_index.items()
                          if storey_filter.lower() in k}
    else:
        spaces_to_search = engine.spatial_index

    for space_name, elements in spaces_to_search.items():
        for el in elements:
            el_type = el.get("type", "").lower()
            if search_type in el_type:
                results.append({
                    "guid": el.get("guid"),
                    "name": el.get("name"),
                    "type": el.get("type"),
                    "location": space_name
                })

    if not results:
        return f"No elements of type '{element_type}' found" + \
               (f" on storey '{storey_filter}'" if storey_filter else "")

    summary = f"Found {len(results)} elements of type '{element_type}'"
    if storey_filter:
        summary += f" on storey '{storey_filter}'"
    summary += ":\n"

    # Group by location
    by_location = {}
    for el in results:
        loc = el["location"]
        if loc not in by_location:
            by_location[loc] = []
        by_location[loc].append(el)

    for location, elements in sorted(by_location.items()):
        summary += f"\n  {location}: {len(elements)} elements"
        for el in elements[:5]:  # Show first 5
            summary += f"\n    - {el['name']} (GUID: {el['guid']})"
        if len(elements) > 5:
            summary += f"\n    ... and {len(elements) - 5} more"

    return summary


@mcp.tool()
def generate_3d_view(guid: str) -> str:
    """
    Generate a visual verification snapshot (3D render) of an element.

    Creates a rendered image of the specified element for visual inspection.
    Useful when users need to 'see' or 'visualize' a defect location.

    Args:
        guid: Global Unique Identifier of the element to visualize

    Returns:
        str: File path to the generated render image, or error message if failed

    Example:
        >>> generate_3d_view("2O2Fr$t4X7Zf8NOew3FLOH")
        "/path/to/outputs/renders/2O2Fr$t4X7Zf8NOew3FLOH_inspection_view.png"
    """
    # Get Blender executable path from environment or use default
    blender_path = os.getenv("BLENDER_PATH", "blender")

    # Get IFC file path from the engine
    ifc_path = str(ifc_engine.ifc_path) if ifc_engine and ifc_engine.ifc_path else None
    if not ifc_path or not os.path.exists(ifc_path):
        return json.dumps({
            "success": False,
            "error": "IFC file not loaded or not found",
            "guid": guid
        })

    # Create output directory
    output_dir = BASE_DIR / "outputs" / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with timestamp to avoid conflicts
    timestamp = int(time.time())
    output_filename = f"{guid[:22]}_{timestamp}_inspection_view.png"
    output_path = output_dir / output_filename

    # Path to render worker script
    render_script = BASE_DIR / "script" / "render_worker.py"
    if not render_script.exists():
        return json.dumps({
            "success": False,
            "error": f"Render script not found: {render_script}",
            "guid": guid
        })

    # Build Blender command
    cmd = [
        blender_path,
        "--background",
        "--python", str(render_script),
        "--",
        str(ifc_path),
        str(output_path),
        guid
    ]

    try:
        print(f"[generate_3d_view] Rendering element {guid}...", file=sys.stderr)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            return json.dumps({
                "success": False,
                "error": f"Blender render failed: {error_msg}",
                "guid": guid
            })

        # Check if output file was created
        if not output_path.exists():
            return json.dumps({
                "success": False,
                "error": "Render completed but output file not created",
                "guid": guid,
                "blender_output": result.stdout[:500] if result.stdout else None
            })

        print(f"[generate_3d_view] Successfully rendered to {output_path}", file=sys.stderr)
        return json.dumps({
            "success": True,
            "image_path": str(output_path),
            "guid": guid,
            "message": f"3D view generated successfully for element {guid}"
        })

    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": "Blender render timed out (>120 seconds)",
            "guid": guid
        })
    except FileNotFoundError:
        return json.dumps({
            "success": False,
            "error": f"Blender executable not found at '{blender_path}'. Set BLENDER_PATH environment variable.",
            "guid": guid
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "guid": guid
        })


@mcp.tool()
def get_query_mode() -> str:
    """
    Get the current query mode (memory or neo4j).

    Useful for experiment tracking and debugging.

    Returns:
        str: Current query mode and connection status
    """
    return json.dumps({
        "query_mode": QUERY_MODE,
        "neo4j_connected": neo4j_graph is not None,
        "description": "Neo4j graph queries" if QUERY_MODE == "neo4j" else "In-memory spatial index"
    }, indent=2)


@mcp.tool()
def query_adjacent_elements(guid: str) -> str:
    """
    Find elements adjacent to or connected with a specific element.

    Uses Neo4j graph traversal when available for semantic relationships,
    falls back to spatial proximity in memory mode.

    Args:
        guid: Global Unique Identifier of the element

    Returns:
        str: List of related/adjacent elements

    Note:
        This tool benefits most from Neo4j mode as it can traverse
        semantic relationships (HAS_OPENING, FILLS, BOUNDED_BY, etc.)
    """
    if QUERY_MODE == "neo4j" and neo4j_graph is not None:
        # Use Neo4j graph query for semantic adjacency
        results = engine.query_adjacent_elements(guid)
        if results:
            return json.dumps({
                "query_mode": "neo4j",
                "source_guid": guid,
                "adjacent_elements": results
            }, indent=2)
        return f"No adjacent elements found for GUID {guid} (Neo4j mode)"
    else:
        # Memory mode: find elements in same space
        for space_name, elements in engine.spatial_index.items():
            for el in elements:
                if el.get("guid") == guid:
                    # Return other elements in same space
                    others = [e for e in elements if e.get("guid") != guid][:10]
                    return json.dumps({
                        "query_mode": "memory",
                        "source_guid": guid,
                        "same_space": space_name,
                        "nearby_elements": others
                    }, indent=2)
        return f"Element with GUID {guid} not found"


# Resource for exposing IFC model metadata
@mcp.resource("ifc://model/metadata")
def get_model_metadata() -> str:
    """
    Get metadata about the loaded IFC model.

    Returns:
        str: JSON-formatted metadata including schema, project info, and statistics
    """
    metadata = {
        "model_path": str(IFC_PATH),
        "schema": engine.file.schema if hasattr(engine, 'file') else "IFC4",
        "query_mode": QUERY_MODE,
        "neo4j_connected": neo4j_graph is not None,
        "num_spaces": len(engine.spatial_index),
        "total_elements": sum(len(elements) for elements in engine.spatial_index.values()),
        "available_spaces": list(engine.spatial_index.keys())
    }

    return json.dumps(metadata, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Analysis Tools (CLIP-based)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_site_image(image_path: str) -> str:
    """
    Analyze a site photo to describe what building elements are visible.

    Uses CLIP multimodal AI to understand the content of site inspection
    photos. Useful for getting an initial understanding of what's in an image
    before matching to specific BIM elements.

    Args:
        image_path: Absolute or relative path to the image file (jpg, png)

    Returns:
        str: Analysis results with likely element types and confidence scores

    Example:
        >>> analyze_site_image("data/ground_truth/gt_1/imgs/crack_window.jpg")
        Analysis of site image:
          Most likely elements visible:
          - Window with visible damage (0.82)
          - Concrete wall surface (0.65)
          - Metal frame structure (0.45)
    """
    aligner = get_visual_aligner()
    if aligner is None:
        return "Error: Visual analysis not available. Install transformers and torch."

    # Resolve path relative to BASE_DIR if not absolute
    resolved_path = image_path
    if not Path(image_path).is_absolute():
        resolved_path = str(BASE_DIR / image_path)

    if not Path(resolved_path).exists():
        return f"Error: Image not found at '{resolved_path}'"

    # Common BIM element descriptions for classification
    element_categories = [
        "Window with glass pane",
        "Concrete wall surface",
        "Metal door frame",
        "Wooden door",
        "Floor slab with tiles",
        "Ceiling with panels",
        "Structural column",
        "Beam or lintel",
        "Crack or damage on surface",
        "Water stain or leak damage",
        "Electrical outlet or switch",
        "HVAC duct or vent",
        "Pipe or plumbing fixture",
        "Staircase or railing"
    ]

    try:
        results = aligner.match_image_to_descriptions(
            resolved_path,
            element_categories,
            top_k=5
        )

        output = f"Analysis of site image ({Path(resolved_path).name}):\n"
        output += "  Most likely elements visible:\n"
        for r in results:
            output += f"    - {r['description']} (confidence: {r['score']:.2f})\n"

        return output

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


@mcp.tool()
def match_image_to_elements(image_path: str, storey_filter: Optional[str] = None, top_k: int = 5) -> str:
    """
    Match a site photo to BIM elements in the IFC model.

    Uses CLIP visual AI to find which BIM elements best match what's shown
    in a site inspection photo. Combines visual understanding with spatial
    filtering by storey for accurate element identification.

    Args:
        image_path: Path to the site photo (jpg, png)
        storey_filter: Optional storey name to narrow search (e.g., "sixth", "Level 1")
        top_k: Number of top matches to return (default: 5)

    Returns:
        str: Ranked list of matching BIM elements with GUIDs and confidence scores

    Example:
        >>> match_image_to_elements("crack_photo.jpg", storey_filter="sixth", top_k=3)
        Top matches for site image:
          #1: Window_North_6F (GUID: 2O2Fr$...) - score: 0.78
          #2: Wall_Ext_6F (GUID: 1X3Gy$...) - score: 0.65
          #3: Slab_6F (GUID: 0P4Hz$...) - score: 0.52
    """
    aligner = get_visual_aligner()
    if aligner is None:
        return "Error: Visual analysis not available. Install transformers and torch."

    # Resolve path
    resolved_path = image_path
    if not Path(image_path).is_absolute():
        resolved_path = str(BASE_DIR / image_path)

    if not Path(resolved_path).exists():
        return f"Error: Image not found at '{resolved_path}'"

    # Get elements from IFC model, optionally filtered by storey
    elements = []
    if storey_filter:
        spaces_to_search = {k: v for k, v in engine.spatial_index.items()
                          if storey_filter.lower() in k.lower()}
    else:
        spaces_to_search = engine.spatial_index

    for space_name, space_elements in spaces_to_search.items():
        for el in space_elements:
            elements.append({
                "guid": el.get("guid", ""),
                "name": el.get("name", ""),
                "type": el.get("type", ""),
                "location": space_name
            })

    if not elements:
        return f"No elements found" + (f" on storey '{storey_filter}'" if storey_filter else "")

    try:
        # Match image to elements
        results = aligner.match_image_to_elements(resolved_path, elements, top_k=top_k)

        output = f"Top {len(results)} matches for site image"
        if storey_filter:
            output += f" (filtered to '{storey_filter}')"
        output += ":\n"

        for r in results:
            output += f"  #{r['rank']}: {r['name']} ({r['type']})\n"
            output += f"       GUID: {r['guid']}\n"
            output += f"       Confidence: {r['score']:.3f}\n"

        return output

    except Exception as e:
        return f"Error matching image to elements: {str(e)}"


@mcp.tool()
def compare_defect_images(image_path1: str, image_path2: str) -> str:
    """
    Compare two site photos to check if they show the same defect/area.

    Uses CLIP visual AI to compute semantic similarity between two images.
    Useful for:
    - Checking if a reported defect matches a previous report
    - Verifying that follow-up photos show the same issue
    - Grouping related defect images

    Args:
        image_path1: Path to first image
        image_path2: Path to second image

    Returns:
        str: Similarity analysis with score and interpretation

    Example:
        >>> compare_defect_images("crack_v1.jpg", "crack_v2.jpg")
        Image Comparison Results:
          Similarity Score: 0.85
          Interpretation: HIGH - Images very likely show the same subject
    """
    aligner = get_visual_aligner()
    if aligner is None:
        return "Error: Visual analysis not available. Install transformers and torch."

    # Resolve paths
    path1 = image_path1 if Path(image_path1).is_absolute() else str(BASE_DIR / image_path1)
    path2 = image_path2 if Path(image_path2).is_absolute() else str(BASE_DIR / image_path2)

    if not Path(path1).exists():
        return f"Error: First image not found at '{path1}'"
    if not Path(path2).exists():
        return f"Error: Second image not found at '{path2}'"

    try:
        similarity = aligner.compare_images(path1, path2)

        # Interpret the score
        if similarity >= 0.85:
            interpretation = "VERY HIGH - Images almost certainly show the same subject"
        elif similarity >= 0.70:
            interpretation = "HIGH - Images very likely show the same subject"
        elif similarity >= 0.50:
            interpretation = "MODERATE - Images may show related subjects"
        elif similarity >= 0.30:
            interpretation = "LOW - Images show somewhat different subjects"
        else:
            interpretation = "VERY LOW - Images show different subjects"

        output = "Image Comparison Results:\n"
        output += f"  Image 1: {Path(path1).name}\n"
        output += f"  Image 2: {Path(path2).name}\n"
        output += f"  Similarity Score: {similarity:.3f}\n"
        output += f"  Interpretation: {interpretation}\n"

        return output

    except Exception as e:
        return f"Error comparing images: {str(e)}"


@mcp.tool()
def match_text_to_elements(description: str, storey_filter: Optional[str] = None, top_k: int = 5) -> str:
    """
    Match a text description to BIM elements using semantic similarity.

    Uses CLIP text embeddings to find BIM elements that best match a
    verbal description. Useful when the user describes a defect or
    location without providing an image.

    Args:
        description: Text description (e.g., "cracked window on the north wall")
        storey_filter: Optional storey name to narrow search
        top_k: Number of top matches to return

    Returns:
        str: Ranked list of matching BIM elements with GUIDs

    Example:
        >>> match_text_to_elements("damaged concrete near the elevator", storey_filter="first")
        Top matches for description:
          #1: Slab_Elevator_1F (GUID: ...) - score: 0.72
          #2: Wall_Core_1F (GUID: ...) - score: 0.68
    """
    aligner = get_visual_aligner()
    if aligner is None:
        return "Error: Visual analysis not available. Install transformers and torch."

    # Get elements from IFC model
    elements = []
    if storey_filter:
        spaces_to_search = {k: v for k, v in engine.spatial_index.items()
                          if storey_filter.lower() in k.lower()}
    else:
        spaces_to_search = engine.spatial_index

    for space_name, space_elements in spaces_to_search.items():
        for el in space_elements:
            elements.append({
                "guid": el.get("guid", ""),
                "name": el.get("name", ""),
                "type": el.get("type", ""),
                "location": space_name
            })

    if not elements:
        return f"No elements found" + (f" on storey '{storey_filter}'" if storey_filter else "")

    try:
        # Build descriptions for elements
        element_descriptions = [aligner.build_element_description(el) for el in elements]

        # Match text to element descriptions
        results = aligner.match_text_to_descriptions(description, element_descriptions, top_k=top_k)

        output = f"Top {len(results)} matches for: \"{description}\"\n"
        if storey_filter:
            output += f"  (filtered to storey: '{storey_filter}')\n"

        for r in results:
            idx = r["index"]
            el = elements[idx]
            output += f"\n  #{r['rank']}: {el['name']} ({el['type']})\n"
            output += f"       GUID: {el['guid']}\n"
            output += f"       Location: {el['location']}\n"
            output += f"       Confidence: {r['score']:.3f}\n"

        return output

    except Exception as e:
        return f"Error matching text to elements: {str(e)}"


if __name__ == "__main__":
    # Run the MCP server
    print("[IFC Server] Starting MCP server via stdio...", file=sys.stderr)
    mcp.run(transport="stdio")
