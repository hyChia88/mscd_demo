"""
IFC Query MCP Server

Exposes IFC/BIM query capabilities as MCP tools.
This server provides spatial queries, element retrieval, and property inspection.

Usage:
    python mcp_servers/ifc_server.py

The server will start and listen for MCP client connections via stdio.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import ifc_engine
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastmcp import FastMCP
from ifc_engine import IFCEngine

# Initialize MCP server
mcp = FastMCP("IFC Query Service")

# Load IFC engine (singleton pattern)
BASE_DIR = Path(__file__).parent.parent
IFC_PATH = os.getenv("IFC_MODEL_PATH", str(BASE_DIR / "data" / "BasicHouse.ifc"))

print(f"[IFC Server] Initializing with model: {IFC_PATH}", file=sys.stderr)
engine = IFCEngine(IFC_PATH)
print(f"[IFC Server] Engine ready. Spatial index contains {len(engine.spatial_index)} spaces.", file=sys.stderr)


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
    return engine.get_element_properties(guid)


@mcp.tool()
def generate_3d_view(guid: str) -> str:
    """
    Generate a visual verification snapshot (3D render) of an element.

    Creates a rendered image of the specified element for visual inspection.
    Useful when users need to 'see' or 'visualize' a defect location.

    Args:
        guid: Global Unique Identifier of the element to visualize

    Returns:
        str: File path to the generated render image

    Note:
        In production, this triggers a headless Blender rendering pipeline.
        For MVP/demo purposes, returns a mock file path.

    Example:
        >>> generate_3d_view("2O2Fr$t4X7Zf8NOew3FLOH")
        "/server/renders/2O2Fr$t4X7Zf8NOew3FLOH_inspection_view.png"
    """
    # MVP Strategy: Mock the rendering pipeline
    # Production implementation would trigger blender_service.py
    return f"/server/renders/{guid}_inspection_view.png"


# Resource for exposing IFC model metadata
@mcp.resource("ifc://model/metadata")
def get_model_metadata() -> str:
    """
    Get metadata about the loaded IFC model.

    Returns:
        str: JSON-formatted metadata including schema, project info, and statistics
    """
    import json

    metadata = {
        "model_path": str(IFC_PATH),
        "schema": engine.file.schema if hasattr(engine, 'file') else "IFC4",
        "num_spaces": len(engine.spatial_index),
        "total_elements": sum(len(elements) for elements in engine.spatial_index.values()),
        "available_spaces": list(engine.spatial_index.keys())
    }

    return json.dumps(metadata, indent=2)


if __name__ == "__main__":
    # Run the MCP server
    print("[IFC Server] Starting MCP server via stdio...", file=sys.stderr)
    mcp.run(transport="stdio")
