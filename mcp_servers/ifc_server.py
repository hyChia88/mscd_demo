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
from pathlib import Path
from typing import Optional

# Add parent directory to path to import ifc_engine
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastmcp import FastMCP
from ifc_engine import IFCEngine

# Initialize MCP server
mcp = FastMCP("IFC Query Service")

# Load IFC engine (singleton pattern - loaded once, reused for all requests)
BASE_DIR = Path(__file__).parent.parent


def load_ifc_path():
    """Load IFC path from config.yaml or environment variable"""
    # First check environment variable (highest priority)
    if os.getenv("IFC_MODEL_PATH"):
        return os.getenv("IFC_MODEL_PATH")

    # Then check config.yaml
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            ifc_path = config.get("ifc", {}).get("model_path")
            if ifc_path:
                return str(BASE_DIR / ifc_path)

    # Fallback to default
    return str(BASE_DIR / "data" / "ifc" / "AdvancedProject" / "IFC" / "AdvancedProject.ifc")


IFC_PATH = load_ifc_path()

print(f"[IFC Server] Initializing with model: {IFC_PATH}", file=sys.stderr)
engine = IFCEngine(IFC_PATH)
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
