"""
IFC GUID index — maps GlobalId → Express ID (once, cached).

Express ID = the integer line number in the STEP file.
Used by tab_result to show the raw STEP text for a specific element.
The 3D viewer uses GUIDs directly (model.getLocalIdsByGuids), so this
index is only needed for the IFC text panel.
"""
from pathlib import Path


def build_index(ifc_path: Path) -> dict[str, int]:
    """
    Build {GlobalId: express_id} for every element in the IFC model.
    Takes ~3-5 seconds on the 43MB AdvancedProject model.
    Result is cached via st.cache_resource in app.py.
    """
    import ifcopenshell  # lazy import — not needed at module load time
    model = ifcopenshell.open(str(ifc_path))
    return {
        e.GlobalId: e.id()
        for e in model
        if hasattr(e, "GlobalId") and e.GlobalId
    }


def get_step_text(ifc_path: Path, guid: str) -> str:
    """
    Return the raw STEP line for one element, e.g.:
      #195164= IFCWALLSTANDARDCASE('3GzoWuxx...',...)
    """
    import ifcopenshell
    model = ifcopenshell.open(str(ifc_path))
    element = model.by_guid(guid)
    if element is None:
        return f"# GUID {guid!r} not found in model"
    return str(element)
