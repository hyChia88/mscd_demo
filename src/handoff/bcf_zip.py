"""
BCFzip Generator - BIM Collaboration Format Issue Files

Generates BCF 2.1 compliant .bcfzip files for interoperability with
BIM tools (Revit, BIMcollab, Navisworks, etc.).

BCFzip Structure:
    <case_id>.bcfzip
    ├── bcf.version
    └── <topic_guid>/
        ├── markup.bcf
        └── viewpoints/
            ├── viewpoint.bcfv
            └── snapshot.png (if available)
"""

import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET


def _generate_uuid() -> str:
    """Generate a BCF-style UUID (uppercase, no hyphens in some contexts)."""
    return str(uuid.uuid4())


def _create_bcf_version() -> str:
    """
    Create bcf.version XML content.

    BCF 2.1 version file format.
    """
    root = ET.Element("Version", VersionId="2.1")
    ET.SubElement(root, "DetailedVersion").text = "2.1"
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _create_markup_bcf(
    topic_guid: str,
    viewpoint_guid: str,
    title: str,
    description: str,
    author: str,
    element_guid: str,
    has_snapshot: bool,
    trace_uri: str,
) -> str:
    """
    Create markup.bcf XML content.

    Contains topic metadata and viewpoint references.
    """
    # Root element
    markup = ET.Element("Markup")

    # Topic element
    topic = ET.SubElement(markup, "Topic", Guid=topic_guid, TopicType="Issue")
    ET.SubElement(topic, "Title").text = title
    ET.SubElement(topic, "CreationDate").text = datetime.now().isoformat()
    ET.SubElement(topic, "CreationAuthor").text = author
    ET.SubElement(topic, "ModifiedDate").text = datetime.now().isoformat()
    ET.SubElement(topic, "ModifiedAuthor").text = author

    # Description with trace reference for reproducibility
    full_description = description
    if trace_uri:
        full_description += f"\n\n[Trace: {trace_uri}]"
    ET.SubElement(topic, "Description").text = full_description

    # Priority based on severity (BCF uses Priority 1-4)
    ET.SubElement(topic, "Priority").text = "2"  # Default to medium

    # Viewpoints section
    viewpoints = ET.SubElement(markup, "Viewpoints")
    viewpoint_ref = ET.SubElement(viewpoints, "ViewPoint", Guid=viewpoint_guid)
    ET.SubElement(viewpoint_ref, "Viewpoint").text = f"viewpoints/viewpoint.bcfv"
    if has_snapshot:
        ET.SubElement(viewpoint_ref, "Snapshot").text = f"viewpoints/snapshot.png"

    return ET.tostring(markup, encoding="unicode", xml_declaration=True)


def _create_viewpoint_bcfv(
    viewpoint_guid: str,
    element_guid: str,
) -> str:
    """
    Create viewpoint.bcfv XML content.

    Contains camera position and component selection.
    The element_guid is placed in Components/Selection to indicate
    which BIM element this issue references.
    """
    # Root element
    vis_info = ET.Element("VisualizationInfo", Guid=viewpoint_guid)

    # Components section - this is where we reference the BIM element
    if element_guid:
        components = ET.SubElement(vis_info, "Components")
        selection = ET.SubElement(components, "Selection")
        component = ET.SubElement(selection, "Component", IfcGuid=element_guid)
        # Optional: Add originating system
        ET.SubElement(component, "OriginatingSystem").text = "MSCD_Demo"

    # Camera section - using default orthogonal view
    # Many BCF consumers don't strictly require this, but it's good practice
    ortho_camera = ET.SubElement(vis_info, "OrthogonalCamera")

    # Camera position (default view looking at origin)
    camera_pos = ET.SubElement(ortho_camera, "CameraViewPoint")
    ET.SubElement(camera_pos, "X").text = "0"
    ET.SubElement(camera_pos, "Y").text = "0"
    ET.SubElement(camera_pos, "Z").text = "10"

    # Camera direction (looking down)
    camera_dir = ET.SubElement(ortho_camera, "CameraDirection")
    ET.SubElement(camera_dir, "X").text = "0"
    ET.SubElement(camera_dir, "Y").text = "0"
    ET.SubElement(camera_dir, "Z").text = "-1"

    # Camera up vector
    camera_up = ET.SubElement(ortho_camera, "CameraUpVector")
    ET.SubElement(camera_up, "X").text = "0"
    ET.SubElement(camera_up, "Y").text = "1"
    ET.SubElement(camera_up, "Z").text = "0"

    # View to world scale
    ET.SubElement(ortho_camera, "ViewToWorldScale").text = "1"

    return ET.tostring(vis_info, encoding="unicode", xml_declaration=True)


def write_bcfzip(out_dir: str, trace: Dict[str, Any]) -> str:
    """
    Create a BCF 2.1 compliant .bcfzip file.

    Creates directory structure: {out_dir}/{run_id}/{case_id}.bcfzip

    The BCFzip contains:
    - bcf.version: BCF version declaration
    - <topic_guid>/markup.bcf: Issue metadata and viewpoint references
    - <topic_guid>/viewpoints/viewpoint.bcfv: Camera and component selection
    - <topic_guid>/viewpoints/snapshot.png: Evidence image (if available)

    Args:
        out_dir: Base output directory (e.g., "outputs/bcf")
        trace: Trace dictionary from build_trace()

    Returns:
        Path to written .bcfzip file
    """
    run_id = trace.get("run_id", "unknown")
    case_id = trace.get("case_id", "unknown")

    # Create output directory
    bcf_dir = Path(out_dir) / run_id
    bcf_dir.mkdir(parents=True, exist_ok=True)

    # Generate GUIDs
    topic_guid = _generate_uuid()
    viewpoint_guid = _generate_uuid()

    # Extract fields from trace
    prediction = trace.get("prediction", {})
    element_guid = prediction.get("element_guid", "")

    inputs = trace.get("inputs", {})
    images = inputs.get("images", [])
    user_input = inputs.get("user_input", "")

    ground_truth = trace.get("ground_truth", {})
    schema = trace.get("schema", {})

    # Build title and description
    title = f"{case_id}: {ground_truth.get('target_element_name', 'BIM Issue')}"
    if len(title) > 80:
        title = title[:77] + "..."

    description_parts = [
        f"Query: {user_input}",
        f"Element GUID: {element_guid or '(none)'}",
    ]

    validation = schema.get("validation", {})
    if validation.get("passed"):
        description_parts.append(f"Validation: PASSED ({validation.get('fill_rate', 0):.0%})")
    elif validation.get("errors"):
        description_parts.append(f"Validation: FAILED - {', '.join(validation.get('errors', []))}")

    description = "\n".join(description_parts)

    # Trace URI for reproducibility
    trace_uri = f"outputs/traces/{run_id}/{case_id}.trace.json"

    # Check for snapshot image
    snapshot_path: Optional[Path] = None
    if images:
        first_image = Path(images[0])
        if first_image.exists():
            snapshot_path = first_image

    # Create bcfzip
    bcfzip_path = bcf_dir / f"{case_id}.bcfzip"

    with zipfile.ZipFile(bcfzip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # bcf.version
        zf.writestr("bcf.version", _create_bcf_version())

        # Topic directory
        topic_dir = topic_guid

        # markup.bcf
        markup_content = _create_markup_bcf(
            topic_guid=topic_guid,
            viewpoint_guid=viewpoint_guid,
            title=title,
            description=description,
            author="MSCD_Demo",
            element_guid=element_guid,
            has_snapshot=snapshot_path is not None,
            trace_uri=trace_uri,
        )
        zf.writestr(f"{topic_dir}/markup.bcf", markup_content)

        # viewpoints/viewpoint.bcfv
        viewpoint_content = _create_viewpoint_bcfv(
            viewpoint_guid=viewpoint_guid,
            element_guid=element_guid,
        )
        zf.writestr(f"{topic_dir}/viewpoints/viewpoint.bcfv", viewpoint_content)

        # viewpoints/snapshot.png (if available)
        if snapshot_path and snapshot_path.exists():
            zf.write(snapshot_path, f"{topic_dir}/viewpoints/snapshot.png")

    return str(bcfzip_path)
