"""
Deterministic mapping from agent output + IFC tool outputs to submission JSON.

LLM should NOT fill form fields directly; it provides:
- selected_guid
- issue_summary
- evidence list

This module handles the deterministic mapping to the schema.
"""

from typing import Any, Dict, List


def _safe_str(x: Any) -> str:
    """Safely convert to string, returning empty string for None."""
    return x if isinstance(x, str) else ""


def build_submission(
    schema_id: str,
    agent_final: Dict[str, Any],
    ifc_element: Dict[str, Any],
    ifc_psets: Dict[str, Any],
    storey_name: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deterministic mapping from (agent_final + ifc_tool outputs) -> submission JSON.

    Args:
        schema_id: Schema identifier string
        agent_final: Parsed FINAL_JSON from agent output
        ifc_element: Element info from get_element_details (GlobalId, Name, Type)
        ifc_psets: Property sets from get_element_details (PropertySets)
        storey_name: Storey name (from agent or ground truth)
        evidence: List of evidence objects

    Returns:
        Submission dict conforming to schema structure
    """
    guid = _safe_str(agent_final.get("selected_guid", ""))

    # Attempt to infer IFC class from tool outputs
    # get_element_details returns: {GlobalId, Name, Type, ObjectType, PropertySets}
    ifc_class = (
        ifc_element.get("Type")
        or ifc_element.get("ifc_class")
        or ifc_element.get("IfcClass")
        or ifc_element.get("type")
        or ifc_element.get("entity")
        or "UNKNOWN"
    )

    issue_type = agent_final.get("issue_type", "unknown")
    severity = agent_final.get("severity", "unknown")
    issue_summary = _safe_str(agent_final.get("issue_summary", ""))

    # Provenance tracking (thesis-friendly)
    provenance = {
        "issue.issue_type": "from_agent_final",
        "issue.severity": "from_agent_final",
        "issue.issue_summary": "from_agent_final",
        "bim_reference.element_guid": "from_agent_final",
        "bim_reference.ifc_class": "from_ifc_tool",
        "bim_reference.storey_name": "from_domain_validator",
        "bim_reference.pset_snapshot": "from_ifc_tool",
        "evidence": "from_agent_final_or_input",
    }

    submission = {
        "submission_id": f"demo_{guid[:6] if guid else 'unknown'}",
        "issue": {
            "issue_type": issue_type if issue_type else "unknown",
            "severity": severity if severity else "unknown",
            "issue_summary": issue_summary if issue_summary else "unspecified",
        },
        "bim_reference": {
            "element_guid": guid,
            "ifc_class": str(ifc_class),
            "storey_name": storey_name,
            "pset_snapshot": ifc_psets or {},
        },
        "evidence": evidence or [],
        "validation_metadata": {
            "schema_id": schema_id,
            "required_fill_rate": 0.0,
            "passed": False,
            "errors": [],
            "provenance": provenance,
        },
    }

    return submission
