"""
RQ2 Post-processing Pipeline.

Orchestrates:
1. FINAL_JSON extraction (done by caller)
2. MCP tool calls to get IFC data
3. Deterministic mapping to submission JSON
4. Validation (JSON Schema + domain)
5. Uncertainty/escalation determination
"""

import json
import re
from typing import Any, Dict, List

from .mapping import build_submission
from .validators import validate_all


def _normalize_tool_result(x: Any) -> Any:
    """
    Normalize MCP tool results.

    MCP tool results may arrive as dict/list OR JSON string.
    This function normalizes to Python objects.

    Args:
        x: Raw tool result

    Returns:
        Normalized Python object (dict, list, or original if unparseable)
    """
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        s = x.strip()
        # Try JSON parse
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def _parse_storeys_from_string(raw: str) -> List[str]:
    """
    Parse storey names from list_available_spaces() string output.

    The output format is:
    Available spaces:
      - 'Kitchen' (12 elements)
      - 'Living Room' (8 elements)

    Args:
        raw: Raw string output from list_available_spaces

    Returns:
        List of storey/space names
    """
    storeys = []
    # Pattern: '...' (N elements)
    pattern = r"'([^']+)'\s*\(\d+\s*elements?\)"
    matches = re.findall(pattern, raw)
    storeys.extend(matches)
    return storeys


async def run_rq2_postprocess(
    schema_id: str,
    schema: Dict[str, Any],
    agent_final: Dict[str, Any],
    parse_error: str,
    rq2_context: Dict[str, Any],
    tool_by_name: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run RQ2 post-processing pipeline.

    Args:
        schema_id: Schema identifier
        schema: JSON Schema dict
        agent_final: Parsed FINAL_JSON from agent (or {} if parse failed)
        parse_error: Error message from FINAL_JSON extraction (or "")
        rq2_context: Context dict with keys:
            - storey_name: str (from agent or ground truth)
            - evidence: list[{type, ref, note}]
        tool_by_name: Dict mapping tool name -> LangChain tool

    Returns:
        Dict with keys:
            - schema_id
            - final_json_parse_error
            - agent_final
            - submission (the validated submission JSON)
            - uncertainty (escalation info)
    """
    storey_name = rq2_context.get("storey_name", "")
    evidence = rq2_context.get("evidence", [])

    guid = agent_final.get("selected_guid", "") if agent_final else ""
    guid = guid or ""

    # === Domain tool checks ===

    # 1) Get available storeys list
    available_storeys = []
    try:
        storeys_raw = await tool_by_name["list_available_spaces"].ainvoke({})
        storeys_obj = _normalize_tool_result(storeys_raw)
        if isinstance(storeys_obj, list):
            available_storeys = storeys_obj
        elif isinstance(storeys_obj, str):
            # Parse from formatted string output
            available_storeys = _parse_storeys_from_string(storeys_obj)
    except Exception:
        pass

    # 2) Check element exists + get properties
    element_exists = False
    ifc_element = {}
    ifc_psets = {}

    if guid:
        try:
            # get_element_details returns JSON with:
            # {GlobalId, Name, Type, ObjectType, Description, PropertySets}
            elem_raw = await tool_by_name["get_element_details"].ainvoke({"guid": guid})
            elem_obj = _normalize_tool_result(elem_raw)
            if isinstance(elem_obj, dict) and elem_obj and "error" not in elem_obj:
                ifc_element = elem_obj
                element_exists = True
                # Extract PropertySets if present
                if "PropertySets" in elem_obj:
                    ifc_psets = elem_obj["PropertySets"]
        except Exception:
            pass

    # === Build submission ===
    submission = build_submission(
        schema_id=schema_id,
        agent_final=agent_final or {},
        ifc_element=ifc_element,
        ifc_psets=ifc_psets,
        storey_name=storey_name,
        evidence=evidence,
    )

    # === Validate ===
    passed, fill_rate, errors = validate_all(
        schema, submission, available_storeys, element_exists
    )

    # Update submission with validation results
    submission["validation_metadata"]["passed"] = passed
    submission["validation_metadata"]["required_fill_rate"] = fill_rate
    submission["validation_metadata"]["errors"] = errors

    # === Determine uncertainty/escalation ===
    uncertainty = {
        "escalated": not passed,
        "reason": (
            parse_error
            if parse_error
            else ("validator_failed" if not passed else "")
        ),
        "confidence": 0.85 if passed else 0.2,
    }

    return {
        "schema_id": schema_id,
        "final_json_parse_error": parse_error,
        "agent_final": agent_final,
        "submission": submission,
        "uncertainty": uncertainty,
    }
