"""
Trace Builder - Single Source of Truth for Evaluation Traces

This module builds and persists trace files that serve as the foundation
for BCF-lite issues and BCFzip generation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.guid import extract_first_ifc_guid


def extract_guid_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract GUID from tool call results.

    Searches through tool call results for IFC GUIDs, prioritizing
    results from element-specific tools.

    Args:
        tool_calls: List of tool call records with 'result' field

    Returns:
        First IFC GUID found in tool results, or None
    """
    # Priority tools that return element details
    priority_tools = ["get_element_by_guid", "get_element_details", "search_elements_by_type"]

    # First pass: check priority tools
    for tool in tool_calls:
        if tool.get("name") in priority_tools:
            result = tool.get("result", "")
            guid = extract_first_ifc_guid(str(result))
            if guid:
                return guid

    # Second pass: check all tool results
    for tool in tool_calls:
        result = tool.get("result", "")
        guid = extract_first_ifc_guid(str(result))
        if guid:
            return guid

    return None


def build_trace(
    run_id: str,
    case_id: str,
    test_case: Dict[str, Any],
    agent_response: str,
    tool_calls: List[Dict[str, Any]],
    eval_result: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a trace dictionary from evaluation components.

    The trace serves as the Single Source of Truth for:
    - BCF-lite issue.json generation
    - BCFzip generation
    - Thesis experiment reproducibility

    Args:
        run_id: Timestamp identifier for this evaluation run
        case_id: Unique identifier for the test case
        test_case: Original test case dictionary from ground truth
        agent_response: Raw text response from the agent
        tool_calls: List of tool call records
        eval_result: Evaluation result dictionary
        config: Configuration dictionary

    Returns:
        Complete trace dictionary
    """
    # Extract GUID with fallback chain
    guid_from_response = extract_first_ifc_guid(agent_response)
    guid_from_tools = extract_guid_from_tool_calls(tool_calls)

    # Determine GUID and source
    if guid_from_response:
        element_guid = guid_from_response
        guid_source = "regex_from_agent_response"
    elif guid_from_tools:
        element_guid = guid_from_tools
        guid_source = "from_tool_call"
    else:
        element_guid = ""
        guid_source = "none"

    # Extract image paths from test case
    images = []
    if "image" in test_case:
        images = [test_case["image"]] if isinstance(test_case["image"], str) else test_case["image"]
    elif "images" in test_case:
        images = test_case["images"]

    # Build schema section (from RQ2 if available)
    schema_section = {
        "schema_id": "",
        "fields": {},
        "validation": {
            "passed": False,
            "errors": [],
            "fill_rate": 0.0,
        },
    }

    # Populate from eval_result if RQ2 data exists
    if "rq2_result" in eval_result and eval_result["rq2_result"]:
        rq2 = eval_result["rq2_result"]
        schema_section["schema_id"] = rq2.get("schema_id", "")
        if "submission" in rq2:
            submission = rq2["submission"]
            schema_section["fields"] = submission.get("bim_reference", {})
            if "validation_metadata" in submission:
                vm = submission["validation_metadata"]
                schema_section["validation"] = {
                    "passed": vm.get("passed", False),
                    "errors": vm.get("errors", []),
                    "fill_rate": vm.get("required_fill_rate", 0.0),
                }

    # Build the trace
    trace = {
        "run_id": run_id,
        "case_id": case_id,
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "user_input": test_case.get("user_input", test_case.get("query", "")),
            "images": images,
            "ifc_path": config.get("ifc", {}).get("model_path", ""),
        },
        "agent": {
            "response_text": agent_response,
            "tool_calls": tool_calls,
        },
        "prediction": {
            "element_guid": element_guid,
            "guid_source": guid_source,
        },
        "schema": schema_section,
        "evaluation": eval_result,
        "ground_truth": test_case.get("ground_truth", {}),
    }

    return trace


def write_trace_json(trace: Dict[str, Any], out_dir: str = "outputs/traces") -> str:
    """
    Write trace to JSON file.

    Creates directory structure: {out_dir}/{run_id}/{case_id}.trace.json

    Args:
        trace: Trace dictionary to write
        out_dir: Base output directory

    Returns:
        Path to written trace file
    """
    run_id = trace.get("run_id", "unknown")
    case_id = trace.get("case_id", "unknown")

    # Create output directory
    trace_dir = Path(out_dir) / run_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Write trace file
    trace_path = trace_dir / f"{case_id}.trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False, default=str)

    return str(trace_path)
