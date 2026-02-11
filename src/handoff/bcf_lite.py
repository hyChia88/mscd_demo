"""
BCF-lite - Lightweight JSON Issue Output

Generates machine-readable issue.json files for each evaluation case.
These serve as a handoff format that can be:
- Consumed by downstream systems
- Converted to full BCFzip if needed
- Used for thesis demonstration
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from handoff.trace import build_issue_title, build_issue_description


def write_issue_json(out_dir: str, trace: Dict[str, Any]) -> str:
    """
    Write BCF-lite issue.json file.

    Creates directory structure: {out_dir}/{run_id}/{case_id}.issue.json

    The issue.json contains:
    - element_guid: Selected BIM element
    - title/description: Human-readable issue summary
    - schema validation info from RQ2 pipeline
    - evidence: Image paths from evaluation
    - trace_uri: Reference to full trace for reproducibility

    Args:
        out_dir: Base output directory (e.g., "outputs/issues")
        trace: Trace dictionary from build_trace()

    Returns:
        Path to written issue.json file
    """
    run_id = trace.get("run_id", "unknown")
    case_id = trace.get("case_id", "unknown")

    # Create output directory
    issue_dir = Path(out_dir) / run_id
    issue_dir.mkdir(parents=True, exist_ok=True)

    # Extract key fields from trace
    prediction = trace.get("prediction", {})
    element_guid = prediction.get("element_guid", "")
    guid_source = prediction.get("guid_source", "none")

    inputs = trace.get("inputs", {})
    images = inputs.get("images", [])

    schema = trace.get("schema", {})
    evaluation = trace.get("evaluation", {})
    ground_truth = trace.get("ground_truth", {})
    validation = schema.get("validation", {})

    # Title and description from shared helpers
    title = build_issue_title(trace)
    description = build_issue_description(trace, compact=False)

    # Build trace URI (relative path)
    trace_uri = f"outputs/traces/{run_id}/{case_id}.trace.json"

    # Determine issue severity from ground truth or default
    severity = ground_truth.get("severity", "medium")
    if severity not in ["low", "medium", "high", "critical"]:
        severity = "medium"

    # Determine issue type
    issue_type = ground_truth.get("issue_type", "defect")
    if issue_type not in ["defect", "compliance", "safety", "information"]:
        issue_type = "defect"

    # Build the issue JSON
    issue = {
        "issue_id": str(uuid.uuid4()),
        "case_id": case_id,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "title": title,
        "description": description,
        "issue_type": issue_type,
        "severity": severity,
        "status": "open",
        "element_guid": element_guid,
        "guid_source": guid_source,
        "bim_reference": {
            "element_guid": element_guid,
            "ifc_class": schema.get("fields", {}).get("ifc_class", ""),
            "storey_name": schema.get("fields", {}).get("storey_name", ""),
            "element_name": ground_truth.get("target_element_name", ""),
        },
        "schema": {
            "schema_id": schema.get("schema_id", ""),
            "fields": schema.get("fields", {}),
            "validation": validation,
        },
        "evidence": images,
        "trace_uri": trace_uri,
        "evaluation": {
            "guid_match": evaluation.get("guid_match", False),
            "name_match": evaluation.get("name_match", False),
            "storey_match": evaluation.get("storey_match", False),
        },
    }

    # Write issue file
    issue_path = issue_dir / f"{case_id}.issue.json"
    with open(issue_path, "w", encoding="utf-8") as f:
        json.dump(issue, f, indent=2, ensure_ascii=False)

    return str(issue_path)
