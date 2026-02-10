"""
Evaluation helpers extracted from main_mcp.py.

Contains:
- get_experiment_description()  (from main_mcp.py:68-75)
- format_test_input()           (from main_mcp.py:165-234)
- evaluate_response()           (from main_mcp.py:261-343)
- compute_gt_matches()          (consolidates eval/runner.py:399-422 + v2/pipeline.py:196-209)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .guid import extract_guids_from_text


def get_experiment_description(experiment_mode, query_mode, visual_enabled):
    """Get human-readable description for experiment mode."""
    if not experiment_mode:
        return "Using config.yaml setting"
    base = "Neo4j graph queries" if query_mode == "neo4j" else "In-memory spatial index"
    if visual_enabled:
        return f"{base} + CLIP visual matching"
    return base


def format_test_input(case, image_dir):
    """
    Format a ground truth case into agent input string.

    Supports both formats:
    - Legacy (gt_1.json): context_payload.meta, image_file
    - Standardized (cases_v2.jsonl): inputs.project_context, inputs.images

    Args:
        case: Ground truth test case dict
        image_dir: Path to directory containing test images

    Returns:
        tuple: (formatted_input_string, list_of_image_paths)
    """
    # Handle both legacy and standardized field names
    if "context_payload" in case:
        # Legacy format (gt_1.json)
        meta = case["context_payload"]["meta"]
        chat_history = case["context_payload"]["chat_history"]
        image_files = case.get("image_file", [])
    else:
        # Standardized format (cases_v2.jsonl)
        inputs = case.get("inputs", {})
        meta = inputs.get("project_context", {})
        chat_history = inputs.get("chat_history", [])
        image_files = [Path(p).name for p in inputs.get("images", [])]

    # Build context string
    input_parts = [
        "=" * 50,
        "[CONTEXT]",
        f"  Timestamp: {meta.get('timestamp', 'N/A')}",
        f"  Sender Role: {meta.get('sender_role', 'N/A')}",
        f"  Project Phase: {meta.get('project_phase', 'N/A')}",
        f"  4D Task Status: {meta.get('4d_task_status', 'N/A')}",
        "",
        "[CHAT HISTORY]"
    ]

    for msg in chat_history:
        input_parts.append(f"  {msg['role']}: {msg['text']}")

    input_parts.extend([
        "",
        "[USER QUERY]",
        f"  {case.get('query_text', '')}",
    ])

    # Build image paths
    image_paths = []
    for img_file in image_files:
        img_path = Path(image_dir) / img_file
        if img_path.exists():
            image_paths.append(str(img_path))
        else:
            print(f"Warning: Image not found: {img_path}")

    # Include image paths in the message so agent can analyze them
    if image_paths:
        input_parts.append("")
        input_parts.append("[ATTACHED IMAGES]")
        for img_path in image_paths:
            input_parts.append(f"  Image path: {img_path}")
        input_parts.append("  Note: Use analyze_site_image(image_path) to analyze these images")

    input_parts.append("=" * 50)
    formatted_input = "\n".join(input_parts)

    return formatted_input, image_paths


def evaluate_response(response_text, ground_truth):
    """
    Evaluate agent response against ground truth with top-k metrics.

    Args:
        response_text: The agent's response string
        ground_truth: Ground truth dict with target_guid, expected_reasoning, etc.

    Returns:
        dict: Evaluation results with top-k metrics, precision, recall
    """
    target_guid = ground_truth.get("target_guid", "")
    target_name = ground_truth.get("target_name", "")
    rq_category = ground_truth.get("rq_category", "")

    # Extract all GUIDs mentioned in response (ordered by appearance)
    mentioned_guids = extract_guids_from_text(response_text)

    results = {
        "guid_match": False,
        "name_match": False,
        "target_guid": target_guid,
        "target_name": target_name,
        "rq_category": rq_category,
        "details": [],
        # Top-k metrics
        "mentioned_guids": mentioned_guids,
        "num_candidates": len(mentioned_guids),
        "top1_hit": False,
        "top3_hit": False,
        "top5_hit": False,
        "target_rank": None,
        # Retrieval metrics
        "precision_at_1": 0.0,
        "precision_at_3": 0.0,
        "recall": 0.0,
    }

    # Skip special target GUIDs
    if target_guid and target_guid not in ["MULTIPLE", "CLARIFICATION_NEEDED", "INSUFFICIENT_DATA", "INVALID_LOCATION"]:
        # Check if target GUID is found in response (backward compatible)
        if target_guid in response_text:
            results["guid_match"] = True
            results["details"].append(f"Target GUID found: {target_guid}")
        else:
            results["details"].append(f"Target GUID not found: {target_guid}")

        # Top-k evaluation
        if mentioned_guids:
            if target_guid in mentioned_guids:
                rank = mentioned_guids.index(target_guid) + 1  # 1-indexed
                results["target_rank"] = rank

                # Top-k hits
                results["top1_hit"] = (rank == 1)
                results["top3_hit"] = (rank <= 3)
                results["top5_hit"] = (rank <= 5)

                results["details"].append(f"Target GUID rank: {rank}/{len(mentioned_guids)}")

                # Precision@k (for single target, precision = 1/k if hit in top-k, else 0)
                results["precision_at_1"] = 1.0 if rank == 1 else 0.0
                results["precision_at_3"] = 1.0 / min(3, rank) if rank <= 3 else 0.0

                # Recall (single target: 1 if found, 0 if not)
                results["recall"] = 1.0
            else:
                results["details"].append(f"Target GUID not in {len(mentioned_guids)} mentioned candidates")
                results["recall"] = 0.0
        else:
            results["details"].append("No GUIDs extracted from response")
            results["recall"] = 0.0 if target_guid else 1.0

    # Check if target name is mentioned
    if target_name:
        name_parts = target_name.split(":")[0] if ":" in target_name else target_name
        if name_parts.lower() in response_text.lower():
            results["name_match"] = True
            results["details"].append(f"Target name found: {name_parts}")
        else:
            results["details"].append(f"Target name not found: {name_parts}")

    return results


def compute_gt_matches(
    target_guid: str,
    target_name: str,
    target_storey: str,
    response_text: str = "",
    candidate_guids: Optional[List[str]] = None,
    candidate_names: Optional[List[str]] = None,
    candidate_storeys: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    Compute guid_match, name_match, storey_match against ground truth.

    Supports two modes:
    - Text-based: check if target appears in response_text (v1 / eval/runner.py style)
    - Candidate-based: check if target appears in candidate lists (v2 style)

    Consolidates:
    - eval/runner.py lines 399-422
    - v2/pipeline.py lines 196-209
    """
    result = {"guid_match": False, "name_match": False, "storey_match": False}

    # GUID match
    if target_guid:
        if candidate_guids is not None:
            result["guid_match"] = target_guid in candidate_guids
        else:
            result["guid_match"] = target_guid in response_text

    # Name match (prefix before ":")
    if target_name:
        name_prefix = target_name.split(":")[0] if ":" in target_name else target_name
        if candidate_names is not None:
            result["name_match"] = any(
                name_prefix.lower() in (n or "").lower() for n in candidate_names
            )
        else:
            result["name_match"] = name_prefix.lower() in response_text.lower()

    # Storey match
    if target_storey:
        if candidate_storeys is not None:
            result["storey_match"] = any(
                target_storey.lower() in (s or "").lower() for s in candidate_storeys
            )
        else:
            result["storey_match"] = target_storey.lower() in response_text.lower()

    return result
