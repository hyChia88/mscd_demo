"""
V2 Diagnostic Metrics

Computes v2-specific metrics for constraints extraction quality,
field-level accuracy, and CLIP reranking effectiveness.
"""

from typing import List, Dict, Any, Optional, Tuple
from .types import Constraints, PipelineTrace, RetrievalResult


def compute_metrics(
    trace: PipelineTrace,
    ground_truth: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute V2-specific diagnostic metrics.

    Args:
        trace: PipelineTrace with constraints and retrieval results
        ground_truth: Ground truth dict from case
        labels: Label constraints from case["labels"]["constraints"]

    Returns:
        Dict with v2 metrics
    """
    metrics = {}

    # 1. Constraints parse success
    metrics["constraints_parsed"] = trace.constraints_parse_success

    # 2. Field-level F1 (only if labels available)
    if labels and trace.constraints and ground_truth:
        metrics["constraints_field_em_f1"] = compute_constraints_field_f1(
            trace.constraints,
            labels.get("constraints", {}),
            ground_truth
        )
    else:
        metrics["constraints_field_em_f1"] = None

    # 3. Rerank gain (only if reranking was applied)
    if trace.retrieval_results and ground_truth:
        metrics["rerank_gain"] = compute_rerank_gain(
            trace.retrieval_results,
            ground_truth.get("target_guid", "")
        )
    else:
        metrics["rerank_gain"] = None

    # 4. Timing breakdowns
    metrics["constraints_extraction_ms"] = trace.constraints_extraction_ms
    metrics["query_planning_ms"] = trace.query_planning_ms
    metrics["retrieval_ms"] = trace.retrieval_ms

    return metrics


def compute_constraints_field_f1(
    predicted: Constraints,
    label: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> float:
    """
    Compute field-level exact match F1 for constraints.

    Compares predicted constraints against ground truth labels.

    Args:
        predicted: Extracted constraints
        label: Label constraints from case
        ground_truth: Ground truth dict

    Returns:
        F1 score (0.0-1.0)
    """
    tp = fp = fn = 0

    # 1. Storey name match
    label_storey = label.get("storey_name")
    if predicted.storey_name:
        if label_storey and _normalize_storey(predicted.storey_name) == _normalize_storey(label_storey):
            tp += 1
        else:
            fp += 1
    else:
        if label_storey:
            fn += 1

    # 2. IFC class match
    label_ifc_class = label.get("ifc_class")
    if predicted.ifc_class:
        if label_ifc_class and predicted.ifc_class == label_ifc_class:
            tp += 1
        else:
            fp += 1
    else:
        if label_ifc_class:
            fn += 1

    # Compute F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def compute_rerank_gain(
    retrieval_results: List[RetrievalResult],
    target_guid: str
) -> Optional[float]:
    """
    Compute rank improvement after CLIP reranking.

    Rerank gain = (original_rank - reranked_rank) / original_rank
    Positive value indicates improvement.

    Args:
        retrieval_results: List of retrieval results
        target_guid: Ground truth GUID

    Returns:
        Rerank gain or None if not applicable
    """
    if not retrieval_results or not target_guid:
        return None

    # Find original and reranked results
    original_result = None
    reranked_result = None

    for i, result in enumerate(retrieval_results):
        if result.rerank_applied:
            reranked_result = result
            # Previous result is original (before rerank)
            if i > 0:
                original_result = retrieval_results[i - 1]
            break

    if not original_result or not reranked_result:
        return None

    # Find target rank in both
    original_rank = _find_guid_rank(original_result.candidates, target_guid)
    reranked_rank = _find_guid_rank(reranked_result.candidates, target_guid)

    if original_rank is None or reranked_rank is None:
        return None

    # Compute gain
    if original_rank > 0:
        gain = (original_rank - reranked_rank) / original_rank
    else:
        gain = 0.0

    return gain


def compute_summary(
    trace_pairs: List[Tuple[Any, PipelineTrace]]
) -> Dict[str, Any]:
    """
    Aggregate v2 metrics across all traces.

    Args:
        trace_pairs: List of (EvalTrace, PipelineTrace) tuples

    Returns:
        Dict with aggregated v2 metrics
    """
    if not trace_pairs:
        return {}

    total_traces = len(trace_pairs)

    # Counters
    parsed_count = 0
    f1_scores = []
    rerank_gains = []

    # Timing stats
    extraction_times = []
    planning_times = []
    retrieval_times = []

    for eval_trace, trace in trace_pairs:
        # Parse success
        if trace.constraints_parse_success:
            parsed_count += 1

        # Field F1 (if available)
        # Note: This requires access to labels, which may not be in trace
        # We'll skip aggregation here and compute in the runner

        # Rerank gain
        if trace.rerank_gain is not None:
            rerank_gains.append(trace.rerank_gain)

        # Timing
        extraction_times.append(trace.constraints_extraction_ms)
        planning_times.append(trace.query_planning_ms)
        retrieval_times.append(trace.retrieval_ms)

    # Compute aggregates
    summary = {
        "total_traces": total_traces,
        "constraints_parse_rate": parsed_count / total_traces if total_traces > 0 else 0.0,
        "avg_rerank_gain": sum(rerank_gains) / len(rerank_gains) if rerank_gains else None,
        "rerank_gain_cases": len(rerank_gains),

        # Timing
        "avg_constraints_extraction_ms": sum(extraction_times) / len(extraction_times) if extraction_times else 0.0,
        "avg_query_planning_ms": sum(planning_times) / len(planning_times) if planning_times else 0.0,
        "avg_retrieval_ms": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0,
    }

    return summary


def _normalize_storey(storey_name: str) -> str:
    """
    Normalize storey name for comparison.

    Args:
        storey_name: Raw storey name

    Returns:
        Normalized storey name
    """
    # Remove case and extra spaces
    normalized = storey_name.strip().lower()

    # Handle common variations
    # "6 - sixth floor" → "6"
    # "level 1" → "1"
    # "-1 - garage" → "-1"

    import re

    # Extract number if present
    match = re.search(r'(-?\d+)', normalized)
    if match:
        return match.group(1)

    return normalized


def _find_guid_rank(candidates: List[Dict[str, Any]], target_guid: str) -> Optional[int]:
    """
    Find 1-indexed rank of target GUID in candidate list.

    Args:
        candidates: List of candidate elements
        target_guid: Target GUID to find

    Returns:
        Rank (1-indexed) or None if not found
    """
    for idx, candidate in enumerate(candidates, 1):
        if candidate.get("guid") == target_guid:
            return idx

    return None
