"""
Metrics Computation for Evaluation Pipeline v2

Implements the core evaluation metrics:
- top1_hit: Exact match on first candidate
- topk_hit: Match within top-k candidates
- search_space_reduction: (1 - final_pool/initial_pool)
- field_population_rate: Fraction of expected fields populated
- count_tool_calls: Count and distribution of tool usage
- escalation_rate: Fraction of scenarios requiring human escalation
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .contracts import EvalTrace, MetricsSummary


def top1_hit(trace: EvalTrace) -> bool:
    """
    Check if the top-1 candidate matches the ground truth GUID.

    Args:
        trace: Evaluation trace with candidates and ground truth

    Returns:
        True if the first candidate's GUID matches ground truth
    """
    if not trace.interpreter_output:
        return trace.guid_match  # Fall back to simple GUID-in-text check

    candidates = trace.interpreter_output.candidates
    target_guid = trace.scenario.ground_truth.target_guid

    if not candidates:
        # Check if GUID appears in mentioned_guids
        mentioned = trace.interpreter_output.mentioned_guids
        return target_guid in mentioned[:1] if mentioned else trace.guid_match

    return candidates[0].guid == target_guid


def topk_hit(trace: EvalTrace, k: int = 3) -> bool:
    """
    Check if the ground truth GUID appears in top-k candidates.

    Args:
        trace: Evaluation trace with candidates and ground truth
        k: Number of top candidates to consider (default: 3)

    Returns:
        True if ground truth GUID is in top-k candidates
    """
    if not trace.interpreter_output:
        return trace.guid_match  # Fall back to simple check

    target_guid = trace.scenario.ground_truth.target_guid
    candidates = trace.interpreter_output.candidates

    if not candidates:
        # Check mentioned_guids
        mentioned = trace.interpreter_output.mentioned_guids
        return target_guid in mentioned[:k] if mentioned else trace.guid_match

    top_k_guids = [c.guid for c in candidates[:k]]
    return target_guid in top_k_guids


def search_space_reduction(trace: EvalTrace) -> Optional[float]:
    """
    Compute search-space reduction ratio.

    Formula: reduction = 1 - (final_pool_size / initial_pool_size)

    A higher value indicates better filtering (closer to 1.0 is better).

    Args:
        trace: Evaluation trace with pool sizes

    Returns:
        Reduction ratio (0.0 to 1.0), or None if pool sizes unavailable
    """
    if trace.initial_pool_size is None or trace.final_pool_size is None:
        return None

    if trace.initial_pool_size == 0:
        return None

    reduction = 1.0 - (trace.final_pool_size / trace.initial_pool_size)
    return max(0.0, min(1.0, reduction))  # Clamp to [0, 1]


def field_population_rate(trace: EvalTrace) -> float:
    """
    Compute the fraction of expected output fields that were populated.

    Expected fields (based on action_required in ground truth):
    - GUID identified
    - Element name identified
    - Storey/location identified
    - Properties retrieved (if Retrieve_Property in action_required)
    - Compliance check performed (if Check_Regulatory in action_required)

    Args:
        trace: Evaluation trace with interpreter output

    Returns:
        Fraction of expected fields populated (0.0 to 1.0)
    """
    if not trace.interpreter_output:
        return 0.0

    fields = trace.interpreter_output.fields_populated

    # Define expected fields based on scenario
    expected_fields = ["guid", "name", "storey"]

    action = trace.scenario.ground_truth.action_required or ""
    if "Retrieve_Property" in action or "Retrieve_IFC_Properties" in action:
        expected_fields.append("properties")
    if "Check_Regulatory" in action or "Validate_IFC_SG" in action:
        expected_fields.append("compliance_check")
    if "Create_BCF" in action:
        expected_fields.append("bcf_issue")

    if not expected_fields:
        return 1.0

    populated_count = sum(1 for f in expected_fields if fields.get(f, False))
    return populated_count / len(expected_fields)


def count_tool_calls(trace: EvalTrace) -> int:
    """
    Count total number of tool invocations in a trace.

    Args:
        trace: Evaluation trace with tool steps

    Returns:
        Number of tool calls
    """
    return len(trace.tool_steps)


def tool_call_distribution(traces: List[EvalTrace]) -> Dict[str, int]:
    """
    Compute distribution of tool calls across all traces.

    Args:
        traces: List of evaluation traces

    Returns:
        Dictionary mapping tool_name -> count
    """
    distribution: Dict[str, int] = {}

    for trace in traces:
        for step in trace.tool_steps:
            tool_name = step.tool_name
            distribution[tool_name] = distribution.get(tool_name, 0) + 1

    return distribution


def is_escalation(trace: EvalTrace) -> bool:
    """
    Check if a trace represents an escalation (agent couldn't resolve).

    Escalation indicators:
    - Agent explicitly asks for clarification
    - Agent says it cannot determine the element
    - No candidates identified
    - Error during execution

    Args:
        trace: Evaluation trace

    Returns:
        True if this scenario escalated
    """
    if not trace.success:
        return True

    if not trace.interpreter_output:
        return True

    return trace.interpreter_output.is_escalation


def escalation_rate(traces: List[EvalTrace]) -> float:
    """
    Compute fraction of scenarios that required escalation.

    Args:
        traces: List of evaluation traces

    Returns:
        Escalation rate (0.0 to 1.0)
    """
    if not traces:
        return 0.0

    escalated = sum(1 for t in traces if is_escalation(t))
    return escalated / len(traces)


def compute_summary(traces: List[EvalTrace], topk: int = 3) -> MetricsSummary:
    """
    Compute aggregated metrics summary from all traces.

    Args:
        traces: List of evaluation traces
        topk: K value for top-k accuracy (default: 3)

    Returns:
        MetricsSummary with all aggregated metrics
    """
    if not traces:
        return MetricsSummary()

    summary = MetricsSummary(
        total_scenarios=len(traces),
        successful_runs=sum(1 for t in traces if t.success),
    )

    # Accuracy metrics
    summary.top1_hits = sum(1 for t in traces if top1_hit(t))
    summary.top1_accuracy = summary.top1_hits / len(traces)

    summary.topk_hits = sum(1 for t in traces if topk_hit(t, topk))
    summary.topk_accuracy = summary.topk_hits / len(traces)

    # Search-space reduction
    reductions = [
        r for r in (search_space_reduction(t) for t in traces) if r is not None
    ]
    summary.avg_search_space_reduction = (
        sum(reductions) / len(reductions) if reductions else 0.0
    )

    # Field population
    pop_rates = [field_population_rate(t) for t in traces]
    summary.avg_field_population_rate = sum(pop_rates) / len(pop_rates)

    # Tool usage
    summary.total_tool_calls = sum(count_tool_calls(t) for t in traces)
    summary.avg_tool_calls_per_scenario = summary.total_tool_calls / len(traces)
    summary.tool_call_distribution = tool_call_distribution(traces)

    # Escalation
    summary.escalation_count = sum(1 for t in traces if is_escalation(t))
    summary.escalation_rate = summary.escalation_count / len(traces)

    # Timing
    latencies = [t.total_latency_ms for t in traces if t.total_latency_ms > 0]
    summary.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    # By RQ category
    by_rq: Dict[str, List[EvalTrace]] = defaultdict(list)
    for t in traces:
        rq = t.scenario.ground_truth.rq_category.value
        by_rq[rq].append(t)

    for rq, rq_traces in by_rq.items():
        rq_top1 = sum(1 for t in rq_traces if top1_hit(t))
        summary.by_rq_category[rq] = {
            "total": len(rq_traces),
            "top1_hits": rq_top1,
            "top1_accuracy": rq_top1 / len(rq_traces),
            "escalation_rate": escalation_rate(rq_traces),
        }

    # RQ2: Schema validation metrics
    rq2_traces = [t for t in traces if t.rq2_result is not None]
    if rq2_traces:
        summary.rq2_total = len(rq2_traces)
        summary.rq2_validation_passed = sum(
            1 for t in rq2_traces if t.rq2_result.submission.validation_metadata.passed
        )
        summary.rq2_validation_pass_rate = (
            summary.rq2_validation_passed / summary.rq2_total
        )
        summary.rq2_avg_fill_rate = sum(
            t.rq2_result.submission.validation_metadata.required_fill_rate
            for t in rq2_traces
        ) / len(rq2_traces)

    return summary
