"""
Evaluation Pipeline v2

Provides structured evaluation with data contracts, metrics, and JSONL trace output.
"""

from .contracts import (
    ScenarioInput,
    ToolStepRecord,
    InterpreterOutput,
    EvalTrace,
    MetricsSummary,
    GroundTruth,
    CandidateElement,
    RQCategory,
)
from .metrics import (
    top1_hit,
    topk_hit,
    search_space_reduction,
    field_population_rate,
    count_tool_calls,
    is_escalation,
    compute_summary,
)
from .runner import run_one_scenario

__all__ = [
    # Contracts
    "ScenarioInput",
    "ToolStepRecord",
    "InterpreterOutput",
    "EvalTrace",
    "MetricsSummary",
    "GroundTruth",
    "CandidateElement",
    "RQCategory",
    # Metrics
    "top1_hit",
    "topk_hit",
    "search_space_reduction",
    "field_population_rate",
    "count_tool_calls",
    "is_escalation",
    "compute_summary",
    # Runner
    "run_one_scenario",
]
