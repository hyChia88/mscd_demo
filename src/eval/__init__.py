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
    RQ2Result,
    RQ2Submission,
    RQ2ValidationMetadata,
    RQ2Uncertainty,
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
from .visualizations import (
    plot_accuracy_by_condition,
    plot_search_space_reduction,
    plot_constraints_parse_rate,
    plot_image_parse_timing,
    plot_vision_impact,
    plot_per_case_heatmap,
    plot_condition_wise_comparison,
    generate_all_plots,
)

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
    "RQ2Result",
    "RQ2Submission",
    "RQ2ValidationMetadata",
    "RQ2Uncertainty",
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
    # Visualizations
    "plot_accuracy_by_condition",
    "plot_search_space_reduction",
    "plot_constraints_parse_rate",
    "plot_image_parse_timing",
    "plot_vision_impact",
    "plot_per_case_heatmap",
    "plot_condition_wise_comparison",
    "generate_all_plots",
]
