"""
V2 Package - Constraints-Driven Pipeline

This package implements the v2 pipeline with explicit constraints extraction
and deterministic query planning for reproducible BIM element retrieval.

Public API exports:
- Data types (Constraints, QueryPlan, RetrievalResult, V2Trace)
- Extractors (PromptConstraintsExtractor, LoRAConstraintsExtractor)
- Query planner (QueryPlanner)
- Retrieval backend (RetrievalBackend)
- Utilities (ConditionMask)
- Metrics (compute_v2_metrics, compute_v2_summary)
"""

from .types import (
    Constraints,
    QueryPlan,
    RetrievalResult,
    V2Trace,
    ConditionOverride,
    ProfileConfig
)

from .condition_mask import ConditionMask

from .constraints_extractor_prompt_only import PromptConstraintsExtractor
from .constraints_extractor_lora import LoRAConstraintsExtractor

from .constraints_to_query import QueryPlanner

from .retrieval_backend import RetrievalBackend

from .metrics_v2 import (
    compute_v2_metrics,
    compute_v2_summary,
    compute_constraints_field_f1,
    compute_rerank_gain
)

__all__ = [
    # Data types
    "Constraints",
    "QueryPlan",
    "RetrievalResult",
    "V2Trace",
    "ConditionOverride",
    "ProfileConfig",

    # Core components
    "ConditionMask",
    "PromptConstraintsExtractor",
    "LoRAConstraintsExtractor",
    "QueryPlanner",
    "RetrievalBackend",

    # Metrics
    "compute_v2_metrics",
    "compute_v2_summary",
    "compute_constraints_field_f1",
    "compute_rerank_gain",
]

__version__ = "2.0.0"
