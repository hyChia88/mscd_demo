"""
V2 Data Structures

Core Pydantic models for constraints extraction, query planning, and retrieval.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict, Any


class Constraints(BaseModel):
    """
    Extracted constraints from chat + images + 4D context.

    Represents structured semantic understanding of the user's query
    in terms of spatial and semantic filters.
    """

    storey_name: Optional[str] = None  # e.g., "6 - Sixth Floor", "Level 1"
    ifc_class: Optional[str] = None  # e.g., "IfcWindow", "IfcWall", "IfcDoor"
    near_keywords: List[str] = Field(default_factory=list)  # e.g., ["north", "elevator"]
    relations: List[str] = Field(default_factory=list)  # e.g., ["adjacent_to", "in_room"]

    # Diagnostics
    confidence: float = 0.0  # Confidence score (0.0-1.0)
    source: str = "unknown"  # "prompt" | "lora" | "prompt_failed"


class QueryPlan(BaseModel):
    """Deterministic query plan for retrieval."""

    priority: int  # 1 (highest) to 5 (fallback)
    strategy: str  # "storey+type", "storey_only", "type_only", "keyword", "fallback"
    params: Dict[str, Any]  # Parameters for execution
    expected_pool_size: Optional[int] = None


class RetrievalResult(BaseModel):
    """Result from retrieval backend execution."""

    candidates: List[Dict[str, Any]]
    pool_size: int
    query_plan_used: QueryPlan
    backend: str  # "memory", "neo4j", "memory+clip", "neo4j+clip"
    rerank_applied: bool = False


class V2Trace(BaseModel):
    """V2-specific trace data (augments EvalTrace)."""

    constraints: Optional[Constraints] = None
    query_plans: List[QueryPlan] = Field(default_factory=list)
    retrieval_results: List[RetrievalResult] = Field(default_factory=list)

    # Diagnostics
    constraints_parse_success: bool = False
    constraints_parse_error: Optional[str] = None
    rerank_gain: Optional[float] = None  # For condition B2/C2 (rank improvement)

    # Timing
    constraints_extraction_ms: float = 0.0
    query_planning_ms: float = 0.0
    retrieval_ms: float = 0.0


class ConditionOverride(BaseModel):
    """Condition-specific overrides (from profiles.yaml conditions section)."""

    model_config = ConfigDict(populate_by_name=True)

    use_images: bool = True
    use_floorplan: bool = False
    chat_blur: bool = False
    four_d_metadata: bool = Field(True, alias="4d_metadata")
    four_d_enhanced: bool = Field(False, alias="4d_enhanced")
    force_clip: bool = False


class ProfileConfig(BaseModel):
    """Profile configuration from profiles.yaml."""

    pipeline: str  # "v1" or "v2"
    constraints_model: Optional[str] = None  # "prompt" or "lora" (v2 only)
    retrieval: str  # "memory" or "neo4j"
    use_clip: bool = False
    thin_agent: bool = False
    rq2_schema: bool = True
    bcf: bool = False
    description: str = ""
