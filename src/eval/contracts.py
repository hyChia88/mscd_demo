"""
Data Contracts for Evaluation Pipeline v2

Pydantic models that define the schema for:
- Input scenarios (from ground truth)
- Tool execution records
- Agent output parsing
- Complete evaluation traces
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RQCategory(str, Enum):
    """Research Question categories from the thesis"""

    RQ1 = "RQ1"  # Visual + Contextual Disambiguation
    RQ2 = "RQ2"  # Regulatory Compliance
    RQ3 = "RQ3"  # Abductive Reasoning
    UNKNOWN = "Unknown"


class ContextMeta(BaseModel):
    """Metadata from the 4D context payload"""

    timestamp: str
    sender_id: Optional[str] = None
    sender_role: str
    project_phase: str
    task_status: Optional[str] = Field(None, alias="4d_task_status")

    class Config:
        populate_by_name = True


class ChatMessage(BaseModel):
    """A single message in the chat history"""

    role: str
    text: str


class GroundTruth(BaseModel):
    """Ground truth data for a scenario"""

    target_guid: str
    target_name: str
    target_storey: str
    expected_reasoning: Optional[str] = None
    action_required: Optional[str] = None
    rq_category: RQCategory = RQCategory.UNKNOWN


class ScenarioInput(BaseModel):
    """
    Complete input scenario for evaluation.
    Maps directly to ground truth JSON structure.
    """

    id: str
    image_files: List[str] = Field(default_factory=list, alias="image_file")
    context_meta: ContextMeta
    chat_history: List[ChatMessage]
    query_text: str
    ground_truth: GroundTruth

    # Computed at runtime
    formatted_input: Optional[str] = None
    image_paths: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True

    @classmethod
    def from_ground_truth_dict(cls, data: dict, image_dir: str) -> "ScenarioInput":
        """Factory method to create from ground truth JSON entry"""
        context = data.get("context_payload", {})
        gt = data.get("ground_truth", {})

        # Build image paths
        image_files = data.get("image_file", [])
        image_paths = []
        for img in image_files:
            img_path = Path(image_dir) / img
            if img_path.exists():
                image_paths.append(str(img_path))

        # Parse RQ category safely
        rq_raw = gt.get("rq_category", "Unknown")
        try:
            rq_category = RQCategory(rq_raw)
        except ValueError:
            rq_category = RQCategory.UNKNOWN

        return cls(
            id=data.get("id", "unknown"),
            image_files=image_files,
            context_meta=ContextMeta(
                timestamp=context.get("meta", {}).get("timestamp", ""),
                sender_id=context.get("meta", {}).get("sender_id"),
                sender_role=context.get("meta", {}).get("sender_role", ""),
                project_phase=context.get("meta", {}).get("project_phase", ""),
                task_status=context.get("meta", {}).get("4d_task_status"),
            ),
            chat_history=[
                ChatMessage(**msg) for msg in context.get("chat_history", [])
            ],
            query_text=data.get("query_text", ""),
            ground_truth=GroundTruth(
                target_guid=gt.get("target_guid", ""),
                target_name=gt.get("target_name", ""),
                target_storey=gt.get("target_storey", ""),
                expected_reasoning=gt.get("expected_reasoning"),
                action_required=gt.get("action_required"),
                rq_category=rq_category,
            ),
            image_paths=image_paths,
        )


class ToolStepRecord(BaseModel):
    """
    Record of a single tool invocation during agent execution.
    Captures timing, inputs, outputs, and extracted candidates.
    """

    step_index: int
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Optional[str] = None

    # Timing information
    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: Optional[float] = None

    # Candidate extraction from tool result
    candidates_extracted: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_count: int = 0

    # Error handling
    error: Optional[str] = None
    success: bool = True


class CandidateElement(BaseModel):
    """
    A candidate BIM element extracted from agent reasoning.
    Represents elements the agent considered or returned.
    """

    guid: str
    name: Optional[str] = None
    element_type: Optional[str] = None
    storey: Optional[str] = None
    confidence: Optional[float] = None  # If agent provides ranking
    source_step: Optional[int] = None  # Which tool step produced this


class InterpreterOutput(BaseModel):
    """
    Parsed output from the agent's final response.
    Extracts structured data from natural language response.
    """

    raw_response: str

    # Extracted elements
    mentioned_guids: List[str] = Field(default_factory=list)
    mentioned_names: List[str] = Field(default_factory=list)
    candidates: List[CandidateElement] = Field(default_factory=list)

    # Classification
    is_clarification_request: bool = False
    is_escalation: bool = False
    escalation_reason: Optional[str] = None

    # Field population tracking
    fields_populated: Dict[str, bool] = Field(default_factory=dict)


class EvalTrace(BaseModel):
    """
    Complete evaluation trace for a single scenario.
    This is the primary output artifact, written as JSONL.
    """

    # Identifiers
    scenario_id: str
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Input reference
    scenario: ScenarioInput

    # Execution trace
    tool_steps: List[ToolStepRecord] = Field(default_factory=list)
    interpreter_output: Optional[InterpreterOutput] = None

    # Timing
    total_latency_ms: float = 0.0

    # Ground truth comparison
    guid_match: bool = False
    name_match: bool = False
    storey_match: bool = False

    # Pool sizes for search-space reduction
    initial_pool_size: Optional[int] = None
    final_pool_size: Optional[int] = None

    # Error tracking
    error: Optional[str] = None
    success: bool = True

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSONL line"""
        return json.dumps(self.model_dump(mode="json"), default=str)


class MetricsSummary(BaseModel):
    """
    Aggregated metrics across multiple traces.
    Used for the final CSV summary output.
    """

    total_scenarios: int = 0
    successful_runs: int = 0

    # Accuracy metrics
    top1_hits: int = 0
    top1_accuracy: float = 0.0
    topk_hits: int = 0  # k=3 by default
    topk_accuracy: float = 0.0

    # Search-space reduction
    avg_search_space_reduction: float = 0.0

    # Field population
    avg_field_population_rate: float = 0.0

    # Tool usage
    total_tool_calls: int = 0
    avg_tool_calls_per_scenario: float = 0.0
    tool_call_distribution: Dict[str, int] = Field(default_factory=dict)

    # Escalation tracking
    escalation_count: int = 0
    escalation_rate: float = 0.0

    # By RQ category
    by_rq_category: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Timing
    avg_latency_ms: float = 0.0
