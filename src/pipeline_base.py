"""
Pipeline Abstraction Layer

Provides a common interface for v1 (agent-driven) and v2 (constraints-driven)
pipelines so the unified runner (script/run.py) can invoke either.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.eval.contracts import EvalTrace
from src.v2.types import V2Trace


class PipelineBase(ABC):
    """Abstract base for v1 / v2 evaluation pipelines."""

    @abstractmethod
    async def run_case(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any],
        run_id: str,
    ) -> Tuple[EvalTrace, Optional[V2Trace]]:
        """
        Run pipeline on a single case.

        Returns:
            (EvalTrace, V2Trace | None)
            V2Trace is None when running a v1 pipeline.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# V1 PIPELINE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class V1Pipeline(PipelineBase):
    """Thin wrapper around the existing v1 agent-driven evaluation."""

    def __init__(
        self,
        engine: Any,
        llm: Any,
        visual_aligner: Optional[Any],
        profile: Dict[str, Any],
        config: Dict[str, Any],
        agent_executor: Any,
        tool_by_name: Optional[Dict[str, Any]] = None,
        rq2_schema: Optional[Dict[str, Any]] = None,
        rq2_schema_id: Optional[str] = None,
    ):
        self.engine = engine
        self.llm = llm
        self.visual_aligner = visual_aligner
        self.profile = profile
        self.config = config
        self.agent_executor = agent_executor
        self.tool_by_name = tool_by_name
        self.rq2_schema = rq2_schema
        self.rq2_schema_id = rq2_schema_id

    async def run_case(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any],
        run_id: str,
    ) -> Tuple[EvalTrace, Optional[V2Trace]]:
        from src.v2.condition_mask import ConditionMask
        from src.v2.pipeline import _build_scenario_input

        # Apply condition mask (even v1 supports modality control)
        masked = ConditionMask.apply(case, condition_overrides)
        image_dir = self.config.get("ground_truth", {}).get("image_dir", "")
        scenario = _build_scenario_input(masked, image_dir)

        # Delegate to v1 runner
        from src.eval.runner import run_one_scenario

        trace = await run_one_scenario(
            scenario=scenario,
            agent_executor=self.agent_executor,
            engine=self.engine,
            run_id=run_id,
            rq2_enabled=self.profile.get("rq2_schema", False),
            rq2_schema=self.rq2_schema,
            rq2_schema_id=self.rq2_schema_id,
            tool_by_name=self.tool_by_name,
        )

        return trace, None          # no V2Trace for v1


# ─────────────────────────────────────────────────────────────────────────────
# V2 PIPELINE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class V2Pipeline(PipelineBase):
    """Constraints-driven pipeline (wraps run_v2_case)."""

    def __init__(
        self,
        engine: Any,
        llm: Any,
        visual_aligner: Optional[Any],
        profile: Dict[str, Any],
        config: Dict[str, Any],
        adapter_path: Optional[str] = None,
        tool_by_name: Optional[Dict[str, Any]] = None,
        rq2_schema: Optional[Dict[str, Any]] = None,
        rq2_schema_id: Optional[str] = None,
    ):
        self.engine = engine
        self.llm = llm
        self.visual_aligner = visual_aligner
        self.profile = profile
        self.config = config
        self.adapter_path = adapter_path
        self.tool_by_name = tool_by_name
        self.rq2_schema = rq2_schema
        self.rq2_schema_id = rq2_schema_id

        # Build retrieval backend from profile
        from src.v2.retrieval_backend import RetrievalBackend

        self.retrieval_backend = RetrievalBackend(
            engine=engine,
            retrieval_mode=profile.get("retrieval", "memory"),
            visual_aligner=visual_aligner,
            use_clip=profile.get("use_clip", False),
        )

        # Build centralized image parser (VLM-based)
        from src.visual.image_parser import ImageParserReader

        self.image_parser = ImageParserReader(llm)

    async def run_case(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any],
        run_id: str,
    ) -> Tuple[EvalTrace, Optional[V2Trace]]:
        from src.v2.pipeline import run_v2_case

        image_dir = self.config.get("ground_truth", {}).get("image_dir", "")

        trace, v2_trace = await run_v2_case(
            case=case,
            condition_overrides=condition_overrides,
            constraints_model=self.profile.get("constraints_model", "prompt"),
            retrieval_backend=self.retrieval_backend,
            llm=self.llm,
            run_id=run_id,
            image_dir=image_dir,
            engine=self.engine,
            adapter_path=self.adapter_path,
            rq2_enabled=self.profile.get("rq2_schema", False),
            rq2_schema=self.rq2_schema,
            rq2_schema_id=self.rq2_schema_id,
            tool_by_name=self.tool_by_name,
            image_parser=self.image_parser,
        )

        return trace, v2_trace
