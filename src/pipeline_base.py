"""
Pipeline Abstraction Layer

Provides a common interface for v1 (agent-driven) and v2 (constraints-driven)
pipelines so the unified runner (script/run.py) can invoke either.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.evaluation_infra.contracts import EvalTrace


class PipelineBase(ABC):
    """Abstract base for v1 / v2 / v3+ evaluation pipelines."""

    @abstractmethod
    async def run_case(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any],
        run_id: str,
    ) -> EvalTrace:
        """
        Run pipeline on a single case.

        Returns a single EvalTrace containing all inputs, outputs, evaluation
        results, and pipeline-specific internals (in trace.internals, keyed
        by pipeline_type: "v1" | "v2" | "v3" …).
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
    ) -> EvalTrace:
        from src.neurosym.condition_mask import ConditionMask
        from src.neurosym.pipeline import _build_scenario_input

        # Apply condition mask (even v1 supports modality control)
        masked = ConditionMask.apply(case, condition_overrides)
        image_dir = self.config.get("ground_truth", {}).get("image_dir", "")
        scenario = _build_scenario_input(masked, image_dir)

        # Delegate to v1 runner
        from src.evaluation_infra.runner import run_one_scenario

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

        return trace


# ─────────────────────────────────────────────────────────────────────────────
# V2 PIPELINE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class V2Pipeline(PipelineBase):
    """Constraints-driven pipeline (wraps run_pipeline_case)."""

    def __init__(
        self,
        engine: Any,
        llm: Any,
        visual_aligner: Optional[Any],
        profile: Dict[str, Any],
        config: Dict[str, Any],
        adapter_path: Optional[str] = None,
        lora_prompt_key: Optional[str] = None,
        tool_by_name: Optional[Dict[str, Any]] = None,
        rq2_schema: Optional[Dict[str, Any]] = None,
        rq2_schema_id: Optional[str] = None,
        precomputed_constraints: Optional[Dict] = None,
    ):
        self.engine = engine
        self.llm = llm
        self.visual_aligner = visual_aligner
        self.profile = profile
        self.config = config
        self.adapter_path = adapter_path
        self.lora_prompt_key = lora_prompt_key
        self.tool_by_name = tool_by_name
        self.rq2_schema = rq2_schema
        self.rq2_schema_id = rq2_schema_id
        self.precomputed_constraints = precomputed_constraints

        # Build retrieval backend from profile
        from src.neurosym.retrieval_backend import RetrievalBackend

        self.retrieval_backend = RetrievalBackend(
            engine=engine,
            retrieval_mode=profile.get("retrieval", "memory"),
            visual_aligner=visual_aligner,
            use_clip=profile.get("use_clip", False),
            p0_strategy=config.get("p0_strategy", "p0_intersect_p1"),
            size_cluster_mode=(config.get("retrieval", {}) or {}).get("size_cluster_mode", "soft"),
            size_band_mode=(config.get("retrieval", {}) or {}).get("size_band_mode", "hard"),
        )

        # Build centralized image parser (VLM-based)
        from src.visual.image_parser import ImageParserReader

        self.image_parser = ImageParserReader(llm)

        # Pre-build LoRA extractor if profile uses lora (load model once)
        # Skip model loading when precomputed constraints are available
        self.lora_extractor = None
        if profile.get("constraints_model") == "lora" and adapter_path and not precomputed_constraints:
            from src.neurosym.constraints_extractor_lora import LoRAConstraintsExtractor

            image_dir = config.get("ground_truth", {}).get("image_dir", "")
            self.lora_extractor = LoRAConstraintsExtractor(
                adapter_path=adapter_path,
                image_dir=image_dir,
                prompt_key=lora_prompt_key,
            )

    async def run_case(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any],
        run_id: str,
    ) -> EvalTrace:
        from src.neurosym.pipeline import run_pipeline_case

        image_dir = self.config.get("ground_truth", {}).get("image_dir", "")

        trace, _ = await run_pipeline_case(
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
            lora_extractor=self.lora_extractor,
            precomputed_constraints=self.precomputed_constraints,
        )

        return trace
