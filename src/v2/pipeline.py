"""
V2 Pipeline — Constraints-driven retrieval with optional thin-agent refinement.

Flow:
  case_v2 → ConditionMask → ConstraintsExtractor → QueryPlanner
          → RetrievalBackend → (opt. thin agent) → EvalTrace
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.eval.contracts import (
    CandidateElement,
    ChatMessage,
    ContextMeta,
    EvalTrace,
    GroundTruth,
    InterpreterOutput,
    ScenarioInput,
)

from .condition_mask import ConditionMask
from .constraints_extractor_lora import LoRAConstraintsExtractor
from .constraints_extractor_prompt_only import PromptConstraintsExtractor
from .constraints_to_query import QueryPlanner
from .metrics_v2 import compute_v2_metrics
from .retrieval_backend import RetrievalBackend
from .types import V2Trace
from common.evaluation import compute_gt_matches


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_scenario_input(case: Dict[str, Any], image_dir: str) -> ScenarioInput:
    """Build a v1-compatible ScenarioInput from a v2 case dict."""
    inputs = case.get("inputs", {})
    gt = case.get("ground_truth", {})
    ctx = inputs.get("project_context", {})

    # Build chat_history
    chat_msgs = [
        ChatMessage(role=m.get("role", ""), text=m.get("text", ""))
        for m in inputs.get("chat_history", [])
    ]

    # Build query_text from last substantive chat message
    query_text = ""
    for m in reversed(inputs.get("chat_history", [])):
        text = m.get("text", "")
        if len(text) > 15:          # skip short acks like "Copy that."
            query_text = text
            break
    if not query_text and chat_msgs:
        query_text = chat_msgs[-1].text

    # Resolve image paths
    images = inputs.get("images", [])
    image_paths = []
    for img in images:
        # images may be relative to data_curation; resolve to image_dir
        from pathlib import Path
        candidate = Path(image_dir) / Path(img).name
        image_paths.append(str(candidate))

    context_meta = ContextMeta(
        timestamp=ctx.get("timestamp", ""),
        sender_role=ctx.get("sender_role", ""),
        project_phase=ctx.get("project_phase", ""),
        task_status=ctx.get("4d_task_status", "N/A"),
    )

    ground_truth = GroundTruth(
        target_guid=gt.get("target_guid", ""),
        target_name=gt.get("target_name", gt.get("target_ifc_class", "")),
        target_storey=gt.get("target_storey", ""),
    )

    return ScenarioInput(
        id=case.get("case_id", str(uuid.uuid4())[:8]),
        image_file=images,
        context_meta=context_meta,
        chat_history=chat_msgs,
        query_text=query_text,
        ground_truth=ground_truth,
        image_paths=image_paths,
    )


# ─────────────────────────────────────────────────────────────────────────────
# main pipeline entry
# ─────────────────────────────────────────────────────────────────────────────

async def run_v2_case(
    case: Dict[str, Any],
    condition_overrides: Dict[str, Any],
    *,
    constraints_model: str,             # "prompt" | "lora"
    retrieval_backend: RetrievalBackend,
    llm: Any,
    run_id: str,
    image_dir: str,
    engine: Any,                        # IFCEngine – for pool-size calc
    adapter_path: Optional[str] = None,
    rq2_enabled: bool = False,
    rq2_schema: Optional[Dict] = None,
    rq2_schema_id: Optional[str] = None,
    tool_by_name: Optional[Dict] = None,
    image_parser: Optional[Any] = None,  # ImageParserReader instance
) -> Tuple[EvalTrace, V2Trace]:
    """
    Run the V2 pipeline on a single case.

    Returns (EvalTrace, V2Trace) — the first is v1-compatible, the second
    carries v2-only diagnostics.
    """
    t0 = time.perf_counter()

    # ── 1. condition mask ──────────────────────────────────────────────────
    masked_case = ConditionMask.apply(case, condition_overrides)
    scenario = _build_scenario_input(masked_case, image_dir)

    # ── 1.5 parse images (VLM) ────────────────────────────────────────────
    image_parse_result = None
    image_parse_ms = 0.0
    if image_parser is not None:
        t_img = time.perf_counter()
        image_parse_result = await image_parser.parse_case_images(
            masked_case, condition_overrides, image_dir
        )
        image_parse_ms = (time.perf_counter() - t_img) * 1000

    # ── 2. extract constraints ─────────────────────────────────────────────
    t_ext = time.perf_counter()
    if constraints_model == "lora":
        extractor = LoRAConstraintsExtractor(adapter_path)
    else:
        extractor = PromptConstraintsExtractor(llm)
    constraints = await extractor.extract(
        masked_case, condition_overrides, image_context=image_parse_result
    )
    constraints_ms = (time.perf_counter() - t_ext) * 1000

    # ── 3. plan queries ────────────────────────────────────────────────────
    t_plan = time.perf_counter()
    planner = QueryPlanner()
    query_plans = planner.plan(constraints)
    planning_ms = (time.perf_counter() - t_plan) * 1000

    # ── 4. execute retrieval (try plans in priority order) ─────────────────
    t_ret = time.perf_counter()

    # Decide whether to pass images (depends on condition)
    use_images = condition_overrides.get("use_images", False)
    img_for_clip = scenario.image_paths if use_images else None
    # force_clip override
    if condition_overrides.get("force_clip", False) and scenario.image_paths:
        img_for_clip = scenario.image_paths

    retrieval_results = []
    final_candidates: List[Dict[str, Any]] = []

    for plan in query_plans:
        result = await retrieval_backend.execute_plan(plan, image_paths=img_for_clip)
        retrieval_results.append(result)
        if result.pool_size > 0:
            final_candidates = result.candidates
            break

    retrieval_ms = (time.perf_counter() - t_ret) * 1000

    # ── 5. build v1-compatible EvalTrace ───────────────────────────────────
    mentioned_guids = [c["guid"] for c in final_candidates[:10]]

    interpreter_output = InterpreterOutput(
        raw_response=f"V2 pipeline: {len(final_candidates)} candidates via {constraints_model}",
        mentioned_guids=mentioned_guids,
        mentioned_names=[c.get("name", "") for c in final_candidates[:10]],
        candidates=[
            CandidateElement(
                guid=c["guid"],
                name=c.get("name"),
                element_type=c.get("type"),
                storey=c.get("storey"),
                confidence=c.get("clip_score"),
                source_step=0,
            )
            for c in final_candidates[:10]
        ],
        is_escalation=(len(final_candidates) == 0),
        escalation_reason="no_candidates" if len(final_candidates) == 0 else None,
    )

    gt_matches = compute_gt_matches(
        target_guid=scenario.ground_truth.target_guid,
        target_name=scenario.ground_truth.target_name or "",
        target_storey=scenario.ground_truth.target_storey or "",
        candidate_guids=mentioned_guids,
        candidate_names=[c.get("name", "") for c in final_candidates[:10]],
        candidate_storeys=[c.get("storey", "") for c in final_candidates[:10]],
    )
    guid_match = gt_matches["guid_match"]
    name_match = gt_matches["name_match"]
    storey_match = gt_matches["storey_match"]

    # pool sizes
    initial_pool_size = 0
    for elems in engine.spatial_index.values():
        initial_pool_size += len(elems)

    total_ms = (time.perf_counter() - t0) * 1000

    trace = EvalTrace(
        scenario_id=scenario.id,
        run_id=run_id,
        scenario=scenario,
        tool_steps=[],                   # v2 has no agent tool calls
        interpreter_output=interpreter_output,
        total_latency_ms=total_ms,
        guid_match=guid_match,
        name_match=name_match,
        storey_match=storey_match,
        initial_pool_size=initial_pool_size,
        final_pool_size=len(final_candidates),
        success=True,
    )

    # ── 6. RQ2 post-processing (optional) ──────────────────────────────────
    if rq2_enabled and rq2_schema and tool_by_name:
        try:
            from src.rq2_schema.pipeline import run_rq2_postprocess
            from src.eval.contracts import RQ2Result

            rq2_raw = await run_rq2_postprocess(
                schema_id=rq2_schema_id or "corenetx_min_v0",
                schema=rq2_schema,
                agent_final={},             # v2 doesn't produce FINAL_JSON yet
                parse_error="",
                rq2_context={
                    "storey_name": constraints.storey_name or "",
                    "evidence": [],
                },
                tool_by_name=tool_by_name,
            )
            trace.rq2_result = RQ2Result.from_pipeline_result(rq2_raw)
        except Exception as exc:
            print(f"⚠️  RQ2 post-processing failed: {exc}")

    # ── 7. build V2Trace ───────────────────────────────────────────────────
    v2_trace = V2Trace(
        constraints=constraints,
        query_plans=query_plans,
        retrieval_results=retrieval_results,
        constraints_parse_success=(constraints.confidence > 0.5),
        constraints_parse_error=(
            None if constraints.confidence > 0.5
            else constraints.source
        ),
        image_parse_result=image_parse_result,
        image_parse_ms=image_parse_ms,
        constraints_extraction_ms=constraints_ms,
        query_planning_ms=planning_ms,
        retrieval_ms=retrieval_ms,
    )

    # compute rerank gain if applicable
    gt_dict = case.get("ground_truth", {})
    labels = case.get("labels")
    v2_metrics = compute_v2_metrics(v2_trace, gt_dict, labels)
    v2_trace.rerank_gain = v2_metrics.get("rerank_gain")

    return trace, v2_trace
