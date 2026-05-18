"""Live wrapper around the offline Graph-RAG reranker.

The offline experiment in `evaluation.experiments.graph_rag_rerank_ap` ships
all the heavy lifting (Cypher fingerprinting, fusion-score scoring, Gemini
prompt building, response parsing). This module exposes a single
`rerank_topk` entry point so the live demo can reuse the exact same logic
without duplicating helpers.

Failure modes return a result with `failed=True` and `ordered_guids` left as
the input order, so callers can fall back gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Reuse the offline experiment helpers verbatim — they already handle
# fingerprint Cypher, fusion scoring, prompt assembly, and answer parsing.
from evaluation.experiments.graph_rag_rerank_ap import (  # noqa: E402
    DEFAULT_MODEL,
    _build_prompt,
    _call_gemini,
    _candidate_match_signals,
    _format_candidate_description,
    _init_gemini,
    _parse_rerank_response,
    _query_candidate_contexts,
)


@dataclass
class RerankResult:
    ordered_guids: List[str]
    fusion_scores: Dict[str, float] = field(default_factory=dict)
    letter_to_guid: Dict[str, str] = field(default_factory=dict)
    descriptions: List[str] = field(default_factory=list)
    raw_output: str = ""
    prompt_text: str = ""
    cot_reasoning: str = ""
    failed: bool = False
    reason: str = ""
    winner_guid: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ordered_guids": list(self.ordered_guids),
            "fusion_scores": dict(self.fusion_scores),
            "letter_to_guid": dict(self.letter_to_guid),
            "descriptions": list(self.descriptions),
            "raw_output": self.raw_output,
            "prompt_text": self.prompt_text,
            "cot_reasoning": self.cot_reasoning,
            "failed": self.failed,
            "reason": self.reason,
            "winner_guid": self.winner_guid,
        }


_GEMINI_CACHE: Dict[str, Any] = {}


def _get_gemini(model_name: str) -> Any:
    """Cache the Gemini client per process — rebuilding it per call costs ~1s."""
    if model_name not in _GEMINI_CACHE:
        _GEMINI_CACHE[model_name] = _init_gemini(model_name)
    return _GEMINI_CACHE[model_name]


def rerank_topk(
    *,
    graph: Any,
    candidate_guids: Sequence[str],
    candidate_fallbacks: Sequence[Dict[str, Any]],
    constraints: Dict[str, Any],
    query_text: str,
    site_image_paths: Sequence[str] = (),
    floorplan_path: Optional[str] = None,
    top_k: int = 10,
    prompt_mode: str = "single_shot",
    model_name: str = DEFAULT_MODEL,
) -> RerankResult:
    """Run Graph-RAG rerank on a live top-K shortlist.

    Args:
        graph: connected `py2neo.Graph` (from IFCEngine.neo4j_conn).
        candidate_guids: ordered top-K GUIDs to rerank.
        candidate_fallbacks: per-GUID dicts (name/type/storey) used when the
            graph fingerprint lookup misses; aligned with `candidate_guids`.
        constraints: structured fields from the VLM/perception layer
            (storey_name, ifc_class, target_name_keyword, size_band,
            size_cluster, position_context, spatial_relations, …).
        query_text: the user's chat message.
        site_image_paths: paths to attached site photos.
        floorplan_path: optional path to the attached floorplan patch.
        top_k: shortlist size (typically 10).
        prompt_mode: "single_shot" or "cot".
        model_name: Gemini model id (defaults to gemini-2.5-flash).

    Returns:
        `RerankResult` with reordered GUIDs and diagnostic fields. On any
        failure, `failed=True` and `ordered_guids` mirrors `candidate_guids`.
    """
    guids = [g for g in candidate_guids if g][:top_k]
    if len(guids) <= 1:
        return RerankResult(
            ordered_guids=list(guids),
            failed=True,
            reason="pool_too_small",
        )

    # 1. Fetch graph fingerprints (host wall, NEXT_TO neighbours, slot, etc.).
    try:
        contexts = _query_candidate_contexts(graph, guids)
    except Exception as exc:
        return RerankResult(
            ordered_guids=list(guids),
            failed=True,
            reason=f"context_query_failed: {exc}",
        )

    # 2. Compute per-candidate fusion score and pre-rank by it (matches
    #    Fix 2 from the offline experiment).
    fallback_by_guid = {}
    for idx, guid in enumerate(guids):
        if idx < len(candidate_fallbacks):
            fallback_by_guid[guid] = candidate_fallbacks[idx] or {}
        else:
            fallback_by_guid[guid] = {}

    scored: List[tuple] = []
    for orig_idx, guid in enumerate(guids):
        ctx = contexts.get(guid, {})
        fb = fallback_by_guid.get(guid, {})
        signals = _candidate_match_signals(ctx, fb, constraints)
        scored.append((-signals["fusion_score"], orig_idx, guid, ctx, fb, signals))
    scored.sort()
    sorted_guids = [t[2] for t in scored]
    fusion_scores = {t[2]: float(t[5]["fusion_score"]) for t in scored}
    letters = [chr(ord("A") + i) for i in range(len(sorted_guids))]
    letter_to_guid = dict(zip(letters, sorted_guids))
    descriptions: List[str] = []
    for letter, (_neg, _idx, _guid, ctx, fb, signals) in zip(letters, scored):
        descriptions.append(_format_candidate_description(letter, ctx, fb, signals))

    # 3. Build the Gemini prompt + call. The case dict mimics the offline
    #    experiment's shape (only the fields used by `_build_prompt` matter).
    case_for_prompt = {
        "query_text": query_text,
        "inputs": {"chat_history": [{"role": "User", "text": query_text}]},
    }
    site_image_paths_p = [Path(p) for p in site_image_paths if p and Path(p).exists()]
    floorplan_path_p = (
        Path(floorplan_path) if floorplan_path and Path(floorplan_path).exists() else None
    )

    cot_reasoning = ""
    try:
        gemini_model = _get_gemini(model_name)
    except SystemExit as exc:  # _init_gemini raises SystemExit on missing key
        return RerankResult(
            ordered_guids=sorted_guids,
            fusion_scores=fusion_scores,
            letter_to_guid=letter_to_guid,
            descriptions=descriptions,
            failed=True,
            reason=f"gemini_unavailable: {exc}",
        )

    try:
        if prompt_mode == "cot":
            cot_prompt = _build_prompt(
                case_for_prompt, descriptions, letters,
                prompt_mode="cot", constraints=constraints,
            )
            cot_reasoning = _call_gemini(
                gemini_model, cot_prompt, site_image_paths_p, floorplan_path_p,
            )
            prompt_text = _build_prompt(
                case_for_prompt, descriptions, letters,
                prompt_mode="cot", constraints=constraints,
                cot_reasoning=cot_reasoning,
            )
        else:
            prompt_text = _build_prompt(
                case_for_prompt, descriptions, letters,
                prompt_mode="single_shot", constraints=constraints,
            )
        raw_output = _call_gemini(
            gemini_model, prompt_text, site_image_paths_p, floorplan_path_p,
        )
    except Exception as exc:
        return RerankResult(
            ordered_guids=sorted_guids,
            fusion_scores=fusion_scores,
            letter_to_guid=letter_to_guid,
            descriptions=descriptions,
            failed=True,
            reason=f"gemini_call_failed: {exc}",
        )

    ordered_letters, winner_letter, reason = _parse_rerank_response(raw_output, letters)
    if not ordered_letters:
        return RerankResult(
            ordered_guids=sorted_guids,
            fusion_scores=fusion_scores,
            letter_to_guid=letter_to_guid,
            descriptions=descriptions,
            raw_output=raw_output,
            prompt_text=prompt_text,
            cot_reasoning=cot_reasoning,
            failed=True,
            reason=reason or "parse_failed",
        )

    ordered_guids = [letter_to_guid[l] for l in ordered_letters]
    winner_guid = letter_to_guid.get(winner_letter, ordered_guids[0])
    return RerankResult(
        ordered_guids=ordered_guids,
        fusion_scores=fusion_scores,
        letter_to_guid=letter_to_guid,
        descriptions=descriptions,
        raw_output=raw_output,
        prompt_text=prompt_text,
        cot_reasoning=cot_reasoning,
        failed=False,
        reason=reason,
        winner_guid=winner_guid,
    )
