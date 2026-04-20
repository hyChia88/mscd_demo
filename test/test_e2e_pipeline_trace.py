"""End-to-end topology trace smoke test for Priority-0 execution."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ifc_engine import IFCEngine
from src.neurosym.constraints_to_query import QueryPlanner
from src.neurosym.retrieval_backend import RetrievalBackend
from src.neurosym.types import Constraints, SpatialTriplet


IFC_PATH = REPO_ROOT / "data_curation" / "ifc_models" / "AdvancedProject.ifc"
SKEL_PATH = (
    REPO_ROOT
    / "data_curation"
    / "datasets"
    / "synth_v0.5"
    / "skeletons"
    / "skeletons_v2_5.jsonl"
)
MAX_CASES = 12
PATTERNS = {"FILLS_RELATION", "ADJACENT_TO_RELATION", "CONTINUOUS_SPAN"}


def _load_test_cases() -> list[dict]:
    with SKEL_PATH.open("r", encoding="utf-8") as f:
        skeletons = [json.loads(line) for line in f if line.strip()]
    return [row for row in skeletons if row.get("pattern") in PATTERNS][:MAX_CASES]


def _build_constraints(skeleton: dict) -> Constraints | None:
    target_props = skeleton.get("target_props", {})
    predicate = target_props.get("predicate") or (
        "FILLS"
        if skeleton.get("pattern") == "FILLS_RELATION"
        else "ADJACENT_TO"
        if skeleton.get("pattern") == "ADJACENT_TO_RELATION"
        else "CONTINUOUS"
    )
    subject_type = target_props.get("subject_type") or target_props.get("Type") or ""
    object_type = (
        target_props.get("ref_type")
        or target_props.get("object_type")
        or skeleton.get("ref_element_type")
        or ""
    )
    storey_name = target_props.get("ref_storey") or target_props.get("Storey") or ""
    if predicate == "CONTINUOUS":
        storey_name = target_props.get("top_constraint") or storey_name

    if not subject_type or not object_type:
        return None

    return Constraints(
        ifc_class=subject_type,
        storey_name=storey_name,
        spatial_relations=[
            SpatialTriplet(
                subject_type=subject_type,
                predicate=predicate,
                object_type=object_type,
            )
        ],
    )


@pytest.fixture(scope="module")
def planner() -> QueryPlanner:
    return QueryPlanner()


@pytest.fixture(scope="module")
def neo4j_backend() -> RetrievalBackend:
    py2neo = pytest.importorskip("py2neo")
    try:
        graph = py2neo.Graph("bolt://localhost:7687", auth=("neo4j", "password"))
        graph.run("RETURN 1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Neo4j unavailable for topology trace smoke test: {exc}")
    engine = IFCEngine(str(IFC_PATH), neo4j_conn=graph)
    return RetrievalBackend(engine=engine, retrieval_mode="neo4j")


def test_priority_zero_topology_traces_do_not_majority_fallback(
    planner: QueryPlanner,
    neo4j_backend: RetrievalBackend,
) -> None:
    test_cases = _load_test_cases()
    fallback_count = 0
    gt_found_count = 0
    checked = 0

    for skeleton in test_cases:
        constraints = _build_constraints(skeleton)
        if constraints is None:
            continue

        plan = planner.plan(constraints)[0]
        assert plan.priority == 0, f"{skeleton['id']} did not route to Priority-0"

        result = asyncio.run(neo4j_backend.execute_plan(plan))
        checked += 1
        if result.fallback_triggered:
            fallback_count += 1

        guids = {cand.get("guid") for cand in result.candidates}
        if skeleton.get("target_guid") in guids:
            gt_found_count += 1

    assert checked > 0, "no topology skeletons produced executable constraints"
    assert fallback_count <= checked // 2, (
        f"Priority-0 fell back in {fallback_count}/{checked} topology cases"
    )
    assert gt_found_count > 0, "ground truth never appeared in the candidate pool"
