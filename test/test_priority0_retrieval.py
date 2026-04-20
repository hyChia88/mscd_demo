"""Priority-0 sanity checks for query planning and retrieval fallbacks."""

from __future__ import annotations

import asyncio
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
from src.neurosym.types import Constraints, QueryPlan, SpatialTriplet


IFC_PATH = REPO_ROOT / "data_curation" / "ifc_models" / "AdvancedProject.ifc"


def _fills_constraints() -> Constraints:
    return Constraints(
        ifc_class="IfcDoor",
        storey_name="Floor 1",
        spatial_relations=[
            SpatialTriplet(
                subject_type="IfcDoor",
                predicate="FILLS",
                object_type="IfcWall",
            )
        ],
    )


def _continuous_constraints() -> Constraints:
    return Constraints(
        ifc_class="IfcWall",
        storey_name="Floor 6",
        spatial_relations=[
            SpatialTriplet(
                subject_type="IfcWall",
                predicate="CONTINUOUS",
                object_type="IfcSlab",
            )
        ],
    )


@pytest.fixture(scope="module")
def planner() -> QueryPlanner:
    return QueryPlanner()


@pytest.fixture(scope="module")
def memory_backend() -> RetrievalBackend:
    engine = IFCEngine(str(IFC_PATH))
    return RetrievalBackend(engine=engine, retrieval_mode="memory")


@pytest.fixture(scope="module")
def neo4j_backend() -> RetrievalBackend:
    py2neo = pytest.importorskip("py2neo")
    try:
        graph = py2neo.Graph("bolt://localhost:7687", auth=("neo4j", "password"))
        graph.run("RETURN 1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Neo4j unavailable for Priority-0 execution checks: {exc}")
    engine = IFCEngine(str(IFC_PATH), neo4j_conn=graph)
    return RetrievalBackend(engine=engine, retrieval_mode="neo4j")


def test_query_planner_routes_fills_to_spatial_triplet(planner: QueryPlanner) -> None:
    plan = planner.plan(_fills_constraints())[0]
    assert plan.priority == 0
    assert plan.strategy == "spatial_triplet"
    assert plan.params.get("predicate") == "FILLS"


def test_query_planner_routes_continuous_to_continuous_span(planner: QueryPlanner) -> None:
    plan = planner.plan(_continuous_constraints())[0]
    assert plan.priority == 0
    assert plan.strategy == "continuous_span"
    assert plan.params.get("top_storey") == "Floor 6"


def test_memory_mode_degrades_spatial_triplet_gracefully(
    planner: QueryPlanner,
    memory_backend: RetrievalBackend,
) -> None:
    plan = planner.plan(_fills_constraints())[0]
    degraded = memory_backend._execute_memory(plan)
    type_only = memory_backend._execute_memory(
        QueryPlan(priority=6, strategy="type_only", params={"type": "IfcDoor"})
    )
    assert degraded
    assert len(degraded) <= len(type_only)


def test_memory_mode_degrades_continuous_span_gracefully(
    planner: QueryPlanner,
    memory_backend: RetrievalBackend,
) -> None:
    plan = planner.plan(_continuous_constraints())[0]
    degraded = memory_backend._execute_memory(plan)
    assert degraded, "continuous_span should fall back to a non-empty IfcWall pool in memory mode"


def test_neo4j_priority_zero_executes_without_fallback(
    planner: QueryPlanner,
    neo4j_backend: RetrievalBackend,
) -> None:
    fills_plan = planner.plan(_fills_constraints())[0]
    fills_result = asyncio.run(neo4j_backend.execute_plan(fills_plan))
    assert fills_result.candidates
    assert not fills_result.fallback_triggered

    cont_plan = planner.plan(_continuous_constraints())[0]
    cont_result = asyncio.run(neo4j_backend.execute_plan(cont_plan))
    assert cont_result.candidates
    assert not cont_result.fallback_triggered
