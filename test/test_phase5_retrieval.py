"""Phase 5 sanity checks for RetrievalBackend memory-mode strategy routing."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ifc_engine import IFCEngine
from src.neurosym.retrieval_backend import RetrievalBackend
from src.neurosym.types import QueryPlan


IFC_PATH = REPO_ROOT / "data_curation" / "ifc_models" / "AdvancedProject.ifc"


def _build_backend() -> tuple[IFCEngine, RetrievalBackend]:
    engine = IFCEngine(str(IFC_PATH))
    backend = RetrievalBackend(engine=engine, retrieval_mode="memory")
    return engine, backend


def test_memory_mode_supports_phase5_strategy_branches() -> None:
    engine, backend = _build_backend()

    storey_type = QueryPlan(
        priority=4,
        strategy="storey+type",
        params={"storey": "sixth", "type": "IfcWindow"},
    )
    storey_results = backend._execute_memory(storey_type)
    assert storey_results, "storey+type regression: expected a non-empty IfcWindow pool"

    sample_space = next((key for key, elems in engine.spatial_index.items() if elems), None)
    assert sample_space is not None, "expected at least one populated spatial index key"

    sample_type = next(
        (elem.get("type") for elem in engine.spatial_index[sample_space] if elem.get("type")),
        None,
    )
    assert sample_type is not None, f"space {sample_space!r} did not expose an IFC type"

    space_type = QueryPlan(
        priority=1,
        strategy="space+type",
        params={"space_name": sample_space, "type": sample_type},
    )
    space_results = backend._execute_memory(space_type)
    assert space_results, f"space+type returned 0 candidates for {sample_space!r} / {sample_type!r}"

    sample_name = next(
        (
            elem.get("name", "")
            for elems in engine.spatial_index.values()
            for elem in elems
            if elem.get("name")
        ),
        "",
    )
    keyword = sample_name[:4] or "Wall"
    keyword_plan = QueryPlan(
        priority=2,
        strategy="name_keyword",
        params={"name_keyword": keyword},
    )
    keyword_results = backend._execute_memory(keyword_plan)
    assert keyword_results, f"name_keyword returned 0 candidates for kw={keyword!r}"

    unknown_plan = QueryPlan(priority=99, strategy="unknown_xyz", params={})
    assert backend._execute_memory(unknown_plan) == []
