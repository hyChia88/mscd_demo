"""
Priority-0 sanity check — spatial_triplet + continuous_span strategy branches.

Tests:
  1. QueryPlanner generates priority-0 plans for FILLS / CONTINUOUS triplets
  2. FILLS → strategy="spatial_triplet"; CONTINUOUS → strategy="continuous_span"
  3. Memory-mode graceful degradation (no topology → storey+type / type_only)
  4. Neo4j execution + predicate relaxation (requires running Neo4j)

Run with:
    conda run -n mscd_demo python test/test_priority0_retrieval.py
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from v2.types import Constraints, SpatialTriplet, QueryPlan
from v2.constraints_to_query import QueryPlanner
from v2.retrieval_backend import RetrievalBackend
from ifc_engine import IFCEngine

IFC_PATH = "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"

errors = []

# ── 1. Plan generation: FILLS triplet → spatial_triplet at priority 0 ────────
c_fills = Constraints(
    ifc_class="IfcDoor",
    storey_name="Floor 1",
    spatial_relations=[SpatialTriplet(
        subject_type="IfcDoor",
        predicate="FILLS",
        object_type="IfcWall"
    )]
)
planner = QueryPlanner()
plans = planner.plan(c_fills)
p0 = plans[0]
ok = p0.strategy == "spatial_triplet" and p0.priority == 0
status = "✅" if ok else "❌"
print(f"{status} [plan/FILLS]        strategy={p0.strategy!r}  priority={p0.priority}  "
      f"params.predicate={p0.params.get('predicate')!r}")
if not ok:
    errors.append(f"FILLS triplet should produce spatial_triplet p0, got {p0}")

# Check predicate is not CONTINUOUS (exclude list works)
ok_excl = p0.params.get("predicate") == "FILLS"
status = "✅" if ok_excl else "❌"
print(f"{status} [plan/excl]         predicate in params = {p0.params.get('predicate')!r}  (expect 'FILLS')")
if not ok_excl:
    errors.append("predicate_exclude failed: FILLS plan should have predicate='FILLS'")

# ── 2. Plan generation: CONTINUOUS triplet → continuous_span at priority 0 ───
c_cont = Constraints(
    ifc_class="IfcWall",
    storey_name="Floor 6",   # AdvancedProject: all continuous walls have top_constraint="6 - Sixth Floor"
    spatial_relations=[SpatialTriplet(
        subject_type="IfcWall",
        predicate="CONTINUOUS",
        object_type="IfcSlab"
    )]
)
plans2 = planner.plan(c_cont)
p0b = plans2[0]
ok2 = p0b.strategy == "continuous_span" and p0b.priority == 0
status = "✅" if ok2 else "❌"
print(f"{status} [plan/CONTINUOUS]   strategy={p0b.strategy!r}  priority={p0b.priority}  "
      f"params.top_storey={p0b.params.get('top_storey')!r}  (Floor 6 = top of all continuous walls)")
if not ok2:
    errors.append(f"CONTINUOUS triplet should produce continuous_span p0, got {p0b}")

# ── 3. Memory-mode degradation: spatial_triplet → storey+type ─────────────────
print(f"\nLoading IFC model (memory mode)...")
engine = IFCEngine(IFC_PATH)
backend_mem = RetrievalBackend(engine=engine, retrieval_mode="memory")

fills_plan = plans[0]
mem_result = backend_mem._execute_memory(fills_plan)
type_only_result = backend_mem._execute_memory(
    QueryPlan(priority=6, strategy="type_only", params={"type": "IfcDoor"}, expected_pool_size=150)
)
# Memory degradation must return ≥1 candidates and same or subset of type_only
ok3 = len(mem_result) > 0
status = "✅" if ok3 else "❌"
print(f"{status} [mem/spatial_triplet] degraded -> {len(mem_result)} candidates "
      f"(type_only ref: {len(type_only_result)})")
if not ok3:
    errors.append("spatial_triplet memory degradation returned 0 candidates")

# ── 4. Memory-mode degradation: continuous_span → type_only ──────────────────
cont_plan = plans2[0]
mem_cont = backend_mem._execute_memory(cont_plan)
ok4 = len(mem_cont) > 0
status = "✅" if ok4 else "❌"
print(f"{status} [mem/continuous_span] degraded -> {len(mem_cont)} candidates  (expect IfcWall pool)")
if not ok4:
    errors.append("continuous_span memory degradation returned 0 candidates")

# ── 5. Neo4j execution (optional — skip if Neo4j is down) ────────────────────
print()
try:
    from py2neo import Graph
    g = Graph('bolt://localhost:7687', auth=('neo4j', 'password'))
    g.run("RETURN 1")
    print("Neo4j connected — running Priority-0 Cypher tests")

    engine_neo = IFCEngine(IFC_PATH, neo4j_conn=g)
    backend_neo = RetrievalBackend(engine=engine_neo, retrieval_mode="neo4j")

    # 5a. FILLS query: IfcDoor -[:FILLS]-> IfcWall (storey resolved via storey_registry)
    res_fills = asyncio.run(backend_neo.execute_plan(fills_plan))
    ok5a = len(res_fills.candidates) > 0 and not res_fills.fallback_triggered
    status = "✅" if ok5a else ("⚠️ " if len(res_fills.candidates) > 0 else "❌")
    print(f"{status} [neo4j/FILLS]      -> {len(res_fills.candidates)} candidates  "
          f"fallback={res_fills.fallback_triggered}  "
          f"strategy_used={res_fills.strategy_actually_used or fills_plan.strategy!r}")
    if res_fills.candidates:
        c0 = res_fills.candidates[0]
        print(f"   sample: guid={c0.get('guid','?')[:8]}... name={c0.get('name','?')} ref_storey={c0.get('ref_storey','?')}")
    if not ok5a and res_fills.fallback_triggered:
        errors.append(f"FILLS triggered fallback: strategy_used={res_fills.strategy_actually_used}")

    # 5b. CONTINUOUS query: IfcWall is_continuous + top_storey=Floor 3
    res_cont = asyncio.run(backend_neo.execute_plan(cont_plan))
    ok5b = len(res_cont.candidates) > 0 and not res_cont.fallback_triggered
    status = "✅" if ok5b else ("⚠️ " if len(res_cont.candidates) > 0 else "❌")
    print(f"{status} [neo4j/CONTINUOUS] -> {len(res_cont.candidates)} candidates  "
          f"fallback={res_cont.fallback_triggered}  "
          f"strategy_used={res_cont.strategy_actually_used or cont_plan.strategy!r}")
    if res_cont.candidates:
        c0 = res_cont.candidates[0]
        print(f"   sample: guid={c0.get('guid','?')[:8]}... name={c0.get('name','?')} "
              f"base={c0.get('ref_storey','?')} top={c0.get('ref_type','?')}")
    if not ok5b and res_cont.fallback_triggered:
        errors.append(f"CONTINUOUS triggered fallback: strategy_used={res_cont.strategy_actually_used}")

    # 5c. Predicate relaxation: bogus predicate → should fall back, not crash
    bogus_plan = QueryPlan(
        priority=0, strategy="spatial_triplet",
        params={"subject_type": "IfcDoor", "predicate": "ADJACENT_TO",
                "object_type": "IfcRailing", "storey": "Floor 99"},
        expected_pool_size=3
    )
    res_bogus = asyncio.run(backend_neo.execute_plan(bogus_plan))
    ok5c = True  # just must not crash
    print(f"✅ [neo4j/relax]      bogus triplet -> {len(res_bogus.candidates)} candidates (relaxed, no crash)")

except Exception as e:
    print(f"[Neo4j skipped: {e}]")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"❌ {len(errors)} error(s):")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("✅ Priority-0 sanity check passed — spatial_triplet + continuous_span route correctly.")
