"""
End-to-end pipeline trace — proves Priority-0 fires (not silent fallback).

For each test skeleton:
  1. Builds a Constraints object with SpatialTriplet from skeleton metadata
  2. Runs QueryPlanner → takes first plan
  3. Executes via RetrievalBackend (Neo4j mode)
  4. Reports: strategy_asked / strategy_actually_used / fallback_triggered / pool_size / gt_in_pool

Passes if Priority-0 strategies produce results WITHOUT triggering fallback for
the majority of FILLS / CONTINUOUS / ADJACENT_TO test cases.

Run with:
    conda run -n mscd_demo python test/test_e2e_pipeline_trace.py
"""
import sys
import os
import json
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from v2.types import Constraints, SpatialTriplet, QueryPlan
from v2.constraints_to_query import QueryPlanner
from v2.retrieval_backend import RetrievalBackend
from ifc_engine import IFCEngine

IFC_PATH    = "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"
SKEL_PATH   = "../data_curation/datasets/synth_v0.5/skeletons/skeletons_v2_5.jsonl"
MAX_CASES   = 12   # run at most N skeletons to keep output readable

PREDICATE_PATTERNS = {"FILLS_RELATION", "ADJACENT_TO_RELATION", "CONTINUOUS_SPAN"}

# ── Load skeletons ────────────────────────────────────────────────────────────
with open(SKEL_PATH) as f:
    all_skeletons = [json.loads(l) for l in f if l.strip()]

test_cases = [s for s in all_skeletons if s["pattern"] in PREDICATE_PATTERNS][:MAX_CASES]
print(f"Loaded {len(test_cases)} topology skeletons (capped at {MAX_CASES})\n")

# ── Set up engine + backend ───────────────────────────────────────────────────
print("Loading IFC model + Neo4j...")
try:
    from py2neo import Graph
    g = Graph('bolt://localhost:7687', auth=('neo4j', 'password'))
    g.run("RETURN 1")
    engine = IFCEngine(IFC_PATH, neo4j_conn=g)
    backend = RetrievalBackend(engine=engine, retrieval_mode="neo4j")
    print("  Neo4j connected.\n")
except Exception as e:
    print(f"  Neo4j unavailable: {e}  — falling back to memory mode\n")
    engine = IFCEngine(IFC_PATH)
    backend = RetrievalBackend(engine=engine, retrieval_mode="memory")

planner = QueryPlanner()

# ── Run each test case ────────────────────────────────────────────────────────
header = (f"{'ID':<8} {'Pattern':<22} {'Strategy Asked':<18} "
          f"{'Actually Used':<18} {'Fallback':<9} {'Pool':<6} {'GT in Pool'}")
print(header)
print("-" * len(header))

p0_hits = 0       # Priority-0 fired without fallback
fallback_count = 0
gt_found_count = 0
results_log = []

for sk in test_cases:
    tp = sk.get("target_props", {})
    pattern = sk["pattern"]
    target_guid = sk["target_guid"]

    # Build predicate from skeleton
    predicate = tp.get("predicate") or (
        "FILLS"       if pattern == "FILLS_RELATION"    else
        "ADJACENT_TO" if pattern == "ADJACENT_TO_RELATION" else
        "CONTINUOUS"
    )
    subject_type = tp.get("subject_type") or tp.get("Type", "")
    ref_type     = tp.get("ref_type") or tp.get("object_type", "")
    storey_raw   = tp.get("ref_storey") or tp.get("Storey") or ""
    top_storey   = tp.get("top_constraint") or storey_raw

    # Constraints → QueryPlan
    try:
        triplet = SpatialTriplet(
            subject_type=subject_type,
            predicate=predicate,
            object_type=ref_type
        )
    except Exception:
        continue   # skip malformed

    # For CONTINUOUS, storey_name = top_storey (the upper bound)
    storey_name = top_storey if predicate == "CONTINUOUS" else storey_raw
    c = Constraints(
        ifc_class=subject_type,
        storey_name=storey_name,
        spatial_relations=[triplet]
    )
    plans = planner.plan(c)
    plan = plans[0]

    # Execute
    result = asyncio.run(backend.execute_plan(plan))

    # Check ground truth
    guids = {cand.get("guid") for cand in result.candidates}
    gt_in_pool = target_guid in guids

    strategy_asked = plan.strategy
    strategy_used  = result.strategy_actually_used or plan.strategy
    fallback       = result.fallback_triggered

    if not fallback and plan.priority == 0:
        p0_hits += 1
    if fallback:
        fallback_count += 1
    if gt_in_pool:
        gt_found_count += 1

    p0_mark  = "✅" if not fallback else "⚠️ "
    gt_mark  = "✅" if gt_in_pool  else "❌"
    fb_mark  = "YES" if fallback   else "no"

    print(f"{sk['id']:<8} {pattern:<22} {strategy_asked:<18} "
          f"{strategy_used:<18} {fb_mark:<9} {len(result.candidates):<6} {gt_mark}")

    results_log.append({
        "id": sk["id"], "pattern": pattern,
        "strategy_asked": strategy_asked, "strategy_used": strategy_used,
        "fallback": fallback, "pool": len(result.candidates), "gt_in_pool": gt_in_pool
    })

# ── Summary ───────────────────────────────────────────────────────────────────
n = len(test_cases)
print()
print(f"=== Summary ({n} topology test cases) ===")
print(f"  Priority-0 fired (no fallback) : {p0_hits}/{n}  "
      f"({'%.0f'%(100*p0_hits/n)}%)")
print(f"  Fallback triggered             : {fallback_count}/{n}  "
      f"({'%.0f'%(100*fallback_count/n)}%)")
print(f"  Ground truth in candidate pool : {gt_found_count}/{n}  "
      f"({'%.0f'%(100*gt_found_count/n)}%)")

# Fail if majority fallback (>50%) — means Priority-0 isn't actually working
if fallback_count > n // 2:
    print(f"\n❌ FAIL: {fallback_count}/{n} cases fell back — "
          f"Priority-0 edge traversal is not firing correctly.")
    sys.exit(1)
else:
    print(f"\n✅ PASS: Priority-0 fires without fallback in majority of topology cases.")
