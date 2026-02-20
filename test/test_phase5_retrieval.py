"""
Phase 5 sanity check — RetrievalBackend new strategy branches.

Run with:
    conda run -n mscd_demo python test/test_phase5_retrieval.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from v2.types import QueryPlan
from v2.retrieval_backend import RetrievalBackend
from ifc_engine import IFCEngine

IFC_PATH = "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"

print("Loading IFC model (memory mode)...")
engine = IFCEngine(IFC_PATH)
backend = RetrievalBackend(engine=engine, retrieval_mode="memory")
print(f"  spatial_index: {len(engine.spatial_index)} keys")

sample_spaces = list(engine.spatial_index.keys())[:8]
print(f"  Space keys sample: {sample_spaces}\n")

errors = []

# ── Regression: storey+type (old strategy, must still work) ─────────────────
p = QueryPlan(priority=3, strategy="storey+type",
              params={"storey": "sixth", "type": "IfcWindow"})
res = backend._execute_memory(p)
status = "✅" if len(res) > 0 else "⚠️ "
print(f"{status} [storey+type]   storey='sixth', type=IfcWindow -> {len(res)} candidates")
if len(res) == 0:
    errors.append("storey+type returned 0 candidates (regression)")

# ── NEW: space+type ──────────────────────────────────────────────────────────
space_key = next((k for k in sample_spaces if engine.spatial_index[k]), None)
if space_key:
    types = list({e.get("type") for e in engine.spatial_index[space_key] if e.get("type")})
    tt = types[0] if types else "IfcWindow"
    p2 = QueryPlan(priority=0, strategy="space+type",
                   params={"space_name": space_key, "type": tt})
    res2 = backend._execute_memory(p2)
    status = "✅" if len(res2) > 0 else "⚠️ "
    print(f"{status} [space+type]    space={space_key!r}, type={tt} -> {len(res2)} candidates")
    if len(res2) == 0:
        errors.append(f"space+type returned 0 for space={space_key!r}, type={tt}")
else:
    print("⚠️  [space+type]    SKIP: no non-empty space key found in spatial_index")

# ── NEW: name_keyword ────────────────────────────────────────────────────────
all_names = [e.get("name", "") for elems in engine.spatial_index.values()
             for e in elems if e.get("name")]
kw = all_names[0][:4] if all_names else "Win"
p3 = QueryPlan(priority=1, strategy="name_keyword",
               params={"name_keyword": kw})
res3 = backend._execute_memory(p3)
status = "✅" if len(res3) > 0 else "⚠️ "
print(f"{status} [name_keyword]  kw={kw!r} -> {len(res3)} candidates")
if len(res3) == 0:
    errors.append(f"name_keyword returned 0 for kw={kw!r}")

# ── NEW: neighbor+type → degrades to type_only in memory mode ───────────────
p4 = QueryPlan(priority=2, strategy="neighbor+type",
               params={"neighbor_type": "IfcColumn", "type": "IfcWindow"})
res4 = backend._execute_memory(p4)
p5 = QueryPlan(priority=5, strategy="type_only", params={"type": "IfcWindow"})
res5 = backend._execute_memory(p5)
ok = len(res4) == len(res5)
status = "✅" if ok else "❌"
print(f"{status} [neighbor+type] -> {len(res4)} candidates  "
      f"(type_only ref: {len(res5)})  {'match' if ok else 'MISMATCH'}")
if not ok:
    errors.append(f"neighbor+type count {len(res4)} != type_only count {len(res5)}")

# ── Unknown strategy → must return [] without error ─────────────────────────
p6 = QueryPlan(priority=99, strategy="unknown_xyz", params={})
res6 = backend._execute_memory(p6)
ok6 = len(res6) == 0
status = "✅" if ok6 else "❌"
print(f"{status} [unknown]       -> {len(res6)} candidates  (expect 0)")
if not ok6:
    errors.append("unknown strategy did not return empty list")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"❌ {len(errors)} error(s):")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("✅ Phase 5 sanity check passed — all new strategy branches route correctly.")
