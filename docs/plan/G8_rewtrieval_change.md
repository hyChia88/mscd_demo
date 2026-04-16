### 1. what's in the label vs what's wired into Cypher:

Label field                    → In Cypher?   Value if wired
─────────────────────────────────────────────────────────────
storey_name                    ✅ wired        (already working)
ifc_class                      ✅ wired        (already working)
spatial_relations[].predicate  ✅ wired        (already working)
spatial_relations[].object_type ✅ wired       (already working)
spatial_relations[].object_material ✅ wired  (already working)
spatial_relations[].direction  ✅ wired        wall_position_index compare
position_context (FILLS/NEXT_TO) ✅ partial   position_index slot
─────────────────────────────────────────────────────────────
target_width_mm / height_mm    ✅ wired   HIGH — breaks 46→10 window groups
position_context (CONNECTS_TO) ❌ NOT wired   MEDIUM — degree in COUNT(edges)
position_context (ADJACENT_TO) ❌ NOT wired   MEDIUM — needs edge distance_mm
spatial_relations[].object_subtype ❌ NOT wired MEDIUM — Revit type name filter
spatial_relations[].confidence ❌ NOT wired   LOW — only 0.75/1.0 in practice
spatial_relations[].host_name  ❌ NOT wired   LOW — wall type name, limited use

#### Sample case: AP_SK_102
GT:
{
  "storey_name": "2",
  "ifc_class": "IfcWindow",
  "space_name": null,
  "target_name_keyword": "small bedroom window",
  "spatial_relations": [
    {
      "predicate": "FILLS",
      "object_type": "IfcWall",
      "object_material": "Plaster",
      "host_name": "Basic Wall:MockUp Exterior",
      "confidence": 1.0
    },
    {
      "predicate": "NEXT_TO",
      "object_type": "IfcWindow",
      "direction": "left",
      "object_subtype": "small bedroom window",   // ← L3
      "confidence": 1.0
    },
    {
      "predicate": "NEXT_TO",
      "object_type": "IfcWindow",
      "direction": "right",
      "object_subtype": "bathroom window",         // ← L3
      "confidence": 1.0
    }
  ],
  "position_context": "3rd of 17 openings on the same wall"  // ← L4 slot
}

G7:
"spatial_relations": [
  { "predicate": "FILLS",   "object_type": "IfcWall",   "object_material": "Plaster" },
  { "predicate": "NEXT_TO", "object_type": "IfcWindow",
    "direction": "left",  "object_subtype": "floor-to-ceiling window" }  // ← wrong subtype
],
// position_context: absent


G8:
"spatial_relations": [
  { "predicate": "FILLS",   "object_type": "IfcWall",   "object_material": "Plaster" },
  { "predicate": "NEXT_TO", "object_type": "IfcWindow",
    "direction": "left",  "object_subtype": "small bedroom window" }  // ← correct
],
"target_name_keyword": "wide bedroom window",  // ← wrong keyword
// position_context: absent
// missing: right-neighbor NEXT_TO

### Impact:
Level	Fields added	Pool size
L0	storey + type	107 avg
L1	+ predicate + object_type	73 avg
L2	+ direction	23 avg
L3	+ object_subtype	6 avg → unique if model gets it right
L4	+ position_context slot (3rd/17)	0.7 avg → near-certain

### Notes: Runs needed after enriched graph + G8 training
1. Track A — AP heldout downstream retrieval (enriched graph)
Model	Precomputed file	Phase5 traces	Status
G0 Canonical	g0_canonical__ap_eval.jsonl	❌ not run	skip (no new fields)
G1 FullAug	g1_fullaug__ap_eval.jsonl	❌ not run	skip
G2 FullAug LowLR	g2_fullaug_lowlr__ap_eval.jsonl	❌ not run	skip
G3 FullAug r32	g3_fullaug_r32__ap_eval.jsonl	❌ not run	skip
G4 Ultimate	g4_ultimate__ap_eval.jsonl	❌ not run	skip
G6 Baseline	g6_baseline__ap_eval.jsonl	❌ not run	skip
G7 Position Context	g7_position_context__ap_eval.jsonl	✅ done	
G8 PosCtx+Dim	g8_posctx_dim__ap_eval.jsonl	✅ done	
Gemini AP	gemini_ap__ap_eval.jsonl	✅ done	
G0-G6 don't output object_subtype, distance_mm, or width_mm so the enriched graph doesn't change their retrieval — safe to skip re-running them.

2. Track A — Graph-RAG rerank
Model	Input traces	Status
G7 phase5 + Graph-RAG	ap_e2e_phase5_g8/g7_position_context/traces_...	❌ needed
G8 phase5 + Graph-RAG	ap_e2e_phase5_g8/g8_posctx_dim/traces_...	❌ needed
The existing Graph-RAG results (20260405_top15_g7_v1) used old phase3 traces — those are stale.

3. Track B — comparison models (unified test set)
These use the unified test set (not AP heldout) and don't benefit from graph enrichment — no re-run needed.

4. Oracle
Experiment	Status
L0-L7 waterfall (live-system aligned)	✅ done (oracle_live_aligned/)
Summary — runs still needed
Graph-RAG on G7 phase5 (for fair G7 vs G8 comparison post-rerank)
Graph-RAG on G8 phase5 (main result)
Score/compare Track A table (score_unified_track.py or score_ap_track.py) with G7, G8, Gemini phase5 + their +Graph-RAG variants