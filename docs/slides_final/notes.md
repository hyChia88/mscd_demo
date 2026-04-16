# Final slides for MSCD
### Slide 0: Title

## **Part 1: Motivation: show an ideal system/results**
### Slide 1
- problem framing: Digital & Physical data flow breaking
- Describe actual problem, e.g.: CMs receive report from frontier and confuse where is it, and have to manually track items, once a month

### Slide 2: The Vision: What If We Had This System?
- what if —— we have the system that can do 
show with natural language input/onsite img --can generate --> locating item, show status, reasonning next step (Where, What, Why), and We are focusing on where first
[img](../docs/diagram/key_diagram/fig_1.drawio.png)


### Slide 3 -> academically, what is the gap, Why "Where" Is Hard: Two Gaps
We are focusing on where first -> as this is the first step, and we had encounter 2 gap on "Where" - gap -> 1 semantic gap 2 entropy
Gap	Description	Evidence
Semantic Gap	Natural language ↔ IFC schema vocabulary mismatch	"window next to a door" ≠ IfcWindowStandardCase + IfcRelFillsElement
Entropy Bottleneck	Attribute-only search fails on identical elements	46 identical IfcWindows per floor → attribute Top-1 = 2.2%

### Slide 4 — Why Existing Tools Don't Solve This
- show the current landscape and proof of gap tgt
Keyword/attribute search (BIM360, Solibri): collapses on duplicates — no spatial discriminator
Pure LLM/RAG over IFC text: hallucinates structural relationships, no formal graph reasoning
IFC rule engines: symbolic-only, cannot parse natural language or onsite imagery
Gap: no system combines spatial reasoning from vision with deterministic graph traversal

### Slide 5 — Research Questions
State clearly and hierarchically (from memory: main RQ first, then sub-RQs)
Main RQ: How can AI act as an interpreter middleware to reliably align unstructured site evidence with project data in AEC workflows?
Sub RQ:
RQ1: Multimodal Grounding under Ambiguity.
How can multimodal site evidence be reliably grounded to specific digital project elements (e.g., IFC entities) amidst geometric repetition and noisy data?
- Expect ans: with domain specific fine-tuned model, realiable intermediate layer, that perform spatial reasoning - what plots to cite?

RQ2: Ontology and Schema Alignment.
How can domain-specific ontological constraints (e.g., IFC type hierarchies and spatial relationships) transform probabilistic neural outputs into ontology-compliant retrieval results, thereby mitigating hallucination risks?
- Rationale: data representation that reflects the topological structure of the building from pixel-level only appearance
- Expected ans: Neuro-symbolic architecture, Graph-based query, Cypher as symbolic tools, rounding neural predictions against the actual BIM graph, the system aims to move from plausible generation to verifiable retrieval. 

## **Part 2: approaches/methods**
### Slide 6 — Why Neuro-Symbolic? (Motivation for Architecture)
- Approach/methods (neuro-symbolic) -> controllable output
Don't just say "neuro-symbolic → controllable output." Justify the pairing:

Pure Neural (LLM/RAG)	Pure Symbolic (Cypher/SPARQL)	Neuro-Symbolic
Flexible, understands images	Precise, deterministic	Both
Halluccinates relationships	Cannot parse natural language	Structured extraction → grounded query
Uncontrollable output	Brittle to vocabulary mismatch	Controllable with learned schema
- cite related work like AutoCypher and Lilas

### Slide 7 — System Architecture Overview
- proof by prototype, -> show breif system architecture
- mscd_demo/docs/diagram/explaination diagram/fig_b_middleware_context.png
```
[Onsite Image] [Floorplan] [Natural Language]
         ↓          ↓              ↓
    LoRA6 VLM (Fine-tuned Qwen-VL)
         ↓ SpatialTriplet (storey, ifc_class, spatial_relations[])
    Cypher Query Planner (Priority 0–8)
         ↓ Candidate Pool
    Neo4j Enriched Graph
         ↓ (optional: Graph-RAG Reranker)
    Ranked Elements
```

<!-- - support module (data curation & neo4j database)
    - spatial relationship (show pattern on floorplan)
    - show base vs enrich graph: symbolic method
    - show data curation
    - show VLM: multimodal learning -->
### Slide 8 - zoom in view of supporting module: The Enriched Knowledge Graph
<!-- Slide 8 -->
Base graph: IFC attributes only (type, storey, name, GUID)
Enriched graph: + FILLS, ADJACENT_TO, CONTINUOUS, NEXT_TO edges + width_mm, height_mm, distance_mm properties
Show fig01_topology_overview.png or the floorplan pattern diagram
The uniqueness insight: "FILLS:WallStandardCase | NEXT_TO:Window:left | NEXT_TO:Window:right" — how many elements in 1,257 share this signature? → Very few. This is the spatial fingerprint.
<!-- Slide 9-->
 "show sample of complex SR and how unique it is in graph database" — elevate this to a dedicated slide:

Take one concrete window from the AP model
Show its topology signature: FILLS:IfcWallStandardCase | NEXT_TO:IfcWindow:left | NEXT_TO:IfcWindow:right
Oracle waterfall table: L0=1257 → L1 (attrs)=46 → L3 (fingerprint)=9 → L4 (position slot)=1
This slide makes RQ2's answer visceral: yes, topology breaks the bottleneck — in theory

### Slide 9 - graph reasoning visualization and methods
graph query on enriched ifc graph
graph rerank

### Slide 10 — Data Curation: Teaching the Model
Show the 3-tier labeling strategy and why it matters:

Tier 1 (~30–40%): Topology-annotated cases with spatial_relations populated → "when to extract"
Tier 2 (~60–70%): Attribute-only cases with spatial_relations=[] → "when NOT to hallucinate"
Tier 3: Fresh cases from 3 IFC models (AP, BH, DXA) → generalization
Show: how skeleton mining extracts ground truth spatial relations from IFC geometry (render crops from 3a_render_relation_crops.py)

### Slide 11 — LoRA Fine-tuning & Output Schema
Brief slide:
Base model: Qwen-VL → fine-tuned with LoRA (r=16/r=32)
Input: onsite photo + floorplan crop + structured text prompt
Output schema: {storey_name, ifc_class, spatial_relations: [{predicate, object_type, direction, confidence}]}
Key design choice: confidence gate (threshold ~0.7) controls when symbolic layer fires
- quick show the model training performance g4 vs g8 vs Gemini


## **Part 3: Results:**
- model running overview compare graph
- basline compare: agent vs neuro-symbolic
- group 3 eval, prediction/end-to-end result
- ans RQ1 modality ablation
- ans RQ2 graph query strategy (incl graph rag rerank) and efficiency compare, need to show how "enriched graph" helps (compare with base graph)
    - show sample of complex sr of a target and show "how unique" it is in graph database; show unique represnetaion like left/rihgt, connects to etc. with multimodal input

## **Part 4: Conclusion:**
- ans Main RQ: How to intergrate tech in this situation, how does it helps?
- insights
     - graph
- future RLHP 
— Limitations & Scope

### Plots/diagram location:
`mscd_demo/docs/plots`
`mscd_demo/output/lora6_v2_ap_20260331/topology_analysis`

- Demo run: `streamlit run demo/app.py`