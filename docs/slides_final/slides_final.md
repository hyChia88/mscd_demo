# Final slides for MSCD
Slide 1: Title

Slide 2: Motivation
- Motivation: bridge physcial on site and digital truth
- Describe actual problem, e.g.: CMs receive report from frontier and confuse where is it, and have to manually track items, once a month
Information Loss Points
Semantic Drift via Fragile Manual Translation
Manual typing & interpretation
No traceability
Context Decay and Reporting Latency

Slide 3: Vision
- What if we add interpreter
show with natural language input/onsite img --can generate --> locating item, show status, reasonning next step (Where, What, Why), and We are focusing on where first

Slide 4: Research Questions (core RQ -> subRQs, actual bottleneck to tackle)
- Core Research Questions: How can AI act as an interpreter middleware to reliably align unstructured site evidence with project data in AEC workflows?
Issue/bottleneck of solving this core RQ:
- RQ1, Multimodal Grounding under Ambiguity: How can multimodal site evidence be reliably grounded to specific digital project elements (e.g., IFC entities) amidst geometric repetition and noisy data? 
- RQ2, Ontology-Driven Retrieval: How can domain-specific ontological constraints (e.g., IFC type hierarchies and spatial relationships) transform probabilistic neural outputs into ontology-compliant retrieval results, thereby mitigating hallucination risks?

Slide 5: Current landscape - related works
Current landscape Why Existing Tools Don't Solve This, Issue at:
- Multimodal Grounding & Geometric Semantics (Addressing RQ1)
- Ontology Alignment & Result Determinism (Addressing RQ2)

Bottom-Left (Low Perception, Low Determinism): Fragmented Manual Workflows & Raw Hardware Silos

Top-Left (High Determinism, Low Perception): Digital Platforms. Platforms governing Interoperability and Compliance (like IFC-SG) provide perfect, ontology-based determinism
. However, they are completely rigid. They cannot ingest "egocentric," unstructured site evidence and instead rely entirely on a massive, fragile bottleneck of manual data entry

Current AEC Tech Tools (some) - mid of perception/determinism, Current AEC Tech tools are stuck in the middle because they are built as generative tools for the design phase, producing visually acceptable models that lack deep topological logic. The AEC Interpreter Layer (Top-Right) is the only system built as a retroactive interpreter for the execution phase—pushing perception beyond pixels into spatial topology, and pushing determinism beyond plausible generation into strict, regulatory-compliant retrieval.

Bottom-Right (High Perception, Low Determinism):  /Generic AI Technology. State-of-the-art Vision-Language Models (VLMs) can easily process generic data formats and multimodal inputs
. However, because they are non-ontology based and lack the rigid, allocentric logic of an IFC graph, they hallucinate spatial relationships and fail completely when faced with AEC compliance schema

Slide 6: Gap
Exact gap at perception and logic
- The Visual vs. Geometric Semantics Gap: Pixel based Visual Perception not aligned with geometry.
Blind to the engineering truth, e.g. floorplan, 3D model
- The Mathematical Deadlock (Attribute Entropy): Visual signatures are non-unique, espeacially under high identitical elements database

AEC Interpreter: Ontology-based interpretation and executable graph logic; VLM supporting multimodal input

Slide 7:
Prototype demo video

Slide 8: System architecture intro
What is Neuro-Symbolic?
Why Neuro-Symbolic?
Alternative: Why not Agentic pipeline?

Slide 9: Neuro-symbolic Pipeline overview
overview with 5 module: input -> interprete middleware (json contract) - output
explain how output works in AEC real workflow

Slide 10: Sample case of Neuro-symbolic Pipeline
left: multimodal input
middle: intermidiate output in fixed json schema
right: Graph-based Retrieval + rerank -> Cypher query sample based on json intermediate output

Slide 11: IFC Parse engine
Raw IFC file → Enriched Knowledge Graph
More connections to able to query/differiate a single element (give example?)

Slide 12: Data mining/Synthetic Dataset Generation Pipeline (give a better title)
Deterministic Skeleton Mining from IFC Graph → Generation → LLM as Judge
- support finetune training

Slide 13: Oracle experiment
Under perfect extraction, prove how spatial relationship helps in query and discover the best graph query strategy and ceiling accuracy.`

Slide 14: Neuro Module - Finetuned VLM (results)
under same input:
- plots: compare zero-shot baseline gemini, g3 model, g8 model -> prove finetune learning works
- plots: compare modality ablation experiment results

Slide 15: Symbolic - graph reasoning (results)
comparing a few query startegy -> find out 2 strategies is working: P1 ONLY + GRAPH RAG RERANK and fulltopology + Graph-RAG rerank
Shows Execution example and reasoning for P1 ONLY + GRAPH RAG RERANK and fulltopology

Slide 16: Deployment Architecture: Continuous Supervised Fine-Tuning
Feedback loop in actual workflow: data mining pipeline + finetune supports this for each long-term projects

Slide 17: Insights
Answers to Research Questions
**RQ1 — Multimodal Grounding under Ambiguity**
> Can multimodal site evidence be reliably grounded to specific IFC entities
> under geometric repetition?

Partial yes.
- Fine-tuned LoRA achieves spatial grounding that zero-shot cannot:
  direction accuracy G8 = 82% vs Gemini = 0%
- Spatial direction grounding requires both modalities:
  text-only (MA): G8 direction acc = 29%
  text + floorplan (MC): G8 direction acc = 82%  (+53pp)
- Spatial topology improves retrieval over text-only in 41/60 cases;
  P0∪P1 places 4/60 at rank-1 vs P1 text-only = 0/60
- Scope: grounding is text-dependent — removing text drops G8 from 82% → 48%;
  the system performs text-grounded topology extraction,
  not visual spatial reasoning
"The system demonstrates that floorplan patches and chat text are complementary evidence channels — the floorplan encodes spatial layout, text encodes semantic identity — and fine-tuning aligns them into a structured spatial constraint."

[support](../plots/phase4_lora6_main/fig09_multimodal_weak_proof.png)

**RQ2 — Ontology-Driven Retrieval**
> Can IFC ontological constraints transform probabilistic outputs into
> reliable, hallucination-resistant retrieval?

Yes, for the symbolic layer.
- Oracle: 100% GT-in-Pool; L3 fingerprint compresses pool 45 → 9 (80%)
- p0∪p1 union strategy is safe under noisy extraction:
  preserves full GT-in-Pool while benefiting from spatial compression
- Oracle ceiling (Top-10 = 40%) sits above best model (G8 Top-10 = 30%);
  the remaining gap is extraction accuracy on richer fields,
  not missing graph or planner logic

Slide 18: Contribuation & Future work

## Contributions
1. **Multimodal constraint extractor**
   Fine-tuned VLM that extracts typed spatial constraints from site images,
   floorplan patches, and chat text into a fixed JSON schema —
   linking unstructured field evidence to executable IFC graph queries

2. **Priority-ordered symbolic planner + training data pipeline**
   Fingerprint-aware Cypher execution over an enriched IFC knowledge graph,
   with a deterministic topology skeleton miner that generates
   discriminative multimodal training cases from raw IFC files

3. **Empirical evidence that neuro-symbolic decomposition is
   iteratively improvable in AEC settings**
   G0 → G8 progression, each generation driven by a diagnosed field-level
   failure; worker accept/reject feedback directly produces training signal

---

## Limitations

1. **Text-dependent spatial grounding**
   Spatial fields (direction, predicate) collapse without text input;
   visual ordinal counting fails at current floorplan resolution
   (G8 position_context exact match = 8.5%)

2. **Benchmark scope and structural ambiguity**
   Evaluated on one building model (AdvancedProject);
   33 of 60 cases are structurally ambiguous even under full fingerprinting —
   strict Top-1 is not achievable for those regardless of model quality

3. **Cross-model generalisation not yet solved**
   Cross-model GT-in-Pool = 36–45% (BasicHouse, Duplex);
   storey normalisation and IFC type variation across building models
   remain the primary open challenge

Slide 18: Next
FAILURE DIAGNOSED          →  CONCRETE FIX
────────────────────────────────────────────────────────
Visual ordinal counting    →  OpenCV position_index 
fails (G8: 8.5% pc exact)     (~50 lines, deterministic)

Dimension regression       →  IFC size-cluster 
fails (G8: 11%/8%)             classification (S/M/L label)

Graph-RAG degrades         →  Decomposed reranker:
topology pools                 auto-generate discriminative
                               questions from graph descriptions

Cross-model GT-in-Pool     →  Multi-model fine-tuning
only 36–45%                    (BH + DXA training cases)


---
## PREVIOUS DRAFT:
Slide 16: Insights - the results (refer chapter 07, NO NEED to perform all but perform the strongest, interesting and value. And no need repeat the previous 14, 15 has perform, directly to what is the contributing insights)
Insights：
1. Neuro-symbolic method design, Query strategy under oracle - ways to distunguish under high entropy attr database:
key:
Layer What it adds Median Pool n
L0 no filter 1257 60
L1 storey + ifc type (IFC attribute baseline) 46 60
L2 topology type-only (predicate + obj type) 45 60
L3 + fingerprint (subtype / material / direction / distance) 9 60
L4 + exact position slot (FILLS / NEXT TO) 1 35
L5 + dimensions ±50 mm (IfcWindow / IfcDoor) 12 38
L6 multi-anchor AND (2+ SRs, star pattern) 45 33
L7 p0∪p1 (live default) 46 60

2. AI model learning behaviour, as lora helps, G7 achieves its strongest predicate recall and direction accuracy when floorplan evidence is present, shows the alignement between fp and onsite image, success unstraucture - structured BIM link

3. Spatial r/s helps, ontoalogy based query works:
The P0 spatial strategy (p0∪p1) places 4 of 60 GT elements at rank 1 before any reranking,
whereas the P1 text-only baseline (storey + IFC class) places 0 of 60 at rank 1 (Figure Fig-
ure 7.16). Per-case analysis of all 60 cases shows that G7 spatial ranking outperforms P1
text-only ranking in 41 of 60 cases (spatial helps)
Direction as fingerprint details helps and learned
![alt text](../plots/phase4_lora6_main/fig09_multimodal_weak_proof.png)

Slide 17: Conclusion
Reflects RQs:
Multimodal align, proof and helps, although text dependent
Query strategy proof: spatial helps retrieval, enriched graph (fingerprint details etc.)
Overall, neural-sys provides traceable and realiably advantage in this task, and able to learnable/finatune system in long run

Contributions:
- from unstructure site edvidence 进行 multimodal alignment
on-site image element link with floorplan patch link with BIM 的方法
- graph representation method, query strategy, traceable/actionable failure traces:
"""
Insight 2: The Neuro-Symbolic Design Produces Actionable
Failure Traces
"""
- 进而推动physcial on site and digital truth 的联系

Limitation:

Slide 18: Insights and Future work
Next steps (before 1st May):
**Insights till Phase 5: (in thesis chap 07)**
1. Storey and class reach 100% — but via text, not vision
- Chat input almost always states floor and element type explicitly; VLM likely extracts these from text cues, not from floorplan or site photo
- This is acceptable: in real AEC workflows, reporters specify these verbally
- The harder problem is disambiguating same-type elements on the same floor
Thesis focus: spatial disambiguation under geometric repetition, measured end-to-end

2. Enriched graph + planner redesign lift the retrieval ceiling
- Oracle experiment: 100% GT-in-Pool, pool reduced by 99%+
- Fingerprint ladder: subtype (L3) is the single largest pool-compression step (45 → 9 median), Spatial topology adds measurable gain over attribute-only filtering (P1-only → full topology: +13pp Top-10)

3. Graph-RAG reranking works — but only on coarse pools
- P1-only + rerank: Top-1 0% → 8.3%, MRR tripled
- Full-topology + rerank: Top-1 drops from 6.7% → 1.7%
Key finding: structured extraction and graph-context matching are complementary, not interchangeable

4. The pipeline is learnable — continuous improvement is viable
- G0 → G3 → G7 → G8: each generation responds to a diagnosed field-level failure
- Neuro-symbolic split makes errors attributable (extraction vs. planner vs. graph), Worker accept/reject feedback directly produces training signal
- Architecture supports weekly LoRA retraining without pipeline changes

**Fix:**
1. Spatial reasoning improve:
- Works: direction (left/right)
- Dimension/counting fail - but prove that it could be helps
so I am planning:
- OpenCV counting: ~50 lines, deterministic, feeds position_index directly to planner
- Dimension classification: mine size clusters from IFC property sets, predict group label not mm value
(usually in IFC building the elements are in fixed groups) -> make this a classification task instead of dimension reasoning. 

2. Decomposed Graph-RAG reranker/Learned Cross-Encoder Reranker
Decomposed Graph-RAG reranker:
- CoT / Listing Specific questions, e.g. "What is to the right?" → "Is it a door or window?" → "How many openings on this wall?" or Auto-generate discriminative questions from graph candidate descriptions
- pass answer to Gemini Rerankers

Current prompt:
```
def _build_prompt(case: dict, descriptions: Sequence[str], letters: Sequence[str]) -> str:
    query_text = case.get("query_text") or _flatten_chat(case.get("inputs") or {})
    example = " ".join(letters[: min(len(letters), 8)])
    return (
        "Match the construction-site evidence to the best BIM candidate.\n\n"
        "Use the site photo, floorplan patch, and query text. "
        "Prefer candidates whose type, storey, host wall, slot position, and left/right neighbors best match the evidence.\n\n"
        f"Query:\n{query_text}\n\n"
        "Candidates:\n"
        + "\n".join(descriptions)
        + "\n\nReturn only the ranked candidate letters from best to worst, separated by spaces.\n"
        f"Example: {example}\n"
        "Do not return JSON. Do not explain."
    )
```
### Plots/diagram location:
`mscd_demo/docs/plots`
`mscd_demo/output/lora6_v2_ap_20260331/topology_analysis`

- Demo run: `streamlit run demo/app.py`