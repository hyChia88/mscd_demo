## Insights till Phase 5: (in thesis chap 07)
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

## Fix:
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