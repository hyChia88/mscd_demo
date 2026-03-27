# Result Overview — Neuro-Symbolic IFC Element Retrieval

> **Last updated:** 2026-03-26
> **Test set:** Unified (n=116 cases, 3 IFC models: AP=70, BH=23, DXA=23)
> **Best model:** LoRA5-r32 with p0 union p1 strategy
> **Plots:** `evaluation/experiment_plots/E1–E3`, `evaluation/plots/T1–T5`, `output/unified/plots/U1–U10`

---

## 1  Best Results (Unified Eval, FP condition, p0 union p1)

| System | GT-in-Pool | Top-10 | Top-1 | Avg Pool |
|--------|-----------|--------|-------|----------|
| **LoRA5-r32** | **53.4%** (62/116) | 20.7% | 4.3% | 73 |
| LoRA5-r16 | 52.6% (61/116) | 24.1% | 4.3% | 71 |
| Gemini | 50.9% (59/116) | **25.9%** | 4.3% | 81 |
| LoRA2 | 36.2% (42/116) | 21.6% | 2.6% | 80 |

**Oracle upper bound:** 100% GT-in-Pool, pool compressed from 917 to 31 elements (96.6% reduction). The symbolic layer is sound — all loss comes from VLM extraction errors.

---

## 2  Key Findings

### F1: Symbolic Layer is Sound — Bottleneck is VLM Extraction
The GT-Reverse oracle achieves 100% GT-in-Pool at all spatial stages. The gap between oracle (100%) and best model (53.4%) is entirely from wrong predicates, wrong element types, and wrong storeys.

### F2: ifc_class is the Critical Bottleneck (not storey)
- Per-case flip analysis: **71% of miss-to-hit flips** come from fixing ifc_class
- Wrong ifc_class = guaranteed GT miss (Cypher `WHERE type = wrong` filters GT out)
- All models have similar ifc_class accuracy (62-65%), suggesting a dataset-level ceiling

### F3: Shortcut Learning in LoRA5 Spatial Extraction
LoRA5 uses image presence as a mode switch, not as an information source:
- FP to MC: **81% SR identity** (changing image content barely changes output)
- MA to FP: **20% SR identity** (image presence triggers template switch)
- Only **14 unique SR patterns** out of 116 cases (vs Gemini's 61)
- 48/50 multi-hop extractions = same template: `FILLS->Wall + CONNECTS_TO->Wall`

> See plot: [E1_shortcut_learning_evidence.png](../../evaluation/experiment_plots/E1_shortcut_learning_evidence.png)

### F4: Multi-Hop Extraction is Unreliable
- 0 multi-hop ground-truth cases in eval set (all 40 spatial cases are single-hop)
- Hop-1 predicate accuracy: 30-48% across models
- 100% of LoRA5 multi-hop on attribute-only cases are hallucinated
- Multi-hop Cypher design is architecturally correct (OPTIONAL MATCH, never reduces pool), but extraction quality blocks it

> See plot: [E2_multihop_analysis.png](../../evaluation/experiment_plots/E2_multihop_analysis.png)

### F5: Spatial Adds Pool Compression, Not GT Discovery
For LoRA5, P0 (spatial) is a **strict subset** of P1 (storey+type) for GT recovery:
- P0 alone: 25.9% GT-in-Pool
- P1 alone: 42.2% GT-in-Pool
- P0 union P1: 42.2% GT-in-Pool (P0 never uniquely finds GT that P1 misses)
- P0's value: **pool size reduction** (avg 40 vs 68), helping downstream ranking

### F6: Modality Crossover
| Task | Best Modality | Why |
|------|--------------|-----|
| Attribute extraction | Floorplan (MC) | Annotations, room numbers, labels |
| Spatial extraction | Site photo (MB) | 3D depth, element-in-context |

Adding all modalities (MA) does NOT outperform the best single modality. The model cannot yet fuse cross-modal cues.

### F7: Type Mention is the Biggest User-Side Lever

| Condition | LoRA5-r32 GIP | Gemini GIP |
|-----------|--------------|------------|
| Type mentioned (n=41) | **68.3%** | **53.7%** |
| Type not mentioned (n=75) | 45.3% | 34.7% |
| **Lift** | **+23pp** | **+19pp** |

> See plot: [E3_input_analysis_user_guide.png](../../evaluation/experiment_plots/E3_input_analysis_user_guide.png)

---

## 3  Bottleneck Hierarchy

```
ifc_class accuracy  >>>  SR quality  >  storey accuracy
   (the hard gate)      (pool compress)    (absorbed by p0∪p1)
```

Wrong element type = guaranteed miss. Wrong storey is absorbed by the union strategy.
Wrong spatial relation = wasted pool compression but GT survives via P1.

---

## 4  Discussion Insights

### Gemini vs LoRA5 Paradox
LoRA5 leads on every diagnostic metric (storey 77% vs 66%, ifc_class 76% vs 75%, SR rate 100% vs 93%) but Gemini achieves higher Top-10 / MRR. Why? Gemini's **diverse spatial relations** (61 unique patterns) provide real reranking signal. LoRA5's memorized templates (14 patterns) provide zero discriminative power.

### Two-Layer Hallucination Resistance
```
Layer 1 — Schema: 100% valid JSON output (SOLVED)
Layer 2 — Symbolic: Invalid triplets -> empty Cypher -> fallback (DETECTABLE)
Gap:      Valid-but-wrong triplets -> wrong pool (SILENT failure)
```

### Training Pipeline is Sound
- LoRA2 proves fine-tuning works (+8.4pp Top-1 over Gemini prompt baseline)
- LoRA5 proves spatial output is learnable (0% to 100% SR extraction rate)
- Degradation explained by: capacity conflict, SR ratio too aggressive (75%), label noise

---

## 5  User Input Priority Guide

| Priority | What to Provide | Impact | How |
|----------|----------------|--------|-----|
| Highest | **Element type** | +23pp GIP | Say "the window", "this wall" |
| Medium | **Floor / storey** | +5pp storey acc | Say "on Floor 3" if not in metadata |
| Medium | **Multiple photos** | +3-9pp GIP | Helps class identification |
| Low | **Spatial context** | +/- 0pp (noisy) | Current VLM accuracy too low |
| Lowest | **Material** | Rare signal | Only for disambiguation |

**Design recommendation:** Prompt users for element type when not detected. Auto-extract floor from task metadata. Accept but don't require spatial descriptions.

---

## 6  Next Steps

### Immediate (High ROI)
| Action | Expected Impact |
|--------|----------------|
| **UI prompt for element type** | +23pp GIP (zero cost) |
| **Add rare type training data** (IfcSlab, IfcStair) | ~+4pp GIP |
| **Fix shortcut learning** (more negative examples, balanced SR ratio ~50%) | Unlock real spatial signal |
| **Attribute-matching reranker** | Recover GT from pool (75-82% of GT-in-pool cases lost at ranking) |

### Medium-term
| Action | Expected Impact |
|--------|----------------|
| Higher LoRA rank (r=64) or staged training | Recover ifc_class accuracy while keeping spatial |
| Confidence gating on SR | Only execute hop-2 when confidence >= 0.8 |
| Text-based spatial extraction | Parse "next to", "near the" from user text |

### FP-Only Strategy (viable)
FP-to-MC delta is only +2.6pp. Floorplan-only is deployable with the improvement roadmap:
1. Type prompt (+23pp) -> ~68% GIP
2. Rare type data -> ~72%
3. Fix shortcut learning -> ~75%+

---

## 7  Threats to Validity (Summary)

- **Test set bias:** 97% Tier-3 (hard) cases — results are stress-test performance, not production accuracy
- **IFC type coverage:** Only 6 element types tested (no IfcColumn, IfcBeam, etc.)
- **Synthetic data:** All cases from skeleton mining + LLM augmentation, not real user queries
- **Cross-version confound:** Different LoRA versions tested on different case sets (not directly comparable)
- **No embedding baseline:** No direct comparison with vector-DB/dense retrieval
