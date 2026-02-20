# LoRA Retraining & Dataset Expansion Plan
**Version**: synth_v0.4 + Qwen2.5-VL-7B LoRA v2
**Date**: 2026-02-20
**Status**: Planning

---

## 1. Motivation & Context

### What We Have Now
The current LoRA adapter (`/mscd-lora/final`) was trained on **synth_v0.3** with a **4-field constraint schema**:
```json
{ "storey_name", "ifc_class", "near_keywords", "relations" }
```

### Why Retrain
Phase 2–5 of the Granularity Upgrade added 3 new retrieval strategies:

| Priority | Strategy | Field | Backend |
|---|---|---|---|
| 0 | `space+type` | `space_name` | Neo4j `query_elements_in_space()` |
| 1 | `name_keyword` | `target_name_keyword` | Neo4j `query_elements_by_name_keyword()` |
| 2 | `neighbor+type` | `neighbor_type` | Neo4j `query_elements_by_neighbor()` |

**Problem**: The current LoRA was never trained to output these 3 fields.
When we expanded the inference prompt to 7 fields, the model hallucinated element paths into `target_name_keyword` (e.g., `"Basic Wall:MockUp Wall:945876"`), triggering wrong strategies and **dropping Top-1 from 10.71% → 3.57%**.

**Solution**: Retrain with 7-field ground truth so the model learns:
- *when* to output each new field (conservative → mostly null)
- *what format* each field expects

### Evidence That New Strategies Work
Using `v2_prompt` (Gemini, already 7-field capable) with Phase 5 retrieval:

| Condition | Before Phase 5 | After Phase 5 | Δ |
|---|---|---|---|
| MA (metadata only) | 3.57% | **5.95%** | +67% |
| MC (images + floorplan) | 3.57% | **8.33%** | +133% |

A retrained LoRA (stronger constraint extractor than Gemini for this domain) with Phase 5 is expected to exceed the old 10.71% baseline.

---

## 2. Dataset Expansion: synth_v0.4

### IFC Model Split Strategy

Each IFC model contributes **20% test / 80% train**, so both train and test sets contain cases from both buildings. This is a standard stratified split that:
- Trains the model on diverse building types
- Tests on held-out cases from both buildings (not just one)
- Avoids the bias of "model has never seen BasicHouse at all" (which tests generalization, not accuracy)

| IFC Model | Total Cases | Train (80%) | Test (20%) |
|---|---|---|---|
| `AdvancedProject.ifc` (synth_v0.3) | 84 | **67** | **17** |
| `BasicHouse.ifc` (new, synth_v0.4_bh) | ~25 | **~20** | **~5** |
| **Total** | **~109** | **~87** | **~22** |

```
synth_v0.4/
  train/
    lora_train.jsonl   # ~87 cases: 67 AdvancedProject + ~20 BasicHouse
    lora_test.jsonl    # ~22 cases: 17 AdvancedProject + ~5 BasicHouse
```

> **How to split within each model**: Use a deterministic random seed (seed=42) on the case list sorted by `case_id`. This ensures reproducibility and the same split every time `7_prepare_lora_data.py` is run.

### New Fields: Ground Truth Generation Strategy

The 3 new fields need ground truth labels. They can be **auto-generated from the IFC model** using ifcopenshell — no manual annotation required for the majority of cases.

#### `space_name` — Containing Room/Space
```python
# For each target element (by GUID):
element = ifc.by_guid(target_guid)
# Find IfcSpace containing the element via IfcRelContainedInSpatialStructure
# or IfcRelSpaceBoundary
space_name = space.LongName or space.Name  # e.g., "Living Room", "Module 606"
# If no IfcSpace → null (element is directly in a storey)
```

#### `target_name_keyword` — Equipment ID / Unique Name
```python
# For each target element:
name = element.Name  # e.g., "AHU-03", "Fire Pump FP-01"
# Rule: only populate if name contains a unique equipment ID pattern
# Pattern: alphanumeric code with hyphens (e.g., "AHU-03", "FP-01")
# Generic names like "Basic Wall:MockUp..." → null
# Conservative: null for architectural elements (IfcWall, IfcWindow, etc.)
```

#### `neighbor_type` — Adjacent Reference Element
```python
# For each target element:
# Use adjacency graph from ifc_engine._build_spatial_graph()
# Find neighbors that are NOT the same type as the target
# Pick the most "distinctive" neighbor type (IfcColumn > IfcWall)
# Only populate if neighbor is clearly mentioned in the chat or scene
# Otherwise → null
```

**Expected fill rates in synth_v0.3** (based on current cases):
- `space_name`: ~40–60% of cases (elements inside IfcSpaces)
- `target_name_keyword`: ~5–10% of cases (mostly null — architectural elements)
- `neighbor_type`: ~20–30% of cases (only when topological reference is clear)

---

## 3. Development Plan

### Phase A — Dataset Preparation (synth_v0.4)

#### A1. Run Data Curation Pipeline on BasicHouse.ifc

The existing pipeline (scripts 1–6) generates cases from an IFC model. Run for BasicHouse:

```bash
cd data_curation/scripts/synth

# 1. Build element index from BasicHouse.ifc
python 1_build_index.py --ifc ../ifc_models/BasicHouse.ifc --out ../datasets/synth_v0.4_bh/

# 2. Hunt skeleton cases (query skeletons from element index)
python 2b_hunt_skeletons_v3.py --index ../datasets/synth_v0.4_bh/

# 3. Generate cases with chat + metadata
python 3c_generate_cases_v3.py --skeletons ../datasets/synth_v0.4_bh/

# 4. Validate case quality
python 4_validate.py --cases ../datasets/synth_v0.4_bh/cases_raw.jsonl

# 5. Generate site photos and floorplan patches
python 5b_generate_photoreal.py --cases ../datasets/synth_v0.4_bh/

# 6. Augment text variations
python 6_augment_text.py --cases ../datasets/synth_v0.4_bh/
```

Target: **~25 cases** from BasicHouse. Take the top-quality cases (4_validate.py score > threshold).

#### A2. Add 7-Field Ground Truth Labels

Create a new script: `8_annotate_phase2_fields.py`

```python
"""
8_annotate_phase2_fields.py

Auto-annotates space_name, target_name_keyword, neighbor_type
for existing and new cases using ifcopenshell + IFC spatial data.

Usage:
    python 8_annotate_phase2_fields.py \
        --ifc ../ifc_models/AdvancedProject.ifc \
        --cases ../datasets/synth_v0.3/augmented.jsonl \
        --out   ../datasets/synth_v0.3/augmented_v2.jsonl
"""
```

Logic:
1. For each case, look up `target_guid` in the IFC model
2. Traverse `IfcRelSpaceBoundary` / `IfcRelContainedInSpatialStructure` to find containing IfcSpace
3. Extract `space_name` from `IfcSpace.LongName`
4. Check element name pattern → `target_name_keyword` (null for most architectural elements)
5. Query adjacency graph → `neighbor_type` (only if a clear topological reference in chat)

Output: enriched JSONL with `labels.constraints` containing all 7 fields.

#### A3. Edit `7_prepare_lora_data.py` In-Place

Three changes directly in the existing script:

**System prompt change** (must match `eval.py` SYSTEM_PROMPT):
```python
SYSTEM_PROMPT = """
You are a construction site assistant...
Output ONLY valid JSON with these fields:
{
  "storey_name": "exact floor name or null",
  "ifc_class": "IfcWall|IfcWindow|... or null",
  "near_keywords": ["spatial", "hints"],
  "relations": ["spatial_relationships"],
  "space_name": "room/space name or null",
  "target_name_keyword": "equipment brand/ID/unique name or null",
  "neighbor_type": "IfcClass of nearby reference element or null"
}
Rules:
- space_name: extract if user says 'in the kitchen', 'room 601'; null otherwise
- target_name_keyword: extract specific IDs like 'AHU-03'; null for generic types
- neighbor_type: extract if user says 'next to the column'; must use Ifc prefix; null otherwise
- Be conservative: use null if uncertain
"""
```

**Assistant response format** (must match new labels):
```python
def format_assistant_response(case: dict) -> str:
    c = case["labels"]["constraints"]
    return json.dumps({
        "storey_name": c.get("storey_name"),
        "ifc_class": c.get("ifc_class"),
        "near_keywords": c.get("near_keywords", []),
        "relations": c.get("relations", []),
        "space_name": c.get("space_name"),          # NEW
        "target_name_keyword": c.get("target_name_keyword"),  # NEW
        "neighbor_type": c.get("neighbor_type"),    # NEW
    }, ensure_ascii=False)
```

**Train/test split** (80/20 per IFC model, deterministic):
```python
import random

def split_cases(cases, test_ratio=0.2, seed=42):
    """Deterministic 80/20 split within a model's case list."""
    cases_sorted = sorted(cases, key=lambda c: c["case_id"])
    rng = random.Random(seed)
    rng.shuffle(cases_sorted)
    n_test = max(1, int(len(cases_sorted) * test_ratio))
    return cases_sorted[n_test:], cases_sorted[:n_test]  # train, test

adv_train, adv_test = split_cases(synth_v03_cases)       # 67 train, 17 test
bh_train,  bh_test  = split_cases(basichouse_cases)      # ~20 train, ~5 test

train_cases = adv_train + bh_train   # ~87 total
test_cases  = adv_test  + bh_test    # ~22 total

write_jsonl("lora_train.jsonl", train_cases)
write_jsonl("lora_test.jsonl",  test_cases)
```

#### A4. Dataset Version Manifest

Create `datasets/synth_v0.4/manifest.json`:
```json
{
  "version": "synth_v0.4",
  "created": "2026-02-20",
  "schema_version": "v2",
  "constraint_fields": [
    "storey_name", "ifc_class", "near_keywords", "relations",
    "space_name", "target_name_keyword", "neighbor_type"
  ],
  "sources": [
    {
      "ifc_model": "AdvancedProject.ifc",
      "cases": 84,
      "split": "train",
      "version": "synth_v0.3"
    },
    {
      "ifc_model": "BasicHouse.ifc",
      "cases": 20,
      "split": "test",
      "version": "synth_v0.4_bh"
    }
  ],
  "total_train": 84,
  "total_test": 20
}
```

---

### Phase B — LoRA Retraining

#### B1. Update training/train.py

Changes needed:
1. **Data path**: point to `synth_v0.4` JSONL files
2. **Run name**: `qwen25vl-7b-r16-synth_v04`
3. **Wandb group**: `synth_v04`
4. **Bake new data into Modal image** (same as current pattern)

```python
# In train.py — update image definition
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    ...
    .add_local_file(str(TRAIN_JSONL), remote_path="/data/train/lora_train.jsonl")
    .add_local_file(str(TEST_JSONL),  remote_path="/data/train/lora_test.jsonl")
    .add_local_dir(str(IMGS_DIR),     remote_path="/data/images/imgs")
    .add_local_dir(str(PLANS_DIR),    remote_path="/data/images/plans")
    .add_local_dir(str(BH_IMGS_DIR),  remote_path="/data/images/basichouse/imgs")   # NEW
    .add_local_dir(str(BH_PLANS_DIR), remote_path="/data/images/basichouse/plans")  # NEW
)
```

Hyperparameters: **keep same** as current (r=16, alpha=32, 3 epochs, lr=2e-4) unless ablations suggest otherwise.

#### B2. Training Run Command

```bash
cd mscd_demo

# Full training run (A100, ~2-3 hours)
modal run training/train.py

# Monitor in Wandb: project=mscd-vlm-lora, run=qwen25vl-7b-r16-synth_v04
```

#### B3. Checkpoint Strategy

Save checkpoints at steps 60, 120, 180, 240 (same cadence as before). After training:
- `final` adapter → primary evaluation
- Best checkpoint by eval loss → secondary evaluation

---

### Phase C — Evaluation

#### C1. Run Modal Inference (new adapter)

```bash
# MA condition
modal run training/eval.py \
  --adapter-dir /mscd-lora/final \
  --condition-override MA

modal volume get mscd-checkpoints \
  /mscd-lora/eval_constraints_final_MA.jsonl \
  logs/evaluations/eval_constraints_v2lora_final_MA.jsonl

# MC condition
modal run training/eval.py \
  --adapter-dir /mscd-lora/final \
  --condition-override MC

modal volume get mscd-checkpoints \
  /mscd-lora/eval_constraints_final_MC.jsonl \
  logs/evaluations/eval_constraints_v2lora_final_MC.jsonl
```

#### C2. Run Local Pipeline

```bash
# MA
python -u script/run.py \
  --profile v2_lora \
  --cases ../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl \
  --precomputed logs/evaluations/eval_constraints_v2lora_final_MA.jsonl \
  --condition-override MA

# MC
python -u script/run.py \
  --profile v2_lora \
  --cases ../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl \
  --precomputed logs/evaluations/eval_constraints_v2lora_final_MC.jsonl \
  --condition-override MC
```

#### C3. Cross-Model Generalization Test (BasicHouse)

After retraining, run on the BasicHouse holdout test set (the 20 unseen cases):
```bash
# BasicHouse test cases (unseen IFC model)
python -u script/run.py \
  --profile v2_lora \
  --cases ../data_curation/datasets/synth_v0.4_bh/cases_filtered.jsonl \
  --precomputed logs/evaluations/eval_constraints_v2lora_final_BH.jsonl \
  --condition-override MC
```

This tests whether the model **generalizes** to a completely new building.

---

## 4. Expected Results

### Success Criteria

| Metric | Current Best (v2_lora 4-field) | Target (v2_lora 7-field) |
|---|---|---|
| MA Top-1 | 10.71% | **> 12%** |
| MA Top-K | 13.10% | **> 15%** |
| MC Top-1 | ~10–11% (est.) | **> 14%** |
| MC Top-K | ~13–14% (est.) | **> 17%** |
| MA SSR | 95.55% | **> 95%** (maintain) |
| Parse Rate | 100% | **> 95%** |

### Expected Improvements vs Baselines

```
v2_prompt (Phase 5, Gemini)     MA: 5.95%  MC: 8.33%
v2_lora v1 (4-field, no Ph.5)   MA: 10.71% MC: ~11%
v2_lora v2 (7-field + Ph.5)     MA: ~13%?  MC: ~15%?  ← TARGET
```

The retrained LoRA should benefit from:
1. **Correct field semantics**: model knows `space_name` ≠ storey name
2. **Conservative null behavior**: trained examples show mostly-null Phase 2 fields
3. **Phase 5 strategies**: when a valid `space_name` or `neighbor_type` IS extracted, it fires the right retrieval path

---

## 5. File Change Summary

| File | Change | Why |
|---|---|---|
| `data_curation/scripts/synth/8_annotate_phase2_fields.py` | **NEW script** | Genuinely new functionality — IFC spatial lookup for Phase 2 fields |
| `data_curation/scripts/synth/7_prepare_lora_data.py` | **EDIT in-place** | Add 3 fields to SYSTEM_PROMPT, `format_assistant_response()`, and 80/20 split logic |
| `data_curation/datasets/synth_v0.4/` | **NEW directory** | New dataset version combining both IFC models |
| `mscd_demo/training/train.py` | **EDIT in-place** | Update data paths + run name (`synth_v04`) |
| `mscd_demo/training/eval.py` | Already updated (7-field schema) ✓ | — |
| `mscd_demo/src/v2/retrieval_backend.py` | Already updated (Phase 5) ✓ | — |
| `mscd_demo/script/run.py` | Already updated (Phase 2 fields loading) ✓ | — |

---

## 6. Timeline

```
Week 1:
  ├── Day 1-2: Run data curation pipeline on BasicHouse.ifc (scripts 1-6)
  ├── Day 3:   Write + run 8_annotate_phase2_fields.py on synth_v0.3
  ├── Day 4:   Write 7_prepare_lora_data_v2.py, generate lora_train/test.jsonl
  └── Day 5:   Smoke test: verify 10 training samples look correct

Week 2:
  ├── Day 1:   Launch LoRA retraining on Modal (modal run training/train.py)
  ├── Day 2:   Monitor training (Wandb), save checkpoints
  ├── Day 3:   Run Modal inference for MA + MC (eval.py)
  ├── Day 4:   Run local pipeline evaluation (script/run.py)
  └── Day 5:   Analyze results, compare baselines, write findings
```

---

## 7. Risk & Mitigation

| Risk | Probability | Mitigation |
|---|---|---|
| BasicHouse has too few elements → sparse test set | Medium | Lower quality threshold in 4_validate.py; accept ~15 cases |
| `space_name` auto-annotation fails (no IfcSpace in IFC) | Medium | Fall back to null; validate with `ifcopenshell` inspection first |
| LoRA overfits on mostly-null Phase 2 labels | Low | Monitor eval loss; if overfitting, reduce epochs to 2 |
| Retrained model regresses on storey/class accuracy | Low | Compare parse rate; if worse, review system prompt changes |
| Modal A100 out of quota | Low | Use checkpoint-180 as fallback for evaluation |

---

## 8. Appendix: Data Curation Scripts Reference

| Script | Purpose |
|---|---|
| `1_build_index.py` | Extract all IFC elements, build spatial index |
| `1b_render_wireframes.py` | Generate wireframe renders |
| `1c_quality_gate.py` | Filter low-quality elements |
| `2b_hunt_skeletons_v3.py` | Generate query skeletons (defect scenarios) |
| `3c_generate_cases_v3.py` | Generate full cases with chat + metadata |
| `4_validate.py` | Quality validation |
| `5b_generate_photoreal.py` | Generate photo-realistic site images |
| `6_augment_text.py` | Text augmentation variants |
| `7_prepare_lora_data.py` | **EDIT** — upgrade to 7-field schema + 80/20 split logic |
| **`8_annotate_phase2_fields.py`** | **NEW** — auto-annotate Phase 2 ground truth from IFC |
