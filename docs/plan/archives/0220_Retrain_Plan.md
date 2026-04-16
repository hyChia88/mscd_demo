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
| `data_curation/scripts/synth/1_build_index.py` | **EDIT in-place** | Add `_get_space_name`, `_get_name_keyword`, `_get_neighbor_type` helpers; write 3 new fields into each element record. IFC is already open here — natural place for IFC-level lookups. |
| `data_curation/scripts/synth/3c_generate_cases_v3.py` | **EDIT in-place** | (1) Extend `Constraints` Pydantic model with 3 new optional fields; (2) update `generate_constraints_v3()` to read them from enriched index; (3) remove old `space_name → near_keywords` hack (line 441) |
| `data_curation/scripts/synth/7_prepare_lora_data.py` | **EDIT in-place** | (1) Add `--ifc` arg + retroactive backfill for synth_v0.3 cases; (2) update SYSTEM_PROMPT to 7 fields; (3) update `format_assistant_response()` to 7 fields |
| `data_curation/datasets/synth_v0.4/` | **NEW directory** | Combined dataset (AdvancedProject + BasicHouse). Script 6 runs separately per model, outputs are merged manually (cat). No script change needed. |
| `mscd_demo/training/train.py` | **EDIT in-place** | Update data paths + run name (`synth_v04`) |
| `mscd_demo/training/eval.py` | Already updated (7-field schema) ✓ | — |
| `mscd_demo/src/v2/retrieval_backend.py` | Already updated (Phase 5) ✓ | — |
| `mscd_demo/script/run.py` | Already updated (Phase 2 fields loading) ✓ | — |

> **No script 8**: phase 2 annotation belongs in script 1 (IFC already open). Train/test split belongs in script 6 (already implemented as `stratified_split()`). Script 7 only needs retroactive backfill via `--ifc` for old synth_v0.3 cases that predate the enriched index.

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

| Script | Purpose | Change for v0.4 |
|---|---|---|
| `1_build_index.py` | Extract all IFC elements, build element_index.jsonl | **EDIT** — add 3 new phase 2 fields to each record |
| `1b_render_wireframes.py` | Generate wireframe renders | None |
| `1c_quality_gate.py` | Filter low-quality elements | None |
| `2b_hunt_skeletons_v3.py` | Generate query skeletons | None |
| `3c_generate_cases_v3.py` | Generate cases with chat + metadata | **EDIT** — extend `Constraints` model; read phase 2 fields from enriched index |
| `4_validate.py` | Quality validation | None |
| `5b_generate_photoreal.py` | Generate photo-realistic site images | None |
| `6_augment_text.py` | Text augmentation + **train/test split** | None (run separately per model, merge manually) |
| `7_prepare_lora_data.py` | Format cases into ChatML for LoRA training | **EDIT** — 7-field SYSTEM_PROMPT, response format, `--ifc` backfill |
