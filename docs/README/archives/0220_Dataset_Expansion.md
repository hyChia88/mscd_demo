# LoRA Retraining & Dataset Expansion Plan
**Version**: synth_v0.4 + Qwen2.5-VL-7B LoRA v2
**Date**: 2026-02-20
**Status**: Planning

---

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

**Actual pipeline flow** (verified against source code):

```
1_build_index.py         →  element_index.jsonl       (IFC model is opened HERE)
1b_render_wireframes.py  →  wireframe PNGs
1c_quality_gate.py       →  filtered element_index
2b_hunt_skeletons_v3.py  →  skeletons_v3.jsonl
3c_generate_cases_v3.py  →  cases_v3.jsonl            (reads index; does NOT open IFC)
4_validate.py            →  cases_v3_filtered.jsonl
5b_generate_photoreal.py →  site image PNGs
6_augment_text.py        →  augmented.jsonl            (train/test split LIVES HERE)
7_prepare_lora_data.py   →  lora_train.jsonl
```

Run for BasicHouse:

```bash
cd data_curation/scripts/synth

python 1_build_index.py --ifc ../ifc_models/BasicHouse.ifc --out ../datasets/synth_v0.4_bh/
python 1b_render_wireframes.py --index ../datasets/synth_v0.4_bh/element_index.jsonl
python 2b_hunt_skeletons_v3.py --index ../datasets/synth_v0.4_bh/element_index.jsonl \
    --out ../datasets/synth_v0.4_bh/skeletons/
python 3c_generate_cases_v3.py \
    --skeletons ../datasets/synth_v0.4_bh/skeletons/skeletons_v3.jsonl \
    --index     ../datasets/synth_v0.4_bh/element_index.jsonl \
    --output    ../datasets/synth_v0.4_bh/cases/ \
    --emit-jsonl ../datasets/synth_v0.4_bh/cases_v3.jsonl
python 4_validate.py --cases ../datasets/synth_v0.4_bh/cases_v3.jsonl \
    --out ../datasets/synth_v0.4_bh/cases_v3_filtered.jsonl
python 5b_generate_photoreal.py --cases ../datasets/synth_v0.4_bh/cases_v3_filtered.jsonl
```

Target: **~25 cases** from BasicHouse. Take the top-quality cases (4_validate.py score > threshold).

#### A2. Add 7-Field Ground Truth Labels

**Where the IFC is open matters.** Code audit shows:
- `1_build_index.py` opens the IFC model and writes `element_index.jsonl` — **this is the right place** to add phase 2 fields to the index
- `3c_generate_cases_v3.py` reads the index, never opens the IFC — it already reads `element.get("space_name")` (line 441) but only uses it as a `near_keyword`, not a dedicated field
- `6_augment_text.py` owns the **train/test split** via `stratified_split()` — not script 7

**Strategy: enrich the element index at source (script 1), propagate through script 3**

**Step 1 — Edit `1_build_index.py`** to add 3 new fields to each element record:
```python
# In build_element_record(element, ifc_model):
record["space_name"]           = _get_space_name(ifc_model, element)
record["target_name_keyword"]  = _get_name_keyword(element)
record["neighbor_type"]        = _get_neighbor_type(ifc_model, element)
```

Annotation helpers (added to script 1):
```python
import re, ifcopenshell.util.element as ifc_util

def _get_space_name(ifc_model, element):
    # IfcRelSpaceBoundary → containing IfcSpace chain
    for rel in ifc_model.get_inverse(element):
        if rel.is_a("IfcRelSpaceBoundary"):
            sp = rel.RelatingSpace
            if sp and sp.is_a("IfcSpace"):
                return (sp.LongName or sp.Name or "").strip() or None
    c = ifc_util.get_container(element)
    while c:
        if c.is_a("IfcSpace"):
            return (c.LongName or c.Name or "").strip() or None
        c = ifc_util.get_container(c)
    return None

_ARCH = {"IfcWall","IfcWallStandardCase","IfcSlab","IfcRoof","IfcColumn",
         "IfcBeam","IfcStair","IfcRailing","IfcCovering","IfcPlate","IfcMember"}
_ID_RE = re.compile(r"^[A-Z]{1,5}-\d{1,4}[A-Z]?$")

def _get_name_keyword(element):
    if element.is_a() in _ARCH:
        return None
    for part in (element.Name or "").split(":"):
        if _ID_RE.match(part.strip()):
            return part.strip()
    return None

_NEIGHBOR_PRIORITY = ["IfcColumn","IfcBeam","IfcStair","IfcRamp",
                      "IfcDoor","IfcWindow","IfcRoof","IfcSlab",
                      "IfcWall","IfcWallStandardCase"]

def _get_neighbor_type(ifc_model, element):
    target_type = element.is_a()
    found = set()
    for rel in ifc_model.get_inverse(element):
        if not rel.is_a("IfcRelConnectsElements"):
            continue
        other = rel.RelatedElement if rel.RelatingElement == element else rel.RelatingElement
        if other and other.is_a() != target_type and not other.is_a("IfcOpeningElement"):
            found.add(other.is_a())
    for pt in _NEIGHBOR_PRIORITY:
        if pt in found:
            return pt
    return None
```

**Step 2 — Edit `3c_generate_cases_v3.py`** — two changes:

A. Extend the `Constraints` Pydantic model:
```python
class Constraints(BaseModel):
    storey_name: Optional[str] = None
    ifc_class: Optional[str] = None
    near_keywords: List[str] = Field(default_factory=list)
    relations: List[str] = Field(default_factory=list)
    space_name: Optional[str] = None            # NEW — from element index
    target_name_keyword: Optional[str] = None   # NEW — from element index
    neighbor_type: Optional[str] = None         # NEW — from element index
```

B. Update `generate_constraints_v3()` to populate the new fields from the index and remove the old `space_name → near_keywords` hack:
```python
def generate_constraints_v3(skeleton, element):
    c = Constraints()
    c.storey_name         = element.get("storey_name")
    c.ifc_class           = element.get("ifc_class")
    c.space_name          = element.get("space_name")           # NEW (from index)
    c.target_name_keyword = element.get("target_name_keyword")  # NEW (from index)
    c.neighbor_type       = element.get("neighbor_type")        # NEW (from index)
    # near_keywords: spatial hints from skeleton (NOT space_name — that has its own field now)
    if skeleton.get("requires_relation"):
        hint = skeleton.get("relation_hint", "")
        if "material" in hint and element.get("material"):
            c.relations.append(f"material={element['material']}")
        if "property" in hint and element.get("fire_rating"):
            c.relations.append(f"fire_rating={element['fire_rating']}")
        if "spatial" in hint:
            near_name = skeleton.get("target_props", {}).get("NearElementName", "")
            if near_name:
                c.near_keywords.append(near_name)
    return c
```

**Step 3 — Retroactive annotation for synth_v0.3** (cases already generated without these fields):

Script 7 adds `--ifc` argument for inline backfill before ChatML formatting:
```python
parser.add_argument("--ifc", default=None,
    help="IFC path for retroactive Phase 2 annotation of synth_v0.3 cases.")

# Before case_to_chatml(), in process loop:
if args.ifc and ifc_model:
    c = case.get("labels", {}).get("constraints", {})
    if "space_name" not in c:
        guid = (case.get("ground_truth") or {}).get("target_guid", "")
        el = ifc_model.by_guid(guid)
        if el:
            c["space_name"]          = _get_space_name(ifc_model, el)
            c["target_name_keyword"] = _get_name_keyword(el)
            c["neighbor_type"]       = _get_neighbor_type(ifc_model, el)
            case["labels"]["constraints"] = c
```

**Expected fill rates for AdvancedProject.ifc** (verified against actual IFC structure):
- `space_name`: **0%** — elements are contained in storeys only; no `IfcRelSpaceBoundary` or IfcSpace containment
- `target_name_keyword`: **0%** — all names follow Revit `FamilyName:TypeName:RevitId` format, no equipment IDs
- `neighbor_type`: **~0%** — `IfcRelConnectsElements` is wall-wall only; target elements are slabs/walls/windows

This is **correct and expected** — trains the model to output `null` conservatively for architectural elements. Non-null fill rates will come from BasicHouse.ifc or MEP-rich models.

#### A3. Combine Datasets and Re-split for synth_v0.4

`6_augment_text.py` owns the train/test split (via `stratified_split()`). For combining two IFC models, run script 6 separately on each filtered JSONL, then merge the outputs:

```bash
# AdvancedProject — augment the existing v0.3 filtered cases
python 6_augment_text.py \
    --cases  ../datasets/synth_v0.3/cases_v3_filtered.jsonl \
    --output ../datasets/synth_v0.4/adv/augmented.jsonl \
    --hold-out 17 --seed 42

# BasicHouse — augment the new v0.4_bh filtered cases
python 6_augment_text.py \
    --cases  ../datasets/synth_v0.4_bh/cases_v3_filtered.jsonl \
    --output ../datasets/synth_v0.4/bh/augmented.jsonl \
    --hold-out 5 --seed 42

# Merge augmented train sets
cat ../datasets/synth_v0.4/adv/augmented.jsonl \
    ../datasets/synth_v0.4/bh/augmented.jsonl \
    > ../datasets/synth_v0.4/train/augmented.jsonl

# Merge test hold-outs
cat ../datasets/synth_v0.4/adv/test_holdout.jsonl \
    ../datasets/synth_v0.4/bh/test_holdout.jsonl \
    > ../datasets/synth_v0.4/train/test_holdout.jsonl
```

#### A4. Edit `7_prepare_lora_data.py` In-Place

Three changes:

**Change 1 — System prompt** (must match `eval.py` SYSTEM_PROMPT):
```python
SYSTEM_PROMPT = (
    "You are a construction site assistant that extracts search constraints "
    "from multimodal inputs (photos, floorplans, chat messages, and metadata). "
    "Given the conversation and any attached images, extract structured JSON "
    "constraints to identify the BIM element being discussed.\n\n"
    "Output ONLY valid JSON with these fields:\n"
    "{\n"
    '  "storey_name": "exact floor name or null",\n'
    '  "ifc_class": "IfcWall|IfcWindow|IfcDoor|IfcSlab|... or null",\n'
    '  "near_keywords": ["spatial", "hints"],\n'
    '  "relations": ["spatial_relationships"],\n'
    '  "space_name": "containing room/space name or null",\n'
    '  "target_name_keyword": "unique equipment ID like AHU-03 or null",\n'
    '  "neighbor_type": "IfcClass of adjacent reference element or null"\n'
    "}\n\n"
    "Rules:\n"
    "- storey_name must match exact IFC storey names (e.g., '1 - First Floor')\n"
    "- ifc_class must use Ifc prefix (e.g., 'IfcWindow' not 'window')\n"
    "- space_name: extract if user says 'in the kitchen', 'room 601'; null otherwise\n"
    "- target_name_keyword: extract specific equipment IDs like 'AHU-03'; null for generic names\n"
    "- neighbor_type: extract if user says 'next to the column'; must use Ifc prefix; null otherwise\n"
    "- Be conservative: use null if uncertain\n"
    "- Look at the image carefully for element type and defect clues"
)
```

**Change 2 — Assistant response format**:
```python
def format_assistant_response(case: dict) -> str:
    c = case.get("labels", {}).get("constraints", {})
    return json.dumps({
        "storey_name":          c.get("storey_name"),
        "ifc_class":            c.get("ifc_class"),
        "near_keywords":        c.get("near_keywords", []),
        "relations":            c.get("relations", []),
        "space_name":           c.get("space_name"),           # NEW
        "target_name_keyword":  c.get("target_name_keyword"),  # NEW
        "neighbor_type":        c.get("neighbor_type"),        # NEW
    }, ensure_ascii=False)
```

**Change 3 — Retroactive annotation** (add `--ifc` arg for synth_v0.3 backfill):
```python
parser.add_argument("--ifc", default=None,
    help="IFC path for retroactive Phase 2 field annotation of old cases.")
```
See annotation helpers defined in A2 Step 1. Called inline before `case_to_chatml()`.

Usage for synth_v0.4 final training data:
```bash
python 7_prepare_lora_data.py \
    --train ../datasets/synth_v0.4/train/augmented.jsonl \
    --test  ../datasets/synth_v0.4/train/test_holdout.jsonl \
    --output ../datasets/synth_v0.4/train/lora_train.jsonl \
    --ifc   ../ifc_models/AdvancedProject.ifc \
    --image-root /root/cmu/master_thesis/data_curation
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