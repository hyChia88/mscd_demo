# MSCD: AI Interpreter for Construction Site Data

MSCD tries to link messy site evidence to the correct IFC/BIM element.
It takes floorplans, site photos, chat messages, and task metadata, turns them into structured constraints, and then searches the IFC model with a rule-based retrieval pipeline.

This README follows the latest summary in [RESULT_OVERVIEW.md](RESULT_OVERVIEW.md), last updated on **March 26, 2026**.

## Latest Status

Best current setup from the latest summary:

- Model: `LoRA5-r32`
- Retrieval strategy: `p0_union_p1`
- Best reported condition: `FP` (floorplan only)
- Test set: unified eval, `116` cases, `3` IFC models (`AP`, `BH`, `DXA`)

Short result snapshot:

| Metric | Value |
| --- | --- |
| GT-in-Pool | `53.4%` |
| Top-10 | `20.7%` |
| Top-1 | `4.3%` |
| Avg Pool Size | `73` |

Main takeaway:

- The symbolic retrieval layer is working as intended.
- The main bottleneck is VLM extraction, especially `ifc_class`.
- Spatial relations help shrink the candidate pool, but they do not yet beat `storey + type` for finding the ground truth.
- Asking the user for the element type is still one of the simplest high-impact improvements.

For the full discussion and all tables, see [RESULT_OVERVIEW.md](RESULT_OVERVIEW.md).

## Current Implementation

Current pipeline:

```text
floorplan / site photo / chat / 4D metadata
        -> LoRA constraint extraction
        -> structured JSON constraints
        -> query planner (P0-P8)
        -> Neo4j retrieval
        -> candidate list, traces, and demo views
```

### 1. Constraint Extraction

- Main runtime extractor: [`src/v2/constraints_extractor_lora.py`](src/v2/constraints_extractor_lora.py)
- Latest training script: [`training/train_lora5.py`](training/train_lora5.py)
- Base model family: `Qwen2.5-VL-7B + LoRA`
- Main output fields: `storey_name`, `ifc_class`, `space_name`, `target_name_keyword`, `spatial_relations`

### 2. Query Planning

- Main planner: [`src/v2/constraints_to_query.py`](src/v2/constraints_to_query.py)
- The planner builds fixed priority rules from `P0` to `P8`
- The latest best retrieval setting is handled in [`src/v2/retrieval_backend.py`](src/v2/retrieval_backend.py) with `p0_union_p1`
- In simple terms, this means:
  - use spatial retrieval first
  - also keep the `storey + type` pool as a safety net

### 3. Retrieval and Pipeline

- Main pipeline: [`src/v2/pipeline.py`](src/v2/pipeline.py)
- Local evaluation runner: [`script/run.py`](script/run.py)
- Retrieval backend: [`src/v2/retrieval_backend.py`](src/v2/retrieval_backend.py)
- Neo4j is the main retrieval backend for the current thesis result

### 4. Evaluation and Demo

- Unified evaluation extraction: [`evaluation/inference/eval_unified.py`](evaluation/inference/eval_unified.py)
- Unified batch script: [`evaluation/inference/eval_unified.sh`](evaluation/inference/eval_unified.sh)
- Plot/analysis script: [`evaluation/analysis/experiment_plots.py`](evaluation/analysis/experiment_plots.py)
- Streamlit demo: [`demo/app.py`](demo/app.py)

## Repo Map

| Path | What it is for now |
| --- | --- |
| [`src/v2/`](src/v2/) | Main neuro-symbolic pipeline used by the current system |
| [`training/`](training/) | LoRA training and model-side scripts |
| [`evaluation/`](evaluation/) | Unified evaluation, baselines, and analysis |
| [`demo/`](demo/) | Streamlit app for saved traces and live pipeline inspection |
| [`output/unified/`](output/unified/) | Saved outputs from unified evaluation runs |
| [`plots/experiments/`](plots/experiments/) | Key experiment plots used in summaries |
| [`RESULT_OVERVIEW.md`](RESULT_OVERVIEW.md) | Latest short result summary |
| [`RESULT_DETAILS.md`](RESULT_DETAILS.md) | Longer notes and supporting detail |
| [`legacy/`](legacy/) | Older archived code paths |

## Important Naming Note

Some active folders still use older names like `src/v2/`.
Some demo code still mentions older LoRA naming.
That is expected in this repo.

The latest thesis result to follow is:

- `LoRA5-r32`
- unified evaluation
- `p0_union_p1`
- result summary in [`RESULT_OVERVIEW.md`](RESULT_OVERVIEW.md)

## Quick Start

### 1. Install

```bash
conda activate mscd_demo
pip install -r requirements.txt
```

### 2. Add API Keys

Create `mscd_demo/.env` and add what you need, for example:

```bash
GOOGLE_API_KEY=your_key_here
```

If you want to run Modal jobs, also make sure Modal is already set up on your machine.

### 3. Start Neo4j

```bash
./script/neo4j_init.sh
```

### 4. Run the Demo

```bash
streamlit run demo/app.py
```

### 5. Run the Latest Unified Evaluation

```bash
bash evaluation/inference/eval_unified.sh
```

This batch script assumes the LoRA adapters are already uploaded to the Modal volume.
See the comments at the top of [`evaluation/inference/eval_unified.sh`](evaluation/inference/eval_unified.sh).

### 6. Replay the Best Local Retrieval Setting

```bash
python script/run.py \
  --profile v2_lora \
  --cases evaluation/cases/cases_unified_test.jsonl \
  --precomputed output/unified/eval_constraints_lora5r32_FP.jsonl \
  --p0-strategy p0_union_p1
```

### 7. Train the Latest LoRA5 Model

```bash
modal run training/train_lora5.py --epochs 5 --lr 2e-4 --lora-r 32
```

## Useful Files

- Latest summary: [`RESULT_OVERVIEW.md`](RESULT_OVERVIEW.md)
- Latest unified cases: [`evaluation/cases/cases_unified_test.jsonl`](evaluation/cases/cases_unified_test.jsonl)
- Latest unified outputs: [`output/unified/`](output/unified/)
- Main evaluation runner: [`script/run.py`](script/run.py)
- Main training script: [`training/train_lora5.py`](training/train_lora5.py)
