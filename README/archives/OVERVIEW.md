## README-OVERVIEW

### Overview

This repository hosts the prototype for my Master’s thesis:

> **An Agentic AI “Interpreter Layer” for AEC:
> Cross-Modal Alignment, Schema Mapping, and Compliance.**

The goal is to build a **compound AI system** that sits between:

* **Unstructured site evidence** – photos, issue logs, notes, chats
* **Structured project representations** – IFC/BIM models, IFC-SG / CORENET-X–style schemas

and acts as an **interpreter layer** that can:

1. **T1 – Cross-modal linking**
   Map site evidence to candidate IFC elements (GUIDs) using a neural perception adapter + IFC topological graph.

2. **T2 – Schema-aware mapping**
   Translate issues and context into structured regulatory fields (IFC-SG / CORENET-X style), with validation against explicit schemas.

3. **T3 – Compliance & handoff**
   Orchestrate tools via an agent (LangGraph + LLM) and export submission-ready artefacts such as JSON records and BCF issues.

换句话说，这个项目不是单一“大模型”，而是一个由 **Neural（感知）+ Symbolic（结构与规则）+ Agentic（编排策略）** 组成的解释层。

### Repository structure (planned)

```bash
mscd_interpreter/
├── data/
│   ├── ifc/              # IFC model files (PPVC-style)
│   ├── images/           # Site photos and synthetic renders
│   ├── texts/            # Issue reports and chat logs
│   ├── schemas/          # IFC-SG / CORENET-X-style schemas (JSON/CSV)
│   └── eval/             # Ground-truth labels for T1/T2/T3 experiments
├── src/
│   ├── t1_adapter/       # T1: Neural Perception Adapter (CLIP + PEFT)
│   ├── t2_reasoning/     # T2: IFC graph + schema engine
│   ├── t3_agent/         # T3: Agent orchestration + BCF generator
│   ├── ifc_engine.py     # Shared IFC parsing / graph utilities
│   ├── schema_engine.py  # Shared schema & validator utilities
│   └── main.py           # End-to-end demo entry point
├── experiments/
│   ├── t1_alignment/     # Notebooks/scripts for T1 evaluation
│   ├── t2_mapping/       # Notebooks/scripts for T2 evaluation
│   └── t3_workflow/      # Scenario scripts for T3 / time-to-correct
├── .env.example          # API key template
└── requirements.txt      # Python dependencies
```

---

## README-T1 – Neural Perception Adapter (Cross-Modal Alignment)

### T1 – Goal

**T1 (Cross-modal alignment)** asks:

> Given a site photo (and optional short text),
> can we retrieve the **correct IFC element(s)** (GUIDs) that the evidence refers to?

This is the main **neural fine-tuning** component of the project.
It provides a **Neural Perception Adapter** that aligns:

* Visual evidence: site photos / synthetic renders
* Textual descriptions: IFC-derived labels (element name, module, storey, etc.)

into a shared embedding space.

### Approach

* **Base model.**
  A pre-trained **CLIP-style vision–language model** (e.g., CLIP / SigLIP) is used as the backbone.

* **Parameter-efficient adaptation.**
  Instead of full fine-tuning, we attach:

  * A **small projection head** (MLP) and/or
  * **LoRA layers** on the visual encoder
    to better align project-specific imagery with IFC/PPVC vocabulary.

* **Synthetic training data.**
  A Blender + IfcOpenShell pipeline generates `(image, text)` pairs:

  * **Images**: rendered PPVC modules and details under varied viewpoints/lighting.
  * **Texts**: structured descriptions from IFC properties, e.g.:
    `Element: Wall-W1; Module: M-04; Storey: Level-03; Material: Concrete; State: Installed.`
    These are optionally mixed with a small curated set of real site photos + manually aligned GUIDs.

### Module layout

```bash
src/t1_adapter/
├── dataset_builder.py   # IFC→Blender synthetic data generation
├── train_adapter.py     # CLIP + PEFT training script
├── model_adapter.py     # Adapter loading/inference utilities
└── retrieval_api.py     # API: (image, text?) → Top-k candidate GUIDs
```

### Usage (example)

```python
from t1_adapter.retrieval_api import retrieve_candidates

result = retrieve_candidates(
    image_path="data/images/site_photo_001.jpg",
    text_hint="water pooling on Level 3 slab in Module M-04",
    top_k=5,
)
print(result)  # list of {guid, score}
```

### Evaluation

T1 is evaluated on a held-out scenario set with ground-truth mappings:

* **Metrics.**

  * Top-1 / Top-3 / Top-5 hit rate (guid retrieval)
  * Precision / recall / F1 for “correct element in candidate set”
* **Baselines.**

  * Text-only IFC graph retrieval (no visual adapter)
  * Random / naive room-level assignment

这部分是论文里「Neural 感知层」的主要实验。

---

## README-T2 – Topological & Schema-Aware Mapping

### T2 – Goal

**T2 (Schema-aware mapping)** asks:

> After we know *which* IFC element(s) an issue refers to,
> can we translate the issue and context into **structured regulatory fields**
> (IFC-SG / CORENET-X style), and check if the record is valid?

T2 is where **symbolic structures** become central:
IFC topology, schemas, and rule validators.

### Components

T2 has two main submodules:

1. **Topological Knowledge Graph**

   * Parses IFC into a **NetworkX graph** using `IfcOpenShell`.
   * Nodes: `IfcSite`, `IfcBuildingStorey`, `IfcAssembly`, `IfcWall`, `IfcSlab`, …
   * Edges: `CONNECTED_TO`, `SUPPORTED_BY`, `ADJACENT_TO`, “belongs to module” etc.
   * Provides graph queries such as:

     * `get_elements_in_space("Kitchen Level 03")`
     * `check_element_zone(guid, zone_id)`

2. **Schema & Validator Module**

   * Encodes IFC-SG / CORENET-X-like schemas as JSON/CSV:

     * field name, type, required/optional, unit, allowed values, description
   * Provides functions:

     * `get_schema(issue_type)` → list of fields & constraints
     * `validate_record(record, schema_id)` → pass/fail + errors

An LLM (e.g., GPT-4o) is then used as a **structured translator**:

* Inputs: issue description, T1/T2 context (element GUIDs, graph info), schema definition, few-shot examples.
* Output: a **JSON record** matching the schema, which is then checked by the validator.

### Module layout

```bash
src/t2_reasoning/
├── topo_graph.py        # Build and query IFC topological graph
├── schema_engine.py     # Load schemas, provide field definitions
├── validator.py         # Check JSON records against schema
└── mapping_agent.py     # LLM wrapper to fill fields with structured output
```

### Usage (example)

```python
from t2_reasoning.mapping_agent import map_issue_to_schema

record, validation = map_issue_to_schema(
    issue_text="Water pooling on floor slab in Module M-04, Level 3.",
    element_guid="3h4fG2J7x9...",        # from T1
    issue_type="SLAB_WATER_LEAK"         # schema ID
)

print(record)      # JSON dict with fields
print(validation)  # { "valid": True/False, "errors": [...] }
```

### Evaluation

* **Field-level accuracy.**
  How many fields are correctly filled vs misfilled vs missing?
* **Record-level validity.**
  Share of records that satisfy all schema constraints.
* **Baselines.**
  Simple rule-based parser / regex templates vs LLM + schema engine.

这里是论文里最「Symbolic」的一层：
重点不在训练新大模型，而在 **把 IFC/IFC-SG 变成可查询/可验证的结构**。

---

## README-T3 – Agentic Orchestration & Compliance Handoff

### T3 – Goal

**T3 (Compliance & handoff)** asks:

> How should these components (T1, T2, graph, schemas) be orchestrated
> to support **real workflows**, and what is the impact on
> time-to-correct and information loss?

这里的重点是 **Agentic 行为**，而不是再训练模型权重。

### Agentic Workflow

The orchestrator is built with **LangGraph** and a strong LLM (e.g., GPT-4o).
It models the interpreter as a **small state machine** with tool calls:

1. **Evidence intake**

   * Receive a photo + optional issue text + metadata.

2. **T1 – Candidate retrieval**

   * Call Neural Perception Adapter → Top-k candidate GUIDs.

3. **Context refinement (graph queries)**

   * Query IFC graph to filter by storey, module, zone, etc.

4. **T2 – Schema-aware mapping**

   * Fetch relevant schema (issue type).
   * Ask LLM (with structured output) to fill fields.
   * Run validator; if invalid, attempt correction or mark as “needs review”.

5. **Compliance handoff (BCF + JSON)**

   * Generate a submission record (JSON/CSV) for downstream systems.
   * Generate a **BCF issue (.bcfzip)** that can be opened in BIM tools.

### Module layout

```bash
src/t3_agent/
├── tools.py             # Tool definitions: T1 adapter, graph, schema, BCF generator
├── workflow_graph.py    # LangGraph workflow / state machine
├── bcf_generator.py     # Create .bcfzip issues
└── api.py               # CLI / FastAPI endpoint for end-to-end runs
```

### Usage (end-to-end demo)

```bash
# Example: run a full pipeline on a folder of site photos
python src/main.py --mode batch --input_folder data/demo_photos/ --output_folder out/
```

Or via API:

```python
from t3_agent.api import process_issue

result = process_issue(
    image_path="data/images/site_photo_001.jpg",
    issue_text="Crack at corner of PPVC module on Level 3, Zone A.",
)
print(result.submission_record)  # JSON
print(result.bcf_path)           # path to .bcfzip file
```

### Evaluation

T3 focuses on **system-level behaviour**, not just model metrics:

* **Time-to-correct (proxy).**
  Steps / interactions needed to reach a correct record vs manual baseline.
* **Error patterns.**
  Where does the agent over-trust low-confidence predictions?
  When does it escalate appropriately?
* **Workflow fit.**
  Can reviewers meaningfully use BCF issues in Revit/Navisworks to inspect AI suggestions?

这一层体现的是你论文里说的 **“agentic interpreter layer”**：
不只是“会看图”和“会填表”，而是“知道何时自动、何时请人”。
