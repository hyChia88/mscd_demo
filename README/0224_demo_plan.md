# MSCD V2.5 — Neuro-Symbolic Prototype Plan

> **Last revised: 2026-02-24**
> Sections 1–4 are the original planning notes, reformatted and annotated.
> Deprecated items are struck through with ❌ and replaced in **§5 (the authoritative plan)**.

---

## 1. Current Progress & Bottleneck Analysis

### 1.1 Engineering Foundation ✅

| Component | Status | Notes |
|---|---|---|
| **Data Pipeline** (synth_v0.4) | ✅ Done | Skeleton-Skin architecture; ifcopenshell parses IFC → deterministic skeletons; Gemini wraps with noisy multimodal evidence |
| **Difficulty tiers** | ✅ Done | H1 (high-density), H2 (relational), H3 (conflict injection) implemented |
| **Model infra** | ✅ Done | Qwen2.5-VL-7B-Instruct LoRA via Unsloth; FastMCP + Pydantic validation; Blender/Bonsai headless renderer |
| **Eval harness** | ✅ Done | synth_v0.4: 933 train / 50 test; Top-1, Top-K, SSR, Field F1 metrics |

### 1.2 Core Bottleneck: The Attribute Entropy Crisis

**Current numbers (V2 baseline, synth_v0.4):**

| Metric | Prompt-only | LoRA_2 | Gap |
|---|---|---|---|
| Top-1 Accuracy | 6.0% | 8.0% | Small |
| SSR | 66.2% | 74.1% | +11.9pp |
| Latency | 4,532 ms | 0.3 ms | 15,000× |

**Root cause — Shannon Entropy at maximum:**

In modern industrialised construction (e.g. PPVC prefabricated buildings), BIM elements have extreme geometric and semantic homogeneity. When a single floor contains 46 windows with identical dimensions, material, and IFC class, retrieval based purely on **intrinsic attributes** hits a mathematical ceiling:

$$H(C) = -\sum_{i=1}^{N} \frac{1}{N} \log_2 \frac{1}{N} \approx H_{max} \quad \text{when all candidates are indistinguishable}$$

Neither SQL attribute filtering nor dense vector retrieval (e.g. CLIP cosine similarity) can break this lock — their Top-1 is mathematically bounded at $\frac{1}{N}$.

**The unlock — Topological Orthogonality:**

To break the bottleneck, a second independent information dimension must be introduced: **extrinsic relationships**. Even if element A and element B are intrinsically identical, their physical spatial relationships in the global 3D coordinate system are unique and deterministic. This research’s neuro-symbolic architecture is built on capturing and computing this topological orthogonality.

> **Prototype verification scenario:**
> Floor 3 has 46 identical `IfcWindow` elements. Only one is `ADJACENT_TO` an `IfcRailing`.
> Attribute baseline: Top-1 = **2.2%** (1/46). Neuro-Symbolic target: **60–80%**.

---

## 2. Thesis Statement & Theory Foundation

### 2.1 Thesis Statement (Updated)

> *”By grounding probabilistic Vision-Language Models in deterministic architectural topology graphs, this research proposes a Neuro-Symbolic interpreter layer that bridges the semantic gap between unstructured, egocentric site evidence and structured, allocentric IFC schemas. We demonstrate that **Relation-Region Crops** — a novel training strategy targeting the interface boundary between two co-located elements — enable VLMs to extract long-tail architectural spatial predicates (`ADJACENT_TO`, `CONTINUOUS`, `FILLS`) without shortcut learning, and that compiling these triplets into deterministic Cypher queries achieves zero-hallucination element retrieval in environments of extreme attribute entropy.”*

> ~~”...explicitly extracting long-tail spatial triplets (e.g., `<INTERSECTS>`) via geometric pre-computation...”~~
>
> ❌ **Deprecated (2026-02-24):** `INTERSECTS` (pipe-through-wall) requires MEP elements. `AdvancedProject.ifc` has **zero MEP elements**. Replaced with architectural predicates: `ADJACENT_TO`, `CONTINUOUS`, `FILLS`, `ON_TOP_OF`. See §5.2 for the full predicate vocabulary.

### 2.2 Theoretical Foundation: Attribute Entropy Bottleneck

The core theory is valid and unchanged. Two aspects of the original framing are updated:

| Original framing | Revised framing |
|---|---|
| “被特定机电管线穿透” (penetrated by MEP pipe) as the topological example | “紧邻楼梯栏杆” (adjacent to staircase railing) — same concept, correct model |
| Generic “extrinsic relationships” | Specifically: `FILLS`, `CONTINUOUS`, `ADJACENT_TO`, `ON_TOP_OF` from architectural topology |

### 2.3 RQ1 — Neuro-Perception Layer

**Research Question 1:** How can multimodal site evidence (photo, floorplan, chat) be reliably grounded to architectural spatial predicates, overcoming the inherent shortcut learning of Vision-Language Models?

**Theory anchors:**
- **Wang et al. 2024 (Industrial SGG):** AEC domain requires a predefined predicate vocabulary + multi-expert validation; without it, VLMs produce open-domain hallucinations
- **Wang et al. 2025 (VLM-VG):** Object-Centric Crops physically isolate background context, blocking language-prior shortcuts
- **This work’s extension:** Relation-Region Crop extends anti-shortcut learning from *entity identification* to *relation identification* — cropping the **interface boundary** between two elements rather than a single element

**Implementation:** Local Scene Graph (LSG) as the intermediate representation between the neural and symbolic modules. The model is forced to output structured spatial triplets `(subject, predicate, object)` rather than free-text descriptions.

**Output schema (Pydantic, implemented in `src/v2/types.py`):**

```python
class SpatialTriplet(BaseModel):
    subject_type: str          # e.g. “IfcWindow”
    predicate: Literal[
        “FILLS”,
        “CONTINUOUS”,
        “ADJACENT_TO”,
        “ON_TOP_OF”,
        “PERPENDICULAR_TO”,
        “PARALLEL_TO”,
    ]
    object_type: str           # e.g. “IfcRailing”
    object_material: Optional[str] = None
    confidence: float = 0.0
```

> ~~`Literal[“INTERSECTS”, “ADJACENT_TO”, “CANTILEVERED_OVER”, “CONTAINS”]`~~
>
> ❌ **Deprecated:** `INTERSECTS` and `CANTILEVERED_OVER` removed — no MEP or complex structural geometry in current model. `CONTAINS` is already handled by the existing `storey_name` / `space_name` constraint fields. Replaced with the 6 architectural predicates above.

> ~~`class LocalSceneGraph(BaseModel): target_element_class: str; spatial_relations: List[SpatialTriplet]`~~
>
> ❌ **Deprecated as a standalone schema:** A full `LocalSceneGraph` replacement of `Constraints` would break the entire pipeline. Instead, `spatial_relations: List[SpatialTriplet]` is added as **one new optional field** inside the existing `Constraints` class (backward-compatible). See §5.4 schema change.

### 2.4 RQ2 — Symbolic-Execution Layer

**Research Question 2:** How can deterministic graph traversal eliminate hallucination in the retrieval stage while maintaining 100% ontological compliance?

Even after accurate spatial triplet extraction, allowing an LLM to generate IFC query code introduces uncontrolled **extrinsic hallucination** (fabricated GUIDs, non-existent properties). A deterministic **Symbolic Execution Cage** is required.

**Theory anchors:**
- **Zhu et al. 2023/2025 (IFC-Graph):** Graph database representation of IFC semantic relationships
- **Iranmanesh et al. 2025 (Graph-RAG):** Graph traversal replacing vector retrieval in AEC
- **Lilis et al. 2025:** BIM semantic enrichment via geometric relation checking

**Geometric pre-computation (revised scope):**

> ~~AABB Broad Phase → OCCT Narrow Phase (`BRepAlgoAPI_Common`, `BRepExtrema_DistShapeShape`)~~
>
> ❌ **OCCT narrow-phase not implementing.** Reasons: (1) current IFC model is purely architectural with no MEP — there are no pipe-wall intersections to detect; (2) OCCT B-Rep boolean operations on IFC geometry require careful tessellation and tolerance handling (1–2 weeks). Replaced with lightweight alternatives:
>
> ✅ **Implementing instead:**
> - `CONTINUOUS`: zero geometry — detected from existing IFC `Constraints.Top ≠ storey_name` field
> - `FILLS`: zero geometry — from `IfcRelFillsElement` (already in `ifc_to_neo4j.py`)
> - `ADJACENT_TO`: centroid distance < 1.5m after extracting XYZ via `ifcopenshell.util.placement`
> - `ON_TOP_OF`: Z-axis centroid comparison + AABB XY overlap

**Cypher compiler (Python, zero LLM):**

The VLM-extracted triplet is deterministically compiled to a Cypher query via Python templates:

```cypher
-- Prototype example (ADJACENT_TO, architectural model)
MATCH (target:IfcWindow)-[:ADJACENT_TO]->(ref:IfcRailing)
WHERE toLower(ref.storey) CONTAINS toLower($storey)
RETURN target.guid, target.name
```

> ~~`MATCH (target:IfcPipeSegment)-[:INTERSECTS]->(ref:IfcWall) WHERE ref.Material CONTAINS ‘Concrete’`~~
>
> ❌ **Deprecated example:** No `IfcPipeSegment` exists in current model.

> ~~Multi-hop MEP reasoning: `MATCH (v:IfcValve)-[:CONNECTED_TO*1..3]->(p:IfcPipeSegment)`~~
>
> ❌ **Not implementing:** No valves, no pipe segments. Multi-hop traversal remains architecturally valid for future MEP-rich models; not applicable to current scope.

**Fallback mechanism (✅ implementing):**

When Cypher returns empty results:
1. **Predicate Relaxation:** `ADJACENT_TO` → `ON_STOREY` (drop the topological constraint)
2. **Attribute Relaxation:** remove `material` filter, keep structural topology only
3. **Final fallback:** cascade to existing Priority 3–7 rules (`storey+type`, `type_only`, etc.)

### 2.5 Evaluation Paradigm

Shifting from NLP metrics (BLEU/ROUGE) to a multimodal relational evaluation framework:

| Metric | Formula | What it proves |
|---|---|---|
| **Top-1 on H2 set** | Exact GUID match / total H2 cases | Topological constraints break attribute entropy bottleneck |
| **mR@100 (per-predicate)** | Mean recall across predicate types at K=100 | VLM learned visual topology, not language frequency |
| **SSR** | $(N_{initial} - N_{retrieved}) / N_{initial}$ | Graph traversal efficiency |

**Baselines:**
- **B1 — Dense Vector (CLIP):** proves attribute entropy kills vector retrieval
- **B2 — V2 Prompt-only (current):** proves text-only extraction bottleneck
- **B3 — V2 LoRA_2 (current fine-tune):** proves attribute constraints insufficient
- **B4 — Ours (LoRA_3 + spatial triplets + Neo4j):** full neuro-symbolic pipeline

> **mR@100 note on predicate scope:**
> ~~”罕见谓词（如 `<INTERSECTS>`, `<CANTILEVERED_OVER>`）的召回率”~~
>
> ❌ **Deprecated predicate targets:** With zero MEP elements, `INTERSECTS` and `CANTILEVERED_OVER` have zero test instances — mR@100 denominator is undefined.
>
> ✅ **Revised:** mR@100 is computed over `{ADJACENT_TO, CONTINUOUS, ON_TOP_OF}` (rare/computed) vs `FILLS` (common/schema-derived). **If recall on the former group > recall on FILLS, the model is proven to use visual topology rather than language frequency.**

---

## 3. Demo Plan — V2.5 Architecture

### 3.1 Neuro Layer: Structured Spatial Triplet Extraction

Current VLMs facing complex architectural scenes tend to output flat attribute descriptions (`{“color”: “white”, “material”: “concrete”}`), which are useless in high-homogeneity AEC environments. The Neuro layer’s goal is to **force allocentric spatial reasoning** — outputting scene graphs aligned with the IFC topology ontology.

**Model:** `Qwen2.5-VL-7B-Instruct` + Unsloth LoRA (r=16, alpha=32) → **LoRA_3**

**Input modalities:**

| Modality | Role | Crop strategy |
|---|---|---|
| Chat text | Vague verbal spatial references (“next to the railing”) | NER → candidate predicate shortlist |
| Site photo (global) | Scene context, element type identification | **Object Crop** (Wang et al. 2025) |
| Site photo (relation crop) | Predicate identification at interface boundary | **Relation Crop** ← this work’s innovation |
| Floorplan | Room containment, wall layout | Spatial zone crop |
| 4D metadata | `storey_name` (strongest signal, keep unchanged) | Direct slot-fill |

**Two-type crop strategy:**

```
Object Crop  (inherited from Wang et al. 2025):
  Input : full scene render
  Crop  : tight AABB around single target element (256×256)
  Learns: “this pixel texture = IfcRailing” without background context

Relation Crop  (this work’s innovation):
  Input : full scene render
  Crop  : union AABB of subject + object + 20% padding
  Learns: “window + railing in this spatial configuration = ADJACENT_TO”
  Key   : model cannot use global scene language prior; must rely on
          local pixel topology at the interface boundary
```

> ~~”仅包含管线穿透墙面那个洞口的 256x256 像素块”~~
> ~~”强迫 Qwen2.5-VL 仅凭借’相交处的物理像素特征’来预测出 `predicate: INTERSECTS`”~~
>
> ❌ **Deprecated example:** No pipe-wall intersections in current model. The Relation Crop concept is fully preserved — applied to architectural interface boundaries (window-railing adjacency, railing-stair top surface, wall-wall corner).

### 3.2 Symbolic Layer: Deterministic Graph Traversal

Once the Neuro layer produces a validated `SpatialTriplet`, the system enters fully deterministic symbolic execution. No LLM participates beyond this boundary.

- **Graph engine:** Neo4j with pre-computed topological edges
- **Query compiler:** Pure Python templates — triplet → Cypher (zero LLM)
- **Guarantee:** Zero extrinsic hallucination (no fabricated GUIDs, no invented properties)

### 3.3 Fallback Mechanism ✅

| Trigger | Action |
|---|---|
| Cypher returns 0 results | Predicate Relaxation: `ADJACENT_TO` → `ON_STOREY` |
| Still 0 results | Attribute Relaxation: drop `material` filter |
| Still 0 results | Cascade to Priority 3 (`storey+type`) in existing QueryPlanner |
| No `spatial_relations` extracted | Skip Priority 0 entirely, use existing Priority 1–7 (no regression) |

### 3.4 Evaluation Metrics

See §2.5. Implemented in existing eval harness — no new infrastructure required.

---

## 4. Data Synthetic Plan — synth_v0.5

### 4.1 Phase 1: Skeleton Mining via Geometric Pre-computation

The pipeline abandons text-matching rules in favour of mathematical computation directly on IFC 3D geometry. Framework: Skeleton-Skin Separation Architecture.

> ~~OCCT two-level pipeline: Broad Phase (AABB B-Rep extraction) + Narrow Phase (`BRepAlgoAPI_Common` for `INTERSECTS`, `BRepExtrema_DistShapeShape` for `ADJACENT_TO`, Z-projection IoU for `CANTILEVERED_OVER`)~~
>
> ❌ **OCCT narrow-phase not implementing in current prototype.** The IFC model is purely architectural (762 walls, 263 windows, 43 slabs — zero MEP). OCCT boolean operations are designed for MEP clash detection; applying them to an architectural-only model yields near-zero `INTERSECTS` instances, making the core metric (`mR@100` on INTERSECTS) undefined.
>
> ✅ **Implementing instead (lightweight, no OCCT dependency):**

| Predicate | Mining method | Effort |
|---|---|---|
| `FILLS` | `IfcRelFillsElement` from IFC schema — already in `ifc_to_neo4j.py` | **Free** |
| `CONTINUOUS` | `Constraints.Top ≠ storey_name` field — no geometry needed | **Free** |
| `ADJACENT_TO` | Centroid distance < 1.5m (same storey), after extracting XYZ via `ifcopenshell.util.placement` | 2–3 days |
| `ON_TOP_OF` | `Z_min(subject) > Z_max(object)` + XY AABB overlap | 1 day (after coords) |

### 4.2 Phase 2: Cross-Modal Skin Generation

After mining deterministic topological skeletons `(subject, predicate, object)`, wrap each with realistic multimodal “skin”.

**Step 1 — Text augmentation:**

Gemini 2.5 Flash generates vague, colloquial chat text from ground-truth triplets.

```
System prompt template:
“你是现场勘查的施工员。
 Ground Truth: 目标窗户紧邻（ADJACENT_TO）一段楼梯栏杆。
 请用口语化的聊天记录描述这扇窗的问题。
 禁止提及 GUID 或 IfcClass 名称。
 必须用空间参考（如’栏杆旁边的那扇窗’）。”
```

**Step 2 — Visual skin + anti-shortcut training:**

> ~~全局渲染 + “仅包含管线穿透墙面洞口” 的 256×256 裁剪~~
>
> ❌ **Deprecated crop target:** No pipe penetrations. The crop strategy is preserved — applied to architectural interface regions instead.

✅ **Implementing:**
- **Global render:** existing Blender/Bonsai headless pipeline (unchanged)
- **Relation Crop (new):** compute union AABB of subject + object elements, extract sub-image from Blender camera view
- **Object Crop:** tight AABB around target element only (for element-type training signal)

### 4.3 Dataset Stratification & Hard Negatives

Reusing and upgrading the existing H1/H2/H3 stratification:

| Tier | Definition | Discriminating signal |
|---|---|---|
| **H1** | 20–50 intrinsically identical elements on one storey | None — pure attribute entropy baseline |
| **H2** | N identical elements; only 1 has a topological relation to an anchor | `ADJACENT_TO` / `FILLS` / `CONTINUOUS` |
| **H3** | Conflict injection (chat says Floor 3, GT is Floor 5) | Requires conflict resolution reasoning |

**H2 expected outcome:**

| System | Top-1 on H2 |
|---|---|
| Attribute baseline (CLIP / V2 prompt) | ~2.2% (1/46) |
| LoRA_3 + Neo4j spatial triplet | **60–80%** (limited by VLM extraction accuracy) |

> ~~”其 Top-1 准确率理论上应收敛至 100%”~~
>
> ❌ **Overly optimistic.** Achieving 100% requires: (1) VLM extracts the correct triplet from the Relation Crop, (2) the triplet exists as a Neo4j edge, and (3) the Cypher returns exactly one result. Each step has non-zero failure probability. Realistic target: **60–80% on H2**, which is still a 27–36× improvement over the 2.2% attribute baseline.

### 4.4 Deliverables & Milestones

**Target: 800–1,200 high-quality triplet-annotated samples** (quality over quantity)

> ~~”目标产出：生成约 2000–3000 条多模态数据样本”~~
> ~~”建议规模：将其扩展到 3000–5000 个高质量的增强样本”~~
>
> ❌ **Scale revised downward.** LoRA fine-tuning does not require tens of thousands of samples. Current LoRA_2 trained on 933 samples and produced meaningful results. For the topology-focused LoRA_3, 800–1,200 samples with explicit spatial relation labels provides sufficient signal given the narrow predicate vocabulary (6 types).

| Days | Task | Output |
|---|---|---|
| 1–2 | Fix Neo4j + verify `FILLS` edges load correctly | Neo4j running with ~389 `FILLS` edges |
| 2–3 | Add centroid XYZ to `1_build_index.py` | element_index.jsonl with positions |
| 3–4 | Add `hunt_CONTINUOUS()` + `hunt_ADJACENT_TO()` to `2_hunt_skeletons.py` | ~300–500 skeleton triplets |
| 4–5 | Generate H2 hard-negative test cases (50 cases, ~46 distractors each) | synth_v0.5 eval set |
| 5–6 | Generate Relation Crop images from Blender | image_relation_crop.png per skeleton |
| 6–7 | Generate chat text via Gemini (relation-aware prompts) | text_chat per skeleton |
| 7 | LLM-as-Judge filtering pass (remove ~20–30% ambiguous samples) | 800–1,000 clean samples |
| 8–9 | Merge with synth_v0.4; prepare LoRA_3 training format | merged JSONL for Modal |
| 10 | Launch LoRA_3 training on Modal A100 | LoRA_3 adapter |

**Sample format (each training record):**

```
inputs:
  image_global.png         ← existing Blender render
  image_object_crop.png    ← tight crop around target element
  image_relation_crop.png  ← union AABB of subject + object (NEW)
  text_chat                ← Gemini-generated vague site language
  4d_metadata              ← storey_name (unchanged from V2)

label:
  {
    “storey_name”: “3 - Third Floor”,
    “ifc_class”: “IfcWindow”,
    “spatial_relations”: [
      {
        “subject_type”: “IfcWindow”,
        “predicate”: “ADJACENT_TO”,
        “object_type”: “IfcRailing”,
        “confidence”: 1.0
      }
    ]
  }
```

> **Technical contribution (unchanged):**
> Building a synthetic dataset with explicit spatial topology labels — without manual annotation, using machine-generated deterministic rules (IFC schema + centroid geometry) — is a recognised gap in AEC × multimodal AI. The Skeleton-Skin Separation Architecture is itself a research contribution.

> ⚠️ **[2026-02-24 修订]** 上述 `<INTERSECTS>` 示例基于 MEP 模型假设，已被下方 §5 全面修订。
> 经过 IFC 模型实地审计（AdvancedProject.ifc），当前数据集为纯建筑模型，零 MEP 构件，
> 谓词词典已切换为建筑拓扑谓词（见 §5.2）。原理论框架完全保留，具体实现路径已更新。

---

# 5. Next Step Plan: V2.5 Neuro-Symbolic Prototype
**Updated: 2026-02-24 | Based on: IFC audit + codebase review + reference paper analysis**

---

## 5.0 设计原则 (Design Principles)

**核心不变量（不可妥协）：**
- Neuro Layer 负责概率推断 → Symbolic Layer 负责确定性执行，两层之间用 Pydantic schema 隔离
- 所有拓扑边（graph edges）必须在离线阶段由几何/关系预计算确定，不允许 LLM 在线生成
- Backward-compatible：所有新增字段为可选，已有 Priority 1-7 cascade 在 `spatial_relations` 为空时完整保留

**核心修订（相对原始计划）：**
- ❌ 放弃 `INTERSECTS`（管道穿墙）→ ✅ 替换为 `ADJACENT_TO`（建筑紧邻）
- ❌ 放弃 OCCT 窄相位运算 → ✅ 质心距离 + Z轴比较（更快，无依赖）
- ❌ `LocalSceneGraph` 全量替换 `Constraints` → ✅ 在 `Constraints` 添加 `spatial_relations` 字段（向后兼容）
- ❌ CLIP reranking → ✅ 被图谱遍历替代（理由：同类型元素 CLIP embedding 高度相似，reranking 无法区分 46 扇相同的窗）

---

## 5.1 Codebase Fix List（优先级队列）

| 优先级 | 任务 | 解锁能力 | 影响文件 |
|--------|------|----------|----------|
| **P0** | Fix Neo4j connection（V1 一直 fallback 到 memory） | Graph vs. Vector 对比实验有效化；Priority 2 `neighbor+type` Cypher 生效 | `config.yaml`, `legacy/script/ifc_to_neo4j.py` |
| **P0** | 验证 `FILLS` 边是否真实写入 Neo4j | `FILLS` 是最便宜的拓扑谓词，0成本可用 | `ifc_to_neo4j.py` |
| **P1** | `1_build_index.py` → 添加质心坐标提取（`ifcopenshell.util.placement`） | 解锁所有几何谓词（`ADJACENT_TO`, `ON_TOP_OF`） | `data_curation/scripts/synth/1_build_index.py` |
| **P1** | `types.py` → 添加 `spatial_relations: List[SpatialTriplet]` 字段 | LoRA_3 输出有 schema 承接 | `src/v2/types.py` |
| **P1** | `constraints_to_query.py` → 添加 Priority 0 spatial_triplet Cypher 规则 | 三元组→图查询流水线通路 | `src/v2/constraints_to_query.py` |
| **P2** | Wire VLM visual output → constraint slots（图像解析结果注入 `spatial_relations`） | B/C/D 条件组有意义 | `src/v2/constraints_extractor_lora.py` |
| **P2** | Smarter fallback：Predicate Relaxation（`ADJACENT_TO` → `ON_STOREY`） | 提取失败时从容降级，而非 pool=100 | `constraints_to_query.py` |
| **P3** | DPO on eval failures（用 synth_v04 失败案例构建 negative pairs） | Constraint extraction F1 ↑ | `training/train.py` |
| **P4** | Add D1 condition（clear text + photo + floorplan，无 4D）测试空间接地 | 证明系统不依赖 4D metadata | `profiles.yaml`, eval harness |

**Critical Path: P0 → P1 → P2 → Evaluate**

---

## 5.2 Dataset Reality & Chosen Spatial Relations

### IFC 模型实地审计结果

```
AdvancedProject.ifc 实际构件统计：
  IfcWallStandardCase : 762   每层 ~127 堵
  IfcWindow           : 263   每层 46 扇（完全相同：同材质/尺寸/IFC类型）← 属性熵瓶颈实例
  IfcDoor             : 126
  IfcSlab             : 43
  IfcRailing          : 19
  IfcStair            : 9
  MEP 构件（管道/风管）:   0   ← 纯建筑模型，无机电系统
  元素含 XYZ 坐标      :   0   ← 需 P1 任务补充提取
  多楼层连续墙（1F→6F）: 771   ← 免费的强拓扑信号
```

### 选定的 6 个建筑拓扑谓词

| 谓词 | 定义 | 计算方式 | 预估实例数 | 视觉可观测性 | 优先级 |
|------|------|----------|-----------|------------|--------|
| `FILLS` | 门/窗占据墙体开口 | IFC `IfcRelFillsElement`，已在 `ifc_to_neo4j.py` | ~389 | 开口边界清晰可见 | ⭐⭐⭐ P0 |
| `CONTINUOUS` | 墙体跨越多个楼层 | `Top Constraint ≠ storey_name`，纯属性，**无需坐标** | **771** | 垂直方向贯穿楼层的墙线 | ⭐⭐⭐ P0 |
| `ADJACENT_TO` | 同楼层两元素质心距 < 1.5m | 质心距离（需 P1 坐标提取） | ~200–400 对 | 现场照片中物理紧邻 | ⭐⭐⭐ P1 |
| `ON_TOP_OF` | Z_min(主) > Z_max(客) 且 XY 投影重叠 | Z轴比较 + AABB | ~19–40（栏杆/楼梯） | 构件叠放关系可见 | ⭐⭐ P1 |
| `PERPENDICULAR_TO` | 两堵墙朝向向量点积 ≈ 0 | 墙体法向量 | 常见 | 阴阳角裂缝、转角场景 | ⭐⭐ P2 |
| `PARALLEL_TO` | 朝向向量点积 ≈ 1 | 墙体法向量 | 极常见 | 平行排列管线/墙列 | ⭐ P2 |

### 核心 H2 硬负样本场景（原型验证案例）

```
场景描述：
  第3楼有 46 扇完全相同的 IfcWindow（相同尺寸/材质/IFC类型）
  现场照片中可见：一扇窗旁边有楼梯栏杆（IfcRailing）

Ground Truth 三元组：
  (IfcWindow) -[:ADJACENT_TO]-> (IfcRailing, storey="3 - Third Floor")

基线系统（属性检索）：
  Top-1 = 1/46 = 2.2%  ← 数学死锁

Neuro-Symbolic 系统：
  Relation Crop → VLM 提取三元组 → Cypher 查询 → 唯一 GUID
  理论 Top-1 → ~70–90%（受 VLM 提取准确率限制）
```

---

## 5.3 Theory Foundation & Research Questions（修订版）

### Thesis Statement（更新为建筑拓扑版本）

> "By grounding probabilistic Vision-Language Models in deterministic architectural topology graphs, this research proposes a Neuro-Symbolic interpreter layer that bridges the semantic gap between unstructured, egocentric site evidence and structured, allocentric IFC schemas. We demonstrate that **Relation-Region Crops** — a novel training strategy targeting the interface boundary between two elements — enable VLMs to extract long-tail architectural spatial predicates (e.g., `<ADJACENT_TO>`, `<CONTINUOUS>`) without shortcut learning, and that compiling these triplets into deterministic Cypher queries achieves zero-hallucination element retrieval in environments of extreme attribute entropy (46 geometrically identical elements per floor)."

### RQ1 — Neuro Layer（感知层）
**How can multimodal site evidence (photo, floorplan, chat) be reliably grounded to architectural spatial predicates, overcoming VLM shortcut learning?**

理论锚点：
- Wang et al. 2024 (Industrial SGG)：AEC 领域需预定义谓词词典 + 多专家验证，否则 VLM 产生开放域幻觉
- Wang et al. 2025 (VLM-VG)：Object-Centric Crop 通过物理隔离背景上下文阻断语言先验捷径
- **本研究扩展**：Relation-Region Crop（双元素联合裁剪）将反捷径学习从"元素识别"扩展到"关系识别"

### RQ2 — Symbolic Layer（符号层）
**How can deterministic graph traversal eliminate hallucination in the retrieval stage while maintaining 100% ontological compliance?**

理论锚点：
- Zhu et al. 2023/2025 (IFC-Graph)：IFC 语义关系的图数据库表示方法
- Iranmanesh et al. 2025 (Graph-RAG)：图遍历替代向量检索的 AEC 应用
- **本研究扩展**：Pydantic schema 作为 Neuro→Symbolic 边界的强类型契约；Python 模板编译器将三元组映射为 Cypher，零 LLM 参与

### 评估范式转移（相比传统 VLM 评估）

| 传统指标 | 局限性 | 本研究指标 | 证明什么 |
|---------|--------|-----------|---------|
| BLEU/ROUGE | 衡量文本相似度，与检索精度无关 | Top-1 Accuracy on H2 | 拓扑约束打破属性熵瓶颈 |
| 整体准确率 | 掩盖长尾谓词的失败 | **mR@100（分谓词召回率）** | VLM 真正理解空间几何而非语言频率 |
| Recall@K | 候选池大小不受控 | SSR（搜索空间缩减率） | 图谱检索效率 |

**mR@100 的学术意义**：若稀有谓词（`CONTINUOUS`, `ADJACENT_TO`）的召回率 > 常见谓词（`FILLS`），则证明模型依赖视觉拓扑特征而非语言统计频率——这是克服 Modality Bias 的直接证据。

---

## 5.4 V2.5 Implementation Plan

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multimodal Input                             │
│  [Site Photo]  [Floorplan]  [Chat Text]  [4D Metadata]         │
└──────────┬──────────────────┬──────────────────────────────────┘
           │                  │
  ┌────────▼──────────────────▼────────┐
  │         NEURO LAYER (LoRA_3)       │
  │  Qwen2.5-VL-7B + Unsloth LoRA r=16 │
  │                                    │
  │  Crop Strategy (per modality):     │
  │  · Object Crop    → IFC class ID   │  (Wang et al. 2025)
  │  · Relation Crop  → predicate ID   │  ← THIS WORK
  │  · Floorplan Crop → spatial zone   │
  │                                    │
  │  Output: Constraints (extended)    │
  │  {storey_name, ifc_class,          │
  │   spatial_relations: [             │
  │     {subject, predicate, object}   │
  │   ]}  ← Pydantic validated         │
  └────────────────┬───────────────────┘
                   │ AEC predicate ∈ {FILLS, CONTINUOUS,
                   │                  ADJACENT_TO, ON_TOP_OF}
  ┌────────────────▼───────────────────┐
  │      QUERY COMPILER (Python)       │
  │      Zero LLM / deterministic      │
  │                                    │
  │  Priority 0: spatial_triplet       │ ← NEW
  │  Priority 1–7: existing cascade    │ ← UNCHANGED (fallback)
  │                                    │
  │  Predicate Relaxation:             │
  │  ADJACENT_TO → ON_STOREY → type   │
  └────────────────┬───────────────────┘
                   │ Cypher query
  ┌────────────────▼───────────────────┐
  │         SYMBOLIC LAYER (Neo4j)     │
  │  Pre-computed topological edges:   │
  │  -[:FILLS]->                       │ ← from IFC schema (free)
  │  -[:CONTINUOUS]->                  │ ← from IFC constraints (free)
  │  -[:ADJACENT_TO]->                 │ ← centroid distance < 1.5m
  │  -[:ON_TOP_OF]->                   │ ← Z-axis comparison
  └────────────────┬───────────────────┘
                   │
              Retrieved GUID
```

### Schema Change（最小侵入，向后兼容）

**`src/v2/types.py`** — 在 `Constraints` 末尾添加：
```python
class SpatialTriplet(BaseModel):
    subject_type: str                          # "IfcWindow"
    predicate: Literal[
        "FILLS", "CONTINUOUS",
        "ADJACENT_TO", "ON_TOP_OF",
        "PERPENDICULAR_TO", "PARALLEL_TO"
    ]
    object_type: str                           # "IfcRailing"
    object_material: Optional[str] = None
    confidence: float = 0.0

# In Constraints class, add ONE new field:
spatial_relations: List[SpatialTriplet] = Field(default_factory=list)
```

**`src/v2/constraints_to_query.py`** — 在 PRIORITY_RULES 最前面插入 Priority 0：
```python
{
    "priority": 0,
    "strategy": "spatial_triplet",
    "requires": ["spatial_relations", "ifc_class"],
    "description": "Topological triplet — breaks attribute entropy bottleneck (~1-3 candidates)",
    "template_cypher": """
        MATCH (target:{subject_type})-[:{predicate}]->(ref:{object_type})
        WHERE toLower(ref.storey) CONTAINS toLower($storey)
        RETURN target.guid, target.name, target.ifc_type
    """
}
```

### Two-Type Object-Centric Crop（核心训练创新）

参照 Wang et al. 2025，但将裁剪策略从"单元素"扩展到"关系界面"：

```
Type 1 — Object Crop（继承自 Wang et al. 2025）
  输入：全局场景渲染图
  裁剪：目标元素的紧密包围盒（256×256）
  训练目标：学习"这种像素纹理 = IfcRailing"（屏蔽背景语境）

Type 2 — Relation Crop（本研究创新）
  输入：全局场景渲染图
  裁剪：主体 AABB ∪ 客体 AABB + 20% padding
  训练目标：学习"这两种像素在此空间构型下 = ADJACENT_TO"
  关键：模型无法利用全局场景语言先验（"栏杆通常在楼梯旁"），
        只能依赖局部像素的相对位置特征进行判断
```

每条训练样本格式：
```
input:
  image_global.png       ← Blender 全局渲染
  image_object_crop.png  ← 目标元素紧密裁剪
  image_relation_crop.png← 双元素联合区域裁剪（新增）
  text_chat              ← Gemini 生成的模糊口语描述
  4d_metadata            ← storey_name（保留）

output (label):
  {
    "storey_name": "3 - Third Floor",
    "ifc_class": "IfcWindow",
    "spatial_relations": [
      {"subject_type": "IfcWindow",
       "predicate": "ADJACENT_TO",
       "object_type": "IfcRailing",
       "confidence": 1.0}
    ]
  }
```

### Data Generation Plan for synth_v0.5

**目标：800–1200 条高质量三元组标注样本**（质量优先于数量）

```
Phase 1 — 骨架挖掘（Day 1-3）
  任务：
    1_build_index.py   → 新增 centroid (X,Y,Z) 提取
                         使用 ifcopenshell.util.placement.get_local_placement()
    2_hunt_skeletons.py → 新增 3 个 hunter 函数：
      hunt_CONTINUOUS()   基于 Constraints.Top ≠ storey 字段，零几何
      hunt_FILLS()        harvest 已有 IFC IfcRelFillsElement 关系
      hunt_ADJACENT_TO()  同楼层 centroid 距离 < 1.5m，仅保留异类型对
  产出：~300–500 条原始骨架三元组

Phase 2 — H2 硬负样本构造（Day 3-4）
  策略：对每个 ADJACENT_TO 骨架，找其同楼层同类型的 N-1 个"诱饵"元素
  标注：target=1个有关系的，distractors=45个没有关系的（属性完全相同）
  产出：~50 个 H2 难度测试用例（仅用于评估，不用于训练）

Phase 3 — 多模态证据生成（Day 4-6）
  对每条骨架三元组：
    image_global      ← 已有 Blender/Bonsai headless 渲染流程（不变）
    image_relation_crop ← 新增：计算 union AABB，从 Blender camera 截取
    text_chat         ← Gemini with 关系感知 system prompt：
                        "严禁提及 GUID 和 IFC 类名。
                         Ground Truth: 目标窗紧邻（ADJACENT_TO）一段栏杆。
                         用工地口语描述这扇窗的问题。"
    label             ← 确定性标注（来自骨架，非 LLM 生成）

Phase 4 — LLM-as-Judge 过滤（Day 6-7）
  每条样本送入 Gemini，问："relation_crop 图像是否视觉上展示了该谓词？"
  过滤掉判定为"不明确"的样本（预计过滤率 ~20-30%）
  产出：800–1000 条最终高质量样本

合并策略：
  synth_v0.5 = synth_v0.4（933 条，已有属性约束） + 新增三元组样本（~800条）
  训练时标注两种 label（attribute constraints + spatial_relations 均有值）
```

### Training Plan（LoRA_3）

```
Base model:  unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit（与 LoRA_2 相同）
LoRA config: r=16, alpha=32（与 LoRA_2 相同）
新增输入：   image_relation_crop 作为第 3 张图像输入
新增输出：   spatial_relations 字段（在原有 JSON 结构末尾追加）
训练平台：   Modal A100（与 LoRA_2 相同）
预计时长：   3-5 小时（样本量约为 LoRA_2 的 2倍）

关键超参变化：
  max_seq_length: 4096（从 2048 增加，因多一张图像）
  epochs: 3（保持不变）
```

### Evaluation Framework

**三个指标，每个证明不同的论点：**

| 指标 | 测试集 | 基线预期 | 目标 | 证明 |
|------|--------|---------|------|------|
| **Top-1 on H2** | 50个 ADJACENT_TO 硬负样本（每个 ~46 诱饵） | 2.2%（1/46）| 60–80% | 拓扑约束打破属性熵瓶颈 |
| **mR@100（分谓词）** | 全量测试集，按谓词分组 | FILLS 最高（常见）| ADJACENT_TO ≥ FILLS | VLM 学习了视觉拓扑，非语言频率 |
| **SSR** | 全量测试集 | 92.65%（已有） | ≥ 92%（不退步） | 系统效率维持 |

**对照基线（Baselines）：**
```
B1: Dense Vector (CLIP)             ← 证明属性熵使向量检索失效
B2: V2 Prompt-only（现有系统）       ← 证明无视觉提取时文本检索瓶颈
B3: V2 LoRA_2（现有微调模型）        ← 证明属性约束不足以区分同质元素
B4: Ours — LoRA_3 + spatial_triplet + Neo4j  ← Full neuro-symbolic
```

---

## 5.5 Innovation Summary vs Prior Work

| 维度 | Wang et al. 2024 | Wang et al. 2025 | **本研究** |
|------|-----------------|-----------------|-----------|
| 领域 | 制造业工业 | 通用视觉 | **AEC / BIM 建筑** |
| 三元组落地 | 停留在文本场景图 | 2D 包围盒 | **IFC GlobalId（Cypher 编译）** |
| 反捷径策略 | 5专家共识验证 | 单元素裁剪 | **关系界面裁剪（Relation Crop）** |
| 标注来源 | 人工标注 | PaLI-3 自动生成 | **几何预计算（零标注成本）** |
| 图谱 Ground Truth | 无 | 无 | **IFC 拓扑关系离线预计算** |
| 输出 | 文本场景图 | 包围盒坐标 | **建筑构件 GlobalId** |
| 核心评估指标 | mR@20/100 | RefCOCO REC/RES | **mR@100 + H2-Top-1** |

**不可被复现的组合创新**（三者缺一不可）：
1. 建筑拓扑谓词词典（domain-specific, AEC-native）
2. Relation-Region Crop（将反捷径从实体扩展到关系）
3. Cypher 编译器（Pydantic 契约 + 纯 Python，零 LLM）
