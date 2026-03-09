# MSCD V2.5 — Neuro-Symbolic Prototype Plan

> **Last revised: 2026-03-09**
> §1–3: Theory, architecture, demo plan (stable).
> §4: Removed — superseded by §7.
> §5.0–5.3: Design principles, codebase fixes, dataset reality, RQs (stable).
> §5.4: Removed — superseded by §7.2–7.6.
> §5.5: Innovation summary (stable).
> §6: Implementation log (append-only).
> **§7: Authoritative plan** — data curation & LoRA_3 training (2026-03-09, training complete).
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

> **§4 removed** — original synth_v0.5 plan, superseded by §7 (Revised Plan).
> Original content covered skeleton mining, skin generation, dataset stratification, and milestones.
> All of these are now covered with updated details in §7.0–7.7.

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

> **§5.4 removed** — superseded by §7 (Revised Plan). Original content covered old schema change, crop strategy,
> data generation plan, training plan, and evaluation framework. See §7.2–7.6 for the authoritative versions.

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

---

# 6. Implementation Log

## 6.1 P0 完成记录 — Neo4j Fix (2026-02-25)

### 环境
- Neo4j Community 5.26.0，安装于 `/tmp/neo4j-community-5.26.0`，用 Java 21 启动
- Python 环境：conda env `mscd_demo`（ifcopenshell 0.8.4, py2neo OK）
- 启动命令：`/tmp/neo4j-community-5.26.0/bin/neo4j start`

### 修复的 Bug（`src/ifc_engine.py`）

| Bug | 根因 | 修复方式 |
|-----|------|----------|
| **FILLS = 0** | `_create_element_relationships()` 尝试向 `IfcOpeningElement` 节点创建边，但这类节点不存在于 Neo4j 中（被 `_create_element_nodes()` 跳过） | 预构建 `opening_to_host` 映射，将 Door/Window → FILLS → Wall 直接连接，跳过中间的 IfcOpeningElement 节点 |
| **storey = None on 17/19 railings** | 只检查了 `IfcRelContainedInSpatialStructure`；栏杆是通过 `IfcRelAggregates` 聚合在楼梯组件中，不在直接的空间结构关系里 | 改用 `ifcopenshell.util.element.get_container()`，可遍历完整层级 |
| **`storey` 属性缺失** | `node_props` 中从未添加该字段 | 添加 `"storey": storey_map.get(element.GlobalId)` |
| **IfcRailing/IfcStair 不在图中** | `element_types` 列表缺少这两种类型 | 已添加——ADJACENT_TO 原型场景必须用到 |

### 验证结果

```
FILLS edges:    389  (expected: 389) ✅
CONTAINS edges: 1,238                ✅
46 IfcWindows + 2 IfcRailings on "3 - Third Floor"  ← H2 prototype scenario in graph ✅
neo4j.enabled: true in config.yaml  ✅
```

### H2 原型场景验证

```
场景：3rd Floor 有 46 扇完全相同的 IfcWindow
      2 扇 IfcRailing 在同楼层
      属性基线 Top-1 = 1/46 = 2.2%（数学死锁）

ADJACENT_TO 边（待 P1 centroid 提取后添加）将打破此死锁
```

### P1 完成状态

- [x] `src/v2/types.py` → 添加 `SpatialTriplet` + `spatial_relations: List[SpatialTriplet]`
- [x] `src/v2/constraints_to_query.py` → 在 PRIORITY_RULES 最前面插入 Priority 0 `spatial_triplet`
- [x] `data_curation/scripts/synth/1_build_index.py` → 添加质心坐标提取（1233 元素，100% centroid 覆盖）
- [x] `data_curation/scripts/synth/2_hunt_skeletons.py` → 添加 `hunt_continuous_span()` / `hunt_fills_relation()` / `hunt_adjacent_to()`
- [ ] Neo4j → 添加 `ADJACENT_TO` + `CONTINUOUS` 边（需离线边生成脚本，P2 尚未开始）

---

## 6.2 P1 完成记录 — Schema + Skeleton Pipeline (2026-02-25)

### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `src/v2/types.py` | 添加 `SpatialTriplet`（6谓词 Literal）；`Constraints` 新增 `spatial_relations: List[SpatialTriplet] = []` |
| `src/v2/constraints_to_query.py` | 插入 Priority 0 `spatial_triplet` 规则；原 Priority 0–7 → 1–8；`_build_params` 处理 `spatial_relations` 字段；`_estimate_pool_size` 对 spatial_triplet 返回 3 |
| `data_curation/scripts/synth/1_build_index.py` | 新增 `centroid` 字段（`ifcopenshell.util.placement`）；添加 `target_name_keyword` / `neighbor_type`；17 个无楼层 railing 输出诊断告警 |
| `data_curation/scripts/synth/2_hunt_skeletons.py` | 新增 3 个 PatternType + 4 个 Skeleton 字段；3 个 hunter 函数 + CLI 参数 |

### 骨架挖掘结果（skeletons_v2_5.jsonl）

```
FILLS_RELATION    : 28  (IFC IfcRelFillsElement，门/窗填充墙洞)
ADJACENT_TO_RELATION: 34  (质心距 100mm–1500mm，跨类型对)
CONTINUOUS_SPAN   : 22  (Top Constraint ≠ storey_name)
其他属性型骨架    : 42
总计              : 126
```

---

## 6.3 Phase 2 完成记录 — H2 Hard-Negative Evaluation Set (2026-02-26)

### 阶段目标

构建 H2 硬负样本评估集：N 个属性完全相同的元素只有 1 个具有拓扑关系 → 属性基线 Top-1 = 1/N ≈ 2.2%。
证明 Priority-0 符号检索层（无 LoRA_3）能 100% 找到 GT 元素。

### 新增文件

| 文件 | 作用 |
|------|------|
| `data_curation/scripts/synth/2b_build_h2_hardneg.py` | 从拓扑骨架 + element_index 构建 H2 评估集 |
| `data_curation/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl` | 83 条评估用例 |
| `mscd_demo/test/test_h2_eval.py` | H2 评估主脚本（Priority-0 retrieval） |

### H2 数据集统计

| 模式 | 用例数 | 前池（属性） | 后池（检索） | SSR | 属性 Top-1 |
|------|--------|------------|------------|-----|-----------|
| `ADJACENT_TO_RELATION` | 34 | 3–361（avg 122）| avg 32 | 30% | avg 5.2% |
| `CONTINUOUS_SPAN` | 21 | 167–361（avg 274）| avg 74 | 65% | avg 0.4% |
| `FILLS_RELATION` | 28 | 30–46（avg 43）| avg 43 | 0%¹ | avg 2.4% |
| **全部** | **83** | **3–361（avg 134）**| **avg 46** | **29%** | **avg 3.1%** |

¹ FILLS SSR=0%：每扇窗/门都"填充"某面墙；无具体锚墙 GUID 时整楼层返回，但 GT 元素始终在池中。

### 修复的 Bug（`src/v2/retrieval_backend.py`）

| Bug | 根因 | 修复 |
|-----|------|------|
| **FILLS 返回 263 条**（预期 ~46）| spatial_triplet Cypher 过滤 `ref.storey`（墙的基层楼）而非 `target.storey`（窗的所在楼）；多楼层墙锚定在基层，Floor-2 窗填充这些墙时过滤失败 | `toLower(ref.storey)` → `toLower(target.storey)` |
| **ADJACENT_TO 重复行**（H2_079: pre=42, post=84）| 一扇门 ADJACENT_TO 两堵墙 → 一扇门出现两次 | `RETURN` 子句改为 `RETURN DISTINCT` |
| **CONTINUOUS GT=False**（H2_030）| `test_h2_eval.py` 硬编码 `"6 - Sixth Floor"` 作为所有 CONTINUOUS 用例的楼层；SK_065 的 `top_constraint` 实为 `"1 - First Floor"` | H2 记录添加 `top_constraint` 字段；`build_constraints()` 使用 `h2.get("top_constraint")` |

### 评估结果（Priority-0 符号层，无 LoRA_3）

```
GT-in-pool rate  : 83/83  (100%)   ← 图谱每次都能找到目标元素 ✅
Fallback rate    : 0/83   (0%)     ← Priority-0 边遍历从不降级 ✅
Mean SSR         : 29%             ← 候选池平均缩减 29%
Attr baseline    : 3.1%            ← 属性随机猜测 Top-1
```

100% GT-in-pool 确认符号层逻辑正确。最终 Top-1 指标需要 LoRA_3 在返回的候选池中将 GT 排在首位（Phase 3）。

### 下一步 (Phase 3)
- [ ] Blender 关系区域截图（`image_relation_crop.png`）—— 见 §6.4 质量问题解决方案
- [ ] Gemini 文本生成（关系感知 system prompt + 防幻觉层）—— 见 §6.4
- [ ] LLM-as-Judge 过滤（约过滤 20–30% 模糊样本）

---

## 6.4 Data Quality — 防幻觉 & Blender 遮挡解决方案 (2026-02-26)

Phase 3（皮肤生成）在实施过程中预期面临两类质量问题，以下记录解决策略。

---

### Issue 1 — Gemini 文本生成幻觉

**问题描述：** Gemini 在生成口语化工地描述时，容易捏造 IFC 类名、GUID、或与 ground truth 拓扑不符的空间关系（例如描述了一个不存在的 `ADJACENT_TO` 关系）。

#### 解决方案：双层防幻觉 + LLM-as-Judge

**Layer 1 — 严格 System Prompt（防止生成时幻觉）**

```python
SYSTEM_PROMPT = """你是现场勘查的施工员，正在用手机发送工地消息。

严格规则（违反则重写）：
1. 禁止提及任何 IFC 类名（如 IfcWindow、IfcWall）
2. 禁止提及 GlobalId、GUID、编号等技术标识符
3. 必须使用空间参考描述目标元素（如"栏杆旁边的那扇窗"）
4. 只能描述肉眼可见的空间关系，不能编造不在 Ground Truth 中的关系
5. 语言风格：口语化、简短、带有现场感（可以有错别字/语气词）

Ground Truth 三元组: {subject_type} {predicate} {object_type}
楼层: {storey_name}
"""

generation_config = {
    "temperature": 0.3,   # 低温度抑制随机幻觉
    "max_output_tokens": 200,
}

BANNED_KEYWORDS = ["Ifc", "GlobalId", "guid", "编号", "GUID",
                   "IfcWindow", "IfcWall", "IfcDoor", "IfcRailing"]
```

生成后立即检查 `BANNED_KEYWORDS`；命中则丢弃并重试（最多 3 次）。

**Layer 2 — LLM-as-Judge 一致性验证（事后过滤）**

将线框渲染图 + 生成文本送入第二个 Gemini 实例，要求输出结构化判断：

```python
JUDGE_PROMPT = """你是建筑 AI 质量审查员。

给定：
- 线框渲染图（wireframe render）
- 生成的口语描述: "{generated_text}"
- Ground Truth 三元组: {subject_type} {predicate} {object_type}

判断（JSON 输出，不得有额外文字）：
{
  "triplet_visible": true/false,     // 图像中能否看到该空间关系
  "text_consistent": true/false,     // 文本是否与 GT 三元组一致
  "hallucination_detected": true/false, // 文本是否捏造了 GT 以外的关系
  "verdict": "KEEP" / "DISCARD",
  "reason": "一句话说明"
}
"""
```

过滤规则：`verdict == "DISCARD"` 或 `hallucination_detected == true` → 丢弃该样本。
预期过滤率：20–30%；最终保留 800–1000 条高质量样本。

---

### Issue 2 — Blender 遮挡问题

**问题描述：** 在全局场景渲染中，目标元素（如墙内侧的窗户）可能被其他几何体完全遮挡，导致 `image_relation_crop.png` 中看不到任何有效像素，VLM 无法从中学习空间关系。

#### 三个候选策略

**Strategy A — 透明度隔离（推荐用于 relation_crop）**

将遮挡元素设为半透明（alpha=0.08），目标元素和锚元素保持不透明：

```python
def render_relation_crop(subject_guid, object_guid, all_elements):
    for elem in all_elements:
        mat = elem.active_material
        if elem.name not in [subject_guid, object_guid]:
            # 遮挡元素：极低不透明度，保留几何轮廓
            mat.blend_method = "BLEND"
            mat.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.08
        else:
            # 目标元素：完全不透明
            mat.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 1.0
```

优点：实现简单，保留场景上下文（灰色轮廓）；缺点：室内元素仍可能被楼板遮挡。

**Strategy B — 多角度相机采样（适合全局渲染）**

在以目标元素为中心的球面上均匀采样相机位置，按目标像素可见性评分，选最优视角：

```python
def best_camera_angle(subject_center, search_radius=5.0, n_samples=32):
    """在球面上均匀采样 n_samples 个相机位置，返回 target 像素最多的视角。"""
    best_score, best_camera = 0, None
    for theta, phi in fibonacci_sphere(n_samples):
        cam_pos = subject_center + search_radius * spherical_to_xyz(theta, phi)
        score = count_visible_pixels(cam_pos, subject_guid)
        if score > best_score:
            best_score, best_camera = score, cam_pos
    return best_camera
```

优点：能找到遮挡最少的自然视角；缺点：n_samples=32 渲染耗时高，适合离线批处理。

**Strategy C — Section Plane 切割（适合室内元素）**

对 FILLS（窗/门在墙内）场景，添加 Blender Section Plane，切除遮挡楼板/外墙：

```python
def add_section_plane(cut_z_offset=0.5):
    """在目标元素 Z 中心高度添加水平截面，切除上方几何体。"""
    bpy.ops.object.empty_add(type="SINGLE_ARROW", location=(0, 0, target_z + cut_z_offset))
    section = bpy.context.active_object
    section.name = "SectionPlane"
    # 配合 Bonsai (BlenderBIM) Section Override shader 生效
    bpy.context.scene.BIMProperties.active_section_plane = section
```

优点：专为 FILLS 关系设计，效果最干净；缺点：需要 Bonsai Section API，配置相对复杂。

#### 推荐组合流程

```
relation_crop 生成流程：
  1. 优先 Strategy A（透明度隔离）
     → 若 target 可见像素 < 阈值（如 500px），转 Strategy B 找最优视角
     → 若仍不足（室内 FILLS 场景），追加 Strategy C（section plane）

  2. 可见性验收标准：
     target + anchor 元素在裁剪图中合计可见像素 ≥ 1000px

  3. 记录每个样本使用的策略（"A" / "B" / "C"），便于后续分析
```

---

---

# 7. synth_v0.5 Revised Plan — Data Curation & LoRA_3 Training
**Updated: 2026-03-08 | Based on: data audit, schema review, training signal analysis**

---

## 7.0 Data Reality Audit (as of 2026-03-08)

### Current Dataset Inventory

| Source | Train | Test | IFC Model | Notes |
|---|---|---|---|---|
| synth_v0.4_ap (AdvancedProject) | 690 | 20 | AdvancedProject (44MB) | Attribute-only, 250 skeletons |
| synth_v0.4_bh (BasicHouse) | 33 | 20 | BasicHouse (53MB) | Small model, 2 storeys |
| synth_v0.4_dxa (Duplex) | 210 | 10 | Duplex_A (2.3MB) | Has railings, 4 storeys |
| synth_v0.4_merged | 933 | 50 | All 3 | Current LoRA_2 training set |
| synth_v0.5 topology (new) | 21 | 83 (H2) | AdvancedProject only | **Bug: `relations=None` in train records** |

### v0.4 Skeleton Pattern Distribution (AdvancedProject, 250 skeletons)

```
DIMENSIONAL_OUTLIER    : 80   ← attribute-only, relations=[] correct
VISUAL_MISMATCH        : 49   ← attribute-only, relations=[] correct
SPATIAL_PROXIMITY      : 33   ← HAS NearElement data, can enrich with relations
STOREY_INFERRED        : 33   ← attribute-only
PRISTINE_NEGATIVE      : 14   ← negative examples (no defect)
HIGH_DENSITY_DISAMBIG  : 13   ← attribute-only
UNIQUE_ATTRIBUTE       :  8
MATERIAL_SUBGROUP      :  7
RARE_TYPE_ON_STOREY    :  5
CROSS_PROPERTY_FILTER  :  5
SPATIAL_SCARCITY       :  3
```

### v0.5 Skeleton Mining Results (126 total, quota-capped)

```
ADJACENT_TO_RELATION   : 34   (centroid dist 100mm–1500mm, cross-type pairs)
FILLS_RELATION         : 28   (IFC IfcRelFillsElement, door/window → wall)
CONTINUOUS_SPAN        : 22   (Top Constraint ≠ storey_name)
Other attribute-type   : 42
Total                  : 126
```

**Key gaps:**
- `ON_TOP_OF`: 0 mined (hunter never implemented)
- `PERPENDICULAR_TO`: 0 mined (wall normals never extracted)
- `PARALLEL_TO`: 0 mined (wall normals never extracted)
- Mining quotas artificially low: 389 FILLS edges exist but only 28 skeletons mined

### v0.5 Skin Generation Results (partial run)

| Predicate | Skeletons | Skins generated | KEEP | DISCARD | Pass rate |
|---|---|---|---|---|---|
| FILLS | 28 | 9 | 2 | 7 | 22% |
| ADJACENT_TO | 34 | 32 | 14 | 18 | 44% |
| CONTINUOUS | 22 | 15 | 5 | 10 | 33% |
| **Total** | **84** | **56** | **21** | **35** | **37.5%** |

Main DISCARD reasons: black/empty wireframe renders, incomplete text generation.

### IFC Material Audit

```
IfcWallStandardCase materials (discriminating signal available):
  "Leather, weathered"         : 380  ← too common, low signal
  "Paint"                      : 260
  "Interior Wall A"            :  66  ← discriminating
  "Brick, Engineering"         :  16  ← discriminating (visually identifiable)
  "Plaster"                    :  16  ← discriminating
  "Concrete, Cast In Situ"     :  16  ← discriminating (visually identifiable)
  "Render, Beige, Textured"    :   8  ← discriminating

IfcWindow: "White RAL 9010" × 263     ← zero variance, useless for disambiguation
IfcDoor: "Door - Frame" × 115         ← low variance
IfcRailing: "None" × 17               ← no material data

Wall type names (from element Name field, also discriminating):
  Generic - 200mm, MockUp Exterior, MockUp Interior, MockUp Kitchen,
  MockUp Storage Wall, MockUp Elevator, etc. (10 distinct types)
```

### Cross-IFC Model Audit

| IFC Model | Walls | Windows | Doors | Railings | FILLS edges | Storeys |
|---|---|---|---|---|---|---|
| AdvancedProject | 762 | 263 | 126 | 19 | 389 | 8 |
| BasicHouse | 26 | 19 | 8 | 0 | 27 | 2 |
| Duplex_A | 97 | 24 | 14 | 4 | 38 | 4 |

---

## 7.1 Predicate Scope Decision

**Keep 3 predicates for thesis scope. Drop 3 as future work.**

| Predicate | Status | Rationale |
|---|---|---|
| `FILLS` | **KEEP** | Free from IFC schema, 389 edges, visually clear |
| `CONTINUOUS` | **KEEP** | Free from IFC constraints, 771 instances, strong signal |
| `ADJACENT_TO` | **KEEP** | Centroid-based, the core H2 scenario predicate |
| `ON_TOP_OF` | **DROP** | Hunter not implemented, limited instances (~19 railings) |
| `PERPENDICULAR_TO` | **DROP** | Wall normals not extracted, too common to disambiguate |
| `PARALLEL_TO` | **DROP** | Wall normals not extracted, too common to disambiguate |

Update `SpatialTriplet.predicate` Literal to reflect actual scope:
```python
predicate: Literal["FILLS", "CONTINUOUS", "ADJACENT_TO"]
```
`ON_TOP_OF`, `PERPENDICULAR_TO`, `PARALLEL_TO` remain in schema for forward compatibility
but will have zero training instances in v0.5.

---

## 7.2 LoRA_3 Output Schema (Revised)

### Design Principles

1. **No redundancy**: each field maps to exactly one priority rule in `constraints_to_query.py`
2. **Modality-aware**: the model learns which input modality provides which field
3. **Backward-compatible**: attribute-only cases output `spatial_relations: []`, triggering Priority 1–8 fallback as before

### Schema

```json
{
  "storey_name": "3 - Third Floor",
  "ifc_class": "IfcWindow",
  "space_name": "Master Bedroom",
  "target_name_keyword": null,
  "spatial_relations": [
    {
      "predicate": "ADJACENT_TO",
      "object_type": "IfcRailing",
      "object_material": null,
      "confidence": 0.85
    }
  ]
}
```

### Field Specification

| Field | Source modality | Priority rule | When empty |
|---|---|---|---|
| `storey_name` | 4D metadata (strongest), text, floorplan | P4 storey+type | Falls to P6 type_only |
| `ifc_class` | Text + image + floorplan | P6 type_only | Falls to P8 fallback |
| `space_name` | Image ("bedroom"), floorplan zone | P1 space+type | Skipped |
| `target_name_keyword` | Text ("Daikin", "AHU-03") | P2 name_keyword | Skipped |
| `spatial_relations` | Floorplan + image + text | **P0 spatial_triplet** | Skipped → P1–P8 cascade |

### Spatial Triplet Fields

| Field | Type | Purpose |
|---|---|---|
| `predicate` | `Literal["FILLS","CONTINUOUS","ADJACENT_TO"]` | Determines Neo4j edge type in Cypher |
| `object_type` | `str` (IFC class) | Reference element type, e.g. "IfcRailing", "IfcWall" |
| `object_material` | `Optional[str]` | Material filter on reference element — narrows pool (e.g. "Brick" → 16 walls instead of 381). VLM can identify visually distinctive materials (brick, concrete, plaster). |
| `confidence` | `float [0,1]` | Quality gate before symbolic execution. If `confidence < threshold` → skip Priority 0, fall to attribute cascade. Prevents bad extractions from polluting deterministic Cypher results. |

### Dropped Fields (vs LoRA_2 output)

| Field | Reason |
|---|---|
| `near_keywords` | Vague, overlaps with `spatial_relations` |
| `relations` (old string list) | Superseded by structured `spatial_relations` |
| `neighbor_type` | Superseded by `object_type` inside `spatial_relations` |
| `subject_type` inside triplet | Always equals `ifc_class` (redundant) |

### Confidence as Quality Gate

```python
# In retrieval_backend.py, before executing Priority 0 Cypher:
CONFIDENCE_THRESHOLD = 0.7  # tunable

if triplet.confidence < CONFIDENCE_THRESHOLD:
    # VLM not confident → skip Priority 0, use Priority 1–8 cascade
    # Prevents wrong predicate/object_type from returning wrong candidates
    logger.info(f"Triplet confidence {triplet.confidence:.2f} < {CONFIDENCE_THRESHOLD}, skipping P0")
    pass
else:
    # Trust the extraction → compile to Cypher
    execute_spatial_cypher(triplet)
```

---

## 7.3 Training Data Strategy — Modality-Specific Signal Routing

### Core Learning Goal

The VLM must learn **which modality provides which constraint field**, and when to output `spatial_relations: []` (no spatial signal) vs. a populated triplet.

```
Input modalities → Field routing:

text: "hairline crack on this wall"     → ifc_class: "IfcWall"
floorplan: red TARGET next to window    → spatial_relations: [{ADJACENT_TO, IfcWindow}]
image: bedroom interior scene           → space_name: "Bedroom"
4D metadata: Floor 3, Task #291         → storey_name: "3 - Third Floor"
                                        ──────────────────────────────
                                        Combined JSON output (all fields)
```

### Three-Tier Labeling Strategy

Training data must teach the model both WHEN to extract spatial relations and WHEN NOT TO.

**Tier 1 — "Spatial signal present" → `spatial_relations` populated**
Cases where at least one input modality provides a recoverable spatial signal:
- v0.4 SPATIAL_PROXIMITY cases (33) — text references a near element
- v0.5 topology skeletons (84) — floorplan shows target + anchor, text may reference relation
- Any case where floorplan crop shows target AND distinctive anchor element

**Tier 2 — "No spatial signal" → `spatial_relations: []` (correctly empty)**
Cases where the discriminating signal is purely intrinsic attributes:
- DIMENSIONAL_OUTLIER, VISUAL_MISMATCH, STOREY_INFERRED, UNIQUE_ATTRIBUTE, etc.
- The model learns: "no spatial signal in any input → don't hallucinate relations"
- Critical for preventing false-positive triplet extraction at inference time

**Tier 3 — "New topology cases" → generated fresh for v0.5**
Purpose-built training records with explicit spatial signal:
- Floorplan crops clearly showing both subject + anchor elements
- Text variants: some with spatial reference ("near the railing"), some without (force floorplan reading)
- Wireframe relation crops as additional visual signal

### Why Tier 2 (negative examples) matters

Without Tier 2, the model learns "always output some spatial relation" → hallucination.
The training mix should be approximately:
- **~60–70% attribute-only** (relations=[]) — teaches restraint
- **~30–40% topology** (relations=[...]) — teaches spatial extraction

This matches the natural data distribution: ~900 v0.4 attribute + ~150–200 v0.5 topology.

---

## 7.4 Data Generation Plan
execution order:
Step	Task	Status
1	Raise skeleton mining quotas	Not started
2	Enrich v0.4 SPATIAL_PROXIMITY (33 cases)	Not started
3	Fix renders + re-skin all topology skeletons	Not started
4	Cross-IFC pipeline (BasicHouse + Duplex)	Not started
5	Assemble training records (new LoRA_3 schema)	Not started
6	LLM-as-Judge image quality	Not started
7-8	LoRA_3 training + eval	Not started


### Step 1: Raise Skeleton Mining Quotas (1 day)

Current quotas artificially cap at ~30 skeletons per predicate. Available IFC instances are much larger.

| Predicate | Current skeletons | Available IFC instances | Target |
|---|---|---|---|
| FILLS | 28 | 389 edges | 100+ |
| ADJACENT_TO | 34 | ~200–400 pairs | 100+ |
| CONTINUOUS | 22 | 771 instances | 50+ |

Action: re-run `2_hunt_skeletons.py` with `--max-fills 120 --max-adjacent 120 --max-continuous 60`.

### Step 2: Enrich v0.4 SPATIAL_PROXIMITY Cases (1 day)

33 v0.4 cases have `NearElement` + `NearElementName` in their skeleton.
Convert to proper `spatial_relations` labels:

```python
# For each SPATIAL_PROXIMITY skeleton:
spatial_relations = [{
    "predicate": "ADJACENT_TO",  # NearElement → ADJACENT_TO
    "object_type": skeleton["target_props"]["NearElement_ifc_class"],
    "object_material": None,
    "confidence": 1.0
}]
```

Do NOT force spatial labels onto other v0.4 patterns (DIMENSIONAL_OUTLIER, etc.) —
those correctly have `spatial_relations: []`.

### Step 3: Fix Renders + Re-skin All Topology Skeletons (2 days)

Current 37.5% pass rate is mainly due to black/empty wireframe renders.
- Fix `3a_render_relation_crops.py` rendering failures (likely camera placement or geometry extraction bugs)
- Re-run skins on all topology skeletons (not just 56/84)
- Generate 2–3 text variants per skeleton to increase volume
- Target: 50–60% pass rate → ~150–180 KEEP skins from ~300 skeletons

### Step 4: Cross-IFC Pipeline (2 days)

Run the full pipeline on BasicHouse and Duplex_A:
1. `1_build_index.py` → element_index.jsonl per IFC
2. `2_hunt_skeletons.py` → topology skeletons per IFC
3. `3a_render_relation_crops.py` → wireframe renders
4. `3b_generate_skin.py` → text + images + judge

Expected yield:
- BasicHouse: ~20–30 topology skeletons (small model, 27 FILLS edges)
- Duplex_A: ~40–60 topology skeletons (4 railings → ADJACENT_TO diversity, 38 FILLS edges)

### Step 5: Assemble Training Records (1 day)

Fix `3c_assemble_training_records.py`:
- Topology cases: populate `spatial_relations` from skeleton
- Attribute cases: keep `spatial_relations: []`
- Output format: new LoRA_3 schema (5 fields, no `near_keywords`/`relations`/`neighbor_type`)

### Step 6: LLM-as-Judge for Image Quality (parallel with Steps 3–4)

Add scene plausibility check for site images:
- Binary verdict: "does this look like a plausible building scene?" (not exact 3D match)
- Discard images that are completely unrelated to the IFC model type
- Note: floorplan patches are already geometrically faithful (rendered from IFC), no filtering needed

### Expected Final Dataset: synth_v0.5

| Category | Records | spatial_relations |
|---|---|---|
| v0.4 attribute-only (Tier 2) | ~900 | `[]` (correctly empty) |
| v0.4 SPATIAL_PROXIMITY enriched (Tier 1) | ~33 | populated (ADJACENT_TO) |
| v0.5 AP topology (Tier 3) | ~80–100 | populated |
| v0.5 BH + DXA topology (Tier 3) | ~40–60 | populated |
| **Total** | **~1,050–1,100** | **~150–190 with relations** |

---

## 7.5 Training Record Format

### Input (user message)

```
[4D Task Status] TASK_291: Third Floor QC Check - IN_PROGRESS
[Project Phase] Fit-out
[Chat Log]
  Site Supervisor: 看一下这边
  Site Supervisor: 栏杆旁边那扇窗有裂缝

[Query] Extract search constraints from the above context and attached images.

<image: floorplan_patch.png>    ← IFC-rendered, target marked in red, legend shows element types
<image: site_photo.png>         ← Gemini-generated or wireframe render
```

### Output (assistant message / training label)

```json
{
  "storey_name": "3 - Third Floor",
  "ifc_class": "IfcWindow",
  "space_name": null,
  "target_name_keyword": null,
  "spatial_relations": [
    {
      "predicate": "ADJACENT_TO",
      "object_type": "IfcRailing",
      "object_material": null,
      "confidence": 1.0
    }
  ]
}
```

### Attribute-only case (Tier 2 example)

```json
{
  "storey_name": "-1 - Garage",
  "ifc_class": "IfcWallStandardCase",
  "space_name": null,
  "target_name_keyword": null,
  "spatial_relations": []
}
```

### Material-enriched triplet example

```json
{
  "storey_name": "1 - First Floor",
  "ifc_class": "IfcDoor",
  "space_name": null,
  "target_name_keyword": null,
  "spatial_relations": [
    {
      "predicate": "FILLS",
      "object_type": "IfcWall",
      "object_material": "Brick",
      "confidence": 0.9
    }
  ]
}
```

---

## 7.6 LoRA_3 Training — Plan & Results

### Model Configuration

```
Base model    : unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit (same as LoRA_2)
LoRA config   : r=16, alpha=32 (same as LoRA_2)
Platform      : Modal A100
Epochs        : 5
max_seq_length: 4096 (up from 2048 — multiple images per sample)
save_strategy : epoch (5 checkpoints, load_best_model_at_end=True)
```

### What LoRA_3 Must Learn (vs LoRA_2)

| Capability | LoRA_2 | LoRA_3 |
|---|---|---|
| Extract storey from 4D metadata | Yes | Yes (preserved) |
| Extract ifc_class from text + image | Yes | Yes (preserved) |
| Extract space_name from image/floorplan | Partial | Yes |
| Extract spatial_relations from floorplan + text | **No** | **Yes — core new capability** |
| Decide WHEN to output spatial_relations vs [] | N/A | **Yes — anti-hallucination** |
| Assign confidence score to triplet | N/A | **Yes — quality gate for symbolic layer** |
| Identify object_material from visual features | N/A | **Yes — brick, concrete, plaster** |

### Anti-Shortcut Training Design

Following Wang et al. 2025 (Object-Centric Crops):

1. **Relation Crop** (wireframe, union AABB of subject + object): forces model to learn predicate from local pixel topology, not global scene context
2. **Floorplan with marked target**: model must read spatial layout to identify adjacent/filling elements
3. **Text variants**: some samples have explicit spatial text ("next to railing"), others have generic text ("crack on wall") — model cannot rely solely on text keywords
4. **Negative examples (Tier 2)**: attribute-only cases with relations=[] prevent "always output a triplet" shortcut

### Evaluation Targets

| Metric | Baseline (LoRA_2) | Target (LoRA_3) | What it proves |
|---|---|---|---|
| **Top-1 on H2 eval** | ~2.2% (attribute ceiling) | **60–80%** | Spatial triplets break entropy bottleneck |
| **mR@100 per predicate** | N/A (no spatial extraction) | ADJACENT_TO ≥ FILLS | VLM learned visual topology, not language frequency |
| **SSR** | 74.1% | ≥ 74% (no regression) | System efficiency maintained |
| **Field F1 (spatial_relations)** | 0% | ≥ 60% | Triplet extraction accuracy |
| **False positive rate (relations on Tier 2)** | N/A | < 10% | Anti-hallucination working |

### Training Results (2026-03-09) ✅

**WandB run**: [qwen25vl-7b-r16-lora3-synth_v05](https://wandb.ai/hychia2024-carnegie-mellon-university/mscd-vlm-lora/runs/r5mpka8h)

```
Training data : 1,111 train / 19 test (synth_v0.5)
  v0.4 enriched : 933 records (AP+BH+DXA, 135 with spatial_relations)
  v0.5 topology : 178 train / 19 test (AP+BH+DXA KEEP skins)
  Anti-halluc.  : 72% attribute-only (relations=[]) / 28% topology

Images per record:
  v0.4: site photo + floorplan (2 images)
  v0.5: site photo + floorplan (2 images)
  Global renders are pipeline artifacts (Gemini generation + LLM-as-Judge) — NOT in training

Training curve:
  Epoch 1 : train_loss ≈ 1.2 → 0.15 (steep drop)
  Epoch 2 : train_loss ≈ 0.08, eval_loss ≈ 0.065
  Epoch 3 : train_loss ≈ 0.05, eval_loss ≈ 0.078
  Epoch 4 : eval_loss ≈ 0.090
  Epoch 5 : train_loss = 0.036, eval_loss = 0.106  (best model loaded from epoch 2)
  → Clear overfitting after epoch 2–3 on this dataset size

Final metrics (350 steps, 5 epochs):
  Train loss         : 0.2334 (averaged over all steps)
  Best eval loss     : 0.065 (epoch 2)
  grad_norm          : 0.233 (stable, no explosion)
```

**Inference check on 19 test samples (all topology):**

| Metric | Result | Target |
|---|---|---|
| JSON parse rate | **19/19 (100%)** | — |
| Class accuracy | **19/19 (100%)** | — |
| Storey accuracy | **19/19 (100%)** | — |
| Spatial predicate accuracy | **19/19 (100%)** | ≥ 60% |
| False positive rate | **0/0 (0%)** | < 10% |

Predicate breakdown: FILLS=11, ADJACENT_TO=7, CONTINUOUS=1 — all correct.

**Adapter location**: `modal volume get mscd-checkpoints /mscd-lora-v3/final ./models/adapters/v3_lora_qwen`

### Confidence Threshold Tuning

After LoRA_3 training, sweep `CONFIDENCE_THRESHOLD` on validation set:

```
threshold=0.5 → high recall, more false positives reaching Cypher
threshold=0.7 → balanced (recommended starting point)
threshold=0.9 → high precision, more fallbacks to Priority 1–8
```

Optimal threshold maximizes Top-1 on H2 eval set.

### Known Issues

1. **Eval loss NaN in post-training evaluate()**: Unsloth's `load_best_model_at_end=True` reloads the best checkpoint, but the post-training `trainer.evaluate()` call returns NaN. The epoch-level eval losses from training (logged to WandB) are valid. Not a training issue.
2. **Overfitting signal**: eval_loss rises from 0.065 (epoch 2) to 0.106 (epoch 5). Best model is epoch 2. For future runs, 3 epochs may suffice, or increase training data volume.
3. **Test set is topology-only (19 samples)**: False positive rate on attribute-only cases is untested in inference check. Need H2 eval harness for full picture.

---

## 7.7 Implementation Priority & Timeline

```
Step 1: Raise skeleton mining quotas           — ✅ Done (251 skeletons, AP)
Step 2: Enrich v0.4 SPATIAL_PROXIMITY          — ✅ Done (933 records in v04_enriched.jsonl)
Step 3: Fix renders + re-skin topology         — ✅ Done (312 skins, 197 KEEP, 63% pass rate)
Step 4: Cross-IFC pipeline (BH + DXA)          — ✅ Done (BH=25 KEEP, DXA=39 KEEP)
Step 5: Assemble + fix schema                  — ✅ Done (1,111 train / 19 test)
Step 6: LLM-as-Judge image quality             — ✅ Done (integrated in 3b_generate_skin.py)
Step 7: LoRA_3 training on Modal A100          — ✅ Done (2026-03-09, 5 epochs, 100% test acc)
Step 8: Evaluate on H2 + tune confidence       — Not started
```

**Next**: Step 8 — Download adapter, run H2 hard-negative eval, tune confidence threshold
