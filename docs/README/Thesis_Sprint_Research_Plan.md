# Thesis Improvement Plan: Personal Research Notes
**2-Week Sprint before Defense**

---

## 0. 背景：为什么当前结果不够 persuasive

### 核心数字问题

| 指标 | 当前结果 | 问题所在 |
|------|---------|---------|
| Top-1 Accuracy | **4.3%** | 系统几乎不能自动锁定正确元素 |
| GT-in-Pool | 53.4% | 接近一半案例在检索阶段已经丢失 GT |
| Predicate Accuracy | 40% | Spatial relation 提取极不可靠 |
| Unique Spatial Patterns (LoRA5) | **14种** vs Gemini 61种 | Template collapse 明显 |

**Oracle 实验的意义**：在完美 constraints 下，symbolic backend 100% GT-in-Pool，pool 从 917 → 31（-96.6%）。这证明了**瓶颈不在 symbolic layer，在 neural extraction**。

### 两个根本性问题

**问题 1：Extractor 质量不足（Neural Bottleneck）**
- LoRA5 出现 template collapse：对 image presence 有反应，对 image content 不敏感
- Shortcut learning：同一场景 → 固定模板输出（81% 的 FP→MC spatial_relations 完全一样）
- IFC class accuracy 只有 47–64%，直接 gate 了 pool formation

**问题 2：Reranking 机制缺失**
- 当前系统在 pool 形成后没有 reranking 步骤
- GT-in-Pool 53.4% → Top-1 4.3%，这个 gap 完全没有被系统解决
- 只要 GT 进了候选池，系统就没有能力把它排到第一

---

## 1. 数据质量确认：Preprocessing & IFC Enrichment

### 当前 IFC Enrichment 的内容（已完成）

`ifc_engine.py` 在原始 IFC 基础上派生了以下边：

| 关系类型 | 来源 | 方法 |
|---------|------|------|
| FILLS | IFC schema | 压缩 IfcRelFillsElement → 直接 filler-to-host 边 |
| ADJACENT_TO | Geometry | 同楼层 centroid 距离 100–1500mm |
| CONTINUOUS | Metadata | Revit constraint 属性（跨楼层墙） |
| NEXT_TO | Geometry + ordering | Filler 投影到 wall axis，按投影值排序后连接相邻元素 |
| CONNECTS_TO | IFC schema | IfcRelConnectsPathElements |

**关键：NEXT_TO 边已存储位置信息**（来自 codebase 诊断 Q3）：
- 边属性：`position_index`（0-based），`wall_guid`
- 节点属性：`wall_position_index`，`wall_child_total`
- 即：Neo4j 里已经知道每面墙上每个 filler 的左右顺序

**Enrichment 带来什么好处（量化）**：

| 指标 | Base IFC Graph | Enriched Graph |
|------|---------------|---------------|
| 平均候选池大小（Oracle P1） | 47 | 31（加 spatial 后）|
| Oracle GT-in-Pool | 100% | 100% |
| Pool 压缩率 | 94.9% | **96.6%** |
| 支持的 predicate 类型 | 1（CONTAINS） | 5（+FILLS/NEXT_TO/ADJACENT_TO/CONTINUOUS/CONNECTS_TO）|

*注：enrichment 的主要价值在于让 P0 spatial query 成为可能，而不是独立提升 GT-in-Pool。在 extraction 准确的前提下，enrichment 是 pool 压缩的关键基础设施。*

### ⚠️ 待做实验：Enrichment Ablation（证明 enrichment 的必要性）

**问题**：上面的量化表格是基于 oracle 实验的间接数字，不足以独立证明 enrichment 的价值。答辩时如果被问"enrichment 到底贡献了多少"，需要一个直接的对比实验。

**实验设计：Base Graph vs Enriched Graph（相同 extraction，不同 graph）**

```
条件对照：
  Condition A（Baseline）：
    LoRA5-r32 extraction → P1-only query（storey + type）
    在 Base IFC Graph 上执行（只有 CONTAINS 关系）
    
  Condition B（Enriched）：
    LoRA5-r32 extraction → P0∪P1 query（spatial + storey + type）
    在 Enriched IFC Graph 上执行（含 FILLS/NEXT_TO 等）

评估集：116-case unified test set（已有）
关键指标：GT-in-Pool, Top-1, Avg Pool Size
```

**预期结果**：

| 指标 | Condition A（P1 only, base） | Condition B（P0∪P1, enriched）| Delta |
|------|----------------------------|-------------------------------|-------|
| GT-in-Pool | ~47%（P1 alone 的估计） | 53.4%（已知） | +~6pp |
| Avg Pool Size | ~68（P1 alone） | 73（已知） | 略大（union 策略） |
| Top-1 | ~4.3% | 4.3% | 相近（reranking 缺失） |

**为什么这个实验有价值**：

- 直接量化了 enrichment 对检索的贡献，不再依赖间接推断
- 揭示了一个重要 insight：**enrichment 提升了 GT-in-Pool，但不提升 Top-1**——因为 reranking 缺失，spatial 信息只用于了 pool 保留，没有用于最终排序
- 这个 insight 直接为 Plan B（RTV）的 reranking 机制提供动机：*"enrichment 已经把正确答案保留在 pool 里，但我们还没有用这些拓扑信息来做最终决策"*

**工作量**：约 2–4 小时
- Condition A 只需要修改 query planner config，禁用 P0，在 base graph（或忽略 spatial edges）上重跑 116 个案例
- Condition B 就是当前已有的结果，无需额外计算

**答辩叙事串联**：

```
"Enrichment 将 GT-in-Pool 从 ~47% 提升至 53.4%（+6pp）——
 这证明了 IFC graph 的拓扑信息在 pool 保留阶段是有效的。
 
 但 enrichment 对 Top-1 没有帮助（4.3% → 4.3%）——
 因为当前系统在形成 pool 之后没有利用这些拓扑信息做 reranking。
 
 这正是 Plan B（RTV）的动机：
 IFC graph 的拓扑知识足够丰富，我们不应该只用它来过滤候选池，
 而应该用它来生成候选描述，让 VLM 做最终的 cross-modal 验证。"
```

### Floorplan vs Site Image 的现实

**为什么不用 on-site image（重要，答辩时会被问）**：

Blender 相机问题导致合成 site image 质量极差：
- 相机无法完美模拟人眼高度（约 1.6m）和角度
- Wireframe 渲染时遮挡关系严重失真（最严重的问题）
- 真实现场有脚手架、工人、家具等遮挡，合成图完全没有
- 结果：在这种假图上训练的 LoRA5 在推断时面对任何真实遮挡就崩溃

**Floorplan 作为 pivot 的合理性**：
- 程序化生成，图像干净（颜色编码：红=Target, 蓝=IfcWindow, 绿=IfcDoor, 黑=IfcWall）
- 信息密度高（storey, type, 空间关系都可见）
- 但它是半结构化输入，和 "unstructured evidence" 的论文主张有距离

**这个 gap 是真实的**，需要在答辩中诚实面对，或通过 Plan B（RTV）来解决。

---

## 2. 可能执行方案

---

### Plan A：Contrastive Fine-tuning

#### 核心假设

> LoRA5 的 template collapse 是**训练数据 diversity 不足**导致的，不是 VLM 的根本能力限制。

证据：Gemini zero-shot 有 61 种 unique patterns，说明任务本身是可行的，只是 fine-tuning 走错了方向。

#### 诊断：为什么当前数据导致 template collapse

```
当前训练数据（同一面墙上的 W1, W2, W3）：

Case W1: floorplan(W1高亮) → spatial_relations: [{FILLS, IfcWall}]
Case W2: floorplan(W2高亮) → spatial_relations: [{FILLS, IfcWall}]  ← 和 W1 一样！
Case W3: floorplan(W3高亮) → spatial_relations: [{FILLS, IfcWall}]  ← 和 W1 一样！

模型学到的捷径：看到 floorplan 有高亮窗 → 输出 FILLS IfcWall
而不是：W1 在墙的最左端，NEXT_TO 右边的 W2
```

#### Codebase 诊断结果（决定工作量）

| 关键问题 | 诊断结果 | 工作量影响 |
|---------|---------|-----------|
| Q1: skeleton 是否有 position_index？ | **没有**，但投影代码已存在（line 949-964），只是把排序结果丢弃了 | ~1h 加字段 |
| Q2: 渲染器能否为不同 target 生成高亮？ | **可以**，`_floorplan_renderer.py` 接受 target_guid，只需创建不同 case JSONL | ~2h |
| Q3: Neo4j 有没有排序信息？ | **有**，`position_index` + `wall_child_total` 存在边和节点上 | 直接可用 |
| Q4: training JSONL 有没有 direction 字段？ | **没有**，spatial_relations 只有 predicate/object_type/confidence | ~3h 扩展 schema |

**总工作量：约 2–3 天（基础设施齐全，只差胶水代码）**

#### Contrastive Dataset 结构

```json
{
  "case_id": "AP_W1_contrastive_A",
  "contrastive_group_id": "AP_wall_B7_floor6",
  "contrastive_position": "leftmost",
  "input": {
    "floorplan_patch": "ap/floor6/wall_B7_highlight_W1.png",
    "chat": "window near the railing on sixth floor",
    "4d_context": "Window Installation, Level 6, In Progress"
  },
  "output": {
    "storey_name": "6 - Sixth Floor",
    "ifc_class": "IfcWindow",
    "spatial_relations": [
      {"predicate": "FILLS", "object_type": "IfcWall"},
      {"predicate": "NEXT_TO", "object_type": "IfcWindow", "direction": "right"}
    ]
  },
  "ground_truth_guid": "...",
  "siblings": ["AP_W2_contrastive_B", "AP_W3_contrastive_C"]
}
```

#### 预期结果与评估

**Before（LoRA5 当前）**：
- Unique spatial patterns: 14
- Predicate accuracy: 40%
- Template diversity (FP→MC identical ratio): 81%

**After（Contrastive fine-tuned）**：
- 目标 unique patterns: 30+
- 目标 predicate accuracy: 提升（具体数字 TBD）
- 如果 GT-in-Pool 因此从 53.4% 提升 → 间接证明

#### 论文贡献定位

> "我们诊断了 template collapse 的根本原因（contrastive diversity 不足），通过引入 position-aware contrastive training pairs，验证了 VLM 具备 pixel-level spatial reasoning 能力，但需要 appropriate training signal。"

#### 风险与局限

- Top-1 的直接改善路径不清晰（extraction 改善 → GT-in-Pool 提升，但 Top-1 仍需要 reranking）
- 不解决 "依赖 clean floorplan" 的根本问题
- 贡献定位是"训练方法改进"，不是架构洞察

---

### Plan B：Retrieve-then-Verify（RTV）

#### 核心洞察

> **Spatial relation 的 verification 比 generation 本质上更容易。**

在 NLP 中，Natural Language Inference（判断两句话关系）比 NLG（生成文本）简单得多。同理，判断"这张现场证据是否与'6楼、靠近金属栏杆的 IfcWindow，嵌在朝南墙体中'匹配"，比从证据中**凭空提取**这些关系要容易得多。

这也是对当前系统失败的一种更深层次的理解：我们一直在要求 VLM 做一个 generation 任务，但 IFC graph 本身已经包含了所有 spatial 信息，我们根本不需要 VLM 去生成它。

#### Pipeline 设计

```
阶段1：粗提取（与现有相同）
  输入：site photo + chat text（真正的 unstructured evidence）
  提取：storey_name, ifc_class（只要基础属性，不要求 spatial relations）
  → Symbolic backend → 候选池（~47–73 个）

阶段2：Graph-to-Language（G2L）描述生成 [新增]
  对候选池中 top-K 个元素，从 Neo4j IFC graph 自动生成自然语言描述：
  
  候选A: "6楼的 IfcWindow，嵌在一面朝南的 IfcWallStandardCase 中，
          左边紧邻一扇 IfcDoor（相距约 0.8m），右边是 IfcRailing"
  候选B: "6楼的 IfcWindow，嵌在走廊侧的 IfcWallStandardCase 中，
          附近没有门，右边是另一扇 IfcWindow"
  ...

阶段3：VLM Verification / Reranking [新增]
  输入：原始 site evidence（chat + photo）+ K 个候选描述
  任务：打分/排序——"哪个候选描述最匹配你看到的现场证据？"
  → 重新排序候选池 → Top-1 改善
```

#### 为什么不再依赖 clean floorplan

```
当前（依赖 floorplan）：
  VLM 需要从 floorplan 图像中"看出" spatial relations → template collapse

RTV：
  Spatial relations 来自 IFC graph（确定性，100% 准确）
  VLM 只需要做 text-image matching：
  "这张现场照片，是否和'靠近金属栏杆、嵌在朝南墙体'的描述匹配？"
  → 这是 VLM 擅长的多模态理解，不是空间关系生成
```

#### G2L 模块设计

```python
def generate_candidate_description(guid: str, neo4j) -> str:
    """
    从 Neo4j 查询候选元素的拓扑邻域，生成自然语言描述
    利用 enriched graph 中的 FILLS, NEXT_TO, position_index 等信息
    """
    result = neo4j.query(f"""
        MATCH (t:IFCElement {{guid: '{guid}'}})
        OPTIONAL MATCH (t)-[:FILLS]->(w:IFCElement)
        OPTIONAL MATCH (t)-[r:NEXT_TO]->(n:IFCElement)
        RETURN t.ifc_type, t.storey, t.name,
               w.ifc_type as host_wall_type,
               w.orientation as wall_orientation,
               collect({{
                   type: n.ifc_type,
                   direction: CASE WHEN r.position_index > t.wall_position_index
                               THEN 'right' ELSE 'left' END,
                   distance: r.projected_distance
               }}) as neighbors
    """)
    
    # 生成自然语言（避免技术术语，贴近 site evidence 语言风格）
    desc = f"{result.storey} 的 {humanize(result.ifc_type)}"
    if result.host_wall_type:
        desc += f"，嵌在{result.wall_orientation or ''}墙体中"
    for n in result.neighbors[:2]:
        desc += f"，{n.direction == 'right' and '右' or '左'}边是{humanize(n.type)}"
    
    return desc
```

**关键设计决策**：描述语言要贴近 site evidence 的表达方式（"靠近金属栏杆"），而不是技术术语（"NEXT_TO IfcRailing"）。这个语言 gap 是 G2L 模块需要解决的核心问题。

#### Verification 模块设计

```python
def verify_candidates(evidence: dict, candidates: list, model) -> list:
    """
    让 VLM 对候选描述和现场证据做 matching/scoring
    """
    prompt = f"""
    [现场证据]
    聊天记录: {evidence['chat']}
    4D上下文: {evidence['4d_context']}
    [附图]: {evidence.get('site_photo')}  # 如果有的话
    
    [候选 BIM 元素]
    {format_candidates(candidates)}
    
    请根据现场证据，对以上候选元素打分（0–10），
    分数越高表示越匹配。只返回 JSON 格式的分数列表。
    """
    
    scores = model.score(prompt)
    return rerank_by_scores(candidates, scores)
```

#### 快速验证实验（Day 1，Go/No-Go Decision）

**在构建完整系统之前，必须先验证核心假设是否成立**：

```
操作：
1. 从 116-case test set 取 10 个 GT-in-Pool 但 Top-1 错误的案例
2. 用 Neo4j 为每案例 top-5 候选手动生成自然语言描述
3. 把描述 + 原始 chat log 传给 Gemini（zero-shot）：
   "哪个候选描述最匹配这条现场证据？"
4. 统计 GT 是否被排到第一

判断标准：
≥ 6/10 成功 → 方向可行，继续建系统
< 4/10 成功 → 降级到 Plan A，RTV 作为 future work
```

#### 预期结果与评估

| 指标 | 当前（no reranking） | RTV（预期） |
|------|--------------------|-----------| 
| GT-in-Pool | 53.4%（阶段1不变） | 维持 ~53.4% |
| Top-10 | 25.9% | 提升（reranking 改善排序） |
| Top-1 | **4.3%** | 目标 **15–25%** |
| 对 floorplan 的依赖 | 高（spatial extraction 依赖） | **低**（阶段1只需属性） |

#### 论文贡献定位

> "我们发现系统的核心失败原因是将 spatial grounding 定义为 generation 任务。通过将任务重构为 retrieve-then-verify，利用 IFC graph 提供 spatial descriptions，让 VLM 做 cross-modal verification，Top-1 从 4.3% 提升至 X%，同时摆脱了对 clean floorplan 的依赖。这是 generation vs verification 任务分解在 AEC 领域的首次应用。"

#### 与论文核心主张的一致性

```
论文主张：Unstructured site evidence → Structured BIM

当前系统（Plan A 修复后）：
  Clean floorplan（半结构化）→ Structured BIM  ← 仍有 gap

RTV（Plan B）：
  Chat text + site photo（真正 unstructured）→ 候选描述（来自 IFC graph）→ Structured BIM
  ↑ 这才是真正回应论文主张的系统
```

#### 风险

- **核心假设未验证**：verification 是否真的比 generation 容易，在 AEC 视觉场景中未知
- **site photo 质量问题**：如果 verification 依赖视觉输入，Blender 生成的差图可能同样让 VLM 无法判断
- **语言 gap**：IFC 技术术语 vs 现场口语表达，G2L 模块需要仔细设计
- **时间风险**：如果 Day 1 验证失败，需要立即 pivot

---

### Plan C：Visual Representation Study（解释性实验）

#### 定位

不改进系统，而是通过系统性实验**解释 LoRA5 为什么失败**，以及什么样的视觉输入对 VLM spatial reasoning 最有利。

#### 实验设计

**同一 IFC 案例，四种不同输入表示**：

| Condition | 输入形态 | 目的 |
|-----------|---------|------|
| C1（baseline） | Raw floorplan（当前） | 当前系统的失败基准 |
| C2 | Annotated floorplan（显式箭头/虚线标注） | 测试 VLM 是否因"信息不够显式"失败 |
| C3 | Sketch-style abstraction（只保留 target + 邻居 + 关系线） | 测试 VLM 是否被无关视觉信息干扰 |
| C4 | Graph-as-Image（networkx 画出 spatial graph） | VLM 理解图结构的上限 |

**任何结果都支撑论文**：
- C1 << C2：信息在图里，VLM 需要更显式的呈现 → preprocessing 的价值
- 所有 Condition 差不多：VLM spatial reasoning 有根本限制 → Neuro-Symbolic 的必要性
- C4 >> 其他：VLM 理解图，但不理解工程图纸 → domain-specific representation 的重要性

#### 定位

这个实验作为**最后做**的补充实验，用于在 Discussion/Chapter 6 中为 Plan A 或 Plan B 的结果提供更深层的解释。工作量约 2–3 天，对论文叙事有加分但不是主攻。

---

## 3. 方案对比与决策

| 维度 | Plan A（Contrastive） | Plan B（RTV） | Plan C（Visual Study） |
|------|----------------------|--------------|----------------------|
| 论文贡献 | 训练方法改进 | 架构洞察（generation→verification） | 解释性发现 |
| 对 unstructured evidence 主张的回应 | 部分（仍依赖 floorplan） | 完全 | 无 |
| Top-1 改善路径 | 间接（extraction → pool） | 直接（reranking 是核心） | 无 |
| 技术风险 | 低（基础设施齐全） | 中（Day 1 验证） | 低 |
| 时间确定性 | 高（2–3 天） | 中（go/no-go 决定路径） | 高 |
| 工作量 | ~2–3 天 | ~5–7 天（含验证） | ~2–3 天 |
| 答辩说服力 | "我修了一个问题" | "我发现了更好的框架" | "我解释了为什么" |

### 推荐执行顺序

```
Day 1（今天）：Plan B 快速假设验证
  → 手动 10 个案例，测试 G2L + zero-shot verification
  → 成功 → 主攻 Plan B，Plan A 降为附加实验
  → 失败 → 主攻 Plan A，Plan B 降为 future work

Week 1：
  Plan B 路线：G2L 模块 + Verification 模块 + 端到端评估
  Plan A 路线：skeleton position_index + contrastive JSONL + continue fine-tune

Week 2：
  Plan C（如有余力）+ 结果整合 + Chapter 6 更新 + Chapter 7 初稿
```

---

## 4. 核心论证链（答辩最终叙事）

```
[已有] Oracle 实验
  → 证明：symbolic backend 在完美 constraints 下 100% GT-in-Pool, pool 917→31
  → 结论：瓶颈在 neural extraction，不在 symbolic

[已有] LoRA5 诊断
  → 证明：template collapse（14种模式 vs 61种），shortcut learning
  → 定位：spatial relation generation 任务对 VLM 来说太难

[新增 Plan A 或 Plan B]
  → Plan A：contrastive data 修复 template collapse，验证 VLM 有能力
  → Plan B：重构任务为 verification，Top-1 从 4.3% → X%，摆脱 floorplan 依赖

[可选 Plan C]
  → 解释：什么样的视觉输入对 VLM spatial reasoning 最有利

三者构成完整论证链：
"问题在哪里（Oracle + 诊断）→ 为什么失败（Plan C）
→ 如何解决（Plan A 或 Plan B）→ 改善了多少（量化结果）"
```
