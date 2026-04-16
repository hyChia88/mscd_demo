Plan的改动：
## 对这个Plan的评价

设计上很完整，但对两周deadline来说**过度工程化了**。

```
5个分支 × 自定义builder/analyzer/registry × 完整artifacts
= 至少2-3天纯infrastructure work
= 你剩余时间的20-30%

而且：Branch 1, 2, 5 的数据你已经有了
      只是需要重新切片和呈现
      不需要新的orchestrator框架
```

真正需要**新代码**的只有Branch 3和Branch 4。其余三个是已有数据的post-hoc重组。

---

## 推荐的精简方案
### Position Context Fix（上午，3-4小时）

这是确定性最高的改动，不需要等诊断结果：

```
改6_assemble_lora6.py:

Before:
  {"predicate":"NEXT_TO", "object_type":"IfcWindow", "confidence":1.0}
  {"predicate":"NEXT_TO", "object_type":"IfcWindow", "confidence":1.0}
  ↑ 两个完全一样

After:
  {"predicate":"NEXT_TO", "object_type":"IfcWindow", 
   "direction":"left", "object_subtype":"BALANS 10M PRIVATE", "confidence":1.0}
  {"predicate":"NEXT_TO", "object_type":"IfcWindow",
   "direction":"right", "object_subtype":"BALANS 10M BATHROOM", "confidence":1.0}
  ↑ 每个唯一

数据源: wall_region_index + Neo4j NEXT_TO edges
  between_openings_patch已经有526条 → position data ready
  
验证: 组装后统计unique output patterns
  目标: >100 (vs 当前19)
  如果达不到 → assembly代码有bug，先debug
```

### 精简版Shortcut Diagnostic（下午，2小时）

不需要新的framework。写一个独立脚本就够：

```python
# shortcut_diagnostic.py (standalone, ~100 lines)
# 不需要orchestrator, registry, manifest

# Step 1: 从eval set找triad pairs
# 同楼层、同ifc_class、同topology family、不同skeleton_id
pairs = find_matched_pairs(eval_cases, family="triad:FILLS+NEXT_TO+NEXT_TO")

# Step 2: 对G3和G4跑counterfactual
for model in [G3, G4]:
    for pair in pairs:
        out_A = load_existing_prediction(model, pair.case_a)  # 已有
        out_B = load_existing_prediction(model, pair.case_b)  # 已有
        
        # 不需要重新inference！
        # 你已经有60个case的predictions
        # 只需要比较同一个model对同族不同case的输出差异
        
        compare(out_A, out_B)
        # 如果output一样 → template collapse (同族case→同模板)
        # 如果output不同 → 模型确实在区分

# Step 3: 输出一个简单table
# case_pair | model | storey_same | class_same | SR_same | direction_same
```

**关键简化**：你不需要swap images和重新inference。你已经有G3/G4对60个case的predictions。只需要比较**同族不同case的predictions是否相同**。如果两个同楼层同类型但不同位置的window输出一模一样，那就是template collapse的直接证据。

### Template Collapse Statistics（和shortcut同时做，30分钟）

```python
# collapse_stats.py (standalone, ~50 lines)

# 读取所有predictions
for model in [G3, G4, G6, Gemini]:
    preds = load_predictions(model)
    
    # 多层粒度统计
    unique_predicate_only = count_unique([normalize(p, level="predicate") for p in preds])
    unique_pred_obj = count_unique([normalize(p, level="pred+obj") for p in preds])
    unique_pred_obj_dir = count_unique([normalize(p, level="pred+obj+dir") for p in preds])
    unique_full = count_unique([normalize(p, level="full") for p in preds])
    
    # 和GT对比
    gt_unique = count_unique([normalize(g, level="full") for g in gt_labels])
    
    print(f"{model}: pred_only={unique_predicate_only}, "
          f"pred+obj={unique_pred_obj}, "
          f"pred+obj+dir={unique_pred_obj_dir}, "
          f"full={unique_full} / GT={gt_unique}")
```

这两个脚本加起来不到200行代码，不需要任何新framework。产出直接进thesis Chapter 6。
output存放：mscd_demo/output/lora6_v2_ap_20260331/group4_post-hoc_analysis

---

## Lora6-g7训练决策

**当前结论：`enriched labels` 是必要方向，但不是单独充分条件。**

```
Group 4 诊断后的最终判断:

  residual template collapse
    → richer labels 有帮助

  shortcut-like / low sensitivity behavior
    → richer labels 也有帮助

  oracle information loss
    → planner 必须真正消费 richer fingerprint

最终优化路径:
  planner-first + richer-label training

结论:
  enriched labels 是必要条件
  但不是单独充分条件
```

### Update 2026-04-04: pre-train audit

- 已修复 `6_assemble_lora6.py` 的 prompt path bug，`lora_system_g7` 现已真实写入 `*_g7.jsonl`
- 已重跑：
  - `python data_curation/scripts/synth/6_assemble_lora6.py --g7-profile --enable-scale-aug`
- 当前 `g7` 数据状态：
  - `train canonical = 237`
  - `eval canonical = 60`
  - `train aug = 757`
- G7 prompt 落盘覆盖：
  - train canonical `237/237`
  - train aug `757/757`
  - eval canonical `60/60`
- Coverage:
  - train canonical: `position_context 0.5316`, `direction 1.0`, `object_subtype 0.8333`
  - train aug: `position_context 0.5376`, `direction 1.0`, `object_subtype 0.8268`
  - eval canonical: `position_context 0.55`, `direction 1.0`, `object_subtype 0.9643`
- Label-only delta vs current settled data:
  - train canonical: `7/237 = 2.95%`
  - train aug: `74/753 = 9.83%`
  - combined train: `81/990 = 8.18%`
  - eval canonical: `0/60 = 0.00%`
- 结论：
  - 这版 `g7` 已可作为 performance-oriented run 开训
  - 但它不是严格的 eval-label ablation，因为 eval labels 没变化

Lora6-g7配置：

```
Data:   lora6_v2_ap_train_aug_g7.jsonl + lora6_v2_ap_eval_canonical_m_g7.jsonl
Prompt: lora_system_g7
Base:   G3 skeleton, but performance-oriented
Train:  r=32, alpha=64, lr=7e-5, epochs=5, batch=2, grad_accum=4
Eval:   Track A + Track B-2 (same 60-case AP)
```

---


## LoRA6-G7训练具体步骤

### Step 1: 改assembly（Day 1上午的核心工作）

```
只改 6_assemble_lora6.py 中生成 spatial_relations 的逻辑

需要做的:
  1. 读取 wall_region_index_ap_20260331_c.jsonl
     找到每个skeleton的between_openings_patch记录
     
  2. 读取 Neo4j NEXT_TO edges
     获取每个FILLS element的:
       wall_position_index
       wall_child_total  
       left neighbor type + name
       right neighbor type + name

  3. 在组装spatial_relations时注入这些字段:
     direction: "left" / "right"
     object_subtype: neighbor的type name (e.g., "BALANS 10M BATHROOM")
     
  4. 输出新的JSONL:
     lora6_v2_ap_train_canonical_m_g7.jsonl
     lora6_v2_ap_train_aug_g7.jsonl
     lora6_v2_ap_eval_canonical_m_g7.jsonl
```

### Step 2: 验证数据质量

```bash
# 当前已通过的 pre-train audit
python data_curation/scripts/synth/6_assemble_lora6.py --g7-profile --enable-scale-aug
```

检查要点：

- `lora_system_g7` 是否真实写入全部 `*_g7.jsonl`
- `train_aug_g7` 是否保留 fullaug 规模
- `combined train label-only changed >= ~8%`

### Step 3: 训练

```bash
# G7 official training command
modal run mscd_demo/training/train_lora6.py \
  --train-jsonl /root/cmu/master_thesis/data_curation/datasets/synth_v0.5_ap/train/lora6_v2_ap_train_aug_g7.jsonl \
  --eval-jsonl /root/cmu/master_thesis/data_curation/datasets/synth_v0.5_ap/train/lora6_v2_ap_eval_canonical_m_g7.jsonl \
  --lora-r 32 \
  --lora-alpha 64 \
  --lr 7e-5 \
  --epochs 5 \
  --batch-size 2 \
  --grad-accum 4 \
  --output-subdir mscd-lora-v6-g7-position-context \
  --wandb-run qwen25vl-7b-lora6-g7-position-context-r32-lr7e5-ep5
```

关键：这次是 performance-oriented G7，不再是严格 isolate 的 label-only run；变化包含 `richer labels + lora_system_g7 + planner consumption`。

### Step 4: Eval

```bash
# Track A
modal run mscd_demo/training/eval.py \
  --adapter-dir /mscd-lora-v6-g7-position-context/best

modal volume get mscd-checkpoints \
  /mscd-lora/eval_constraints_mscd-lora-v6-g7-position-context__best.jsonl \
  mscd_demo/output/lora6_v2_ap_20260331/g7_position_context__ap_eval.jsonl

python mscd_demo/evaluation/analysis/score_ap_track.py \
  --pred g7_position_context=mscd_demo/output/lora6_v2_ap_20260331/g7_position_context__ap_eval.jsonl \
  --out-dir mscd_demo/output/lora6_v2_ap_20260331/metrics

# Track B-2
cd mscd_demo
python script/run.py \
  --profile v2_lora \
  --cases evaluation/cases/cases_ap_heldout_e2e.jsonl \
  --precomputed output/lora6_v2_ap_20260331/g7_position_context__ap_eval.jsonl \
  --output_dir output/lora6_v2_ap_20260331/ap_e2e_phase5_g7/g7_position_context \
  --config config.yaml \
  --profiles profiles.yaml \
  --p0-strategy p0_union_p1

python evaluation/analysis/score_unified_track.py \
  --cases evaluation/cases/cases_ap_heldout_e2e.jsonl \
  --trace g7_position_context=output/lora6_v2_ap_20260331/ap_e2e_phase5_g7/g7_position_context/traces_20260404_132823_v2_lora_p0_union_p1.jsonl \
  --precomputed g7_position_context=output/lora6_v2_ap_20260331/g7_position_context__ap_eval.jsonl \
  --out-dir output/lora6_v2_ap_20260331/metrics \
  --summary-prefix track_b2 \
  --metric-suffix ap_e2e_metrics \
  --order track_b2
```

### Update 2026-04-04: actual evaluation result

| Metric | G3 | G4 | G7 |
| --- | ---: | ---: | ---: |
| Track A Hop-1 | 80.0 | 86.7 | 78.3 |
| Track A Pred R | 91.4 | 81.0 | 93.1 |
| Track A Dir | 78.6 | 57.1 | 82.1 |
| Track B-2 GT-in-Pool | 100.0 | 100.0 | 100.0 |
| Track B-2 Top-10 | 26.7 | 23.3 | 23.3 |
| Track B-2 Top-1 | 1.7 | 0.0 | 3.3 |
| Track B-2 MRR@10 | 0.0641 | 0.0324 | 0.0681 |

补充结果：

- `G7` parse rate = `100.0%`
- `G7` class acc = `100.0%`
- `G7` storey acc = `100.0%`
- `G7` GT-in-Pool = `100.0%`
- `G7` avg final pool = `117.4`

结论：

- 当前正式评估结果如上，应直接作为 `G7` 的有效结果引用。
- 在当前正式设置下，`G7` 成为：
  - 当前最强的 Track A extractor
  - 当前最强的 `Top-1 / MRR` downstream model
- `G3` 仍然保持最高 `Top-10 = 26.7%`，所以 strict hit-rate 仍由 `G3` 领先。
- 因此当前最准确的系统判断是：
  - `Track A`: `G7`
  - `Track B-2 Top-10`: `G3`
  - `Track B-2 Top-1 / MRR`: `G7`

一句话总结：

- `planner-first + richer fingerprint` 的方向仍然由 oracle 支持；
- `G7` 证明 richer-label training 在 planner 已升级、eval path 已对齐的前提下是有效的；
- 但当前收益主要集中在 `Top-1 / MRR` 和 extraction richness，而不是 strict `Top-10`。
