# Phase 5 Post-Hoc Analysis

## Status

- Group 4 post-hoc analysis 已完成。
- oracle / information loss 结论已稳定。
- `G7_position_context` 已训练，并在修正 formal eval 路径后完成 AP held-out 评估。
- 当前 best systems 为：
  - `Track A`: `G7`
  - `Track B-2 Top-10`: `G3`
  - `Track B-2 Top-1 / MRR`: `G7`

## Summary

本轮 Group 4 implementation 完成了三类 post-hoc analysis：

1. `Oracle ceiling + unique topology diagnosis`
2. `Model collapse / shortcut-like diagnosis`
3. `Minimal LoRA6 ablation audit`

输出目录统一为：

- `mscd_demo/output/lora6_v2_ap_20260331/group4_post-hoc_analysis/`

对应总索引：

- `mscd_demo/output/lora6_v2_ap_20260331/group4_post-hoc_analysis/group4_index_20260404.md`

---

## 1. Oracle Ceiling: 当前 ceiling 低，不是单一原因

### 1.1 Overall oracle ceiling

本轮统一采用 `AP held-out, bug-fixed oracle (phase3_fixed)` 口径：

| Setting | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: |
| Oracle phase3 fixed | 40.0% | 5.0% | 0.1279 |

核心解释：

- Oracle 已经使用 perfect constraints，但 `Top-1` 仍只有 `5.0%`，说明问题不只是 extraction。
- 这个 ceiling 同时受到两类因素影响：
  - 一部分 case 在图结构上本来就不唯一。
  - 另一部分 case 虽然图里有更强 fingerprint，但当前 query / schema 没有把这些信息用到。

### 1.2 Rank distribution

| Rank bucket | Cases |
| --- | ---: |
| 1 | 3 |
| 2–5 | 11 |
| 6–10 | 10 |
| 51+ / not in shortlist | 36 |

解释逻辑：

- 只有 `3/60` case 是 oracle Top-1。
- `21/60` case 已经进入 Top-10 但没有排到第 1，这部分最值得分析，因为它们说明 “GT 已经被找到了，但没有被排到最前面”。

### 1.3 Unique topology / fingerprint analysis

对每个 target 构造 5 层 fingerprint：

- `L0`: `storey + ifc_class`
- `L1`: `+ predicate + object_type`
- `L2`: `+ direction`
- `L3`: `+ object_subtype`
- `L4`: `+ exact slot fingerprint` (`host_guid + wall_position_index [+ wall_child_total]`)

结果如下：

| Uniqueness class | Cases | Interpretation |
| --- | ---: | --- |
| `never_unique_even_at_L4` | 33 | 即使 full fingerprint 也不唯一，属于真实图歧义 |
| `unique_at_L3` | 9 | 加 `direction + subtype` 已可唯一定位 |
| `unique_at_L4_only` | 18 | 只有 exact slot 才能唯一定位 |

关键结论：

- `33/60` case 并不存在 strict Top-1 的唯一目标，因此当前 benchmark 里有很大一块是 `true graph ambiguity`。
- 但仍有 `27/60` case 在 richer fingerprint 下可以唯一化，其中：
  - `9` 个只要到 `L3`
  - `18` 个必须到 `L4`

这说明：

- 继续提高 schema granularity 仍然有价值。
- 但 thesis 中不能把所有 ceiling gap 都解释成“模型没学好”；约一半 case 是 benchmark/graph 自身的唯一性限制。

### 1.4 Reverse-GT: 反推 extraction 需要做到什么粒度

本轮新增了一个 reverse-GT audit：从 `GT GUID` 直接回到 Neo4j / graph fingerprint，反推“如果 target 在图里可唯一化，模型至少需要抽到什么粒度的 label 才够”。

结果如下：

| Minimal enrichment target | Cases | Meaning |
| --- | ---: | --- |
| `add_direction_and_object_subtype` | 9 | 只要补到 `L3` 就能唯一化 |
| `add_position_context` | 18 | 必须补到 `L4` exact slot 才能唯一化 |
| `current_label_sufficient` | 3 | 当前链路已经足够 |
| `not_fixable_by_label_only` | 30 | 即使 full fingerprint 也不唯一 |

如果只看 `query_not_using_available_info = 24` 这一组最值得优化的 case，则拆分为：

| Root-cause subset | Cases |
| --- | ---: |
| 需要 `position_context` | 16 |
| 需要 `direction + object_subtype` | 8 |

解释逻辑：

- 这说明 G7 的 label enrichment 不应平均发力，而应优先瞄准 `position_context`。
- `direction + object_subtype` 仍然重要，但它更像第二优先级。
- 对 `30` 个 `true_graph_ambiguity` case，label enrichment 本身不能把任务变成 strict Top-1。

### 1.5 Information loss chain

#### All cases

| Level | Coverage | Avg pool | Median pool | Ideal Top-10 | Ideal Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `L0_p1_only` | 100.0% | 107.18 | 76.0 | 3.3% | 0.0% |
| `L1_pred_obj` | 100.0% | 73.47 | 40.0 | 8.3% | 0.0% |
| `L2_pred_obj_dir` | 55.0% | 22.85 | 28.0 | 24.2% | 0.0% |
| `L3_pred_obj_dir_sub` | 100.0% | 24.40 | 6.0 | 58.3% | 15.0% |
| `L4_full_fingerprint` | 58.3% | 0.74 | 1.0 | 100.0% | 74.3% |

#### Position-sensitive subset only

| Level | Coverage | Avg pool | Median pool | Ideal Top-10 | Ideal Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `L0_p1_only` | 100.0% | 58.47 | 46.0 | 0.0% | 0.0% |
| `L1_pred_obj` | 100.0% | 33.65 | 33.0 | 5.9% | 0.0% |
| `L2_pred_obj_dir` | 97.1% | 22.85 | 28.0 | 24.2% | 0.0% |
| `L3_pred_obj_dir_sub` | 100.0% | 3.41 | 2.0 | 91.2% | 23.5% |
| `L4_full_fingerprint` | 97.1% | 0.73 | 1.0 | 100.0% | 72.7% |

解释逻辑：

- 当前 oracle 实际主要停留在 `L1`，即 `predicate + object_type` 这一层。
- 只到 `L1` 时，candidate pool 仍然很大，尤其对 AP 这种高密度 office model，无法支持高 Top-1。
- `L3` 已能把 position-sensitive case 的平均 pool 压到 `3.41`，并把 ideal Top-10 提升到 `91.2%`。
- `L4` 则几乎把 pool 压到 `1`，说明 exact slot 是最强 fingerprint。

对应图表：

- `mscd_demo/output/lora6_v2_ap_20260331/group4_post-hoc_analysis/oracle_ceiling/20260404/oracle_fingerprint_waterfall.png`

这张图与原来的 `oracle progression waterfall` 互补：

- 原图讲的是 `realized oracle / planner progression`
- 新图讲的是 `within-oracle information loss by L-query level`

因此，oracle ceiling analysis 给出的不是“天花板就是 40%/5%”，而是：

- `40%/5%` 是 **当前 schema + planner implementation** 的 ceiling；
- 不是 full-fingerprint graph retrieval 的 theoretical ceiling。

### 1.6 Query semantics: multi-anchor 和 multi-chain 不冲突

当前 `L1-L4` 分析的语义是：

- `target-rooted`
- `multi-anchor neighborhood filter`

也就是说：

- 一个 candidate 必须同时满足多个 relation anchor，才能在该 level 存活。
- 所以它能表达 `multi-anchor`。
- 但它**不是**显式的 `A -> B -> C` path traversal。

这不代表两者冲突。相反，下一步 planner 完全可以把两者组合起来：

1. 先做 `multi-chain / graph traversal`
- 例如从 anchor A 沿图走到 wall / opening，再走到 target neighborhood

2. 再做 `multi-anchor filter`
- 检查 target 周围是否同时满足 `FILLS + NEXT_TO + NEXT_TO`
- 再加上 `direction / subtype / position_context`

因此，更准确的 thesis 表述应是：

> The current L-query audit models target-rooted multi-anchor filtering rather than explicit chained traversal. In future planner design, the two are complementary rather than conflicting: graph traversal can be used to reach a candidate neighborhood, after which multi-anchor fingerprint filters can enforce subtype- and slot-level uniqueness.

### 1.7 oracle_position_test

在 `Top-10 = YES but Top-1 = NO` 的 `21` 个 case 中：

- `16` 个 case 可以应用 exact slot filter
- 其中 `15/16` 会直接变成 Top-1

代表性例子：

| Case | Family | GT rank before | Pool after exact slot | Top-1 after exact slot |
| --- | --- | ---: | ---: | --- |
| `AP_SK_149` | `triad:FILLS+NEXT_TO+NEXT_TO` | 3 | 1 | Yes |
| `AP_SK_091` | `triad:FILLS+NEXT_TO+NEXT_TO` | 7 | 1 | Yes |
| `AP_SK_102` | `triad:FILLS+NEXT_TO+NEXT_TO` | 8 | 1 | Yes |
| `AP_SK_233` | `paired:FILLS+NEXT_TO` | 2 | 1 | Yes |

结论：

- 这组结果强烈支持：当前 oracle ceiling 的主要突破口，不是再加训练数据，而是 **把 graph 中已有的 position fingerprint 显式引入 query / ranking**。

### 1.8 Root-cause attribution

| Root cause | Cases | Thesis interpretation |
| --- | ---: | --- |
| `query_not_using_available_info` | 24 | 图里已有更强信息，但 planner/schema 没用到 |
| `true_graph_ambiguity` | 30 | 即使 full fingerprint 也不唯一，strict Top-1 本身不现实 |
| `ground_truth_not_collected` | 3 | richer GT 本来可恢复，但 canonical label 未稳定记录 |
| `none_top1_success` | 3 | oracle 已成功达到 Top-1 |

可直接写成 thesis insight：

> The low oracle Top-1 is not a pure ranking failure. About half of the AP held-out cases remain structurally ambiguous even under full fingerprinting, while a further 24 cases already contain sufficient graph-side information but the current planner stops at predicate-object level constraints instead of exploiting subtype and exact-slot identity.

---

## 2. Model Diagnostics: G4 更强于 extraction，但更窄于 downstream discrimination

### 2.1 Diversity / collapse statistics

| Source | Predicate only | Pred+Obj | Pred+Obj+Dir | SR full | Label full |
| --- | ---: | ---: | ---: | ---: | ---: |
| GT | 5 | 13 | 16 | 35 | 45 |
| G3 | 4 | 10 | 12 | 30 | 41 |
| G4 | 5 | 10 | 12 | 21 | 38 |
| G7 | 6 | 13 | 15 | 31 | 42 |

解释逻辑：

- LoRA6 已经不是 LoRA5 那种极端 template collapse。
- 但 prediction diversity 仍然明显低于 GT，说明仍有 `residual template collapse`。
- 在更有判别力的层级上，`G4 < G3 < GT`：
  - `SR full`: `21 < 30 < 35`
  - `Label full`: `38 < 41 < 45`

这意味着：

- G4 不是“完全不懂 richer structure”，而是更容易把不同 case 压回到较少的稳定模板。

### 2.2 Field usage: direction / subtype 是否被真正用起来

| Model | Direction non-empty rate | Object subtype non-empty rate | Direction match rate | Subtype match rate |
| --- | ---: | ---: | ---: | ---: |
| G3 | 100.0% | 84.8% | 50.0% | 27.8% |
| G4 | 100.0% | 72.1% | 46.4% | 9.3% |
| G7 | 100.0% | 74.1% | 42.9% | 11.1% |

解释逻辑：

- 两个模型都会输出 direction，但这不等于它们真正“用好了” direction。
- `object_subtype` 是更强的区分信息，而 G4 在 subtype match 上明显弱于 G3。
- 这与前面的 oracle analysis 是一致的：
  - `L3` 对唯一化很重要；
  - 但 G4 对 `L3` 信息的保留和利用更弱。

### 2.3 Matched-case shortcut-like diagnostic

只在 `triad:FILLS+NEXT_TO+NEXT_TO` family 中比较：

- same storey
- same target class
- GT label 不同
- 看模型输出是否还相同

结果如下：

| Model | Pairs with GT difference | Pred+Obj same | Pred+Obj+Dir same | SR full same | Label full same |
| --- | ---: | ---: | ---: | ---: | ---: |
| G3 | 37 | 37.8% | 37.8% | 8.1% | 0.0% |
| G4 | 37 | 51.3% | 51.3% | 16.2% | 5.4% |
| G7 | 37 | 32.4% | 32.4% | 10.8% | 5.4% |

解释逻辑：

- GT 已经不同，但 G4 仍然更容易输出同一个 template。
- G3 在 `label_full_same = 0.0%`，而 G4 为 `5.4%`。
- G4 还存在两个很典型的 case pair：
  - `AP_SK_149` vs `AP_SK_158`
  - `AP_SK_173` vs `AP_SK_337`
  在 GT 已不同的情况下，G4 仍输出完全相同的 `label_full`。

因此，当前更准确的说法是：

- 不是已经证明 `pure shortcut learning`
- 而是已经证明：
  - `residual template collapse`
  - `position-insensitive shortcut-like behavior`
- `G4` 仍然是压缩最明显的一组。
- `G7` 只是在 `G4` 之上明显缓解了 residual collapse / shortcut-like behavior，还没有完全消除这两个问题。

可以直接写成 thesis insight：

> G4 achieves the strongest intermediate extraction metrics, but its output space is more compressed than G3. In matched triad cases where the ground truth differs, G4 is more likely to emit identical structured outputs, indicating residual template collapse and lower sensitivity to position-bearing cues.

---

## 3. G3 vs G4 vs G7: extraction 和 downstream 的张力

| System | Hop-1 | Predicate recall | Direction accuracy | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle phase3 fixed | - | - | - | 40.0 | 5.0 | 0.1279 |
| G3 | 80.0 | 91.4 | 78.6 | 26.7 | 1.7 | 0.0641 |
| G4 | 86.7 | 81.0 | 57.1 | 23.3 | 0.0 | 0.0324 |
| G7 | 78.3 | 93.1 | 82.1 | 23.3 | 3.3 | 0.0681 |

解释逻辑：

- G4 在 `Hop-1` 上更高，说明它更保守、更精确。
- 但 G4 的 `predicate recall` 和 `direction accuracy` 更低。
- 下游 retrieval 需要的是可区分候选的结构信号，而不仅仅是“格式正确、字段保守”。
- 因此：
  - G4 更像 `precision-biased extractor`
  - G3 更像 `better retrieval-oriented extractor`

简短结论：

- 对 Track A，`G7` 是当前 winner。
- 对 Track B-2，如果主指标是 `Top-10`，`G3` 仍然最好。
- 如果主指标是 `Top-1 / MRR`，`G7` 已经优于 `G3`。

---

## 4. Minimal richer-label ablation: G7 已完成

### 4.1 Pre-train audit

| Split | position\_context cov | direction cov | object\_subtype cov | Label-only changed |
| --- | ---: | ---: | ---: | ---: |
| train canonical | 0.5316 | 1.0 | 0.8333 | 2.95% |
| train aug | 0.5376 | 1.0 | 0.8268 | 9.83% |
| eval canonical | 0.55 | 1.0 | 0.9643 | 0.00% |

结论：

- `*_g7.jsonl` 已生成。
- `lora_system_g7` 已真实写入。
- richer-label gate 实际上是通过的。
- 但这不是 strict controlled eval-label ablation，因为 eval labels 没变化。

### 4.2 Corrected G7 result

| System | Hop-1 | Predicate recall | Direction accuracy | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G3 | 80.0 | 91.4 | 78.6 | 26.7 | 1.7 | 0.0641 |
| G4 | 86.7 | 81.0 | 57.1 | 23.3 | 0.0 | 0.0324 |
| G7 | 78.3 | 93.1 | 82.1 | 23.3 | 3.3 | 0.0681 |

补充指标：

- `G7` parse rate = `100.0%`
- `G7` class acc = `100.0%`
- `G7` storey acc = `100.0%`
- `G7` GT-in-Pool = `100.0%`

解释逻辑：

- `G7` 已经成为：
  - 当前最强的 intermediate extractor
  - 一个在 `Top-1 / MRR` 上优于 `G3` 的 downstream candidate
- 但 `G7` 还没有超过 `G3` 的 `Top-10 = 26.7%`，说明 richer fingerprint 更明显改善的是 early-rank precision，而不是 overall hit-rate。
- 因此当前最合理的结论不是“G7 已完全解决 collapse / shortcut-like behavior”，而是：
  - planner 升级是必要的
  - richer-label training 也是必要的
  - 但 richer-label training alone 还不足以完全消除 residual collapse 或恢复最强 `Top-10`

---

## 5. 可直接写进 Chapter 6 的主结论

### 5.1 Result statement

> The post-hoc analyses show that the current LoRA6 bottleneck is not purely a training issue. Under the bug-fixed AP held-out oracle, Top-10 reaches 40.0% but Top-1 remains only 5.0%, indicating a retrieval ceiling even under perfect constraints. However, this ceiling is not final: 24 cases already contain more discriminative graph-side information than the current planner uses, while 15 of 16 applicable Top-10-but-not-Top-1 cases become rank-1 once exact slot identity is injected. This suggests that the current symbolic backend stops too early at predicate-object constraints rather than exploiting full position-bearing fingerprints.

### 5.2 Interpretation statement

> At the model level, G4 achieves the strongest intermediate extraction accuracy but preserves less discriminative structure than G3. Diversity statistics and matched-case comparisons show that G4 is more likely to compress distinct triad cases into identical outputs, especially when subtype- and position-bearing cues are needed. This explains the extraction-to-retrieval tension: better intermediate exactness does not necessarily translate into better downstream ranking when the structured outputs become less distinctive.

### 5.3 Design implication statement

> Taken together, the evidence suggests a two-part improvement path. First, the symbolic backend should be extended to use richer fingerprints such as subtype and exact wall-slot identity, because the graph already contains substantial unused discriminative information. Second, model-side richer-label training is also useful: G7 becomes the strongest extractor and the best early-rank retriever after planner-side and eval-path alignment. However, the current gains are concentrated in Top-1/MRR rather than Top-10, so the next model iteration should aim to preserve the richer fingerprint benefits while recovering stronger retrieval hit-rate.

---

## 6. 下一步建议

优先级建议：

1. `Planner / schema` 优先
- 支持 `direction + subtype + exact slot` query
- 因为 oracle evidence 已经表明这比单纯再训 LoRA 更接近 root cause

2. `G7 follow-up`
- 下一轮模型实验如果继续做，应以：
  - 保持 `Top-1 / MRR` 优势
  - 同时追平或超过 `G3` 的 `Top-10`
  为目标
- 优先可检验的方向：
  - richer-label warm-start from `G3`
  - planner 对 `position_context` 的更强消费
  - `NEXT_TO` / triad slice 的 targeted supervision

3. `Thesis writing` 口径
- 不要写成 “oracle ceiling is fixed at 40%/5%”
- 应写成：
  - `40%/5%` 是 current phase3-fixed planner ceiling
  - 不是 full-fingerprint graph retrieval ceiling

4. `Shortcut learning` 口径
- 不要写成 “已严格证明 pure shortcut learning”
- 更稳妥的说法是：
  - `residual template collapse`
  - `position-insensitive shortcut-like behaviour`

---

## Final Conclusion

- `G7` 没有完全解决 template collapse 或 shortcut-like behavior，但相较 `G4` 已经明显缓解。
- 当前 planner 已不是单纯 single-hop 查询，而是 `multi-chain + multi-anchor + fingerprint-aware filter`，并且已经消费 `position_context`、`direction` 和 `object_subtype`。
- 因此，当前剩余瓶颈不只是 planner 设计，而是 extractor 对 richer fingerprint 的稳定 grounding。
- 最终系统结论应写成：
  - planner upgrade 是必要条件
  - richer-label training 也是必要条件
  - 但 richer-label training alone 还不足以完全消除 residual collapse / shortcut-like behavior
  - 当前最佳结论是：`G7` 最强于 extraction 和 early-rank retrieval，而 `G3` 仍最强于 strict `Top-10`
