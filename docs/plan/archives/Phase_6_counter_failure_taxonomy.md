# Phase 6 T1.1 — Counter failure taxonomy (AP canonical)

**Date:** 2026-04-20
**Cases:** 60 canonical (AP eval) → 33 FILLS/NEXT_TO with teacher `"Nth of M openings"` GT
**Validator:** `mscd_demo/evaluation/h2/validate_phase6_ap_canonical_counts.py`
**Report:** `mscd_demo/output/lora6_v2_ap_20260331/phase6_ap_canonical_counter_report.jsonl`

## Headline: 11/33 exact-match = 33.3%

Baseline comparison:
- Previous H2 run (two-image template-match design): 0/10 exact-match; 44/46 nulls
- New full-storey render + world→pixel pipeline: **33.3% exact-match, 0 nulls, 0 crashes**

Below the 40% T1.2 gate, but **most of the "fails" are a frame-of-reference mismatch**, not counter bugs. See below.

## Teacher label semantics — the frame-of-reference issue

Reverse-engineered from [6_assemble_lora6.py:310-358](data_curation/scripts/synth/6_assemble_lora6.py#L310-L358):

Teacher's `"Nth of M openings on the same wall"` is computed as:
- **Grouped by single `IfcWall` GUID** (the opening's host wall element)
- **Restricted to the target's storey** (multi-storey walls are sliced per floor)
- Ordered by projection along the wall's local-coord X-axis

OpenCV's count is computed as:
- **Grouped by a single Hough line in the image** (can span multiple colinear `IfcWall` elements)
- Already per-storey (we render one PNG per storey)
- Ordered by projection along the detected line direction

**Two walls that look like one continuous line in the floorplan are one IFC wall for the teacher, one line for OpenCV** — but the reverse can also hold. When adjacent `IfcWall` elements at T-junctions or along facades happen to share an axis, OpenCV merges them into one line while teacher counts them separately.

Spot-check confirming the frame:
```
AP_SK_082  teacher=1/5   IFC per-host-wall=1/5   (agreement, isolated wall)
AP_SK_107  teacher=3/3   IFC per-host-wall=11/15 (teacher is storey-slice)
AP_SK_092  teacher=9/10  IFC per-host-wall=45/50 (teacher is storey-slice)
```

## Failure distribution (revised with frame analysis)

| Category | Count | % | Bug? |
|---|---|---|---|
| OK (exact match) | 11 | 33.3 | — |
| Total matches, position inverted | 5 | 15.2 | **YES** — ordering direction not canonicalised |
| Total over-count (OpenCV line merges walls) | 14 | 42.4 | **FRAME MISMATCH** — not a bug |
| Total under-count (Hough short / symbols missed) | 3 | 9.1 | **YES** — detection thresholds too strict |
| Null / crash | 0 | 0.0 | — |

Effective counter bug rate = **8/33 = 24%** (5 ordering + 3 under-count). Frame-mismatch cases need architectural decision, not a code fix.

## Bug 1 — Ordering inversion (5 cases, fixable in 30 min)

Total matches teacher exactly, but position is mirrored:

| case_id | predicate | teacher | opencv |
|---|---|---|---|
| AP_SK_107 | FILLS | 3/3 | 1/3 |
| AP_SK_118 | FILLS | 4/4 | 1/4 |
| AP_SK_329 | NEXT_TO | 9/10 | 10/10 |
| AP_SK_316 | NEXT_TO | 2/4 | 3/4 |
| AP_SK_325 | NEXT_TO | 2/4 | 3/4 |

**Root cause:** Counter sorts openings by projection along `wall_dir`, where `wall_dir = (p2 − p1) / |p2 − p1|`. Sign depends on which Hough endpoint is `p1` first — undefined. Teacher orders by the IFC wall's local X-axis, which has a consistent sign.

**Fix:** canonicalise — always set `wall_origin` = leftmost-px endpoint (smallest x, tiebreak smallest y). This still won't perfectly match the IFC local-X direction, but it provides a deterministic convention that aligns with how labels were likely rendered (matplotlib x-axis increases rightward in world coords → image px increases rightward).

## Bug 2 — Under-count (3 cases, fixable in 1h)

| case_id | teacher | opencv |
|---|---|---|
| AP_SK_217 | 14/14 | 11/11 |
| AP_SK_137 | 11/14 | 10/11 |
| AP_SK_228 | 13/14 | 11/11 |

**Root cause:** `_detect_openings_on_wall` filter thresholds (aspect ≥1.6, area ≥20, angle within 25°) drop 2-3 genuine openings. Full-storey renders use thinner lines than the original patches the thresholds were tuned for.

**Fix:** loosen area to ≥10 and aspect to ≥1.3; add diagnostic `dropped_candidates_count` in debug dict.

## Frame mismatch — 14 cases (architectural decision)

OpenCV's count is a strict superset of teacher's count because OpenCV's visible line spans multiple colinear IFC walls. Example pattern:

```
teacher_total=9    opencv_total=19    × 5 cases
teacher_total=3    opencv_total=11    × 2 cases
teacher_total=5    opencv_total=11    × 1 case
```

Both answers are internally self-consistent. The question is which one the retrieval pipeline wants.

**Retrieval filter consumes Neo4j's `wall_position_index` / `wall_child_total`**, which are the teacher frame. So for hard-filter use, OpenCV output needs conversion to teacher frame. Three options:

**Option F1 — IFC injection at inference**
- Pull host-wall endpoints from IFC/Neo4j, clip the Hough line to those endpoints
- Accurate but requires IFC query per inference
- Breaks "vision-only at inference" design goal

**Option F2 — Detect junctions in the image**
- Find corners / T-junctions → break long lines at junctions
- Pure vision, no IFC dependency
- More engineering; unclear whether junctions are visually distinguishable in the render style

**Option F3 — Accept the mismatch; counter output = "soft descriptor"**
- Feed `(line_position, line_total)` to the reranker as context, not as hard filter
- LoRA / rerank handle the frame conversion implicitly
- Counter contributes as weak signal universally; strong disambiguation only when walls happen to align with IFC walls (like the `17-opening` cases that all passed)

**My recommendation:** **F3 for Phase 6 / thesis,** **F1 as post-thesis work.** F3 keeps the Phase 6 slice shippable; F1 is a larger system change that belongs in a future retrieval upgrade.

## Projected impact of T1.2 fixes

Assuming F3 (no IFC injection):

| Fix | Target failures | Expected new OKs |
|-----|-----------------|------------------|
| Ordering canonicalisation | 5 | +5 → **16/33 = 48%** ✓ passes 40% gate |
| Loosen opening filters | 3 | +1-2 → **17-18/33 = 52-55%** |
| Combined | 8 | **17-18/33 = 52-55%** |

The 14 frame-mismatch cases will not become exact-match. But under F3, the counter still contributes as a **soft** signal in those cases (even if total is wrong, left/right neighbors and relative position can help the reranker).

## Next steps

1. **T1.2.a ordering canonicalisation** (30 min): set `wall_origin` = leftmost endpoint; re-run validator; expect 16/33
2. **T1.2.b loosen detection filters** (1h): re-run validator; expect 17-18/33
3. **T1.2.c soft-signal mode** (1h): when OpenCV confidence high but count likely conflicts with IFC frame (e.g. `total > teacher_typical_max`), emit as low-confidence to avoid poisoning hard filter
4. **T1.3 re-validation** — same validator run, should now pass 40% gate
5. **T1.4 fig03** — only after T1.3 passes gate, run injector on G8 precomputed, add `G8 + OpenCV` row

Frame-mismatch architectural decision (F1 vs F2 vs F3) pushed to T1.5 post-thesis.
