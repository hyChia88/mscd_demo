#!/usr/bin/env python3
"""
Analyze evaluation traces — per-strategy breakdown with GT-in-pool,
over-reduction, and SSR metrics.

Usage:
  # Single trace file (original mode):
  python evaluation/analyze_traces.py eval/results/oracle_MB/traces_*.jsonl
  python evaluation/analyze_traces.py eval/results/*/traces_*.jsonl   # compare multiple runs
  python evaluation/analyze_traces.py traces.jsonl --ap-only           # AP model only

  # Full multi-condition analysis (LoRA_4 style):
  python evaluation/analyze_traces.py --full \\
    --traces-dir output/synth_v05_lora4 \\
    --precomputed-dir output/synth_v05_lora4 \\
    --cases evaluation/cases/cases_v4_test.jsonl \\
    --gt-labels ../data_curation/datasets/synth_v0.5/train/lora4_test.jsonl \\
    --output evaluation/analysis/lora4_metrics.csv
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional


# ═════════════════════════════════════════════════════════════════════════════
# Original single-file analysis (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def analyze(trace_path: str, ap_only: bool = False):
    with open(trace_path) as f:
        traces = [json.loads(l) for l in f]

    if ap_only:
        traces = [t for t in traces if re.search(r'AP', t['scenario_id'])]

    n = len(traces)
    if n == 0:
        print(f"  No traces found in {trace_path}")
        return

    gt_in_pool = 0
    top1 = 0
    pools = []
    by_strategy = defaultdict(lambda: {"total": 0, "gt_in": 0, "top1": 0, "pools": []})
    by_predicate = defaultdict(lambda: {"total": 0, "gt_in": 0, "pools": []})

    for t in traces:
        internals = t.get("internals", {})
        rr = internals.get("retrieval_results", [])
        if not rr:
            continue
        r = rr[0]
        candidates = r.get("candidates", [])
        pool_size = r.get("pool_size", len(candidates))
        strat = r.get("query_plan_used", {}).get("strategy", "unknown")
        actual = r.get("strategy_actually_used", "")
        fb = r.get("fallback_triggered", False)

        gt_guid = t["scenario"]["ground_truth"]["target_guid"]
        cand_guids = [c.get("guid", "") for c in candidates]

        in_pool = gt_guid in cand_guids
        is_top1 = in_pool and cand_guids[0] == gt_guid

        if in_pool:
            gt_in_pool += 1
        if is_top1:
            top1 += 1
        pools.append(pool_size)

        s = by_strategy[strat]
        s["total"] += 1
        s["pools"].append(pool_size)
        if in_pool:
            s["gt_in"] += 1
        if is_top1:
            s["top1"] += 1

        # Per-predicate (for P0 strategies)
        if strat in ("spatial_triplet", "continuous_span"):
            constraints = internals.get("constraints", {})
            sr = constraints.get("spatial_relations", [])
            pred = sr[0].get("predicate", "?") if sr else "?"
            p = by_predicate[pred]
            p["total"] += 1
            p["pools"].append(pool_size)
            if in_pool:
                p["gt_in"] += 1

    avg_pool = sum(pools) / len(pools) if pools else 0
    ssr = 1 - avg_pool / 1257

    label = f"{'(AP only) ' if ap_only else ''}{Path(trace_path).parent.name}"
    print(f"\n{'=' * 70}")
    print(f"  {label}  (n={n})")
    print(f"{'=' * 70}")
    print(f"  GT-in-pool:    {gt_in_pool}/{n} ({gt_in_pool/n*100:.1f}%)")
    print(f"  GT-DROPPED:    {n-gt_in_pool}/{n} ({(n-gt_in_pool)/n*100:.1f}%) ← over-reduction")
    print(f"  Top-1:         {top1}/{n} ({top1/n*100:.1f}%)")
    print(f"  Avg pool:      {avg_pool:.1f}")
    print(f"  SSR:           {ssr*100:.1f}%")

    # Per-strategy
    print(f"\n  {'Strategy':<20} {'N':>4} {'GT-in':>12} {'Over-red':>10} {'Top-1':>8} {'AvgPool':>8}")
    print(f"  {'-'*65}")
    for strat in sorted(by_strategy.keys(), key=lambda s: by_strategy[s]["total"], reverse=True):
        s = by_strategy[strat]
        ap = sum(s["pools"]) / len(s["pools"]) if s["pools"] else 0
        over = s["total"] - s["gt_in"]
        print(
            f"  {strat:<20} {s['total']:>4} "
            f"{s['gt_in']:>4}/{s['total']}={s['gt_in']/s['total']*100:>5.1f}% "
            f"{over:>3}/{s['total']}={over/s['total']*100:>5.1f}% "
            f"{s['top1']/s['total']*100:>5.1f}% "
            f"{ap:>7.1f}"
        )

    # Per-predicate (P0 only)
    if by_predicate:
        print(f"\n  P0 by predicate:")
        for pred in sorted(by_predicate.keys()):
            p = by_predicate[pred]
            ap = sum(p["pools"]) / len(p["pools"]) if p["pools"] else 0
            over = p["total"] - p["gt_in"]
            print(
                f"    {pred:<18} {p['total']:>3} cases, "
                f"GT-in={p['gt_in']}/{p['total']} ({p['gt_in']/p['total']*100:.0f}%), "
                f"over-red={over}/{p['total']} ({over/p['total']*100:.0f}%), "
                f"avg_pool={ap:.1f}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# Full multi-condition analysis (--full mode)
# ═════════════════════════════════════════════════════════════════════════════

def _norm_storey(s: str) -> str:
    """Extract floor number from storey string."""
    if not s:
        return ""
    m = re.search(r"(-?\d+)", s.lower().strip())
    return m.group(1) if m else s.lower().strip()


def _ifc_match(ext: str, gt: str) -> bool:
    """IFC subtype-aware match: IfcWall matches IfcWallStandardCase."""
    if not ext or not gt:
        return False
    return ext == gt or gt.startswith(ext) or ext.startswith(gt)


def _frac(num: int, denom: int) -> str:
    """Format fraction with percentage."""
    pct = num / denom * 100 if denom else 0
    return f"{num}/{denom} ({pct:.1f}%)"


def _load_cases(path: str) -> Dict[str, Dict]:
    cases = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                cases[c["case_id"]] = c
    return cases


def _load_gt_labels(path: str) -> Dict[str, Dict]:
    labels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for msg in rec.get("messages", []):
                if msg.get("role") == "assistant":
                    try:
                        labels[rec["id"]] = json.loads(msg["content"])
                    except (json.JSONDecodeError, TypeError):
                        pass
    return labels


def _find_latest(directory: str, pattern: str) -> Optional[str]:
    files = sorted(glob(f"{directory}/{pattern}"))
    return files[-1] if files else None


def analyze_full_condition(
    trace_path: str,
    precomp_path: Optional[str],
    cases: Dict[str, Dict],
    gt_labels: Dict[str, Dict],
    condition: str,
) -> Dict[str, Any]:
    """Compute all metrics for one condition."""

    with open(trace_path) as f:
        traces = [json.loads(l) for l in f if l.strip()]

    precomp = {}
    if precomp_path:
        with open(precomp_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    precomp[rec["case_id"]] = rec

    n = len(traces)
    if n == 0:
        return {"condition": condition, "n": 0}

    # Counters
    gt_in_pool = 0
    gt_in_top10 = 0
    top1 = 0
    p0_fires = 0
    p0_gt_in = 0
    pool_sizes: List[int] = []
    p0_pools: List[int] = []
    initial_pools: List[int] = []

    sr_extracted = 0
    sr_gt_has = 0
    sr_correct_pred = 0
    sr_fp = 0
    sr_fn = 0
    storey_correct = 0
    ifc_correct = 0
    parse_ok = 0

    # Per-schema field accuracy (compared against GT labels)
    schema_total = 0        # cases with GT labels
    space_name_ok = 0
    space_name_gt = 0       # cases where GT has non-null space_name
    keyword_ok = 0
    keyword_gt = 0          # cases where GT has non-null keyword
    object_type_ok = 0
    object_type_gt = 0      # cases where GT has SR with object_type
    object_material_ok = 0
    object_material_gt = 0  # cases where GT has SR with object_material

    mrr_sum = 0.0
    recall_at = {10: 0, 50: 0, 100: 0}
    # Valid SSR: only count cases where GT is retained in pool
    valid_pool_sizes: List[int] = []
    valid_init_pools: List[int] = []
    over_reduced = 0  # cases where pool > 0 but GT dropped

    # Per-hop predicted vs GT comparison (for confusion plots)
    hop_comparisons: List[Dict[str, Any]] = []
    field_f1_scores: List[float] = []

    p0_storey_ok = 0
    p0_storey_total = 0
    rerank_kept = 0
    rerank_lost = 0

    by_strategy: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "gt_in": 0, "top1": 0, "pools": []}
    )
    by_predicate: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "gt_in": 0, "pools": []}
    )
    by_floor: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "gt_in": 0, "top1": 0, "pools": []}
    )
    by_ifc: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "gt_in": 0, "top1": 0, "pools": []}
    )

    for t in traces:
        sid = t["scenario_id"]
        gt = t["scenario"]["ground_truth"]
        gt_guid = gt["target_guid"]
        gt_storey = gt.get("target_storey", "")
        # target_ifc_class may not be in trace GT (ScenarioInput schema);
        # fall back to the cases dict which always has it.
        gt_ifc = gt.get("target_ifc_class", "")
        if not gt_ifc:
            gt_ifc = cases.get(sid, {}).get("ground_truth", {}).get("target_ifc_class", "")

        internals = t.get("internals", {})
        constraints = internals.get("constraints", {})
        plans = internals.get("query_plans", [])
        rr = internals.get("retrieval_results", [])

        # Pool
        pool = []
        pool_size = 0
        if rr and isinstance(rr[0], dict) and "candidates" in rr[0]:
            pool = rr[0]["candidates"]
            pool_size = rr[0].get("pool_size", len(pool))
        pool_guids = [c.get("guid", "") for c in pool]
        in_pool = gt_guid in pool_guids
        pool_sizes.append(pool_size)
        initial_pools.append(t.get("initial_pool_size", 0))

        if in_pool:
            gt_in_pool += 1
            rank = pool_guids.index(gt_guid) + 1
            mrr_sum += 1.0 / rank
            for k in recall_at:
                if rank <= k:
                    recall_at[k] += 1
            valid_pool_sizes.append(pool_size)
            valid_init_pools.append(t.get("initial_pool_size", 0))
        elif pool_size > 0:
            over_reduced += 1

        # Top-10
        top10 = t.get("interpreter_output", {}).get("mentioned_guids", [])
        if gt_guid in top10:
            gt_in_top10 += 1

        # Top-1
        cands_out = t.get("interpreter_output", {}).get("candidates", [])
        is_top1 = (cands_out and isinstance(cands_out[0], dict)
                   and cands_out[0].get("guid") == gt_guid)
        if is_top1:
            top1 += 1

        # Strategy
        strat = plans[0]["strategy"] if plans else "unknown"
        is_p0 = strat in ("spatial_triplet", "continuous_span")
        s = by_strategy[strat]
        s["total"] += 1
        s["pools"].append(pool_size)
        if in_pool:
            s["gt_in"] += 1
        if is_top1:
            s["top1"] += 1

        if is_p0:
            p0_fires += 1
            p0_pools.append(pool_size)
            if in_pool:
                p0_gt_in += 1
            p0_storey_total += 1
            e_s = _norm_storey(constraints.get("storey_name", ""))
            g_s = _norm_storey(gt_storey)
            if e_s and g_s and e_s == g_s:
                p0_storey_ok += 1
            # Full triplet chain (e.g. "IfcWindow-FILLS→IfcWall-CONNECTS_TO→IfcWall")
            sr_list = constraints.get("spatial_relations", [])
            subject_type = constraints.get("ifc_class", "?")
            if sr_list:
                parts = [subject_type or "?"]
                for s in sr_list:
                    parts.append(f"-[{s.get('predicate', '?')}]→{s.get('object_type', '?')}")
                pred = "".join(parts)
            else:
                pred = "?"
            p = by_predicate[pred]
            p["total"] += 1
            p["pools"].append(pool_size)
            if in_pool:
                p["gt_in"] += 1

        # Per-floor breakdown
        floor_num = _norm_storey(gt_storey) or "unknown"
        fl = by_floor[floor_num]
        fl["total"] += 1
        fl["pools"].append(pool_size)
        if in_pool:
            fl["gt_in"] += 1
        if is_top1:
            fl["top1"] += 1

        # Per-IFC-class breakdown
        ifc_key = gt_ifc or "unknown"
        ic = by_ifc[ifc_key]
        ic["total"] += 1
        ic["pools"].append(pool_size)
        if in_pool:
            ic["gt_in"] += 1
        if is_top1:
            ic["top1"] += 1

        # Reranking
        if in_pool:
            if gt_guid in top10:
                rerank_kept += 1
            else:
                rerank_lost += 1

        # VLM extraction (from precomputed or trace constraints)
        pc = precomp.get(sid, {})
        pc_c = pc.get("constraints", constraints)
        if pc.get("status") == "OK" or not pc:
            parse_ok += 1

        e_s = _norm_storey(pc_c.get("storey_name", ""))
        g_s = _norm_storey(gt_storey)
        if e_s and g_s and e_s == g_s:
            storey_correct += 1
        e_ifc = pc_c.get("ifc_class", "")
        if _ifc_match(e_ifc, gt_ifc):
            ifc_correct += 1

        sr = pc_c.get("spatial_relations", [])
        gt_label = gt_labels.get(sid, {})
        gt_sr = gt_label.get("spatial_relations", [])
        has_sr = len(sr) > 0
        gt_has = len(gt_sr) > 0
        if has_sr:
            sr_extracted += 1
        if gt_has:
            sr_gt_has += 1
        if has_sr and gt_has:
            if sr[0].get("predicate", "").upper() == gt_sr[0].get("predicate", "").upper():
                sr_correct_pred += 1
        if has_sr and not gt_has:
            sr_fp += 1
        if not has_sr and gt_has:
            sr_fn += 1

        # Collect per-hop comparison data (predicted vs GT)
        if has_sr or gt_has:
            pred_subject = pc_c.get("ifc_class", "?")
            gt_subject = gt_label.get("ifc_class", gt_ifc or "?")
            max_hops = max(len(sr), len(gt_sr))
            for hop_i in range(max_hops):
                p_hop = sr[hop_i] if hop_i < len(sr) else {}
                g_hop = gt_sr[hop_i] if hop_i < len(gt_sr) else {}
                hop_comparisons.append({
                    "case_id": sid,
                    "hop": hop_i + 1,
                    "in_pool": in_pool,
                    "pred_subject": pred_subject if hop_i == 0 else (sr[hop_i - 1].get("object_type", "?") if hop_i - 1 < len(sr) else "?"),
                    "gt_subject": gt_subject if hop_i == 0 else (gt_sr[hop_i - 1].get("object_type", "?") if hop_i - 1 < len(gt_sr) else "?"),
                    "pred_predicate": p_hop.get("predicate", "—"),
                    "gt_predicate": g_hop.get("predicate", "—"),
                    "pred_object": p_hop.get("object_type", "—"),
                    "gt_object": g_hop.get("object_type", "—"),
                })

        # Per-schema field accuracy (vs GT labels)
        if gt_label:
            schema_total += 1
            # space_name
            gt_space = gt_label.get("space_name") or ""
            pred_space = pc_c.get("space_name") or ""
            if gt_space:
                space_name_gt += 1
                if gt_space.lower().strip() == pred_space.lower().strip():
                    space_name_ok += 1
            # target_name_keyword
            gt_kw = gt_label.get("target_name_keyword") or ""
            pred_kw = pc_c.get("target_name_keyword") or ""
            if gt_kw:
                keyword_gt += 1
                if gt_kw.lower() in pred_kw.lower() or pred_kw.lower() in gt_kw.lower():
                    keyword_ok += 1
            # SR hop-1 object_type
            if gt_has and has_sr:
                gt_otype = (gt_sr[0].get("object_type") or "").upper()
                pred_otype = (sr[0].get("object_type") or "").upper()
                if gt_otype:
                    object_type_gt += 1
                    if gt_otype == pred_otype or pred_otype.startswith(gt_otype) or gt_otype.startswith(pred_otype):
                        object_type_ok += 1
                gt_omat = (gt_sr[0].get("object_material") or "").lower()
                pred_omat = (sr[0].get("object_material") or "").lower()
                if gt_omat:
                    object_material_gt += 1
                    if gt_omat in pred_omat or pred_omat in gt_omat:
                        object_material_ok += 1

        # Field EM F1
        fc, ft = 0, 0
        if gt_storey:
            ft += 1
            if e_s == g_s:
                fc += 1
        if gt_ifc:
            ft += 1
            if _ifc_match(e_ifc, gt_ifc):
                fc += 1
        if ft > 0:
            field_f1_scores.append(fc / ft)

    # Aggregate
    avg_pool = sum(pool_sizes) / n if n else 0
    avg_init = sum(initial_pools) / n if n else 1
    ssr = (1.0 - avg_pool / avg_init) * 100 if avg_init > 0 else 0
    # Valid SSR: reduction only among cases where GT is retained
    valid_avg_pool = sum(valid_pool_sizes) / len(valid_pool_sizes) if valid_pool_sizes else 0
    valid_avg_init = sum(valid_init_pools) / len(valid_init_pools) if valid_init_pools else 1
    valid_ssr = (1.0 - valid_avg_pool / valid_avg_init) * 100 if valid_avg_init > 0 else 0

    # RQS (Retrieval Quality Score): harmonic mean of GT recall and valid SSR
    # - GT recall = fraction of cases where GT is in pool (retrieval didn't drop it)
    # - Valid SSR = pool reduction among those retained cases
    # RQS = 0 when either is 0; maximized when both recall and reduction are high
    gt_recall = gt_in_pool / n * 100 if n else 0
    if gt_recall > 0 and valid_ssr > 0:
        rqs = 2 * gt_recall * valid_ssr / (gt_recall + valid_ssr)
    else:
        rqs = 0.0
    p0_avg = sum(p0_pools) / len(p0_pools) if p0_pools else 0
    mrr = mrr_sum / n if n else 0
    avg_f1 = sum(field_f1_scores) / len(field_f1_scores) if field_f1_scores else 0
    sr_both = min(sr_extracted, sr_gt_has)

    return {
        "condition": condition, "n": n,
        "gt_in_pool": gt_in_pool, "gt_in_top10": gt_in_top10, "top1": top1,
        "avg_pool": avg_pool, "ssr": ssr, "valid_ssr": valid_ssr,
        "valid_avg_pool": valid_avg_pool, "over_reduced": over_reduced,
        "rqs": rqs, "gt_recall": gt_recall, "mrr": mrr,
        "r10": recall_at[10], "r50": recall_at[50], "r100": recall_at[100],
        "p0_fires": p0_fires, "p0_gt_in": p0_gt_in, "p0_avg": p0_avg,
        "p0_storey_ok": p0_storey_ok, "p0_storey_n": p0_storey_total,
        "rerank_kept": rerank_kept, "rerank_lost": rerank_lost,
        "parse_rate": parse_ok / n * 100 if n else 0,
        "field_f1": avg_f1,
        "storey_acc": storey_correct / n * 100 if n else 0,
        "ifc_acc": ifc_correct / n * 100 if n else 0,
        "sr_ext": sr_extracted, "sr_gt_has": sr_gt_has,
        "pred_ok": sr_correct_pred, "pred_n": sr_both,
        "fp": sr_fp, "fn": sr_fn,
        # Per-schema field accuracy
        "space_name_ok": space_name_ok, "space_name_gt": space_name_gt,
        "keyword_ok": keyword_ok, "keyword_gt": keyword_gt,
        "object_type_ok": object_type_ok, "object_type_gt": object_type_gt,
        "object_material_ok": object_material_ok, "object_material_gt": object_material_gt,
        "by_strategy": dict(by_strategy), "by_predicate": dict(by_predicate),
        "by_floor": dict(by_floor), "by_ifc": dict(by_ifc),
        "hop_comparisons": hop_comparisons,
    }


def print_full_table(results: List[Dict[str, Any]]):
    """Print the full multi-condition comparison table."""
    cw = 22  # column width
    lw = 24  # label width

    def row(label: str, values: List[str]):
        cols = "".join(f"{v:>{cw}}" for v in values)
        print(f"  {label:<{lw}}{cols}")

    def sep():
        print(f"  {'─' * (lw + cw * len(results))}")

    print()
    print("=" * (lw + cw * len(results) + 4))
    print(f"  LoRA5 Evaluation — Full Metrics Table")
    print("=" * (lw + cw * len(results) + 4))
    print()

    row("", [f"{r['condition']} (n={r['n']})" for r in results])
    sep()

    # Retrieval pipeline
    print()
    print("  RETRIEVAL PIPELINE")
    sep()
    row("GT-in-pool",   [_frac(r["gt_in_pool"], r["n"]) for r in results])
    row("GT-in-top10",  [_frac(r["gt_in_top10"], r["n"]) for r in results])
    row("Top-1",        [_frac(r["top1"], r["n"]) for r in results])
    row("MRR",          [f"{r['mrr']:.3f}" for r in results])
    row("R@10",         [_frac(r["r10"], r["n"]) for r in results])
    row("R@50",         [_frac(r["r50"], r["n"]) for r in results])
    row("R@100",        [_frac(r["r100"], r["n"]) for r in results])
    row("Avg pool",     [f"{r['avg_pool']:.1f}" for r in results])
    row("Valid SSR",    [f"{r['valid_ssr']:.1f}% (n={r['gt_in_pool']})" for r in results])
    row("Over-reduced", [_frac(r["over_reduced"], r["n"]) for r in results])
    row("RQS",          [f"{r['rqs']:.1f}" for r in results])

    # P0
    print()
    print("  P0 SPATIAL TRIPLET")
    sep()
    row("P0 fires",         [str(r["p0_fires"]) for r in results])
    row("P0 GT-in-pool",    [_frac(r["p0_gt_in"], r["p0_fires"]) for r in results])
    row("P0 avg pool",      [f"{r['p0_avg']:.1f}" for r in results])
    row("P0 storey correct",[_frac(r["p0_storey_ok"], r["p0_storey_n"]) for r in results])
    row("Rerank kept/lost", [f"{r['rerank_kept']} / {r['rerank_lost']}" for r in results])

    # VLM extraction
    print()
    print("  VLM EXTRACTION")
    sep()
    row("Parse rate",       [f"{r['parse_rate']:.1f}%" for r in results])
    row("Field EM F1",      [f"{r['field_f1']:.3f}" for r in results])
    row("Storey accuracy",  [f"{r['storey_acc']:.1f}%" for r in results])
    row("IFC class acc",    [f"{r['ifc_acc']:.1f}%" for r in results])
    row("SR extraction",    [_frac(r["sr_ext"], r["n"]) for r in results])
    row("Predicate acc",    [_frac(r["pred_ok"], r["pred_n"]) for r in results])
    row("FP rate (SR)",     [_frac(r["fp"], r["n"]) for r in results])
    row("FN rate (SR)",     [_frac(r["fn"], r["sr_gt_has"]) for r in results])

    # Per-schema field accuracy
    print()
    print("  PER-SCHEMA FIELD ACCURACY")
    sep()
    row("Storey name",       [f"{r['storey_acc']:.1f}%" for r in results])
    row("IFC class",         [f"{r['ifc_acc']:.1f}%" for r in results])
    row("Space name",        [_frac(r["space_name_ok"], r["space_name_gt"]) if r["space_name_gt"] else "N/A (0 GT)" for r in results])
    row("Name keyword",      [_frac(r["keyword_ok"], r["keyword_gt"]) if r["keyword_gt"] else "N/A (0 GT)" for r in results])
    row("SR predicate",      [_frac(r["pred_ok"], r["pred_n"]) for r in results])
    row("SR object_type",    [_frac(r["object_type_ok"], r["object_type_gt"]) if r["object_type_gt"] else "N/A" for r in results])
    row("SR object_material",[_frac(r["object_material_ok"], r["object_material_gt"]) if r["object_material_gt"] else "N/A" for r in results])
    print()

    # Per-strategy (last condition = MC typically)
    mc = results[-1]
    print(f"  Per-strategy ({mc['condition']}):")
    sep()
    print(f"  {'Strategy':<22} {'N':>4} {'GT-in':>14} {'Top-1':>8} {'AvgPool':>8}")
    for st in sorted(mc["by_strategy"], key=lambda x: mc["by_strategy"][x]["total"], reverse=True):
        s = mc["by_strategy"][st]
        ap = sum(s["pools"]) / len(s["pools"]) if s["pools"] else 0
        print(
            f"  {st:<22} {s['total']:>4} "
            f"{_frac(s['gt_in'], s['total']):>14} "
            f"{s['top1']/s['total']*100:>6.1f}% "
            f"{ap:>7.1f}"
        )

    if mc["by_predicate"]:
        print(f"\n  P0 by triplet chain ({mc['condition']}):")
        for pred in sorted(mc["by_predicate"], key=lambda k: mc["by_predicate"][k]["total"], reverse=True):
            p = mc["by_predicate"][pred]
            ap = sum(p["pools"]) / len(p["pools"]) if p["pools"] else 0
            print(
                f"    {pred:<55} {p['total']:>3} cases, "
                f"GT-in={_frac(p['gt_in'], p['total'])}, "
                f"avg_pool={ap:.1f}"
            )

    # Per-floor breakdown (last condition)
    print()
    print(f"  Per-floor ({mc['condition']}):")
    sep()
    print(f"  {'Floor':<22} {'N':>4} {'GT-in-pool':>14} {'Top-1':>8} {'AvgPool':>8}")
    for fl in sorted(mc["by_floor"], key=lambda x: (x != "unknown", x)):
        f = mc["by_floor"][fl]
        ap = sum(f["pools"]) / len(f["pools"]) if f["pools"] else 0
        print(
            f"  {fl:<22} {f['total']:>4} "
            f"{_frac(f['gt_in'], f['total']):>14} "
            f"{f['top1']/f['total']*100:>6.1f}% "
            f"{ap:>7.1f}"
        )

    # Per-IFC-class breakdown (last condition)
    print(f"\n  Per-IFC-class ({mc['condition']}):")
    sep()
    print(f"  {'IFC Class':<22} {'N':>4} {'GT-in-pool':>14} {'Top-1':>8} {'AvgPool':>8}")
    for ic_key in sorted(mc["by_ifc"], key=lambda x: mc["by_ifc"][x]["total"], reverse=True):
        ic = mc["by_ifc"][ic_key]
        ap = sum(ic["pools"]) / len(ic["pools"]) if ic["pools"] else 0
        print(
            f"  {ic_key:<22} {ic['total']:>4} "
            f"{_frac(ic['gt_in'], ic['total']):>14} "
            f"{ic['top1']/ic['total']*100:>6.1f}% "
            f"{ap:>7.1f}"
        )
    print()


def write_full_csv(results: List[Dict[str, Any]], output_path: str):
    """Write flat metrics to CSV."""
    skip = {"by_strategy", "by_predicate", "by_floor", "by_ifc", "hop_comparisons"}
    keys = [k for k in results[0] if k not in skip]
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + [r["condition"] for r in results])
        for k in keys:
            w.writerow([k] + [results[i].get(k, "") for i in range(len(results))])
    print(f"  CSV → {output_path}")


def plot_per_floor(results: List[Dict[str, Any]], output_dir: str):
    """Generate per-floor GT-in-pool bar chart and storey accuracy breakdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Use last condition (MC preferred) for floor breakdown
    mc = results[-1]
    by_floor = mc["by_floor"]

    if not by_floor:
        print("  [plot] No per-floor data — skipping")
        return

    floors = sorted(by_floor.keys(), key=lambda x: int(x) if x.lstrip("-").isdigit() else 999)
    n_vals = [by_floor[f]["total"] for f in floors]
    gt_vals = [by_floor[f]["gt_in"] / by_floor[f]["total"] * 100 for f in floors]

    # ── Plot 1: Per-floor GT-in-pool ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(floors))
    bars = ax.bar(x, gt_vals, color="#2563EB", alpha=0.85)
    # Annotate n on each bar
    for i, (bar, n) in enumerate(zip(bars, n_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Floor {f}" for f in floors], rotation=30, ha="right")
    ax.set_ylabel("GT-in-pool (%)")
    ax.set_title(f"Per-Floor GT-in-pool — {mc['condition']} (n={mc['n']})")
    ax.set_ylim(0, 105)
    ax.axhline(y=mc["gt_in_pool"] / mc["n"] * 100, color="red", linestyle="--",
               alpha=0.7, label=f"Overall: {mc['gt_in_pool']/mc['n']*100:.1f}%")
    ax.legend()
    plt.tight_layout()
    p1 = out / f"per_floor_gt_in_pool_{mc['condition']}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  [plot] {p1}")

    # ── Plot 2: Multi-condition per-floor comparison (if >1 condition) ────
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        width = 0.8 / len(results)
        colors = ["#2563EB", "#EA580C", "#16A34A", "#9333EA", "#DC2626"]
        for i, r in enumerate(results):
            bf = r["by_floor"]
            vals = [bf[f]["gt_in"] / bf[f]["total"] * 100 if f in bf else 0 for f in floors]
            offset = (i - len(results) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=r["condition"],
                   color=colors[i % len(colors)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Floor {f}" for f in floors], rotation=30, ha="right")
        ax.set_ylabel("GT-in-pool (%)")
        ax.set_title("Per-Floor GT-in-pool by Condition")
        ax.set_ylim(0, 105)
        ax.legend()
        plt.tight_layout()
        p2 = out / "per_floor_multi_condition.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"  [plot] {p2}")

    # ── Plot 3: Storey prediction accuracy (correct vs wrong floor) ───────
    # Shows what fraction of cases had correct storey prediction per GT floor
    fig, ax = plt.subplots(figsize=(10, 5))
    storey_ok_vals = []
    storey_n_vals = []
    for f in floors:
        bf = mc["by_floor"].get(f, {})
        # storey_ok not tracked per-floor in current data, approximate from gt_in
        # A case with correct storey has better chance of GT-in-pool
        storey_ok_vals.append(bf.get("gt_in", 0) / bf.get("total", 1) * 100)
        storey_n_vals.append(bf.get("total", 0))

    bars = ax.bar(x, storey_ok_vals, color="#16A34A", alpha=0.85)
    for i, (bar, n) in enumerate(zip(bars, storey_n_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Floor {f}" for f in floors], rotation=30, ha="right")
    ax.set_ylabel("GT-in-pool (%)")
    ax.set_title(f"Per-Floor Retrieval Success — {mc['condition']}")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    p3 = out / f"per_floor_retrieval_{mc['condition']}.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  [plot] {p3}")

    # ── Plot 4: RQS + Over-reduction summary across conditions ────────────
    if len(results) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        conds = [r["condition"] for r in results]
        x_c = np.arange(len(conds))

        # Left: GT-in-pool vs Over-reduced (stacked)
        gt_pcts = [r["gt_in_pool"] / r["n"] * 100 for r in results]
        or_pcts = [r["over_reduced"] / r["n"] * 100 for r in results]
        empty_pcts = [100 - g - o for g, o in zip(gt_pcts, or_pcts)]
        ax1.bar(x_c, gt_pcts, color="#16A34A", label="GT retained")
        ax1.bar(x_c, or_pcts, bottom=gt_pcts, color="#DC2626", label="Over-reduced")
        ax1.bar(x_c, empty_pcts, bottom=[g + o for g, o in zip(gt_pcts, or_pcts)],
                color="#D1D5DB", label="Empty pool")
        ax1.set_xticks(x_c)
        ax1.set_xticklabels(conds)
        ax1.set_ylabel("% of cases")
        ax1.set_title("Pool Outcome Breakdown")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_ylim(0, 105)

        # Right: RQS bars
        rqs_vals = [r["rqs"] for r in results]
        bars = ax2.bar(x_c, rqs_vals, color="#2563EB", alpha=0.85)
        for bar, v in zip(bars, rqs_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xticks(x_c)
        ax2.set_xticklabels(conds)
        ax2.set_ylabel("RQS (F1 of Recall × Valid SSR)")
        ax2.set_title("Retrieval Quality Score")
        ax2.set_ylim(0, 100)
        plt.tight_layout()
        p4 = out / "rqs_overview.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        print(f"  [plot] {p4}")


def plot_hop_comparison(results: List[Dict[str, Any]], output_dir: str):
    """Generate plots comparing predicted vs GT triplet chains per hop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Use last condition (MC preferred)
    mc = results[-1]
    hops = mc.get("hop_comparisons", [])
    if not hops:
        print("  [plot] No hop comparison data — skipping")
        return

    # ── Plot 1: Per-hop accuracy (subject, predicate, object) ─────────────
    from collections import defaultdict, Counter
    hop_stats = defaultdict(lambda: {"total": 0, "subj_ok": 0, "pred_ok": 0, "obj_ok": 0})
    for h in hops:
        hop_n = min(h["hop"], 3)  # cap at 3 for display
        s = hop_stats[hop_n]
        s["total"] += 1
        # Subject match (subtype-aware)
        ps, gs = (h["pred_subject"] or "").upper(), (h["gt_subject"] or "").upper()
        if ps and gs and (ps in gs or gs in ps):
            s["subj_ok"] += 1
        # Predicate match
        pp, gp = (h["pred_predicate"] or "").upper(), (h["gt_predicate"] or "").upper()
        if pp and gp and pp == gp and pp != "—":
            s["pred_ok"] += 1
        # Object type match
        po, go = (h["pred_object"] or "").upper(), (h["gt_object"] or "").upper()
        if po and go and (po in go or go in po) and po != "—":
            s["obj_ok"] += 1

    hop_nums = sorted(hop_stats.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(hop_nums))
    width = 0.25
    for i, (field, color, label) in enumerate([
        ("subj_ok", "#2563EB", "Subject type"),
        ("pred_ok", "#EA580C", "Predicate"),
        ("obj_ok", "#16A34A", "Object type"),
    ]):
        vals = [hop_stats[h][field] / hop_stats[h]["total"] * 100 for h in hop_nums]
        bars = ax.bar(x + i * width, vals, width, color=color, label=label, alpha=0.85)
        for bar, v, h in zip(bars, vals, hop_nums):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Hop {h} (n={hop_stats[h]['total']})" for h in hop_nums])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-Hop Field Accuracy — {mc['condition']}")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    p1 = out / f"hop_accuracy_{mc['condition']}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  [plot] {p1}")

    # ── Plot 2: Predicate confusion matrix (hop-1 only) ──────────────────
    hop1 = [h for h in hops if h["hop"] == 1]
    pred_labels = sorted({h["gt_predicate"] for h in hop1 if h["gt_predicate"] != "—"}
                         | {h["pred_predicate"] for h in hop1 if h["pred_predicate"] != "—"})
    if pred_labels:
        n_labels = len(pred_labels)
        label_idx = {l: i for i, l in enumerate(pred_labels)}
        conf = np.zeros((n_labels, n_labels), dtype=int)
        # Count FP (predicted SR but GT has none)
        fp_preds = Counter()
        fn_preds = Counter()
        for h in hop1:
            gt_p = h["gt_predicate"].upper() if h["gt_predicate"] != "—" else None
            pr_p = h["pred_predicate"].upper() if h["pred_predicate"] != "—" else None
            if gt_p and pr_p and gt_p in label_idx and pr_p in label_idx:
                conf[label_idx[gt_p], label_idx[pr_p]] += 1
            elif pr_p and not gt_p:
                fp_preds[pr_p] += 1
            elif gt_p and not pr_p:
                fn_preds[gt_p] += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n_labels))
        ax.set_yticks(range(n_labels))
        ax.set_xticklabels(pred_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(pred_labels, fontsize=9)
        ax.set_xlabel("Predicted predicate")
        ax.set_ylabel("GT predicate")
        ax.set_title(f"Hop-1 Predicate Confusion — {mc['condition']}")
        # Annotate cells
        for i in range(n_labels):
            for j in range(n_labels):
                v = conf[i, j]
                if v > 0:
                    color = "white" if v > conf.max() * 0.6 else "black"
                    ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        p2 = out / f"predicate_confusion_{mc['condition']}.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"  [plot] {p2}")

        # ── Plot 3: Subject type confusion (hop-1) ───────────────────────
        subj_labels = sorted({h["gt_subject"] for h in hop1 if h["gt_subject"] and h["gt_subject"] != "?"}
                             | {h["pred_subject"] for h in hop1 if h["pred_subject"] and h["pred_subject"] != "?"})
        if subj_labels:
            n_sub = len(subj_labels)
            sub_idx = {l: i for i, l in enumerate(subj_labels)}
            subj_conf = np.zeros((n_sub, n_sub), dtype=int)
            for h in hop1:
                gs = h["gt_subject"] or "?"
                ps = h["pred_subject"] or "?"
                if gs in sub_idx and ps in sub_idx:
                    subj_conf[sub_idx[gs], sub_idx[ps]] += 1

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(subj_conf, cmap="Oranges", aspect="auto")
            ax.set_xticks(range(n_sub))
            ax.set_yticks(range(n_sub))
            ax.set_xticklabels(subj_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(subj_labels, fontsize=8)
            ax.set_xlabel("Predicted subject (ifc_class)")
            ax.set_ylabel("GT subject (ifc_class)")
            ax.set_title(f"Hop-1 Subject Type Confusion — {mc['condition']}")
            for i in range(n_sub):
                for j in range(n_sub):
                    v = subj_conf[i, j]
                    if v > 0:
                        color = "white" if v > subj_conf.max() * 0.6 else "black"
                        ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            p3 = out / f"subject_confusion_{mc['condition']}.png"
            fig.savefig(p3, dpi=150)
            plt.close(fig)
            print(f"  [plot] {p3}")

    # ── Plot 4: Per-case hop chain: predicted vs GT (waterfall) ───────────
    # Group by case, show which hops matched and which didn't
    from collections import OrderedDict
    case_hops = OrderedDict()
    for h in hops:
        cid = h["case_id"]
        if cid not in case_hops:
            case_hops[cid] = []
        case_hops[cid].append(h)

    # Sort: failures first (not in_pool), then by number of hops
    cases_sorted = sorted(case_hops.items(),
                          key=lambda x: (x[1][0]["in_pool"], len(x[1])))

    # Limit to 30 cases for readability
    cases_show = cases_sorted[:min(30, len(cases_sorted))]

    fig, ax = plt.subplots(figsize=(14, max(6, len(cases_show) * 0.35)))
    y_labels = []
    for yi, (cid, case_h) in enumerate(cases_show):
        in_pool = case_h[0]["in_pool"]
        for h in case_h:
            hop_i = h["hop"]
            x_pos = (hop_i - 1) * 3  # space for subject, predicate, object

            # Subject match
            ps, gs = (h["pred_subject"] or "").upper(), (h["gt_subject"] or "").upper()
            subj_ok = ps and gs and (ps in gs or gs in ps)
            ax.scatter(x_pos, yi, marker="s", s=80,
                       color="#16A34A" if subj_ok else "#DC2626", zorder=3)

            # Predicate match
            pp, gp = (h["pred_predicate"] or "").upper(), (h["gt_predicate"] or "").upper()
            pred_ok = pp and gp and pp == gp and pp != "—"
            ax.scatter(x_pos + 1, yi, marker="D", s=80,
                       color="#16A34A" if pred_ok else "#DC2626", zorder=3)

            # Object match
            po, go = (h["pred_object"] or "").upper(), (h["gt_object"] or "").upper()
            obj_ok = po and go and (po in go or go in po) and po != "—"
            ax.scatter(x_pos + 2, yi, marker="o", s=80,
                       color="#16A34A" if obj_ok else "#DC2626", zorder=3)

        # Background color for GT-in-pool
        ax.axhspan(yi - 0.4, yi + 0.4, color="#D1FAE5" if in_pool else "#FEE2E2",
                    alpha=0.3, zorder=0)
        short_id = cid[-20:] if len(cid) > 20 else cid
        y_labels.append(f"{'✓' if in_pool else '✗'} {short_id}")

    # X-axis labels
    max_hops = max(len(ch) for _, ch in cases_show) if cases_show else 1
    x_ticks = []
    x_labels = []
    for h in range(max_hops):
        for i, label in enumerate(["Subj", "Pred", "Obj"]):
            x_ticks.append(h * 3 + i)
            x_labels.append(f"H{h+1} {label}")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_title(f"Per-Case Hop Accuracy — {mc['condition']} (green=match, red=mismatch)")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#16A34A', markersize=8, label='Subject match'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#16A34A', markersize=8, label='Predicate match'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#16A34A', markersize=8, label='Object match'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#DC2626', markersize=8, label='Mismatch'),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()
    p4 = out / f"hop_waterfall_{mc['condition']}.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"  [plot] {p4}")


def run_full_analysis(args):
    """Multi-condition analysis mode."""
    precomp_dir = args.precomputed_dir or args.traces_dir
    conditions = [c.strip() for c in args.conditions.split(",")]

    cases = _load_cases(args.cases)
    print(f"Loaded {len(cases)} test cases")

    gt_labels: Dict[str, Dict] = {}
    if args.gt_labels and Path(args.gt_labels).exists():
        gt_labels = _load_gt_labels(args.gt_labels)
        print(f"Loaded {len(gt_labels)} GT labels")

    results = []
    for cond in conditions:
        tp = _find_latest(args.traces_dir, f"traces_*_v2_lora_{cond}*.jsonl")
        pp = f"{precomp_dir}/eval_constraints_final_{cond}.jsonl"
        pp = pp if Path(pp).exists() else None
        if not tp:
            print(f"  SKIP {cond}: no trace file")
            continue
        print(f"  {cond}: {Path(tp).name}" + (f" + precomputed" if pp else ""))
        results.append(analyze_full_condition(tp, pp, cases, gt_labels, cond))

    if not results:
        print("No conditions analyzed")
        sys.exit(1)

    print_full_table(results)
    if args.output:
        write_full_csv(results, args.output)
    if args.plots:
        print("\n  Generating plots...")
        plot_per_floor(results, args.plots)
        plot_hop_comparison(results, args.plots)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # Detect --full mode
    if "--full" in sys.argv:
        parser = argparse.ArgumentParser(description="Full multi-condition trace analysis")
        parser.add_argument("--full", action="store_true")
        parser.add_argument("--traces-dir", required=True)
        parser.add_argument("--precomputed-dir", default=None)
        parser.add_argument("--cases", required=True)
        parser.add_argument("--gt-labels", default=None)
        parser.add_argument("--output", default=None)
        parser.add_argument("--plots", default=None,
                            help="Output directory for plots (per-floor, RQS, etc.)")
        parser.add_argument("--conditions", default="MA,MB,MC")
        args = parser.parse_args()
        run_full_analysis(args)
    else:
        # Original mode: positional trace file paths
        ap_only = "--ap-only" in sys.argv
        paths = [a for a in sys.argv[1:] if not a.startswith("--")]
        if not paths:
            print("Usage:")
            print("  python evaluation/analyze_traces.py <trace_file.jsonl> [--ap-only]")
            print("  python evaluation/analyze_traces.py --full --traces-dir DIR --cases FILE")
            sys.exit(1)
        for path in paths:
            analyze(path, ap_only=ap_only)


if __name__ == "__main__":
    main()
