#!/usr/bin/env python3
"""
Unified Evaluation Runner

One entry-point for v1 (agent-driven) and v2 (constraints-driven) pipelines.
Driven by profiles.yaml — each profile defines pipeline type, retrieval mode,
constraints model, CLIP usage, RQ2, BCF, etc.

Usage:
  python script/run.py --profile v2_prompt \\
    --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl

  python script/run.py --profile v1_baseline \\
    --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl \\
    --condition A2

  python script/run.py --profile best_v2 \\
    --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl \\
    --adapter_path models/qwen3-vl-8b-lora/checkpoint-1000
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.contracts import EvalTrace
from src.eval.metrics import compute_summary
from src.v2.metrics_v2 import compute_v2_metrics, compute_v2_summary
from src.v2.types import V2Trace


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_cases_jsonl(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def write_jsonl(traces: List[EvalTrace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(t.to_jsonl_line() + "\n")


def write_csv_summary(
    v1_summary: Any,
    v2_summary: Optional[Dict[str, Any]],
    v2_per_case: Optional[List[Dict[str, Any]]],
    path: Path,
) -> None:
    """Write combined v1 + v2 metrics CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # ── Section 1: overall v1 metrics ──
        w.writerow(["=== OVERALL METRICS ==="])
        w.writerow(["Metric", "Value"])
        w.writerow(["Total Scenarios", v1_summary.total_scenarios])
        w.writerow(["Successful Runs", v1_summary.successful_runs])
        w.writerow(["Top-1 Accuracy", f"{v1_summary.top1_accuracy:.4f}"])
        w.writerow(["Top-K Accuracy", f"{v1_summary.topk_accuracy:.4f}"])
        w.writerow(["Avg Search-Space Reduction", f"{v1_summary.avg_search_space_reduction:.4f}"])
        w.writerow(["Escalation Rate", f"{v1_summary.escalation_rate:.4f}"])
        w.writerow(["Avg Latency (ms)", f"{v1_summary.avg_latency_ms:.1f}"])

        # RQ2 block
        if v1_summary.rq2_total > 0:
            w.writerow(["RQ2 Total", v1_summary.rq2_total])
            w.writerow(["RQ2 Pass Rate", f"{v1_summary.rq2_validation_pass_rate:.4f}"])
            w.writerow(["RQ2 Avg Fill Rate", f"{v1_summary.rq2_avg_fill_rate:.4f}"])

        w.writerow([])

        # ── Section 2: v2 diagnostic metrics ──
        if v2_summary:
            w.writerow(["=== V2 DIAGNOSTIC METRICS ==="])
            w.writerow(["Metric", "Value"])
            w.writerow(["Constraints Parse Rate", f"{v2_summary.get('constraints_parse_rate', 0):.4f}"])
            avg_rg = v2_summary.get("avg_rerank_gain")
            w.writerow(["Avg Rerank Gain", f"{avg_rg:.4f}" if avg_rg is not None else "N/A"])
            w.writerow(["Avg Constraints Extraction (ms)", f"{v2_summary.get('avg_constraints_extraction_ms', 0):.1f}"])
            w.writerow(["Avg Query Planning (ms)", f"{v2_summary.get('avg_query_planning_ms', 0):.1f}"])
            w.writerow(["Avg Retrieval (ms)", f"{v2_summary.get('avg_retrieval_ms', 0):.1f}"])
            w.writerow([])

        # ── Section 3: per-case v2 detail ──
        if v2_per_case:
            w.writerow(["=== PER-CASE V2 DETAIL ==="])
            headers = [
                "case_id", "constraints_parsed",
                "constraints_field_em_f1", "rerank_gain",
            ]
            w.writerow(headers)
            for row in v2_per_case:
                w.writerow([
                    row.get("case_id", ""),
                    row.get("constraints_parsed", ""),
                    _fmt(row.get("constraints_field_em_f1")),
                    _fmt(row.get("rerank_gain")),
                ])


def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_engine(config: Dict[str, Any]):
    """Return an IFCEngine (v1 component, reused)."""
    from src.ifc_engine import IFCEngine

    ifc_path = config.get("ifc", {}).get("model_path", "")
    return IFCEngine(ifc_path)


def init_llm(config: Dict[str, Any]):
    """Return a LangChain LLM from config."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm_cfg = config.get("llm", {})
    return ChatGoogleGenerativeAI(
        model=llm_cfg.get("model", "gemini-2.5-flash"),
        temperature=llm_cfg.get("temperature", 0),
        max_retries=llm_cfg.get("max_retries", 2),
    )


def init_visual_aligner(use_clip: bool):
    """Lazy-load VisualAligner only when needed."""
    if not use_clip:
        return None
    try:
        from src.visual.aligner import VisualAligner
        return VisualAligner()
    except Exception as e:
        print(f"⚠️  Failed to load VisualAligner: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    # ── 1. load configs ────────────────────────────────────────────────────
    config = load_yaml(args.config)
    profiles_data = load_yaml(args.profiles)

    profile_name = args.profile
    all_profiles = profiles_data.get("profiles", {})
    if profile_name not in all_profiles:
        print(f"ERROR: Profile '{profile_name}' not found. Available: {list(all_profiles)}")
        sys.exit(1)

    profile = all_profiles[profile_name]
    conditions_map = profiles_data.get("conditions", {})

    print(f"Profile : {profile_name}")
    print(f"Desc    : {profile.get('description', '')}")
    print(f"Pipeline: {profile.get('pipeline', 'v1')}")
    print()

    # ── 2. load cases ──────────────────────────────────────────────────────
    cases = load_cases_jsonl(args.cases)

    if args.condition:
        cases = [c for c in cases if c.get("bench", {}).get("condition") == args.condition]
        print(f"Filtered to {len(cases)} cases with condition={args.condition}")
    else:
        print(f"Loaded {len(cases)} cases (all conditions)")

    if not cases:
        print("No cases matched — exiting.")
        return

    # ── 3. initialise shared components ────────────────────────────────────
    engine = init_engine(config)
    llm = init_llm(config)
    visual_aligner = init_visual_aligner(profile.get("use_clip", False))

    # ── 4. build pipeline ──────────────────────────────────────────────────
    pipeline_type = profile.get("pipeline", "v1")

    if pipeline_type == "v2":
        from src.pipeline_base import V2Pipeline

        pipeline = V2Pipeline(
            engine=engine,
            llm=llm,
            visual_aligner=visual_aligner,
            profile=profile,
            config=config,
            adapter_path=args.adapter_path,
        )
    elif pipeline_type == "v1":
        # For v1, we need an MCP agent executor — expensive to set up.
        # If MCP is not available, fall back to a lightweight "no-agent" stub
        # that simply returns empty traces.
        print("NOTE: V1 pipeline requires MCP server. "
              "Use `python src/main_mcp.py` for full V1 evaluation.")
        print("      Running V1 stub (no agent) for structure testing.\n")

        from src.pipeline_base import V1Pipeline

        # Agent executor stub — returns empty response
        class _StubAgent:
            async def ainvoke(self, payload):
                return {"messages": []}

        pipeline = V1Pipeline(
            engine=engine,
            llm=llm,
            visual_aligner=visual_aligner,
            profile=profile,
            config=config,
            agent_executor=_StubAgent(),
        )
    else:
        print(f"ERROR: Unknown pipeline type '{pipeline_type}'")
        sys.exit(1)

    # ── 5. run evaluation ──────────────────────────────────────────────────
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{profile_name}"
    if args.condition:
        run_id += f"_{args.condition}"

    traces: List[EvalTrace] = []
    v2_traces: List[V2Trace] = []
    v2_per_case_metrics: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{idx}")
        case_cond = case.get("bench", {}).get("condition", "")
        cond_overrides = conditions_map.get(case_cond, {})

        # force_clip override from condition
        if cond_overrides.get("force_clip"):
            profile_copy = {**profile, "use_clip": True}
        else:
            profile_copy = profile

        print(f"[{idx:>3}/{len(cases)}] {case_id}  cond={case_cond}", end="")

        try:
            trace, v2_trace = await pipeline.run_case(case, cond_overrides, run_id)
            traces.append(trace)

            hit = "HIT" if trace.guid_match else "miss"
            pool = trace.final_pool_size or 0
            print(f"  pool={pool:<5}  {hit}")

            if v2_trace:
                v2_traces.append(v2_trace)
                # Per-case v2 metrics
                labels = case.get("labels")
                gt_dict = case.get("ground_truth", {})
                m = compute_v2_metrics(v2_trace, gt_dict, labels)
                m["case_id"] = case_id
                v2_per_case_metrics.append(m)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            traces.append(EvalTrace(
                scenario_id=case_id,
                run_id=run_id,
                scenario=None,      # type: ignore  – error trace
                error=str(exc),
                success=False,
            ))

    # ── 6. compute summaries ───────────────────────────────────────────────
    valid_traces = [t for t in traces if t.success and t.scenario is not None]
    v1_summary = compute_summary(valid_traces)

    v2_summary = None
    if v2_traces:
        v2_summary = compute_v2_summary(list(zip(valid_traces, v2_traces)))

        # enrich with field-F1 average
        f1_scores = [
            m["constraints_field_em_f1"]
            for m in v2_per_case_metrics
            if m.get("constraints_field_em_f1") is not None
        ]
        if f1_scores:
            v2_summary["avg_constraints_field_em_f1"] = sum(f1_scores) / len(f1_scores)

    # ── 7. write outputs ───────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ts}_{profile_name}"

    traces_file = output_dir / f"traces_{tag}.jsonl"
    summary_file = output_dir / f"summary_{tag}.csv"

    write_jsonl(valid_traces, traces_file)
    write_csv_summary(v1_summary, v2_summary, v2_per_case_metrics or None, summary_file)

    # ── 8. print quick summary ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Profile        : {profile_name}")
    print(f"Cases          : {v1_summary.total_scenarios}")
    print(f"Top-1 Accuracy : {v1_summary.top1_accuracy:.4f}")
    print(f"Top-K Accuracy : {v1_summary.topk_accuracy:.4f}")
    print(f"Search Space   : {v1_summary.avg_search_space_reduction:.4f}")
    if v2_summary:
        print(f"Parse Rate     : {v2_summary.get('constraints_parse_rate', 0):.4f}")
        avg_f1 = v2_summary.get("avg_constraints_field_em_f1")
        print(f"Field EM F1    : {avg_f1:.4f}" if avg_f1 else "Field EM F1    : N/A")
        avg_rg = v2_summary.get("avg_rerank_gain")
        print(f"Rerank Gain    : {avg_rg:.4f}" if avg_rg else "Rerank Gain    : N/A")
    print("=" * 60)
    print(f"Traces  → {traces_file}")
    print(f"Summary → {summary_file}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified evaluation runner (v1 + v2 pipelines)",
    )
    p.add_argument(
        "--profile", required=True,
        help="Profile name from profiles.yaml (e.g., v2_prompt, v1_baseline, best_v2)",
    )
    p.add_argument(
        "--cases", required=True,
        help="Path to cases JSONL file",
    )
    p.add_argument(
        "--condition", default=None,
        choices=["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"],
        help="Filter cases by experimental condition (optional)",
    )
    p.add_argument(
        "--adapter_path", default=None,
        help="LoRA adapter checkpoint path (for v2 lora mode)",
    )
    p.add_argument(
        "--output_dir", default="logs/evaluations",
        help="Output directory for traces + summary",
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    p.add_argument(
        "--profiles", default="profiles.yaml",
        help="Path to profiles.yaml",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(cli()))
