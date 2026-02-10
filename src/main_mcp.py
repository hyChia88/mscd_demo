"""
MCP-Based AI Agent Orchestrator for BIM Inspection

This is the production-ready version using Model Context Protocol (MCP) for tool integration.
Instead of hardcoded LangChain tools, this version dynamically connects to MCP servers
that expose BIM analysis capabilities as standardized services.

Architecture:
    LangGraph ReAct Agent → MCP Clients → MCP Servers (IFC, Visual, Rendering)

Benefits over legacy approach:
    1. Tools are decoupled and reusable across different AI applications
    2. MCP servers can be developed/deployed/scaled independently
    3. Standard protocol enables ecosystem integration (Claude Desktop, VS Code, etc.)
    4. Better separation of concerns: orchestration vs. domain logic

Usage:
    python src/main_mcp.py
"""

import os
import sys
import time
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("Warning: MCP dependencies not installed. Please run: pip install -r requirements.txt")
    MCP_AVAILABLE = False
    sys.exit(1)

# Try to import langchain-mcp-adapters, fall back to custom adapter
try:
    from langchain_mcp_adapters import convert_mcp_to_langchain_tools
    print("[MCP] Using official langchain-mcp-adapters", file=sys.stderr)
except ImportError:
    from mcp_langchain_adapter import convert_mcp_to_langchain_tools
    print("[MCP] Using custom MCP-LangChain adapter", file=sys.stderr)

from chat_logger import ConversationLogger

# Common utilities (consolidated from former local functions)
from common.config import load_config, load_system_prompt, load_ground_truth, get_base_dir
from common.guid import extract_guids_from_text
from common.response_parser import extract_response_content, apply_guid_fallback, handle_empty_response
from common.evaluation import format_test_input, evaluate_response, get_experiment_description

# RQ2: Schema-aware mapping and validation
from rq2_schema.extract_final_json import extract_final_json
from rq2_schema.schema_registry import SchemaRegistry
from rq2_schema.pipeline import run_rq2_postprocess

# P2: BCF handoff - trace, issue, and BCFzip generation
from handoff.trace import build_trace, write_trace_json
from handoff.bcf_lite import write_issue_json
from handoff.bcf_zip import write_bcfzip


async def main_async(args=None):
    """Main asynchronous orchestrator"""
    load_dotenv()

    # Determine experiment mode
    experiment_mode = args.experiment if args and args.experiment else None

    # Load centralized configuration
    config = load_config()
    base_dir = get_base_dir()

    # RQ2: Load schema configuration
    rq2_cfg = config.get("rq2", {})
    rq2_enabled = rq2_cfg.get("enabled", True)
    rq2_schema_path = str(base_dir / rq2_cfg.get("schema_path", "schemas/corenetx_min/v0.schema.json"))
    schema_registry = None
    rq2_schema = None
    rq2_schema_id = None
    if rq2_enabled:
        try:
            schema_registry = SchemaRegistry(rq2_schema_path)
            rq2_schema = schema_registry.schema
            rq2_schema_id = schema_registry.schema_id
            print(f"[RQ2] Schema loaded: {rq2_schema_id}")
        except FileNotFoundError as e:
            print(f"⚠️  Warning: RQ2 schema not found: {e}")
            rq2_enabled = False

    # Initialize conversation logger
    logger = ConversationLogger()

    # Setup LLM from config
    llm_config = config.get("llm", {})
    llm = ChatGoogleGenerativeAI(
        model=llm_config.get("model", "gemini-2.5-flash"),
        temperature=llm_config.get("temperature", 0),
        max_retries=llm_config.get("max_retries", 2),
    )

    # Load system prompt from config
    agent_config = config.get("agent", {})
    try:
        system_prompt = load_system_prompt(agent_config.get("system_prompt_file", "prompts/system_prompt.yaml"))
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        system_prompt = "You are a helpful BIM inspection assistant. Use the available tools to help users query the IFC model."

    # Define MCP server configuration
    python_exe = sys.executable
    ifc_server_path = str(base_dir / "mcp_servers" / "ifc_server.py")

    # Ground truth / dataset configuration
    # Priority: --cases > --dataset > config.yaml
    gt_config = config.get("ground_truth", {})

    # Dataset presets
    DATASET_PRESETS = {
        "gt1": {
            "file": "data/ground_truth/gt_1/gt_1.json",
            "image_dir": "data/ground_truth/gt_1/imgs"
        },
        "synth": {
            "file": "../data_curation/datasets/synth_v0.2/cases_v2.jsonl",
            "image_dir": "../data_curation/datasets/synth_v0.2/cases/imgs"
        }
    }

    # Resolve dataset path
    if args and args.cases:
        # Custom path specified
        gt_file = args.cases
        gt_image_dir = base_dir / gt_config.get("image_dir", "data/ground_truth/gt_1/imgs")
        dataset_name = "custom"
    elif args and args.dataset:
        # Preset selected
        preset = DATASET_PRESETS.get(args.dataset, DATASET_PRESETS["gt1"])
        gt_file = preset["file"]
        gt_image_dir = base_dir / preset["image_dir"]
        dataset_name = args.dataset
    else:
        # Use config.yaml
        gt_file = gt_config.get("file", "data/ground_truth/gt_1/gt_1.json")
        gt_image_dir = base_dir / gt_config.get("image_dir", "data/ground_truth/gt_1/imgs")
        dataset_name = "config"

    # Print active dataset to terminal
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name.upper()}  ({gt_file})")
    print(f"  IMAGES:  {gt_image_dir}")
    print(f"{'='*60}\n")

    # Output configuration from config
    output_config = config.get("output", {})
    evaluations_dir = base_dir / output_config.get("evaluations_dir", "logs/evaluations")

    # Delay between tests from config
    delay_between_tests = agent_config.get("delay_between_tests", 7)

    # Build environment for MCP server (inherit current env + add QUERY_MODE if specified)
    server_env = dict(os.environ)  # Inherit current environment

    # Parse experiment mode: "memory", "neo4j", "memory+clip", "neo4j+clip"
    query_mode = None
    visual_enabled = False
    if experiment_mode:
        if "+clip" in experiment_mode:
            query_mode = experiment_mode.replace("+clip", "")
            visual_enabled = True
        else:
            query_mode = experiment_mode
            visual_enabled = False
        server_env["QUERY_MODE"] = query_mode
        server_env["VISUAL_ENABLED"] = "true" if visual_enabled else "false"

    # Determine display mode
    display_mode = experiment_mode or "config"  # "config" means using config.yaml setting

    print("\n" + "="*70)
    print("🚀 Initializing MCP-based Agent")
    print(f"   IFC Model: {config.get('ifc', {}).get('model_path', 'N/A')}")
    print(f"   Dataset: {dataset_name.upper()} → {gt_file}")
    print(f"   Image Dir: {gt_image_dir}")
    if experiment_mode:
        print(f"   Experiment: {display_mode.upper()}")
        print(f"   Query Mode: {query_mode.upper()}")
        print(f"   Visual (CLIP): {'ENABLED' if visual_enabled else 'DISABLED'}")
    else:
        print(f"   Query Mode: CONFIG (from config.yaml)")
    print("="*70)

    server_params = StdioServerParameters(
        command=python_exe,
        args=[ifc_server_path],
        env=server_env
    )

    # Connect to MCP server and run agent INSIDE the session context
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to IFC Query Service")

            # List available tools
            tools_result = await session.list_tools()

            if not tools_result or not tools_result.tools:
                print("❌ No tools found. Cannot proceed.")
                return

            print(f"   Found {len(tools_result.tools)} tools:")
            for tool in tools_result.tools:
                desc = tool.description[:60] if tool.description else "No description"
                print(f"      - {tool.name}: {desc}...")

            # Convert MCP tools to LangChain tools
            langchain_tools = convert_mcp_to_langchain_tools(tools_result.tools, session)

            # RQ2: Build tool lookup map for post-processing
            tool_by_name = {t.name: t for t in langchain_tools}

            print(f"\n✅ {len(langchain_tools)} tools loaded from MCP server")
            print("="*70 + "\n")

            # Create LangGraph ReAct agent with MCP tools
            agent_executor = create_react_agent(
                llm.bind(system=system_prompt),
                langchain_tools
            )

            # Load ground truth test cases
            test_cases = load_ground_truth(gt_file)
            print(f"✅ Agent initialized. Loaded {len(test_cases)} ground truth test cases.\n")
            logger.log_agent_message(f"MCP-based agent initialized with {len(langchain_tools)} tools")

            # P2: Create run_id for this evaluation session (shared across all cases)
            # Include dataset and experiment mode in run_id for easy identification
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id_parts = [timestamp, dataset_name]
            if experiment_mode:
                run_id_parts.append(experiment_mode)
            run_id = "_".join(run_id_parts)

            # Track evaluation results
            evaluation_results = []

            for i, case in enumerate(test_cases, 1):
                case_id = case.get("case_id", case.get("id", f"case_{i}"))
                rq_category = case.get("ground_truth", {}).get("rq_category", "Unknown")

                print(f"\n{'='*70}")
                print(f"📋 Test Case {i}/{len(test_cases)}: {case_id}")
                print(f"   RQ Category: {rq_category}")
                print(f"{'='*70}")

                # Format input from ground truth case
                user_input, image_paths = format_test_input(case, gt_image_dir)

                print(f"📥 Input:\n{user_input}\n")
                if image_paths:
                    print(f"📷 Images: {', '.join([Path(p).name for p in image_paths])}\n")

                logger.log_user_message(user_input)

                try:
                    start_time = time.time()

                    # TODO: Add image support when multimodal is enabled
                    response = await agent_executor.ainvoke({"messages": [("user", user_input)]})

                    elapsed = time.time() - start_time

                    print(f"\n📤 Final Response ({elapsed:.2f}s):")
                    print("-" * 70)

                    # Extract and parse response using common utilities
                    parsed = extract_response_content(response)
                    tool_calls = parsed.tool_calls

                    # Handle empty responses, then apply GUID fallback chain
                    output = handle_empty_response(parsed, extract_guids_from_text)
                    parsed.final_text = output
                    output = apply_guid_fallback(parsed, extract_guids_from_text)

                    print(output)
                    print("-" * 70)

                    logger.log_agent_message(output, tool_calls=tool_calls if tool_calls else None)

                    # RQ2: Parse FINAL_JSON and run post-processing
                    agent_final, parse_err = extract_final_json(output)

                    # Build evidence list
                    evidence = [{"type": "chat", "ref": case_id, "note": "ground truth chat context + query"}]
                    for p in image_paths:
                        evidence.append({"type": "image", "ref": p, "note": "ground truth image"})

                    rq2_result = None
                    if rq2_enabled and rq_category == "RQ2" and rq2_schema is not None:
                        rq2_context = {
                            "storey_name": agent_final.get("selected_storey_name", "") if agent_final else "",
                            "evidence": evidence
                        }
                        rq2_result = await run_rq2_postprocess(
                            schema_id=rq2_schema_id,
                            schema=rq2_schema,
                            agent_final=agent_final,
                            parse_error=parse_err,
                            rq2_context=rq2_context,
                            tool_by_name=tool_by_name
                        )

                    # Evaluate response against ground truth
                    eval_result = evaluate_response(output, case.get("ground_truth", {}))
                    eval_result["case_id"] = case_id
                    eval_result["elapsed_time"] = elapsed
                    eval_result["tool_calls"] = tool_calls
                    eval_result["rq2"] = rq2_result  # Attach RQ2 results (may be None for non-RQ2 cases)
                    # Capture ambiguity tier for synthetic dataset analysis
                    tier = case.get("ambiguity_tier") or case.get("difficulty_tags", {}).get("tier")
                    if tier:
                        eval_result["ambiguity_tier"] = tier

                    # P2: Build trace and generate handoff artifacts
                    # Prepare rq2_result for trace (convert to dict format if needed)
                    eval_result_for_trace = dict(eval_result)
                    if rq2_result:
                        eval_result_for_trace["rq2_result"] = rq2_result

                    trace = build_trace(
                        run_id=run_id,
                        case_id=case_id,
                        test_case=case,
                        agent_response=output,
                        tool_calls=tool_calls,
                        eval_result=eval_result_for_trace,
                        config=config
                    )

                    # Write trace, issue.json, and bcfzip
                    trace_path = write_trace_json(trace, out_dir=str(base_dir / "outputs/traces"))
                    issue_path = write_issue_json(out_dir=str(base_dir / "outputs/issues"), trace=trace)
                    bcf_path = write_bcfzip(out_dir=str(base_dir / "outputs/bcf"), trace=trace)

                    # Add handoff paths to eval_result
                    eval_result["handoff"] = {
                        "trace": trace_path,
                        "issue_json": issue_path,
                        "bcfzip": bcf_path
                    }

                    evaluation_results.append(eval_result)

                    # Print evaluation results
                    print(f"\n📊 Evaluation:")
                    for detail in eval_result["details"]:
                        print(f"   {detail}")

                    # P2: Print handoff artifact paths
                    print(f"\n📁 Handoff Artifacts:")
                    print(f"   Trace:  {trace_path}")
                    print(f"   Issue:  {issue_path}")
                    print(f"   BCFzip: {bcf_path}")

                except Exception as e:
                    error_msg = f"Error during execution: {e}"
                    print(f"\n❌ {error_msg}")
                    logger.log_agent_message(f"ERROR: {error_msg}")
                    evaluation_results.append({
                        "case_id": case_id,
                        "guid_match": False,
                        "name_match": False,
                        "error": str(e)
                    })

                # Pause between test cases
                if i < len(test_cases):
                    print(f"\n⏳ Proceeding to next test case in {delay_between_tests} seconds...")
                    await asyncio.sleep(delay_between_tests)

            # Print summary report
            print(f"\n{'='*70}")
            print("📊 GROUND TRUTH EVALUATION SUMMARY")
            print(f"{'='*70}")

            total = len(evaluation_results)
            guid_matches = sum(1 for r in evaluation_results if r.get("guid_match", False))
            name_matches = sum(1 for r in evaluation_results if r.get("name_match", False))

            # Top-k retrieval metrics
            top1_hits = sum(1 for r in evaluation_results if r.get("top1_hit", False))
            top3_hits = sum(1 for r in evaluation_results if r.get("top3_hit", False))
            top5_hits = sum(1 for r in evaluation_results if r.get("top5_hit", False))
            avg_precision_at_1 = sum(r.get("precision_at_1", 0) for r in evaluation_results) / total if total > 0 else 0
            avg_recall = sum(r.get("recall", 0) for r in evaluation_results) / total if total > 0 else 0
            # F1 score
            f1_score = 2 * (avg_precision_at_1 * avg_recall) / (avg_precision_at_1 + avg_recall) if (avg_precision_at_1 + avg_recall) > 0 else 0

            print(f"   Total Test Cases: {total}")
            print(f"\n   📈 Retrieval Metrics:")
            print(f"      Top-1 Accuracy: {top1_hits}/{total} ({100*top1_hits/total:.1f}%)")
            print(f"      Top-3 Accuracy: {top3_hits}/{total} ({100*top3_hits/total:.1f}%)")
            print(f"      Top-5 Accuracy: {top5_hits}/{total} ({100*top5_hits/total:.1f}%)")
            print(f"      Precision@1:    {avg_precision_at_1:.3f}")
            print(f"      Recall:         {avg_recall:.3f}")
            print(f"      F1 Score:       {f1_score:.3f}")

            print(f"\n   📋 Legacy Metrics (backward compatible):")
            print(f"      GUID Matches: {guid_matches}/{total} ({100*guid_matches/total:.1f}%)")
            print(f"      Name Matches: {name_matches}/{total} ({100*name_matches/total:.1f}%)")

            # Group by RQ category with top-k metrics
            rq_stats = {}
            for r in evaluation_results:
                rq = r.get("rq_category", "Unknown")
                if rq not in rq_stats:
                    rq_stats[rq] = {"total": 0, "guid_match": 0, "top1_hit": 0, "top3_hit": 0}
                rq_stats[rq]["total"] += 1
                if r.get("guid_match", False):
                    rq_stats[rq]["guid_match"] += 1
                if r.get("top1_hit", False):
                    rq_stats[rq]["top1_hit"] += 1
                if r.get("top3_hit", False):
                    rq_stats[rq]["top3_hit"] += 1

            print(f"\n   📊 Results by RQ Category:")
            for rq, stats in sorted(rq_stats.items()):
                top1_pct = 100 * stats["top1_hit"] / stats["total"] if stats["total"] > 0 else 0
                top3_pct = 100 * stats["top3_hit"] / stats["total"] if stats["total"] > 0 else 0
                print(f"      {rq}: Top-1={stats['top1_hit']}/{stats['total']} ({top1_pct:.1f}%) | Top-3={stats['top3_hit']}/{stats['total']} ({top3_pct:.1f}%)")

            # Ambiguity Tier statistics (for synthetic dataset)
            tier_stats = {}
            for r in evaluation_results:
                tier = r.get("ambiguity_tier", None)
                if tier:
                    if tier not in tier_stats:
                        tier_stats[tier] = {"total": 0, "top1_hit": 0, "top3_hit": 0}
                    tier_stats[tier]["total"] += 1
                    if r.get("top1_hit", False):
                        tier_stats[tier]["top1_hit"] += 1
                    if r.get("top3_hit", False):
                        tier_stats[tier]["top3_hit"] += 1

            if tier_stats:
                print(f"\n   🎯 Results by Ambiguity Tier (Enough Thinking):")
                for tier, stats in sorted(tier_stats.items()):
                    top1_pct = 100 * stats["top1_hit"] / stats["total"] if stats["total"] > 0 else 0
                    top3_pct = 100 * stats["top3_hit"] / stats["total"] if stats["total"] > 0 else 0
                    print(f"      {tier}: Top-1={stats['top1_hit']}/{stats['total']} ({top1_pct:.1f}%) | Top-3={stats['top3_hit']}/{stats['total']} ({top3_pct:.1f}%)")

            # RQ2 Schema Validation summary
            rq2_rows = [r for r in evaluation_results if r.get("rq_category") == "RQ2" and r.get("rq2")]
            if rq2_rows:
                passed = sum(1 for r in rq2_rows if r["rq2"]["submission"]["validation_metadata"]["passed"])
                avg_fill = sum(r["rq2"]["submission"]["validation_metadata"]["required_fill_rate"] for r in rq2_rows) / len(rq2_rows)
                print(f"\n   📝 RQ2 Schema Validation:")
                print(f"      Validation Pass Rate: {passed}/{len(rq2_rows)} ({100*passed/len(rq2_rows):.1f}%)")
                print(f"      Avg Field Fill Rate:  {avg_fill:.3f}")

            # Save evaluation results to JSON with enhanced metrics
            evaluations_dir.mkdir(parents=True, exist_ok=True)
            eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Include dataset and experiment mode in filename for easy identification
            filename_parts = ["eval", eval_timestamp, dataset_name]
            if experiment_mode:
                filename_parts.append(experiment_mode)
            results_file = evaluations_dir / f"{'_'.join(filename_parts)}.json"

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": eval_timestamp,
                    "run_id": run_id,
                    "dataset": {
                        "name": dataset_name,
                        "path": gt_file,
                        "image_dir": str(gt_image_dir),
                        "total_cases": total
                    },
                    "experiment": {
                        "mode": experiment_mode or "config",
                        "query_mode": query_mode if experiment_mode else "config",
                        "visual_enabled": visual_enabled if experiment_mode else False,
                        "description": get_experiment_description(experiment_mode, query_mode, visual_enabled)
                    },
                    "ground_truth_file": gt_file,  # Kept for backward compatibility
                    "summary": {
                        "total": total,
                        # Retrieval metrics (new)
                        "retrieval": {
                            "top1_accuracy": top1_hits / total if total > 0 else 0,
                            "top3_accuracy": top3_hits / total if total > 0 else 0,
                            "top5_accuracy": top5_hits / total if total > 0 else 0,
                            "precision_at_1": avg_precision_at_1,
                            "recall": avg_recall,
                            "f1_score": f1_score
                        },
                        # Legacy metrics (backward compatible)
                        "guid_matches": guid_matches,
                        "name_matches": name_matches,
                        # RQ2 Schema metrics
                        "rq2_schema": {
                            "total": len(rq2_rows),
                            "passed": sum(1 for r in rq2_rows if r["rq2"]["submission"]["validation_metadata"]["passed"]) if rq2_rows else 0,
                            "pass_rate": (sum(1 for r in rq2_rows if r["rq2"]["submission"]["validation_metadata"]["passed"]) / len(rq2_rows)) if rq2_rows else 0,
                            "avg_fill_rate": (sum(r["rq2"]["submission"]["validation_metadata"]["required_fill_rate"] for r in rq2_rows) / len(rq2_rows)) if rq2_rows else 0
                        },
                        "by_rq_category": rq_stats,
                        # Ambiguity tier metrics (for synthetic dataset)
                        "by_ambiguity_tier": tier_stats if tier_stats else None
                    },
                    "results": evaluation_results
                }, f, indent=2)

            print(f"\n   Results saved to: {results_file}")

            # P2: Print handoff summary
            print(f"\n   Handoff Artifacts (run_id: {run_id}):")
            print(f"      Traces:  outputs/traces/{run_id}/")
            print(f"      Issues:  outputs/issues/{run_id}/")
            print(f"      BCFzips: outputs/bcf/{run_id}/")

            # Save conversation summary
            logger.save_summary(f"Completed {len(test_cases)} ground truth test cases using MCP architecture")
            print(f"\n📊 Session complete. Check logs/ directory for conversation history.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MCP-Based AI Agent Orchestrator for BIM Inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main_mcp.py                         # Default: use config.yaml setting
  python src/main_mcp.py --experiment memory     # In-memory spatial index (baseline)
  python src/main_mcp.py --experiment neo4j      # Neo4j graph queries
  python src/main_mcp.py --experiment memory+clip  # In-memory + CLIP visual matching
  python src/main_mcp.py --experiment neo4j+clip   # Neo4j + CLIP visual matching
  python src/main_mcp.py -e neo4j+clip           # Short form

Dataset selection:
  python src/main_mcp.py --dataset synth         # Use synthetic dataset (synth_v0.2)
  python src/main_mcp.py --dataset gt1           # Use original ground truth (gt_1)
  python src/main_mcp.py -d synth -e neo4j       # Combine dataset + experiment mode

Run all experiments in sequence:
  for mode in memory neo4j memory+clip neo4j+clip; do
    python src/main_mcp.py -e $mode
    sleep 10
  done
        """
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        choices=["memory", "neo4j", "memory+clip", "neo4j+clip"],
        default=None,
        help="Experiment mode: 'memory', 'neo4j', 'memory+clip', 'neo4j+clip'. The '+clip' variants enable CLIP visual matching tools."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["gt1", "synth"],
        default=None,
        help="Dataset to use: 'gt1' (original ground truth), 'synth' (synthetic dataset synth_v0.2). Default: use config.yaml setting."
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Custom path to cases file or directory (overrides --dataset and config.yaml)"
    )
    return parser.parse_args()


def main():
    """Synchronous entry point"""
    if not MCP_AVAILABLE:
        print("❌ MCP dependencies not available. Install with: pip install -r requirements.txt")
        return

    args = parse_args()

    # Run the async main function with args
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
