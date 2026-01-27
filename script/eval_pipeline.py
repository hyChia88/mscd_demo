#!/usr/bin/env python3
"""
Evaluation Pipeline CLI v2

Orchestrates evaluation runs and produces:
- JSONL trace file: One EvalTrace per line for detailed analysis
- CSV summary: Aggregated metrics for quick comparison

Usage:
    python script/eval_pipeline.py --config config.yaml --output logs/evaluations
    python script/eval_pipeline.py --gt data/ground_truth/gt_1/gt_1.json --output logs/evaluations
"""

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

load_dotenv()

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("MCP dependencies not installed. Please run: pip install -r requirements.txt")
    sys.exit(1)

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# MCP adapter
try:
    from langchain_mcp_adapters import convert_mcp_to_langchain_tools
except ImportError:
    from mcp_langchain_adapter import convert_mcp_to_langchain_tools

# Evaluation modules
from eval.contracts import EvalTrace, MetricsSummary, ScenarioInput
from eval.metrics import compute_summary
from eval.runner import run_one_scenario


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ground_truth(gt_path: str) -> list:
    """Load ground truth test cases from JSON"""
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_system_prompt(prompt_file: str, base_dir: Path) -> str:
    """Load system prompt from YAML file"""
    prompt_path = base_dir / prompt_file
    if not prompt_path.exists():
        return "You are a helpful BIM inspection assistant."

    with open(prompt_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("system_prompt", "")


class JSONLWriter:
    """Append-only JSONL file writer for streaming trace output"""

    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "w", encoding="utf-8")

    def write(self, trace: EvalTrace):
        line = trace.to_jsonl_line()
        self.file.write(line + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def write_csv_summary(
    summary: MetricsSummary, traces: List[EvalTrace], output_path: str
):
    """Write metrics summary to CSV file"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header section - Overall metrics
        writer.writerow(["=== OVERALL METRICS ===", ""])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Scenarios", summary.total_scenarios])
        writer.writerow(["Successful Runs", summary.successful_runs])
        writer.writerow(["Top-1 Accuracy", f"{summary.top1_accuracy:.2%}"])
        writer.writerow(["Top-3 Accuracy", f"{summary.topk_accuracy:.2%}"])
        writer.writerow(
            ["Avg Search-Space Reduction", f"{summary.avg_search_space_reduction:.2%}"]
        )
        writer.writerow(
            ["Avg Field Population Rate", f"{summary.avg_field_population_rate:.2%}"]
        )
        writer.writerow(["Escalation Rate", f"{summary.escalation_rate:.2%}"])
        writer.writerow(
            ["Avg Tool Calls per Scenario", f"{summary.avg_tool_calls_per_scenario:.2f}"]
        )
        writer.writerow(["Avg Latency (ms)", f"{summary.avg_latency_ms:.1f}"])
        writer.writerow([])

        # By RQ category
        writer.writerow(["=== BY RESEARCH QUESTION ===", ""])
        writer.writerow(
            ["RQ Category", "Total", "Top-1 Hits", "Top-1 Accuracy", "Escalation Rate"]
        )
        for rq, stats in sorted(summary.by_rq_category.items()):
            writer.writerow(
                [
                    rq,
                    stats["total"],
                    stats["top1_hits"],
                    f"{stats['top1_accuracy']:.2%}",
                    f"{stats['escalation_rate']:.2%}",
                ]
            )
        writer.writerow([])

        # Tool usage distribution
        writer.writerow(["=== TOOL USAGE ===", ""])
        writer.writerow(["Tool", "Call Count"])
        for tool, count in sorted(
            summary.tool_call_distribution.items(), key=lambda x: -x[1]
        ):
            writer.writerow([tool, count])
        writer.writerow([])

        # Per-scenario results
        writer.writerow(["=== PER-SCENARIO RESULTS ===", ""])
        writer.writerow(
            [
                "Scenario ID",
                "RQ",
                "GUID Match",
                "Name Match",
                "Storey Match",
                "Tool Calls",
                "Latency (ms)",
                "Pool Reduction",
                "Error",
            ]
        )
        for trace in traces:
            # Compute pool reduction for this trace
            pool_reduction = ""
            if trace.initial_pool_size and trace.final_pool_size:
                reduction = 1 - (trace.final_pool_size / trace.initial_pool_size)
                pool_reduction = f"{reduction:.2%}"

            writer.writerow(
                [
                    trace.scenario_id,
                    trace.scenario.ground_truth.rq_category.value,
                    "Yes" if trace.guid_match else "No",
                    "Yes" if trace.name_match else "No",
                    "Yes" if trace.storey_match else "No",
                    len(trace.tool_steps),
                    f"{trace.total_latency_ms:.1f}",
                    pool_reduction,
                    trace.error or "",
                ]
            )


async def run_pipeline(args):
    """Main pipeline execution"""

    base_dir = PROJECT_ROOT

    # Load configuration
    config_path = base_dir / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Load ground truth
    gt_path = args.gt or config.get("ground_truth", {}).get(
        "file", "data/ground_truth/gt_1/gt_1.json"
    )
    gt_full_path = base_dir / gt_path
    if not gt_full_path.exists():
        print(f"Ground truth file not found: {gt_full_path}")
        sys.exit(1)

    test_cases = load_ground_truth(str(gt_full_path))

    # Limit scenarios if requested
    if args.scenarios:
        test_cases = test_cases[: args.scenarios]

    image_dir = base_dir / config.get("ground_truth", {}).get(
        "image_dir", "data/ground_truth/gt_1/imgs"
    )

    # Output setup
    output_dir = Path(
        args.output
        or config.get("output", {}).get("evaluations_dir", "logs/evaluations")
    )
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"traces_{timestamp}.jsonl"
    csv_path = output_dir / f"summary_{timestamp}.csv"

    # LLM setup
    llm_config = config.get("llm", {})
    llm = ChatGoogleGenerativeAI(
        model=llm_config.get("model", "gemini-2.5-flash"),
        temperature=llm_config.get("temperature", 0),
        max_retries=llm_config.get("max_retries", 2),
    )

    # System prompt
    agent_config = config.get("agent", {})
    system_prompt = load_system_prompt(
        agent_config.get("system_prompt_file", "prompts/system_prompt.yaml"), base_dir
    )

    # MCP server setup
    python_exe = sys.executable
    ifc_server_path = str(base_dir / "mcp_servers" / "ifc_server.py")

    server_params = StdioServerParameters(
        command=python_exe, args=[ifc_server_path], env=None
    )

    print(f"\n{'='*70}")
    print("Evaluation Pipeline v2")
    print(f"{'='*70}")
    print(f"  Ground Truth: {gt_path} ({len(test_cases)} scenarios)")
    print(f"  Output Dir:   {output_dir}")
    print(f"  LLM Model:    {llm_config.get('model', 'gemini-2.5-flash')}")
    print(f"{'='*70}\n")

    traces: List[EvalTrace] = []
    jsonl_writer = JSONLWriter(str(jsonl_path))

    delay_between_tests = agent_config.get("delay_between_tests", 7)

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("[MCP] Connected to IFC Query Service")

                # Get tools
                tools_result = await session.list_tools()
                langchain_tools = convert_mcp_to_langchain_tools(
                    tools_result.tools, session
                )
                print(f"[MCP] Loaded {len(langchain_tools)} tools\n")

                # Create agent
                agent_executor = create_react_agent(
                    llm.bind(system=system_prompt), langchain_tools
                )

                # Run scenarios
                for i, case in enumerate(test_cases, 1):
                    case_id = case.get("id", f"case_{i}")
                    rq = case.get("ground_truth", {}).get("rq_category", "Unknown")

                    print(f"[{i}/{len(test_cases)}] {case_id} ({rq})")

                    # Create scenario
                    scenario = ScenarioInput.from_ground_truth_dict(
                        case, str(image_dir)
                    )

                    # Run evaluation
                    trace = await run_one_scenario(
                        scenario=scenario, agent_executor=agent_executor, run_id=timestamp
                    )

                    traces.append(trace)
                    jsonl_writer.write(trace)

                    # Print result
                    if trace.error:
                        status = "ERROR"
                    elif trace.guid_match:
                        status = "PASS"
                    else:
                        status = "FAIL"

                    print(
                        f"         -> {status} ({trace.total_latency_ms:.0f}ms, "
                        f"{len(trace.tool_steps)} tool calls)"
                    )

                    # Rate limiting
                    if i < len(test_cases):
                        await asyncio.sleep(delay_between_tests)

    finally:
        jsonl_writer.close()

    # Compute and write summary
    summary = compute_summary(traces)
    write_csv_summary(summary, traces, str(csv_path))

    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Scenarios:     {summary.total_scenarios}")
    print(f"  Successful Runs:     {summary.successful_runs}")
    print(f"  Top-1 Accuracy:      {summary.top1_accuracy:.1%}")
    print(f"  Top-3 Accuracy:      {summary.topk_accuracy:.1%}")
    print(f"  Escalation Rate:     {summary.escalation_rate:.1%}")
    print(f"  Avg Tool Calls:      {summary.avg_tool_calls_per_scenario:.1f}")
    print(f"  Avg Latency:         {summary.avg_latency_ms:.0f}ms")

    if summary.avg_search_space_reduction > 0:
        print(f"  Avg Pool Reduction:  {summary.avg_search_space_reduction:.1%}")

    print(f"\n  Results by RQ:")
    for rq, stats in sorted(summary.by_rq_category.items()):
        print(
            f"    {rq}: {stats['top1_hits']}/{stats['total']} ({stats['top1_accuracy']:.1%})"
        )

    print(f"\n  Outputs:")
    print(f"    JSONL: {jsonl_path}")
    print(f"    CSV:   {csv_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation Pipeline v2 - Run BIM inspection agent evaluation"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Config file path (default: config.yaml)"
    )
    parser.add_argument(
        "--gt", help="Ground truth JSON path (overrides config ground_truth.file)"
    )
    parser.add_argument(
        "--output", help="Output directory (overrides config output.evaluations_dir)"
    )
    parser.add_argument(
        "--scenarios", type=int, help="Limit number of scenarios to run"
    )

    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
