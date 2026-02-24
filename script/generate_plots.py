#!/usr/bin/env python3
"""
Generate evaluation plots from trace files.

Usage:
  # Standard mode — from JSONL trace files
  python script/generate_plots.py --latest
  python script/generate_plots.py --traces logs/evaluations/traces_*.jsonl
  python script/generate_plots.py \\
    --traces logs/evaluations/new_pipeline/traces_*.jsonl \\
    --before logs/evaluations/old_pipeline/traces_*.jsonl \\
    --output logs/plots/comparison

  # Modality analysis mode — loads *.trace.json files directly
  python script/generate_plots.py --modality
  python script/generate_plots.py --modality --output docs/plots/modality
  python script/generate_plots.py --modality --run-filter lora
"""

import argparse
import sys
from glob import glob
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.eval.visualizations import generate_all_plots, generate_modality_plots


def find_latest_traces(base_dir: str = "logs/evaluations") -> str:
    """Find the most recent traces file."""
    pattern = f"{base_dir}/**/traces_*.jsonl"
    files = sorted(glob(pattern, recursive=True), reverse=True)
    if not files:
        raise FileNotFoundError(f"No traces files found in {base_dir}")
    return files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation visualizations from trace files"
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        help="Path(s) to traces JSONL file(s). Supports glob patterns and multiple values."
    )
    parser.add_argument(
        "--before",
        nargs="+",
        help="Path(s) to 'before VLM' traces for comparison (optional). Supports globs."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for plots (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest traces file from logs/evaluations"
    )
    parser.add_argument(
        "--modality",
        action="store_true",
        help="Generate modality importance plots (Charts 9-11) from *.trace.json files"
    )
    parser.add_argument(
        "--run-filter",
        default=None,
        metavar="SUBSTRING",
        help="With --modality: only load runs whose directory name contains SUBSTRING (e.g. 'lora')"
    )

    args = parser.parse_args()

    def expand_patterns(patterns: list) -> list:
        """Expand a list of glob patterns into a deduplicated sorted file list."""
        matches = []
        for p in patterns:
            matches.extend(glob(p))
        return sorted(set(matches))

    # ── Modality analysis mode ────────────────────────────────────────────────
    if args.modality:
        output_dir = args.output or "docs/plots/modality"
        traces_jsonl = None
        if args.traces:
            matches = expand_patterns(args.traces)
            if not matches:
                print(f"ERROR: No files match: {args.traces}")
                sys.exit(1)
            traces_jsonl = matches
            print(f"Using {len(matches)} JSONL files for modality analysis:")
            for f in matches:
                print(f"  - {f}")
        generate_modality_plots(
            output_dir=output_dir,
            run_filter=args.run_filter,
            traces_jsonl=traces_jsonl,
        )
        return

    # ── Standard JSONL mode ───────────────────────────────────────────────────
    # Determine traces path
    if args.latest:
        traces_path = find_latest_traces()
        print(f"Using latest traces: {traces_path}")
    elif args.traces:
        matches = expand_patterns(args.traces)
        if not matches:
            print(f"ERROR: No files match: {args.traces}")
            sys.exit(1)
        traces_path = sorted(matches)
        if len(traces_path) == 1:
            traces_path = traces_path[0]
            print(f"Using traces: {traces_path}")
        else:
            print(f"Using {len(traces_path)} trace files:")
            for f in traces_path:
                print(f"  - {f}")
    else:
        print("ERROR: Must specify --traces or --latest")
        parser.print_help()
        sys.exit(1)

    # Determine before traces (optional)
    before_traces_path = None
    if args.before:
        matches = expand_patterns(args.before)
        if matches:
            before_traces_path = sorted(matches)
            if len(before_traces_path) == 1:
                before_traces_path = before_traces_path[0]
                print(f"Using before traces: {before_traces_path}")
            else:
                print(f"Using {len(before_traces_path)} before trace files:")
                for f in before_traces_path:
                    print(f"  - {f}")
        else:
            print(f"WARNING: No files match --before: {args.before}")

    # Auto-generate output directory with timestamp if not specified
    output_dir = args.output
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Extract run info from traces filename if possible
        first_trace = traces_path[0] if isinstance(traces_path, list) else traces_path
        traces_basename = Path(first_trace).stem  # e.g., "traces_20240210_143022_v2_prompt"
        if traces_basename.startswith("traces_"):
            # Extract the profile name and timestamp from traces filename
            parts = traces_basename.split("_")
            if len(parts) >= 4:
                profile_name = "_".join(parts[3:])  # e.g., "v2_prompt"
                output_dir = f"logs/plots/{timestamp}_{profile_name}"
            else:
                output_dir = f"logs/plots/{timestamp}"
        else:
            output_dir = f"logs/plots/{timestamp}"

        print(f"Auto-generated output directory: {output_dir}")

    # Generate plots
    generate_all_plots(
        traces_path=traces_path,
        v2_traces_path=traces_path,  # Same file for V2 metrics
        before_traces_path=before_traces_path,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
