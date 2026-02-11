#!/usr/bin/env python3
"""
Experiment Management System for Master Thesis

Manages running, tracking, and comparing evaluation experiments.

Usage:
  # Run a single experiment
  python script/experiment.py run vlm_integration

  # Run multiple experiments
  python script/experiment.py run baseline_v2 vlm_integration

  # List available experiments
  python script/experiment.py list

  # Compare experiments
  python script/experiment.py compare vlm_impact

  # Run and compare in one go
  python script/experiment.py run baseline_v2 vlm_integration --compare vlm_impact
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class ExperimentManager:
    """Manages experiment execution and tracking."""

    def __init__(self, config_path: str = "experiments.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.experiments = self.config.get("experiments", {})
        self.comparisons = self.config.get("comparisons", {})
        self.tracking = self.config.get("tracking", {})

    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration from YAML."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def list_experiments(self):
        """Print all available experiments."""
        print("=" * 70)
        print("Available Experiments")
        print("=" * 70)
        for name, config in self.experiments.items():
            tags = ", ".join(config.get("tags", []))
            print(f"\n{name}")
            print(f"  Description: {config.get('description', 'N/A')}")
            print(f"  Profile: {config.get('profile', 'N/A')}")
            print(f"  Conditions: {config.get('conditions', [])}")
            print(f"  Tags: {tags}")
        print("\n" + "=" * 70)

    def list_comparisons(self):
        """Print all available comparison groups."""
        print("=" * 70)
        print("Available Comparisons")
        print("=" * 70)
        for name, config in self.comparisons.items():
            print(f"\n{name}")
            print(f"  Description: {config.get('description', 'N/A')}")
            if "before" in config:
                print(f"  Before: {config['before']}")
                print(f"  After: {config['after']}")
            elif "experiments" in config:
                print(f"  Experiments: {', '.join(config['experiments'])}")
        print("\n" + "=" * 70)

    def get_git_info(self) -> Dict[str, str]:
        """Get current git information."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT
            ).decode().strip()
            diff = subprocess.check_output(
                ["git", "diff", "HEAD"], cwd=PROJECT_ROOT
            ).decode()
            has_changes = len(diff) > 0
            return {
                "commit": commit,
                "branch": branch,
                "has_uncommitted_changes": has_changes,
                "diff": diff if has_changes else ""
            }
        except Exception as e:
            return {"error": str(e)}

    def save_metadata(self, experiment_name: str, output_dir: Path):
        """Save experiment metadata."""
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": self.experiments[experiment_name],
        }

        if self.tracking.get("log_git_info", True):
            metadata["git"] = self.get_git_info()

        if self.tracking.get("log_conda_env", True):
            try:
                env_info = subprocess.check_output(
                    ["conda", "env", "export", "-n", "mscd_demo"]
                ).decode()
                metadata["conda_env"] = env_info
            except Exception:
                metadata["conda_env"] = "unavailable"

        # Save metadata
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "experiment_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Metadata saved to {metadata_path}")

        # Register experiment in central registry
        self._register_experiment(experiment_name, output_dir, metadata)

    def _register_experiment(
        self,
        experiment_name: str,
        output_dir: Path,
        metadata: Dict[str, Any]
    ):
        """Add experiment to central registry."""
        registry_path = Path("logs/experiments/registry.jsonl")
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        registry_entry = {
            "experiment_name": experiment_name,
            "timestamp": metadata["timestamp"],
            "output_dir": str(output_dir),
            "profile": metadata["config"].get("profile"),
            "conditions": metadata["config"].get("conditions", []),
            "tags": metadata["config"].get("tags", []),
            "git_commit": metadata.get("git", {}).get("commit", "unknown")[:8],
        }

        with open(registry_path, "a") as f:
            f.write(json.dumps(registry_entry) + "\n")

    def run_experiment(
        self,
        experiment_name: str,
        dry_run: bool = False
    ) -> Optional[Path]:
        """Run a single experiment."""
        if experiment_name not in self.experiments:
            print(f"ERROR: Experiment '{experiment_name}' not found")
            return None

        config = self.experiments[experiment_name]
        print("\n" + "=" * 70)
        print(f"Running Experiment: {experiment_name}")
        print("=" * 70)
        print(f"Description: {config.get('description', 'N/A')}")
        print(f"Profile: {config.get('profile')}")
        print(f"Conditions: {config.get('conditions', [])}")
        print("=" * 70)

        output_dir = Path(config["output_dir"])

        # Save metadata
        if not dry_run:
            self.save_metadata(experiment_name, output_dir)

        # Run each condition
        for condition in config.get("conditions", []):
            print(f"\n  → Running condition {condition}...")

            cmd = [
                "conda", "run", "-n", "mscd_demo",
                "python", "script/run.py",
                "--profile", config["profile"],
                "--cases", config["cases"],
                "--condition", condition,
                "--output_dir", str(output_dir)
            ]

            if "limit" in config:
                cmd.extend(["--limit", str(config["limit"])])

            if dry_run:
                print(f"    [DRY RUN] {' '.join(cmd)}")
            else:
                try:
                    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
                    print(f"    ✓ Condition {condition} completed")
                except subprocess.CalledProcessError as e:
                    print(f"    ✗ Condition {condition} failed: {e}")
                    return None

        if not dry_run:
            print(f"\n  ✓ Experiment '{experiment_name}' completed")
            print(f"  Results saved to: {output_dir}")

            # Auto-generate plots for this experiment
            self._generate_plots_for_experiment(experiment_name, output_dir, dry_run=False)

        return output_dir

    def _generate_plots_for_experiment(
        self,
        experiment_name: str,
        output_dir: Path,
        dry_run: bool = False
    ):
        """Auto-generate plots for a completed experiment."""
        print(f"\n  → Auto-generating plots...")

        plots_dir = output_dir / "plots"
        traces_pattern = f"{output_dir}/traces_*.jsonl"

        cmd = [
            "conda", "run", "-n", "mscd_demo",
            "python", "script/generate_plots.py",
            "--traces", traces_pattern,
            "--output", str(plots_dir)
        ]

        if dry_run:
            print(f"    [DRY RUN] {' '.join(cmd)}")
        else:
            try:
                subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
                print(f"  ✓ Plots saved to: {plots_dir}")
                return plots_dir
            except subprocess.CalledProcessError as e:
                print(f"  ⚠ Plot generation failed: {e}")
                return None

    def compare_experiments(
        self,
        comparison_name: str,
        dry_run: bool = False
    ):
        """Generate comparison plots between experiments."""
        if comparison_name not in self.comparisons:
            print(f"ERROR: Comparison '{comparison_name}' not found")
            return

        config = self.comparisons[comparison_name]
        output_dir = Path(config["output_dir"])

        print("\n" + "=" * 70)
        print(f"Generating Comparison: {comparison_name}")
        print("=" * 70)
        print(f"Description: {config.get('description', 'N/A')}")

        if "before" in config and "after" in config:
            # Before/After comparison
            before_exp = config["before"]
            after_exp = config["after"]

            before_dir = Path(self.experiments[before_exp]["output_dir"])
            after_dir = Path(self.experiments[after_exp]["output_dir"])

            print(f"Before: {before_exp} ({before_dir})")
            print(f"After: {after_exp} ({after_dir})")

            cmd = [
                "conda", "run", "-n", "mscd_demo",
                "python", "script/generate_plots.py",
                "--traces", f"{after_dir}/traces_*.jsonl",
                "--before", f"{before_dir}/traces_*.jsonl",
                "--output", str(output_dir)
            ]

        else:
            # Multi-experiment comparison
            experiments = config.get("experiments", [])
            print(f"Experiments: {', '.join(experiments)}")

            # Use first experiment as main, others as comparison
            main_exp = experiments[0]
            main_dir = Path(self.experiments[main_exp]["output_dir"])

            cmd = [
                "conda", "run", "-n", "mscd_demo",
                "python", "script/generate_plots.py",
                "--traces", f"{main_dir}/traces_*.jsonl",
                "--output", str(output_dir)
            ]

        if dry_run:
            print(f"\n[DRY RUN] {' '.join(cmd)}")
        else:
            try:
                subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
                print(f"\n  ✓ Comparison plots saved to: {output_dir}")
            except subprocess.CalledProcessError as e:
                print(f"\n  ✗ Comparison failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment management for master thesis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List experiments
    subparsers.add_parser("list", help="List all available experiments")

    # List comparisons
    subparsers.add_parser("list-comparisons", help="List all available comparisons")

    # Run experiments
    run_parser = subparsers.add_parser("run", help="Run one or more experiments")
    run_parser.add_argument(
        "experiments",
        nargs="+",
        help="Name(s) of experiment(s) to run"
    )
    run_parser.add_argument(
        "--compare",
        help="Comparison to run after experiments complete"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )

    # Compare experiments
    compare_parser = subparsers.add_parser(
        "compare",
        help="Generate comparison plots"
    )
    compare_parser.add_argument(
        "comparison",
        help="Name of comparison to generate"
    )
    compare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    # Initialize manager
    manager = ExperimentManager()

    # Execute command
    if args.command == "list":
        manager.list_experiments()
    elif args.command == "list-comparisons":
        manager.list_comparisons()
    elif args.command == "run":
        for exp_name in args.experiments:
            manager.run_experiment(exp_name, dry_run=args.dry_run)

        # Run comparison if requested
        if args.compare:
            manager.compare_experiments(args.compare, dry_run=args.dry_run)
    elif args.command == "compare":
        manager.compare_experiments(args.comparison, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
