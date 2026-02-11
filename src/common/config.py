"""
Unified configuration and prompt loading.

All path resolution uses get_base_dir() which returns the project root
(parent of src/). This replaces scattered Path(__file__).parent.parent
patterns throughout the codebase.

Consolidates:
- main_mcp.py:load_config(), load_system_prompt(), load_scenarios(), load_ground_truth()
- chat_cli.py:load_config(), load_system_prompt(), load_ground_truth()
- v2/constraints_extractor_prompt_only.py:_load_prompts() path resolution
- visual/image_parser.py:_load_prompts() path resolution
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_base_dir() -> Path:
    """
    Return the project root directory (parent of src/).

    This is the single source of truth for the base directory calculation,
    replacing scattered Path(__file__).parent.parent calls.
    """
    return Path(__file__).parent.parent.parent


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """Load centralized configuration from YAML file."""
    base_dir = get_base_dir()
    config_path = base_dir / config_file

    if not config_path.exists():
        print(f"Warning: Config file '{config_path}' not found. Using defaults.")
        return {
            "ifc": {"model_path": "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"},
            "ground_truth": {
                "file": "data/ground_truth/gt_1/gt_1.json",
                "image_dir": "data/ground_truth/gt_1/imgs"
            },
            "llm": {"model": "gemini-2.5-flash", "temperature": 0, "max_retries": 2},
            "agent": {"delay_between_tests": 7, "system_prompt_file": "prompts/system_prompt.yaml"},
            "output": {"evaluations_dir": "logs/evaluations", "logs_dir": "logs"}
        }

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompt(prompt_file: str = "prompts/system_prompt.yaml") -> str:
    """Load system prompt from YAML file."""
    base_dir = get_base_dir()
    prompt_path = base_dir / prompt_file

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("system_prompt", "")


def load_yaml_prompts(prompts_path: str) -> Dict[str, Any]:
    """
    Load a YAML prompts file, resolving path relative to project root.

    This replaces the _load_prompts() path-resolution pattern in:
    - v2/constraints_extractor_prompt_only.py
    - visual/image_parser.py

    Args:
        prompts_path: Path relative to project root (e.g., "prompts/constraints_extraction.yaml")

    Returns:
        Parsed YAML dict

    Raises:
        FileNotFoundError: If the prompts file does not exist
    """
    base_dir = get_base_dir()
    full_path = base_dir / prompts_path

    if not full_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scenarios(file_path: str = "prompts/tests/test_2.yaml") -> List[Dict[str, Any]]:
    """Load test scenarios from YAML file."""
    if not os.path.exists(file_path):
        print(f"Error: Configuration file '{file_path}' not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            return []


def load_ground_truth(file_path: str = "data/ground_truth/gt_1/gt_1.json") -> List[Dict[str, Any]]:
    """
    Load ground truth test cases from JSON or JSONL.

    Supports:
    1. Single JSON array file (gt_1.json)
    2. JSONL file with one case per line (cases_v2.jsonl)
    """
    base_dir = get_base_dir()
    gt_path = base_dir / file_path

    if not gt_path.exists():
        print(f"Error: Ground truth path '{gt_path}' not found.")
        return []

    # JSONL format (cases_v2.jsonl)
    if gt_path.suffix == ".jsonl":
        cases = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        print(f"Loaded {len(cases)} cases from JSONL")
        return cases

    # Single JSON array file (gt_1.json)
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_llm(config: Dict[str, Any]):
    """
    Create a LangChain LLM instance from config.

    Consolidates identical ChatGoogleGenerativeAI initialisation from:
    - main_mcp.py
    - chat_cli.py
    - script/run.py

    Args:
        config: Top-level config dict (reads the ``llm`` section).

    Returns:
        ChatGoogleGenerativeAI instance.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm_cfg = config.get("llm", {})
    return ChatGoogleGenerativeAI(
        model=llm_cfg.get("model", "gemini-2.5-flash"),
        temperature=llm_cfg.get("temperature", 0),
        max_retries=llm_cfg.get("max_retries", 2),
    )
