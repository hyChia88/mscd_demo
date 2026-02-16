#!/usr/bin/env python3
"""
RQ2 Schema Pipeline Smoke Test

This test validates the RQ2 pipeline WITHOUT requiring MCP server.
It uses dummy tool implementations to simulate MCP responses.

Usage:
    python script/rq2_schema_smoke_test.py
"""

import asyncio
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rq2_schema.schema_registry import SchemaRegistry
from rq2_schema.pipeline import run_rq2_postprocess
from rq2_schema.extract_final_json import extract_final_json


class DummyTool:
    """Mock MCP tool that returns predefined responses."""

    def __init__(self, name: str, return_value):
        self.name = name
        self.return_value = return_value

    async def ainvoke(self, args):
        return self.return_value


async def test_basic_pass():
    """Test case: Valid agent output, element exists, storey valid → should PASS."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Pass Case")
    print("=" * 60)

    schema_path = PROJECT_ROOT / "schemas" / "corenetx_min" / "v0.schema.json"
    reg = SchemaRegistry(str(schema_path))

    # Simulate agent FINAL_JSON output
    agent_final = {
        "selected_guid": "0Um_J2ClP45uPRcRbJqhxe",
        "selected_storey_name": "6 - Sixth Floor",
        "issue_summary": "Window alignment check during module installation.",
        "issue_type": "coordination",
        "severity": "medium",
        "evidence": [{"type": "chat", "ref": "GT_007", "note": "mock"}],
    }

    # Mock MCP tool responses
    tool_by_name = {
        "list_available_spaces": DummyTool(
            "list_available_spaces",
            "Available spaces:\n  - '6 - Sixth Floor' (45 elements)\n  - '7 - Seventh Floor' (32 elements)",
        ),
        "get_element_details": DummyTool(
            "get_element_details",
            {
                "GlobalId": "0Um_J2ClP45uPRcRbJqhxe",
                "Name": "BALANS Window",
                "Type": "IfcWindow",
                "PropertySets": {"Pset_WindowCommon": {"FireRating": "E30"}},
            },
        ),
    }

    rq2_context = {
        "storey_name": agent_final["selected_storey_name"],
        "evidence": [{"type": "image", "ref": "test.png", "note": "mock image"}],
    }

    result = await run_rq2_postprocess(
        schema_id=reg.schema_id,
        schema=reg.schema,
        agent_final=agent_final,
        parse_error="",
        rq2_context=rq2_context,
        tool_by_name=tool_by_name,
    )

    # Assertions
    submission = result["submission"]
    metadata = submission["validation_metadata"]

    print(f"  Schema ID: {metadata['schema_id']}")
    print(f"  Passed: {metadata['passed']}")
    print(f"  Fill Rate: {metadata['required_fill_rate']:.3f}")
    print(f"  Errors: {metadata['errors']}")
    print(f"  Escalated: {result['uncertainty']['escalated']}")

    assert metadata["passed"] is True, f"Expected passed=True, got {metadata['passed']}"
    assert (
        metadata["required_fill_rate"] >= 0.8
    ), f"Expected fill_rate>=0.8, got {metadata['required_fill_rate']}"
    assert len(metadata["errors"]) == 0, f"Expected no errors, got {metadata['errors']}"
    assert (
        result["uncertainty"]["escalated"] is False
    ), "Expected escalated=False for passing case"

    print("  ✅ TEST 1 PASSED")
    return True


async def test_missing_guid():
    """Test case: Missing GUID → should FAIL and escalate."""
    print("\n" + "=" * 60)
    print("TEST 2: Missing GUID Case")
    print("=" * 60)

    schema_path = PROJECT_ROOT / "schemas" / "corenetx_min" / "v0.schema.json"
    reg = SchemaRegistry(str(schema_path))

    # Agent couldn't determine GUID
    agent_final = {
        "selected_guid": "",
        "selected_storey_name": "Level 1",
        "issue_summary": "Cannot determine specific element from image.",
        "issue_type": "unknown",
        "severity": "unknown",
    }

    tool_by_name = {
        "list_available_spaces": DummyTool(
            "list_available_spaces",
            "Available spaces:\n  - 'Level 1' (20 elements)",
        ),
        "get_element_details": DummyTool(
            "get_element_details", {"error": "Element not found"}
        ),
    }

    rq2_context = {"storey_name": "Level 1", "evidence": []}

    result = await run_rq2_postprocess(
        schema_id=reg.schema_id,
        schema=reg.schema,
        agent_final=agent_final,
        parse_error="",
        rq2_context=rq2_context,
        tool_by_name=tool_by_name,
    )

    metadata = result["submission"]["validation_metadata"]

    print(f"  Passed: {metadata['passed']}")
    print(f"  Fill Rate: {metadata['required_fill_rate']:.3f}")
    print(f"  Errors: {metadata['errors']}")
    print(f"  Escalated: {result['uncertainty']['escalated']}")

    # Should fail due to low fill rate (missing guid, ifc_class)
    assert metadata["passed"] is False, f"Expected passed=False, got {metadata['passed']}"
    assert (
        result["uncertainty"]["escalated"] is True
    ), "Expected escalated=True for failing case"

    print("  ✅ TEST 2 PASSED")
    return True


async def test_invalid_storey():
    """Test case: Storey not in model → domain validation error."""
    print("\n" + "=" * 60)
    print("TEST 3: Invalid Storey Case")
    print("=" * 60)

    schema_path = PROJECT_ROOT / "schemas" / "corenetx_min" / "v0.schema.json"
    reg = SchemaRegistry(str(schema_path))

    agent_final = {
        "selected_guid": "abc123def456ghi789jkl012",
        "selected_storey_name": "Basement 99",  # Invalid storey
        "issue_summary": "Test issue.",
        "issue_type": "defect",
        "severity": "high",
    }

    tool_by_name = {
        "list_available_spaces": DummyTool(
            "list_available_spaces",
            "Available spaces:\n  - 'Level 1' (20 elements)\n  - 'Level 2' (30 elements)",
        ),
        "get_element_details": DummyTool(
            "get_element_details",
            {"GlobalId": "abc123def456ghi789jkl012", "Type": "IfcWall"},
        ),
    }

    rq2_context = {"storey_name": "Basement 99", "evidence": []}

    result = await run_rq2_postprocess(
        schema_id=reg.schema_id,
        schema=reg.schema,
        agent_final=agent_final,
        parse_error="",
        rq2_context=rq2_context,
        tool_by_name=tool_by_name,
    )

    metadata = result["submission"]["validation_metadata"]

    print(f"  Passed: {metadata['passed']}")
    print(f"  Errors: {metadata['errors']}")

    # Should fail due to domain validation error
    assert metadata["passed"] is False, f"Expected passed=False, got {metadata['passed']}"
    assert any(
        "storey_name not in model" in e for e in metadata["errors"]
    ), f"Expected storey error, got {metadata['errors']}"

    print("  ✅ TEST 3 PASSED")
    return True


async def test_extract_final_json():
    """Test FINAL_JSON extraction from agent output."""
    print("\n" + "=" * 60)
    print("TEST 4: FINAL_JSON Extraction")
    print("=" * 60)

    # Test successful extraction
    text1 = """
Based on the 4D context and spatial filtering, I found the window.

FINAL_JSON={"selected_guid": "xyz123", "issue_type": "defect", "severity": "low", "issue_summary": "Test"}
"""
    result, error = extract_final_json(text1)
    print(f"  Test 4a - Valid JSON:")
    print(f"    Extracted: {result}")
    print(f"    Error: '{error}'")
    assert result.get("selected_guid") == "xyz123", f"Expected xyz123, got {result}"
    assert error == "", f"Expected no error, got '{error}'"

    # Test missing tag
    text2 = "No JSON block here."
    result, error = extract_final_json(text2)
    print(f"  Test 4b - Missing tag:")
    print(f"    Error: '{error}'")
    assert result == {}, "Expected empty dict for missing tag"
    assert "not found" in error.lower(), f"Expected 'not found' error, got '{error}'"

    # Test malformed JSON
    text3 = "FINAL_JSON={not valid json"
    result, error = extract_final_json(text3)
    print(f"  Test 4c - Malformed JSON:")
    print(f"    Error: '{error}'")
    assert result == {}, "Expected empty dict for malformed JSON"
    assert error != "", "Expected error for malformed JSON"

    print("  ✅ TEST 4 PASSED")
    return True


async def main():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("RQ2 SCHEMA PIPELINE SMOKE TEST")
    print("=" * 70)

    all_passed = True

    try:
        all_passed &= await test_basic_pass()
        all_passed &= await test_missing_guid()
        all_passed &= await test_invalid_storey()
        all_passed &= await test_extract_final_json()
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL SMOKE TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
