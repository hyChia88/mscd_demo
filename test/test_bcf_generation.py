#!/usr/bin/env python3
"""
BCF Generation Sanity Test

Tests the BCF handoff pipeline without requiring MCP server or LLM.
Verifies that trace, issue.json, and bcfzip files are generated correctly.

Usage:
    python script/test_bcf_generation.py
"""

import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from handoff.trace import build_trace, write_trace_json, extract_ifc_guid
from handoff.bcf_lite import write_issue_json
from handoff.bcf_zip import write_bcfzip


def test_extract_ifc_guid():
    """Test IFC GUID extraction from text."""
    print("=" * 60)
    print("TEST 1: IFC GUID Extraction")
    print("=" * 60)

    # Valid IFC GUID (22 chars, Base64 alphabet)
    text_with_guid = "The element GUID is 0Um_J2ClP45uPRcRbJqhxe and it's a window."
    guid = extract_ifc_guid(text_with_guid)
    assert guid == "0Um_J2ClP45uPRcRbJqhxe", f"Expected '0Um_J2ClP45uPRcRbJqhxe', got '{guid}'"
    print(f"  Extracted GUID: {guid}")

    # Text without GUID
    text_no_guid = "No GUID here, just text."
    guid = extract_ifc_guid(text_no_guid)
    assert guid is None, f"Expected None, got '{guid}'"
    print(f"  No GUID found (correct): {guid}")

    # Empty text
    guid = extract_ifc_guid("")
    assert guid is None
    print(f"  Empty text: {guid}")

    print("  ✅ TEST 1 PASSED\n")


def test_build_trace():
    """Test trace building."""
    print("=" * 60)
    print("TEST 2: Trace Building")
    print("=" * 60)

    trace = build_trace(
        run_id="20260127_120000",
        case_id="GT_TEST_001",
        test_case={
            "id": "GT_TEST_001",
            "query_text": "Find the cracked wall",
            "image_file": ["test_image.png"],
            "ground_truth": {
                "target_guid": "0cRoQU_sD5R8MkkMkeodzx",
                "target_element_name": "Basic Wall:Interior",
                "rq_category": "RQ1"
            }
        },
        agent_response="Based on my analysis, the wall with GUID 0cRoQU_sD5R8MkkMkeodzx is the cracked wall.",
        tool_calls=[
            {"name": "get_element_by_guid", "args": {"guid": "0cRoQU_sD5R8MkkMkeodzx"}, "result": "Found wall"}
        ],
        eval_result={
            "guid_match": True,
            "name_match": True,
            "rq_category": "RQ1"
        },
        config={"ifc": {"model_path": "test.ifc"}}
    )

    assert trace["run_id"] == "20260127_120000"
    assert trace["case_id"] == "GT_TEST_001"
    assert trace["prediction"]["element_guid"] == "0cRoQU_sD5R8MkkMkeodzx"
    assert trace["prediction"]["guid_source"] == "regex_from_agent_response"
    print(f"  Run ID: {trace['run_id']}")
    print(f"  Case ID: {trace['case_id']}")
    print(f"  Predicted GUID: {trace['prediction']['element_guid']}")
    print(f"  GUID Source: {trace['prediction']['guid_source']}")

    print("  ✅ TEST 2 PASSED\n")
    return trace


def test_write_trace_json(trace):
    """Test trace JSON writing."""
    print("=" * 60)
    print("TEST 3: Trace JSON Writing")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = write_trace_json(trace, out_dir=tmpdir)
        path = Path(trace_path)

        assert path.exists(), f"Trace file not created: {trace_path}"
        assert path.suffix == ".json"
        assert "GT_TEST_001.trace.json" in path.name

        print(f"  Trace file created: {trace_path}")
        print(f"  File size: {path.stat().st_size} bytes")

    print("  ✅ TEST 3 PASSED\n")


def test_write_issue_json(trace):
    """Test BCF-lite issue.json writing."""
    print("=" * 60)
    print("TEST 4: BCF-lite Issue JSON")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        issue_path = write_issue_json(out_dir=tmpdir, trace=trace)
        path = Path(issue_path)

        assert path.exists(), f"Issue file not created: {issue_path}"
        assert path.suffix == ".json"
        assert "GT_TEST_001.issue.json" in path.name

        # Read and verify content
        import json
        with open(issue_path) as f:
            issue = json.load(f)

        assert "issue_id" in issue
        assert issue["case_id"] == "GT_TEST_001"
        assert issue["element_guid"] == "0cRoQU_sD5R8MkkMkeodzx"
        assert "trace_uri" in issue

        print(f"  Issue file created: {issue_path}")
        print(f"  Issue ID: {issue['issue_id']}")
        print(f"  Element GUID: {issue['element_guid']}")
        print(f"  Trace URI: {issue['trace_uri']}")

    print("  ✅ TEST 4 PASSED\n")


def test_write_bcfzip(trace):
    """Test BCFzip generation."""
    print("=" * 60)
    print("TEST 5: BCFzip Generation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        bcf_path = write_bcfzip(out_dir=tmpdir, trace=trace)
        path = Path(bcf_path)

        assert path.exists(), f"BCFzip not created: {bcf_path}"
        assert path.suffix == ".bcfzip"

        # Verify zip contents
        with zipfile.ZipFile(bcf_path, "r") as zf:
            names = zf.namelist()
            print(f"  BCFzip contents:")
            for name in names:
                print(f"    - {name}")

            # Check required files
            assert "bcf.version" in names, "Missing bcf.version"

            # Find topic directory (it's a UUID)
            topic_dirs = [n for n in names if "/markup.bcf" in n]
            assert len(topic_dirs) == 1, "Missing markup.bcf"

            topic_dir = topic_dirs[0].split("/")[0]
            assert f"{topic_dir}/viewpoints/viewpoint.bcfv" in names, "Missing viewpoint.bcfv"

            # Verify bcf.version XML
            bcf_version = zf.read("bcf.version").decode("utf-8")
            root = ET.fromstring(bcf_version)
            assert root.get("VersionId") == "2.1", f"Wrong BCF version: {root.get('VersionId')}"
            print(f"  BCF Version: {root.get('VersionId')}")

            # Verify markup.bcf XML
            markup = zf.read(f"{topic_dir}/markup.bcf").decode("utf-8")
            markup_root = ET.fromstring(markup)
            topic = markup_root.find("Topic")
            assert topic is not None, "Missing Topic element"

            title = topic.find("Title")
            assert title is not None and title.text, "Missing Topic Title"
            print(f"  Topic Title: {title.text}")

            # Verify viewpoint references
            viewpoints = markup_root.find("Viewpoints")
            assert viewpoints is not None, "Missing Viewpoints section"
            viewpoint_ref = viewpoints.find("ViewPoint/Viewpoint")
            assert viewpoint_ref is not None, "Missing Viewpoint reference"
            print(f"  Viewpoint ref: {viewpoint_ref.text}")

            # Verify viewpoint.bcfv XML
            viewpoint = zf.read(f"{topic_dir}/viewpoints/viewpoint.bcfv").decode("utf-8")
            vp_root = ET.fromstring(viewpoint)

            # Check component selection
            components = vp_root.find("Components/Selection")
            if components is not None:
                component = components.find("Component")
                if component is not None:
                    ifc_guid = component.get("IfcGuid")
                    print(f"  Component IfcGuid: {ifc_guid}")
                    assert ifc_guid == "0cRoQU_sD5R8MkkMkeodzx"

    print("  ✅ TEST 5 PASSED\n")


def main():
    """Run all sanity tests."""
    print("=" * 60)
    print("BCF GENERATION SANITY TEST")
    print("=" * 60)
    print()

    test_extract_ifc_guid()
    trace = test_build_trace()
    test_write_trace_json(trace)
    test_write_issue_json(trace)
    test_write_bcfzip(trace)

    print("=" * 60)
    print("✅ ALL SANITY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
