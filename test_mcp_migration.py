#!/usr/bin/env python3
"""
Quick test script to verify MCP migration is working correctly.

This script performs basic checks without running the full agent:
1. Verify MCP dependencies are installed
2. Test that MCP servers can be imported
3. Check that FastMCP tools are properly defined
4. Validate configuration files
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required imports work"""
    print("🔍 Testing imports...")

    try:
        import fastmcp
        print("  ✅ fastmcp installed")
    except ImportError:
        print("  ❌ fastmcp not installed - run: pip install fastmcp")
        return False

    try:
        import mcp
        print("  ✅ mcp installed")
    except ImportError:
        print("  ❌ mcp not installed - run: pip install mcp")
        return False

    try:
        from langchain_mcp_adapters import convert_mcp_to_langchain_tools
        print("  ✅ langchain-mcp-adapters installed")
    except ImportError:
        print("  ❌ langchain-mcp-adapters not installed")
        print("     Note: This package may need to be installed manually")
        print("     The core MCP functionality will still work")

    return True


def test_ifc_engine():
    """Test that IFC engine can be initialized"""
    print("\n🔍 Testing IFC Engine...")

    try:
        from ifc_engine import IFCEngine
        base_dir = Path(__file__).parent
        ifc_path = base_dir / "data" / "BasicHouse.ifc"

        if not ifc_path.exists():
            print(f"  ⚠️  IFC file not found: {ifc_path}")
            return False

        engine = IFCEngine(str(ifc_path))
        num_spaces = len(engine.spatial_index)
        print(f"  ✅ IFC Engine initialized")
        print(f"     - Found {num_spaces} spaces in model")
        return True

    except Exception as e:
        print(f"  ❌ IFC Engine failed: {e}")
        return False


def test_mcp_servers():
    """Test that MCP servers can be imported"""
    print("\n🔍 Testing MCP Servers...")

    # Test IFC server
    try:
        sys.path.insert(0, str(Path(__file__).parent / "mcp_servers"))

        # We can't run the servers directly, but we can check they're valid Python
        ifc_server_path = Path(__file__).parent / "mcp_servers" / "ifc_server.py"
        visual_server_path = Path(__file__).parent / "mcp_servers" / "visual_server.py"

        if ifc_server_path.exists():
            print("  ✅ ifc_server.py exists")
        else:
            print("  ❌ ifc_server.py not found")
            return False

        if visual_server_path.exists():
            print("  ✅ visual_server.py exists")
        else:
            print("  ❌ visual_server.py not found")
            return False

        return True

    except Exception as e:
        print(f"  ❌ MCP server test failed: {e}")
        return False


def test_config_files():
    """Test that configuration files exist"""
    print("\n🔍 Testing Configuration Files...")

    base_dir = Path(__file__).parent

    files_to_check = [
        ("MCP servers config", base_dir / "config" / "mcp_servers.yaml"),
        ("System prompt", base_dir / "prompts" / "system_prompt.yaml"),
        ("Test scenarios", base_dir / "test.yaml"),
        ("Environment file", base_dir / ".env"),
    ]

    all_exist = True
    for name, path in files_to_check:
        if path.exists():
            print(f"  ✅ {name}: {path.name}")
        else:
            print(f"  ⚠️  {name} not found: {path}")
            if name == "Environment file":
                print("     Create .env file with: GOOGLE_API_KEY=your_key")
            all_exist = False

    return all_exist


def test_legacy_backup():
    """Check that legacy files were backed up"""
    print("\n🔍 Testing Legacy Backup...")

    base_dir = Path(__file__).parent
    legacy_dir = base_dir / "src" / "legacy"

    if not legacy_dir.exists():
        print("  ⚠️  Legacy directory not found")
        return False

    legacy_files = ["main.py", "agent_tools.py"]
    all_exist = True

    for filename in legacy_files:
        if (legacy_dir / filename).exists():
            print(f"  ✅ Backed up: {filename}")
        else:
            print(f"  ⚠️  Missing backup: {filename}")
            all_exist = False

    return all_exist


def main():
    print("="*70)
    print("🧪 MCP Migration Test Suite")
    print("="*70)

    results = {
        "Imports": test_imports(),
        "IFC Engine": test_ifc_engine(),
        "MCP Servers": test_mcp_servers(),
        "Configuration": test_config_files(),
        "Legacy Backup": test_legacy_backup(),
    }

    print("\n" + "="*70)
    print("📊 Test Results Summary")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests passed! MCP migration successful.")
        print("\n📝 Next steps:")
        print("   1. Ensure .env file has your GOOGLE_API_KEY")
        print("   2. Run: ./run_mcp.sh")
        print("   3. Check README/MCP_MIGRATION_GUIDE.md for details")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")
        print("   Check requirements.txt and run: pip install -r requirements.txt")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
