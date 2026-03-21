#!/usr/bin/env python3
"""CLI wrapper for IFC → Neo4j export. Used by neo4j_init.sh."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from py2neo import Graph
from ifc_engine import IFCEngine


class _GeminiLLMClient:
    """Minimal wrapper exposing .complete(prompt) -> str for IFCEngine registries."""
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def complete(self, prompt: str) -> str:
        resp = self._model.generate_content(prompt)
        return resp.text


def _make_llm_client():
    """Create Gemini LLM client if API key is available, else None."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No GOOGLE_API_KEY — registry parsing will use regex fallback")
        return None
    try:
        client = _GeminiLLMClient(api_key)
        print("✅  Gemini LLM client configured for registry parsing")
        return client
    except Exception as e:
        print(f"⚠️  Gemini client init failed ({e}) — using regex fallback")
        return None


def main():
    # Load .env from project root (mscd_demo/)
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", required=True, help="Path to IFC file")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--no-clear", action="store_true",
                        help="Additive mode: don't clear existing graph (for multi-model loading)")
    args = parser.parse_args()

    if not os.path.exists(args.ifc):
        print(f"Error: IFC file not found: {args.ifc}", file=sys.stderr)
        sys.exit(1)

    llm_client = _make_llm_client()
    g = Graph(args.uri, auth=(args.user, args.password))
    engine = IFCEngine(args.ifc, neo4j_conn=g, llm_client=llm_client)
    clear = not args.no_clear
    stats = engine.export_to_neo4j(clear_existing=clear)

    nodes = g.run("MATCH (n:IFCElement) RETURN count(n) AS c").data()[0]["c"]
    fills = g.run("MATCH ()-[:FILLS]->() RETURN count(*) AS c").data()[0]["c"]
    print(f"Exported: {nodes} nodes (total), {fills} FILLS edges (total)")

    if nodes == 0:
        print("Error: export produced 0 nodes", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
