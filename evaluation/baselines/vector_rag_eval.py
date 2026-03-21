"""
Vector RAG Baseline — Step 2: Evaluate Retrieval

Runs cosine similarity retrieval on the test cases using the FAISS index
built by build_vector_index.py. Measures GT-in-pool and Top-1 accuracy.

Demonstrates that dense embedding retrieval cannot solve the attribute entropy
problem — identical elements produce near-identical embeddings.

Usage:
    python evaluation/baselines/vector_rag_eval.py
    python evaluation/baselines/vector_rag_eval.py \
        --index evaluation/baselines/faiss_index/ \
        --cases evaluation/cases/cases_v3_test.jsonl \
        --output logs/evaluation_output/vector_rag_results.jsonl \
        --plot docs/plots/vector_rag_baseline.png
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def query_to_text(case: dict) -> str:
    """Convert a test case to a query text for embedding."""
    parts = []

    # Use chat history to build the query
    inputs = case.get("inputs", {})
    chat = inputs.get("chat_history", [])
    for msg in chat:
        parts.append(msg.get("text", ""))

    query = case.get("query_text", "")
    if query:
        parts.append(query)

    # Add project context if available
    ctx = inputs.get("project_context", {})
    phase = ctx.get("project_phase", "")
    if phase:
        parts.append(f"Project phase: {phase}")

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="evaluation/baselines/faiss_index/")
    parser.add_argument("--cases", default="evaluation/cases/cases_v3_test.jsonl")
    parser.add_argument("--h2", default="",
                        help="H2 hard-negative JSONL (alternative eval set)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of candidates to retrieve")
    parser.add_argument("--output", default="", help="Save results JSONL")
    parser.add_argument("--plot", default="", help="Save comparison plot")
    args = parser.parse_args()

    # ── Load FAISS index + metadata ──────────────────────────────────────
    print(f"Loading index from: {args.index}")

    with open(os.path.join(args.index, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    guids = meta["guids"]
    embed_model = meta["model"]
    dim = meta["dim"]
    print(f"  {meta['n_elements']} elements, dim={dim}, model={embed_model}")

    embeddings = np.load(os.path.join(args.index, "embeddings.npy"))

    try:
        import faiss
        index = faiss.read_index(os.path.join(args.index, "index.faiss"))
        use_faiss = True
        print(f"  FAISS index loaded: {index.ntotal} vectors")
    except (ImportError, Exception):
        use_faiss = False
        print("  Using numpy fallback (no FAISS)")

    # ── Load cases ───────────────────────────────────────────────────────
    if args.h2:
        # H2 hard-negative format
        print(f"Loading H2 cases from: {args.h2}")
        with open(args.h2) as f:
            raw_cases = [json.loads(l) for l in f if l.strip()]
        cases = []
        for h2 in raw_cases:
            cases.append({
                "case_id": h2["h2_id"],
                "query_text": f"Find the {h2['subject_type']} on {h2['storey_name']}",
                "inputs": {"chat_history": []},
                "bench": {"gt_guid": h2["target_guid"]},
                "_pool_size": h2["pool_size"],
            })
    else:
        print(f"Loading cases from: {args.cases}")
        with open(args.cases) as f:
            cases = [json.loads(l) for l in f if l.strip()]

    print(f"  {len(cases)} cases, Top-K={args.top_k}")

    # ── Embed queries ────────────────────────────────────────────────────
    query_texts = [query_to_text(c) for c in cases]

    print(f"\nEmbedding {len(query_texts)} queries...")
    if embed_model == "openai":
        from eval.baselines.build_vector_index import embed_openai
        query_embeds = np.array(embed_openai(query_texts), dtype=np.float32)
    else:
        from eval.baselines.build_vector_index import embed_local
        query_embeds = np.array(embed_local(query_texts), dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(query_embeds, axis=1, keepdims=True)
    query_embeds = query_embeds / (norms + 1e-10)

    # ── Retrieve ─────────────────────────────────────────────────────────
    print(f"\nRetrieving Top-{args.top_k} for each case...")

    results = []
    n_gt_in_pool = 0
    n_gt_top1 = 0

    hdr = f"{'Case':<12} {'GT GUID':<26} {'Top-1 GUID':<26} {'GT@K':>5} {'Sim':>6}"
    print(hdr)
    print("-" * len(hdr))

    for i, case in enumerate(cases):
        case_id = case.get("case_id", f"case_{i}")
        gt_guid = case.get("bench", {}).get("gt_guid", "")

        q = query_embeds[i:i + 1]

        if use_faiss:
            scores, indices = index.search(q, args.top_k)
            top_guids = [guids[idx] for idx in indices[0]]
            top_scores = scores[0].tolist()
        else:
            sims = (embeddings @ q.T).flatten()
            top_indices = np.argsort(sims)[::-1][:args.top_k]
            top_guids = [guids[idx] for idx in top_indices]
            top_scores = sims[top_indices].tolist()

        gt_in_pool = gt_guid in top_guids
        gt_top1 = top_guids[0] == gt_guid if top_guids else False

        if gt_in_pool:
            n_gt_in_pool += 1
        if gt_top1:
            n_gt_top1 += 1

        gt_mark = "Y" if gt_in_pool else "N"
        top1_sim = top_scores[0] if top_scores else 0.0

        print(f"{case_id:<12} {gt_guid[:24]:<26} {top_guids[0][:24] if top_guids else 'N/A':<26} "
              f"{gt_mark:>5} {top1_sim:>6.3f}")

        results.append({
            "case_id": case_id,
            "gt_guid": gt_guid,
            "top_guids": top_guids,
            "top_scores": [round(s, 4) for s in top_scores],
            "gt_in_pool": gt_in_pool,
            "gt_top1": gt_top1,
            "gt_rank": top_guids.index(gt_guid) + 1 if gt_in_pool else -1,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    n = len(results)
    print()
    print(f"{'='*60}")
    print(f"  Vector RAG Baseline ({n} cases, Top-{args.top_k})")
    print(f"{'='*60}")
    print(f"  GT-in-pool (Top-{args.top_k}): {n_gt_in_pool}/{n} "
          f"({100*n_gt_in_pool/n:.1f}%)")
    print(f"  Top-1 accuracy:      {n_gt_top1}/{n} "
          f"({100*n_gt_top1/n:.1f}%)")

    gt_ranks = [r["gt_rank"] for r in results if r["gt_rank"] > 0]
    if gt_ranks:
        print(f"  Mean GT rank (when found): {sum(gt_ranks)/len(gt_ranks):.1f}")

    print()
    print(f"  Conclusion: Dense embedding retrieval achieves "
          f"{100*n_gt_top1/n:.1f}% Top-1 —")
    print(f"  confirming semantic collapse: identical elements produce")
    print(f"  near-identical embeddings that cannot be disambiguated.")

    # ── Save results ─────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\n  Results saved: {args.output}")

    if args.plot:
        _generate_plot(results, args.top_k, args.plot)


def _generate_plot(results, top_k, out_path):
    """Generate comparison bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    gt_pool = 100 * sum(1 for r in results if r["gt_in_pool"]) / n
    gt_top1 = 100 * sum(1 for r in results if r["gt_top1"]) / n

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [f"GT in Top-{top_k}", "Top-1 Accuracy"],
        [gt_pool, gt_top1],
        color=["#f59e0b", "#ef4444"],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, [gt_pool, gt_top1]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Vector RAG Baseline ({n} cases)\nDense Embedding Retrieval",
                 fontweight="bold")
    ax.set_ylim(0, max(gt_pool, gt_top1) * 1.3 + 5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


if __name__ == "__main__":
    main()
