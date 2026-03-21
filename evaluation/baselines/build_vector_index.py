"""
Vector RAG Baseline — Step 1: Build FAISS Index

Embeds all IFC elements from element_index.jsonl as text chunks and stores
them in a FAISS index for cosine similarity retrieval.

Demonstrates "semantic collapse" — identical elements produce near-identical
embeddings, making dense retrieval unable to disambiguate.

Usage:
    # Using OpenAI embeddings (requires OPENAI_API_KEY)
    python evaluation/baselines/build_vector_index.py \
        --elements ../data_curation/references/element_index.jsonl \
        --output evaluation/baselines/faiss_index/ \
        --model openai

    # Using local sentence-transformers (free, no API key)
    python evaluation/baselines/build_vector_index.py \
        --elements ../data_curation/references/element_index.jsonl \
        --output evaluation/baselines/faiss_index/ \
        --model local
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path


def element_to_text(elem: dict) -> str:
    """Convert an IFC element record to a text chunk for embedding."""
    parts = []
    parts.append(f"{elem.get('ifc_class', 'Unknown')} element")

    name = elem.get("name", "")
    if name:
        parts.append(f"named '{name}'")

    storey = elem.get("storey_name", "")
    if storey:
        parts.append(f"on storey '{storey}'")

    material = elem.get("material", "")
    if material:
        parts.append(f"made of {material}")

    obj_type = elem.get("object_type", "")
    if obj_type:
        parts.append(f"type: {obj_type}")

    dims = elem.get("dimensions", {})
    if dims:
        dim_parts = []
        for k in ("Width", "Height", "Length"):
            v = dims.get(k)
            if v is not None:
                dim_parts.append(f"{k}={v:.0f}mm")
        if dim_parts:
            parts.append(f"dimensions: {', '.join(dim_parts)}")

    fire = elem.get("fire_rating", "")
    if fire:
        parts.append(f"fire rating: {fire}")

    return ". ".join(parts) + "."


def embed_openai(texts: list, model_name: str = "text-embedding-3-small") -> list:
    """Embed texts using OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    # Batch in groups of 100
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model_name)
        all_embeddings.extend([d.embedding for d in response.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return all_embeddings


def embed_local(texts: list, model_name: str = "all-MiniLM-L6-v2") -> list:
    """Embed texts using sentence-transformers (local, free)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    print(f"  Using local model: {model_name}")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements",
                        default="../data_curation/references/element_index.jsonl")
    parser.add_argument("--output", default="evaluation/baselines/faiss_index/")
    parser.add_argument("--model", default="local",
                        choices=["openai", "local"],
                        help="Embedding model: openai (API) or local (free)")
    args = parser.parse_args()

    # ── Load elements ────────────────────────────────────────────────────
    print(f"Loading elements from: {args.elements}")
    elements = []
    with open(args.elements) as f:
        for line in f:
            if line.strip():
                elements.append(json.loads(line))
    print(f"  Loaded {len(elements)} elements")

    # ── Convert to text ──────────────────────────────────────────────────
    texts = [element_to_text(e) for e in elements]
    guids = [e["global_id"] for e in elements]

    # Show a few examples
    print(f"\n  Example text chunks:")
    for i in range(min(3, len(texts))):
        print(f"    [{i}] {texts[i][:120]}...")

    # ── Check for duplicates (the semantic collapse proof) ───────────────
    unique_texts = set(texts)
    print(f"\n  Unique text chunks: {len(unique_texts)}/{len(texts)}")
    if len(unique_texts) < len(texts):
        from collections import Counter
        text_counts = Counter(texts)
        top_dupes = text_counts.most_common(5)
        print(f"  Top duplicate chunks:")
        for txt, count in top_dupes:
            print(f"    {count}x: {txt[:100]}...")

    # ── Embed ────────────────────────────────────────────────────────────
    print(f"\nEmbedding {len(texts)} elements (model={args.model})...")
    if args.model == "openai":
        embeddings = embed_openai(texts)
    else:
        embeddings = embed_local(texts)

    # ── Build FAISS index ────────────────────────────────────────────────
    import numpy as np

    embed_matrix = np.array(embeddings, dtype=np.float32)
    dim = embed_matrix.shape[1]
    print(f"  Embedding dim: {dim}")

    # Normalize for cosine similarity (FAISS IP = cosine on normalized vectors)
    norms = np.linalg.norm(embed_matrix, axis=1, keepdims=True)
    embed_matrix = embed_matrix / (norms + 1e-10)

    try:
        import faiss
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized
        index.add(embed_matrix)
        print(f"  FAISS index built: {index.ntotal} vectors")
    except ImportError:
        print("  FAISS not installed — saving raw numpy instead")
        print("  Install: pip install faiss-cpu")
        index = None

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    # Save metadata
    meta = {
        "guids": guids,
        "texts": texts,
        "model": args.model,
        "dim": dim,
        "n_elements": len(elements),
    }
    with open(os.path.join(args.output, "metadata.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # Save embeddings
    np.save(os.path.join(args.output, "embeddings.npy"), embed_matrix)

    # Save FAISS index
    if index is not None:
        faiss.write_index(index, os.path.join(args.output, "index.faiss"))
        print(f"\n  FAISS index saved to: {args.output}")
    else:
        print(f"\n  Embeddings saved to: {args.output} (no FAISS)")

    # ── Duplicate embedding analysis ─────────────────────────────────────
    print(f"\n  Embedding similarity analysis (semantic collapse check):")
    # Find max similarity between different elements of the same type+storey
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, e in enumerate(elements):
        key = (e.get("ifc_class"), e.get("storey_name"))
        buckets[key].append(i)

    max_sim_examples = []
    for key, indices in sorted(buckets.items(), key=lambda x: -len(x[1]))[:5]:
        if len(indices) < 2:
            continue
        # Compute pairwise similarity within bucket
        bucket_vecs = embed_matrix[indices]
        sims = bucket_vecs @ bucket_vecs.T
        # Exclude self-similarity
        np.fill_diagonal(sims, -1)
        max_sim = sims.max()
        avg_sim = (sims.sum() + len(indices)) / (len(indices) * (len(indices) - 1))
        ifc_class, storey = key
        print(f"    {ifc_class} on {storey}: {len(indices)} elements, "
              f"max_sim={max_sim:.4f}, avg_sim={avg_sim:.4f}")

    print(f"\n  Done. Next: python evaluation/baselines/vector_rag_eval.py")


if __name__ == "__main__":
    main()
