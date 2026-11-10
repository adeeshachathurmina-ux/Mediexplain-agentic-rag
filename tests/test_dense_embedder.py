"""
Unit & Demonstration Tests for Member 2: DenseEmbedder
DS205.3 - Data Science with Python

Run with:
    python tests/test_dense_embedder.py
"""

import sys
import os
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mathematics.dense_embedder import DenseEmbedder
from src.mathematics.vector_math import VectorMathEngine


def test_dense_embedder():
    print("=" * 70)
    print("[*] RUNNING DENSE EMBEDDER VERIFICATION SUITE (MEMBER 2)")
    print("=" * 70)

    # 1. Load sample chunks from data/sample_chunks.json
    chunks_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_chunks.json")
    )
    assert os.path.exists(chunks_path), f"Cannot find sample chunks at {chunks_path}"

    with open(chunks_path, "r", encoding="utf-8") as f:
        sample_chunks = json.load(f)

    print(f"\n[Step 1] Loaded {len(sample_chunks)} sample chunks from sample_chunks.json.")

    # 2. Initialize DenseEmbedder
    embedder = DenseEmbedder()
    print("\n[Step 2] Initialized DenseEmbedder (model: sentence-transformers/all-MiniLM-L6-v2).")

    # 3. Test single query encoding & unit normalization
    query = "elevated blood sugar and glucose tolerance"
    q_vec = embedder.encode_text(query, normalize=True)
    q_magnitude = VectorMathEngine.vector_magnitude(q_vec)

    print(f"\n[Step 3] Encoded query '{query}':")
    print(f"  Embedding Dimension: {len(q_vec)} (Expected: 384)")
    print(f"  Unit Magnitude ||q||: {q_magnitude:.6f} (Expected: 1.000000)")
    assert len(q_vec) == 384, "Embedding dimensionality must be 384!"
    assert np.isclose(q_magnitude, 1.0, atol=1e-5), "Normalized embedding magnitude must be 1.0!"

    # 4. Batch encode document chunks
    print("\n[Step 4] Batch encoding sample chunks...")
    doc_matrix = embedder.encode_documents(sample_chunks, normalize=True)
    print(f"  Generated document embedding matrix with shape: {doc_matrix.shape}")
    assert doc_matrix.shape == (len(sample_chunks), 384)

    # 5. Semantic Search Test 1: Cholesterol & Arterial Plaque
    # Query uses natural paraphrasing ("fatty blockage in heart blood vessels")
    # without explicitly using the exact term "atherosclerosis" or "LDL"
    q1 = "fatty blockage in heart blood vessels"
    print(f"\n[Step 5] Paraphrased Semantic Query: '{q1}'")
    results_q1 = embedder.search(q1, top_k=2)

    for r in results_q1:
        print(f"  -> Chunk ID: {r['chunk_id']} | Dense Score: {r['dense_score']:.4f}")
        print(f"     Dot Product: {r['dot_product']:.4f} | ||q||: {r['query_magnitude']:.4f} | ||d||: {r['doc_magnitude']:.4f}")
        print(f"     Source: {r['source']} (Page {r['page']})")
        print(f"     Excerpt: {r['text'][:90]}...\n")

    # Top chunk should semantically match the Lipid Profile guideline (chunk_04, 05, or 06)
    assert results_q1[0]["source"] == "Clinical_Lipid_Profile_Guideline.pdf", (
        "Expected Clinical_Lipid_Profile_Guideline.pdf for artery plaque / blood fat query!"
    )

    # 6. Verify mathematical property:
    # For unit-normalized vectors: Dot Product == Cosine Similarity
    top_r = results_q1[0]
    print("[Step 6] Verification of Mathematical Property for Unit Hypersphere:")
    print(f"  Dot Product       : {top_r['dot_product']:.6f}")
    print(f"  Cosine Similarity : {top_r['dense_score']:.6f}")
    assert np.isclose(top_r['dot_product'], top_r['dense_score'], atol=1e-5), (
        "For unit vectors, dot product must equal cosine similarity!"
    )
    print("  -> Proved: dot_product == cosine_similarity when ||u|| = ||v|| = 1.0")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL DENSE EMBEDDER TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_dense_embedder()
