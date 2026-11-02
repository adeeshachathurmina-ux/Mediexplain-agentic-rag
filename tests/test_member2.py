"""
Master Verification Suite for Member 2: Mathematical Foundations & Hybrid Search
DS205.3 - Data Science with Python

Tests the end-to-end integration of:
1. VectorMathEngine (Vector magnitude, Dot product, Cosine similarity)
2. SparseTFIDFRetriever (Vocabulary, TF, IDF, Sparse scoring)
3. DenseEmbedder (SentenceTransformers dense embeddings)
4. HybridMedicalRetriever (Convex combination fusion and explainable math logs)

Run with:
    python tests/test_member2.py
"""

import sys
import os
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.hybrid_retriever import HybridMedicalRetriever
from src.mathematics.vector_math import VectorMathEngine


def run_member2_master_test():
    print("=" * 75)
    print("[*] MediExplain AI - MEMBER 2: HYBRID RETRIEVAL & MATH ENGINE SUITE")
    print("=" * 75)

    # 1. Load sample chunks from data/sample_chunks.json
    chunks_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_chunks.json")
    )
    assert os.path.exists(chunks_path), f"Cannot find sample chunks at {chunks_path}"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n[Step 1] Loaded {len(chunks)} clinical guideline chunks from sample_chunks.json.")

    # 2. Instantiate and Index HybridMedicalRetriever (alpha = 0.55 for balanced hybrid search)
    print("\n[Step 2] Initializing HybridMedicalRetriever with alpha = 0.55...")
    retriever = HybridMedicalRetriever(chunks=chunks, alpha=0.55)
    print("  -> Sparse TF-IDF vocabulary built successfully.")
    print("  -> Dense SentenceTransformer embeddings cached successfully.")

    # 3. Test Clinical Query 1: Diabetes & HbA1c
    q1 = "HbA1c 8.2% uncontrolled diabetes and blood glucose management"
    print("\n" + "-" * 75)
    print(f"[Clinical Query 1 - Diabetes]: '{q1}'")
    print("-" * 75)
    results_q1 = retriever.retrieve(q1, top_k=2)

    for res in results_q1:
        print(retriever.format_math_log(res))
        print(f"  Excerpt: {res['text'][:95]}...\n")

    # Contract Assertions for Member 3 & 4 compatibility
    assert len(results_q1) == 2
    for r in results_q1:
        assert "chunk_id" in r
        assert "tfidf_score" in r
        assert "dense_score" in r
        assert "dot_product" in r
        assert "query_magnitude" in r
        assert "doc_magnitude" in r
        assert "hybrid_score" in r
    assert results_q1[0]["chunk_id"] == "chunk_01", "Expected chunk_01 for diabetes HbA1c query!"

    # 4. Test Clinical Query 2: Renal / Kidney Function & Creatinine
    q2 = "Serum creatinine 1.8 mg/dL and reduced eGFR kidney damage"
    print("-" * 75)
    print(f"[Clinical Query 2 - Renal/Kidney]: '{q2}'")
    print("-" * 75)
    results_q2 = retriever.retrieve(q2, top_k=2)

    for res in results_q2:
        print(retriever.format_math_log(res))
        print(f"  Excerpt: {res['text'][:95]}...\n")

    assert results_q2[0]["chunk_id"] in ["chunk_07", "chunk_08"]

    # 5. Test Clinical Query 3: Lipid Profile / Atherosclerosis
    q3 = "High LDL cholesterol plaque buildup in blood vessels"
    print("-" * 75)
    print(f"[Clinical Query 3 - Lipid Profile]: '{q3}'")
    print("-" * 75)
    results_q3 = retriever.retrieve(q3, top_k=2)

    for res in results_q3:
        print(retriever.format_math_log(res))
        print(f"  Excerpt: {res['text'][:95]}...\n")

    assert results_q3[0]["source"] == "Clinical_Lipid_Profile_Guideline.pdf"

    # 6. Verify Mathematical Integrity (Convex Combination Check)
    alpha = retriever.alpha
    for r in results_q1 + results_q2 + results_q3:
        expected_hybrid = (alpha * r["dense_score"]) + ((1.0 - alpha) * r["tfidf_score"])
        assert abs(r["hybrid_score"] - round(expected_hybrid, 4)) <= 0.001, (
            f"Convex combination failed: {r['hybrid_score']} != {expected_hybrid}"
        )

    print("=" * 75)
    print("[SUCCESS] ALL MATHEMATICAL & HYBRID RETRIEVAL CHECKS PASSED!")
    print("=" * 75)


if __name__ == "__main__":
    run_member2_master_test()
