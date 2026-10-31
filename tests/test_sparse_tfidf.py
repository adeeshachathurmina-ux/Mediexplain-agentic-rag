"""
Unit & Demonstration Tests for Member 2: SparseTFIDFRetriever
DS205.3 - Data Science with Python

Run with:
    python tests/test_sparse_tfidf.py
"""

import sys
import os
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mathematics.sparse_tfidf import SparseTFIDFRetriever


def test_sparse_tfidf_retriever():
    print("=" * 70)
    print("[*] RUNNING SPARSE TF-IDF RETRIEVER VERIFICATION SUITE (MEMBER 2)")
    print("=" * 70)

    # 1. Load sample chunks from data/sample_chunks.json
    chunks_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_chunks.json")
    )
    assert os.path.exists(chunks_path), f"Cannot find sample chunks at {chunks_path}"

    with open(chunks_path, "r", encoding="utf-8") as f:
        sample_chunks = json.load(f)

    print(f"\n[Step 1] Loaded {len(sample_chunks)} sample clinical chunks from sample_chunks.json.")

    # 2. Fit the SparseTFIDFRetriever
    retriever = SparseTFIDFRetriever(min_df=1, lowercase=True)
    retriever.fit(sample_chunks)

    vocab_size = len(retriever.vocabulary_)
    print(f"\n[Step 2] Built Clinical Vocabulary successfully.")
    print(f"  Vocabulary size: {vocab_size} distinct terms.")
    
    # Inspect key clinical terms in vocabulary
    key_terms = ["hba1c", "creatinine", "diabetes", "ldl", "mg/dl"]
    print("  Inspection of key clinical term IDs and precomputed IDF weights:")
    for term in key_terms:
        if term in retriever.vocabulary_:
            idx = retriever.vocabulary_[term]
            idf = retriever.idf_weights_[idx]
            print(f"    - '{term:12s}' -> Index: {idx:3d} | IDF: {idf:.4f}")
        else:
            print(f"    - '{term:12s}' -> [NOT FOUND]")

    # 3. Test Clinical Query 1: Diabetes & HbA1c
    q1 = "What does HbA1c level 6.5% mean for diabetes?"
    print(f"\n[Step 3] Query 1: '{q1}'")
    top_terms_q1 = retriever.get_top_terms_for_query(q1, top_n=3)
    print(f"  Most informative query terms (TF-IDF): {top_terms_q1}")

    results_q1 = retriever.search(q1, top_k=2)
    print("  Top Retrieved Chunks:")
    for r in results_q1:
        print(f"    -> Chunk ID: {r['chunk_id']} | TF-IDF Cosine Score: {r['tfidf_score']:.4f}")
        print(f"       Dot Product: {r['dot_product']:.4f} | ||q||: {r['query_magnitude']:.4f} | ||d||: {r['doc_magnitude']:.4f}")
        print(f"       Source: {r['source']} (Page {r['page']})")
        print(f"       Excerpt: {r['text'][:90]}...\n")

    # The top retrieved chunk for HbA1c should be chunk_01
    assert results_q1[0]["chunk_id"] == "chunk_01", "Expected chunk_01 for HbA1c query!"

    # 4. Test Clinical Query 2: Kidney & Creatinine
    q2 = "Serum creatinine 1.8 mg/dL kidney filtration function"
    print(f"[Step 4] Query 2: '{q2}'")
    results_q2 = retriever.search(q2, top_k=2)
    print("  Top Retrieved Chunks:")
    for r in results_q2:
        print(f"    -> Chunk ID: {r['chunk_id']} | TF-IDF Cosine Score: {r['tfidf_score']:.4f}")
        print(f"       Dot Product: {r['dot_product']:.4f} | ||q||: {r['query_magnitude']:.4f} | ||d||: {r['doc_magnitude']:.4f}")
        print(f"       Source: {r['source']} (Page {r['page']})")
        print(f"       Excerpt: {r['text'][:90]}...\n")

    # Top result should be chunk_07 (Creatinine / kidney)
    assert results_q2[0]["chunk_id"] in ["chunk_07", "chunk_08"], "Expected kidney chunk for creatinine query!"

    print("=" * 70)
    print("[SUCCESS] ALL SPARSE TF-IDF RETRIEVER TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_sparse_tfidf_retriever()

