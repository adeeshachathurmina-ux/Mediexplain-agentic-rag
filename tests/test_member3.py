"""
Standalone Unit Test & Verification for Member 3: OOP Medical Agent.
DS205.3 - Data Science with Python

Uses Dependency Injection with a MockRetriever so Member 3 can test
the full Agentic Loop independently without waiting for Member 1 or 2.
"""

import os
import sys
import json
from pathlib import Path

# Configure stdout and stderr for UTF-8 compatibility
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.interfaces.base_interfaces import BaseRetriever
from src.agent.medical_agent import MedicalReportAgent


class MockMedicalRetriever(BaseRetriever):
    """
    Mock retriever loading data/sample_chunks.json.
    Simulates Member 2's Hybrid Retriever with full mathematical explainability outputs.
    """

    def __init__(self, sample_data_path: str = None):
        if not sample_data_path:
            sample_data_path = str(ROOT_DIR / "data" / "sample_chunks.json")

        self.sample_chunks = []
        if os.path.exists(sample_data_path):
            with open(sample_data_path, "r", encoding="utf-8") as f:
                self.sample_chunks = json.load(f)

    def retrieve(self, query: str, top_k: int = 2):
        """
        Simulates hybrid retrieval and mathematical scoring (TF-IDF + Dense Sim).
        """
        query_words = set(query.lower().split())
        scored = []

        for chunk in self.sample_chunks:
            text = chunk.get("text", "").lower()
            overlap = sum(1 for w in query_words if len(w) > 3 and w in text)

            # Generate realistic explainable math metrics
            tfidf_sim = min(0.95, overlap * 0.22 + 0.15)
            dense_sim = min(0.98, overlap * 0.18 + 0.35)
            dot_prod = dense_sim  # Normalized vectors: dot product == cosine similarity
            hybrid_score = (0.5 * tfidf_sim) + (0.5 * dense_sim)

            scored.append({
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "Guideline.pdf"),
                "page": chunk.get("page", 1),
                "tfidf_score": round(tfidf_sim, 3),
                "dense_score": round(dense_sim, 3),
                "dot_product": round(dot_prod, 3),
                "query_magnitude": 1.000,
                "doc_magnitude": 1.000,
                "hybrid_score": round(hybrid_score, 3)
            })

        # Sort by hybrid score descending
        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:top_k]


def run_member3_verification():
    print("=" * 75)
    print("🩺 MEDIEXPLAIN AI - MEMBER 3 AGENTIC LOOP VERIFICATION")
    print("=" * 75)

    # 1. Initialize Mock Retriever (Dependency Injection)
    retriever = MockMedicalRetriever()
    print(f"[*] Loaded {len(retriever.sample_chunks)} sample chunks into MockRetriever.")

    # 2. Instantiate Medical Agent
    agent = MedicalReportAgent(retriever=retriever)
    print("[*] MedicalReportAgent initialized successfully.")

    # 3. Test Multi-Condition Query (Diabetes + Renal)
    query = "Patient blood report shows HbA1c: 8.2% and Serum Creatinine: 1.8 mg/dL. What does this indicate?"
    print(f"\n[?] User Input Query:\n    \"{query}\"\n")

    result = agent.run(query)

    # Verify Observations
    print("=" * 75)
    print("👀 STEP 1: OBSERVATIONS (Clinical Biomarkers Extracted)")
    print("=" * 75)
    for obs in result["observations"]:
        print(f"  • {obs}")

    # Verify Reasoning Steps
    print("\n" + "=" * 75)
    print("🧠 STEP 2 & 3: REASONING & ACTION LOGS (Chain-of-Thought + Math Logs)")
    print("=" * 75)
    for step in result["reasoning_steps"]:
        print(f"  {step}")

    # Verify Retrieved Chunks & Math Explainability
    print("\n" + "=" * 75)
    print("⚡ RETRIEVED CHUNKS & MATHEMATICAL EXPLAINABILITY SCORES")
    print("=" * 75)
    for chunk in result["retrieved_chunks"]:
        print(f"  [{chunk['chunk_id']}] Source: {chunk['source']} (Page {chunk['page']})")
        print(f"      TF-IDF: {chunk['tfidf_score']} | Dense: {chunk['dense_score']} | Dot Prod: {chunk['dot_product']} | Hybrid: {chunk['hybrid_score']}")
        print(f"      Text: {chunk['text'][:90]}...")

    # Verify Final Answer
    print("\n" + "=" * 75)
    print("💾 STEP 4: PERSISTENCE & FINAL GROUNDED CLINICAL ANSWER")
    print("=" * 75)
    print(result["final_answer"])
    print("=" * 75)

    # Automated assertions
    assert len(result["observations"]) >= 2, "Assertion failed: Failed to extract 2 clinical markers."
    assert len(result["reasoning_steps"]) >= 2, "Assertion failed: Reasoning loop did not log steps."
    assert len(result["retrieved_chunks"]) > 0, "Assertion failed: No chunks retrieved."
    assert len(result["final_answer"]) > 50, "Assertion failed: Final answer too short or missing."

    print("\n✅ ALL MEMBER 3 UNIT TESTS PASSED SUCCESSFULLY! 🚀")


if __name__ == "__main__":
    run_member3_verification()
