# 🤝 Team Contribution & Module Specification Guide (DS205.3)

To ensure all 4 members can work in parallel without merge conflicts or integration errors, please follow the exact file names, class names, and data formats below.

---

## 👤 Member 1: PDF Ingestion & Persistent Vector Store
* **Branch:** `member1-ingestion`
* **Files to Create:**
  1. `src/ingestion/pdf_loader.py` → Class `PDFDocumentLoader` (uses `pdfplumber` to extract text & tables from `data/raw_pdfs/*.pdf`)
  2. `src/ingestion/chunker.py` → Class `SmartClinicalChunker(BaseDocumentIngestor)` (splits text with overlap and returns chunk dictionaries)
  3. `src/retrieval/chroma_store.py` → Class `ChromaVectorStore` (saves and loads chunks persistently in `./data/chroma_db`)

* **Required Output Format for Each Chunk:**
```python
{
    "chunk_id": "chunk_01",
    "text": "Extracted medical guideline text...",
    "source": "WHO_Diabetes_Guideline.pdf",
    "page": 1
}
```

---

## 👤 Member 2: Mathematical Foundations & Hybrid Retriever
* **Branch:** `member2-math`
* **Files to Create:**
  1. `src/mathematics/vector_math.py` → Class `VectorMathEngine` (explicit functions for `dot_product`, `vector_magnitude`, `l2_normalize`, and `cosine_similarity`)
  2. `src/mathematics/sparse_tfidf.py` → Class `SparseTFIDFRetriever` (calculates Vocabulary, Term Frequency, IDF, and sparse similarity scores)
  3. `src/retrieval/hybrid_retriever.py` → Class `HybridMedicalRetriever(BaseRetriever)` (combines Sparse TF-IDF + Dense Embeddings and logs mathematical values)

* **Required Output Format for `retrieve(query, top_k=3)`:**
```python
[
    {
        "chunk_id": "chunk_01",
        "text": "HbA1c >= 6.5% indicates diabetes...",
        "source": "WHO_Diabetes_Guideline.pdf",
        "page": 4,
        "tfidf_score": 0.682,
        "dense_score": 0.854,
        "dot_product": 0.854,
        "query_magnitude": 1.000,
        "doc_magnitude": 1.000,
        "hybrid_score": 0.785
    }
]
```

---

## 👤 Member 3: OOP Architecture & Multi-Step Medical Agent
* **Branch:** `member3-agent`
* **Files to Create:**
  1. `src/agent/agent_state.py` → Class `AgentState` (stores query, observations, reasoning steps, retrieved chunks, and final answer)
  2. `src/agent/medical_agent.py` → Class `MedicalReportAgent(BaseAgent)` (implements `observe()`, `reason()`, `act()`, `persist()`, and `run(query)`)

* **Required Output Format for `run(query)`:**
```python
{
    "query": "User question...",
    "observations": ["Detected HbA1c=8.2%...", "..."],
    "reasoning_steps": ["Step 1: Query diabetes guideline...", "..."],
    "retrieved_chunks": [...],
    "final_answer": "Grounded simplified explanation with citations..."
}
```

---

## 👤 Member 4: Evaluation Framework, Ground-Truth Dataset & Main Demo
* **Branch:** `member4-evaluation`
* **Files to Create:**
  1. `data/ground_truth.json` → 10+ clinical Question / Reference Context / Expected Answer pairs
  2. `src/evaluation/evaluator.py` → Class `RAGMetricsEvaluator(BaseEvaluator)` (calculates `context_relevance`, `groundedness`, and `answer_relevance`)
  3. `run_evaluation.py` & `main.py` / `app.py` → CLI / Streamlit interface and automated evaluation table runner
