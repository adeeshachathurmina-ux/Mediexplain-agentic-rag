# MediExplain AI: Explainable Agentic Medical Lab Report & Guideline Interpreter

**Module:** DS205.3 – Data Science with Python (NSBM Green University)  
**Assessment:** Coursework (40%)

## Overview
MediExplain AI is a modular, Object-Oriented **Explainable Agentic RAG System** designed to interpret medical laboratory results (Diabetes, Lipid Profile, and Renal/Liver Function panels) grounded strictly in official clinical practice guidelines.

### Key Architectural Highlights
1. **Modular OOP & Dependency Injection (`src/interfaces/`):** Clean separation of concerns via Abstract Base Classes (`BaseDocumentIngestor`, `BaseRetriever`, `BaseAgent`, `BaseEvaluator`).
2. **Mathematical Foundations & Hybrid Retrieval (`src/mathematics/`, `src/retrieval/`):** Combines sparse mathematical text representations (`TF-IDF` / `BoW`) with dense semantic embeddings (`all-MiniLM-L6-v2`), explicitly logging **Dot Products ($\mathbf{q} \cdot \mathbf{d}$)**, **Vector Magnitudes ($\|\mathbf{v}\|$)**, **L2 Normalization**, and **Cosine Similarity**.
3. **Multi-Step Agentic Loop (`src/agent/`):** Implements the **Observation → Reasoning → Action → Persistence** cycle to decompose multi-marker medical queries.
4. **Persistent Vector Store (`data/chroma_db/`):** Uses ChromaDB for persistent storage across sessions.
5. **Automated Evaluation Framework (`src/evaluation/`):** Evaluates 10+ ground-truth clinical queries across **Context Relevance**, **Groundedness (Faithfulness)**, and **Answer Relevance**.

## Quick Start (Reproducibility)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```
