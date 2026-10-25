"""
Shared Abstract Base Classes (Interfaces) for MediExplain Agentic RAG System.
DS205.3 - Data Science with Python

This file defines the OOP contract so all 4 team members can work in parallel
without blocking each other (Dependency Injection & Interface Segregation).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseDocumentIngestor(ABC):
    """Interface for Member 1: PDF Ingestion & Smart Chunking."""

    @abstractmethod
    def load_and_chunk(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Reads a PDF and returns a list of chunk dictionaries:
        [{"chunk_id": str, "text": str, "source": str, "page": int}]
        """
        pass


class BaseRetriever(ABC):
    """Interface for Member 2: Mathematical Hybrid Retriever (TF-IDF + Dense Embeddings)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant chunks with full mathematical explainability scores:
        [
            {
                "chunk_id": str,
                "text": str,
                "source": str,
                "page": int,
                "tfidf_score": float,
                "dense_score": float,
                "dot_product": float,
                "query_magnitude": float,
                "doc_magnitude": float,
                "hybrid_score": float
            }
        ]
        """
        pass


class BaseAgent(ABC):
    """Interface for Member 3: Multi-Step Agentic Loop (Observe -> Reason -> Act -> Persist)."""

    @abstractmethod
    def run(self, query: str) -> Dict[str, Any]:
        """
        Executes the Agentic Loop and returns:
        {
            "query": str,
            "observations": List[str],
            "reasoning_steps": List[str],
            "retrieved_chunks": List[Dict[str, Any]],
            "final_answer": str
        }
        """
        pass


class BaseEvaluator(ABC):
    """Interface for Member 4: Automated RAG Evaluation Framework."""

    @abstractmethod
    def evaluate_single(
        self, query: str, retrieved_context: str, generated_answer: str, expected_answer: str
    ) -> Dict[str, float]:
        """
        Calculates the 3 mandatory RAG metrics:
        {
            "context_relevance": float,
            "groundedness": float,
            "answer_relevance": float
        }
        """
        pass
