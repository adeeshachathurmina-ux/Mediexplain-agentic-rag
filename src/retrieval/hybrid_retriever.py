"""
Hybrid Clinical Retriever for MediExplain AI.
DS205.3 - Data Science with Python (Coursework Component - Member 2)

This module implements the HybridMedicalRetriever adhering to BaseRetriever interface.
It fuses foundational Sparse Lexical TF-IDF representations with Dense Contextual
Transformer Embeddings to achieve mathematically explainable, high-precision retrieval.

Mathematical Hybrid Fusion Formulation:
    Score_hybrid(q, d) = α * Score_dense(q, d) + (1 - α) * Score_sparse(q, d)

Where:
    - α ∈ [0.0, 1.0] (default = 0.5): Convex combination weighting parameter.
    - Score_dense(q, d): Cosine similarity computed over 384-dimensional dense vectors:
        cos(θ_dense) = (q_dense · d_dense) / (||q_dense|| * ||d_dense||)
    - Score_sparse(q, d): Cosine similarity computed over high-dimensional sparse TF-IDF vectors:
        cos(θ_sparse) = (q_sparse · d_sparse) / (||q_sparse|| * ||d_sparse||)
    - Both scoring components are computed via the foundational VectorMathEngine.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.interfaces.base_interfaces import BaseRetriever
from src.mathematics.vector_math import VectorMathEngine
from src.mathematics.sparse_tfidf import SparseTFIDFRetriever
from src.mathematics.dense_embedder import DenseEmbedder


class HybridMedicalRetriever(BaseRetriever):
    """
    Explainable Hybrid Clinical Retriever fusing Sparse TF-IDF and Dense Transformer Embeddings.
    
    Implements Dependency Injection, allowing custom sparse and dense engines,
    and returns full mathematical audit trails (dot product, magnitudes, hybrid score).
    """

    def __init__(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        alpha: float = 0.5,
        sparse_retriever: Optional[SparseTFIDFRetriever] = None,
        dense_embedder: Optional[DenseEmbedder] = None
    ):
        """
        Initializes the HybridMedicalRetriever.
        
        Args:
            chunks: Optional list of document chunk dictionaries to index immediately.
            alpha: Convex fusion weight (0.0 <= alpha <= 1.0).
                   alpha = 1.0 -> purely dense (semantic).
                   alpha = 0.0 -> purely sparse (lexical keyword).
                   alpha = 0.5 -> balanced hybrid.
            sparse_retriever: Injected SparseTFIDFRetriever instance (or None to instantiate).
            dense_embedder: Injected DenseEmbedder instance (or None to instantiate).
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha parameter must be in [0.0, 1.0], got {alpha}.")

        self.alpha = float(alpha)
        self.sparse_retriever = sparse_retriever or SparseTFIDFRetriever(min_df=1, lowercase=True)
        self.dense_embedder = dense_embedder or DenseEmbedder()
        self.chunks: List[Dict[str, Any]] = []

        if chunks:
            self.index_documents(chunks)

    def index_documents(self, chunks: List[Dict[str, Any]]) -> "HybridMedicalRetriever":
        """
        Indexes chunks into both the Sparse TF-IDF engine and the Dense Embedding model.
        
        Args:
            chunks: List of chunk dictionaries containing 'chunk_id', 'text', 'source', 'page'.
            
        Returns:
            self: Configured retriever instance.
        """
        if not chunks:
            raise ValueError("Cannot index empty chunks list.")

        self.chunks = chunks

        # 1. Fit Sparse TF-IDF vocabulary & document matrix
        self.sparse_retriever.fit(self.chunks)

        # 2. Encode and cache normalized Dense Embeddings
        self.dense_embedder.encode_documents(self.chunks, normalize=True)

        return self

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant chunks with full mathematical explainability scores.
        
        Matches the exact output contract required by Member 3 (Agent) and Member 4 (Evaluator):
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
        
        Args:
            query: Clinical question or laboratory value query.
            top_k: Number of highest ranking chunks to return.
            
        Returns:
            List[Dict[str, Any]]: Top k chunks sorted descending by hybrid_score.
        """
        if not self.chunks:
            raise RuntimeError("HybridMedicalRetriever has no documents indexed. Call index_documents() first.")

        # 1. Transform query into sparse TF-IDF and dense embedding vectors
        q_sparse = self.sparse_retriever.transform(query)
        q_dense = self.dense_embedder.encode_text(query, normalize=True)

        # Query vector magnitude for dense embedding (is 1.0 under unit normalization)
        q_dense_mag = VectorMathEngine.vector_magnitude(q_dense)

        scored_chunks: List[Dict[str, Any]] = []

        # 2. Score every chunk against both vector spaces
        for idx, chunk in enumerate(self.chunks):
            # A. Sparse TF-IDF Scoring
            d_sparse = self.sparse_retriever.doc_vectors_[idx]
            tfidf_sim = VectorMathEngine.cosine_similarity(q_sparse, d_sparse)

            # B. Dense Embedding Scoring
            d_dense = self.dense_embedder.doc_embeddings_[idx]
            dense_sim = VectorMathEngine.cosine_similarity(q_dense, d_dense)
            dense_dot = VectorMathEngine.dot_product(q_dense, d_dense)
            d_dense_mag = VectorMathEngine.vector_magnitude(d_dense)

            # C. Convex Combination Hybrid Score:
            # Score_hybrid = α * Score_dense + (1 - α) * Score_sparse
            hybrid_score = (self.alpha * dense_sim) + ((1.0 - self.alpha) * tfidf_sim)

            scored_chunk = {
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "unknown"),
                "page": chunk.get("page", 0),
                "tfidf_score": round(float(tfidf_sim), 4),
                "dense_score": round(float(dense_sim), 4),
                "dot_product": round(float(dense_dot), 4),
                "query_magnitude": round(float(q_dense_mag), 4),
                "doc_magnitude": round(float(d_dense_mag), 4),
                "hybrid_score": round(float(hybrid_score), 4)
            }
            scored_chunks.append(scored_chunk)

        # 3. Sort descending by hybrid_score
        scored_chunks.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return scored_chunks[:top_k]

    def format_math_log(self, result: Dict[str, Any]) -> str:
        """
        Formats a single retrieval result into an explainable mathematical log string.
        Perfect for console printing and video presentation / Viva defense.
        
        Args:
            result: Result dictionary from retrieve().
            
        Returns:
            str: Pretty formatted mathematical proof log.
        """
        return (
            f"[Chunk #{result['chunk_id']}] ({result['source']}, p.{result['page']})\n"
            f"  |-- TF-IDF Score (Sparse) : {result['tfidf_score']:.4f}\n"
            f"  |-- Dense Score (Cosine)  : {result['dense_score']:.4f}\n"
            f"  |-- Vector Dot Product    : {result['dot_product']:.4f}\n"
            f"  |-- Magnitudes            : ||q||={result['query_magnitude']:.4f}, ||d||={result['doc_magnitude']:.4f}\n"
            f"  \\-- Hybrid Score (alpha={self.alpha:.2f}): {result['hybrid_score']:.4f}"
        )
