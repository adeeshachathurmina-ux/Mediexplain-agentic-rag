"""
Dense Semantic Embedding Engine for MediExplain AI.
DS205.3 - Data Science with Python (Coursework Component - Member 2)

This module implements the Dense Embedding layer using contextual transformer
models (SentenceTransformers).

Why Dense Embeddings in Medical RAG?
While Sparse TF-IDF (Day 2) matches exact clinical numbers and acronyms, it cannot
connect semantically equivalent concepts that use different words:
- "high blood sugar" <---> "hyperglycemia" / "elevated plasma glucose"
- "kidney filtration failure" <---> "renal impairment" / "reduced eGFR"
- "bad cholesterol" <---> "atherogenic low-density lipoprotein (LDL)"

Dense embeddings project text into a continuous 384-dimensional vector space:
v ∈ R^384, where cosine similarity reflects semantic closeness.

Mathematical Formulations:
1. Contextual Embedding:
   v = Transformer_Encoder(tokens) ∈ R^d (d = 384)

2. Unit Sphere Projection (L2 Normalization):
   v̂ = v / ||v||_2   (computed via VectorMathEngine)

3. Dense Cosine Similarity:
   cos(θ) = (q · d) / (||q||_2 * ||d||_2)
   When q and d are L2-normalized unit vectors: cos(θ) = q · d
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from typing import List, Dict, Any, Optional, Union
import numpy as np

from src.mathematics.vector_math import VectorMathEngine


class DenseEmbedder:
    """
    Explainable Dense Embedding Retriever & Semantic Encoder.
    
    Wraps SentenceTransformer with explicit L2 normalization,
    caching, and mathematical explainability logging via VectorMathEngine.
    """

    DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        """
        Initializes the DenseEmbedder.
        Model weights are lazily loaded upon first encoding request.
        
        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu', 'cuda', or None for automatic selection.
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        
        # Precomputed document embeddings cache (Shape: [N_docs, 384])
        self.doc_embeddings_: Optional[np.ndarray] = None
        self.chunks_: List[Dict[str, Any]] = []

    def _load_model(self):
        """Lazily loads the SentenceTransformer model on demand."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load SentenceTransformer model '{self.model_name}'. "
                    f"Ensure internet connectivity or a valid local model cache. Error: {exc}"
                )

    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encodes a single text string into a 1D dense vector.
        
        Args:
            text: Input string (e.g., clinical query).
            normalize: Whether to apply L2 normalization to unit length.
            
        Returns:
            np.ndarray: 1D float64 embedding vector (dimension: 384).
        """
        if not text or not text.strip():
            # Return zero vector for empty input
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float64)

        self._load_model()
        raw_vec = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        vec_1d = raw_vec.astype(np.float64).flatten()

        if normalize:
            return VectorMathEngine.l2_normalize(vec_1d)
        return vec_1d

    def encode_documents(
        self,
        chunks: List[Dict[str, Any]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Batch encodes a list of document chunk dictionaries and caches their vectors.
        
        Args:
            chunks: List of chunk dictionaries containing at least the 'text' key.
            normalize: Whether to apply L2 normalization to unit length.
            
        Returns:
            np.ndarray: 2D matrix of shape [len(chunks), 384].
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunk list.")

        self.chunks_ = chunks
        texts = [chunk.get("text", "") for chunk in chunks]

        self._load_model()
        raw_embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        embeddings = raw_embeddings.astype(np.float64)

        if normalize:
            # Normalize each document vector to unit length
            normalized = np.zeros_like(embeddings)
            for i in range(len(embeddings)):
                normalized[i] = VectorMathEngine.l2_normalize(embeddings[i])
            self.doc_embeddings_ = normalized
        else:
            self.doc_embeddings_ = embeddings

        return self.doc_embeddings_

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic search against cached document embeddings.
        
        Calculates cosine similarity and dot product explicitly via VectorMathEngine.
        
        Args:
            query: Clinical query string.
            top_k: Number of most semantically relevant chunks to return.
            
        Returns:
            List[Dict[str, Any]]: Top scoring chunks with mathematical breakdown.
        """
        if self.doc_embeddings_ is None or not self.chunks_:
            raise RuntimeError("DenseEmbedder has not indexed any documents. Call encode_documents() first.")

        query_vec = self.encode_text(query, normalize=True)
        q_magnitude = VectorMathEngine.vector_magnitude(query_vec)

        scored_results: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(self.chunks_):
            doc_vec = self.doc_embeddings_[idx]
            d_magnitude = VectorMathEngine.vector_magnitude(doc_vec)

            # Compute similarity and dot product via VectorMathEngine
            similarity = VectorMathEngine.cosine_similarity(query_vec, doc_vec)
            dot_prod = VectorMathEngine.dot_product(query_vec, doc_vec)

            result = {
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "unknown"),
                "page": chunk.get("page", 0),
                "dense_score": round(similarity, 6),
                "dot_product": round(dot_prod, 6),
                "query_magnitude": round(q_magnitude, 6),
                "doc_magnitude": round(d_magnitude, 6),
            }
            scored_results.append(result)

        # Sort descending by dense semantic similarity score
        scored_results.sort(key=lambda x: x["dense_score"], reverse=True)
        return scored_results[:top_k]
