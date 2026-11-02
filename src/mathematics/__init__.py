"""Mathematical foundations package for MediExplain AI."""

from src.mathematics.vector_math import VectorMathEngine
from src.mathematics.sparse_tfidf import SparseTFIDFRetriever
from src.mathematics.dense_embedder import DenseEmbedder

__all__ = ["VectorMathEngine", "SparseTFIDFRetriever", "DenseEmbedder"]
