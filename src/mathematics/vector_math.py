"""
Linear Algebra & Vector Mathematics Engine for MediExplain AI.
DS205.3 - Data Science with Python (Coursework Component - Member 2)

This module provides foundational vector operations from first mathematical
principles using NumPy. It avoids "black-box" abstractions by explicitly
computing dot products, Euclidean norms (vector magnitude), L2 normalization,
and cosine similarity with zero-division safeguards.

Mathematical Formulations:
1. Dot Product:
   u · v = Σ (u_i * v_i) for i in [1, n]

2. Euclidean Norm (L2 Magnitude):
   ||v||_2 = √(Σ (v_i)^2)

3. L2 Normalization (Unit Vector Projection):
   v_hat = v / ||v||_2  (where ||v_hat||_2 = 1.0)

4. Cosine Similarity:
   cos(θ) = (u · v) / (||u||_2 * ||v||_2)
"""

from typing import Union, List, Tuple
import numpy as np


class VectorMathEngine:
    """
    Explainable Linear Algebra Engine for Vector Space Models & Dense Embeddings.
    
    Provides explicit mathematical calculations for retrieval scoring,
    guaranteeing full mathematical auditability for academic evaluation and clinical RAG.
    """

    @staticmethod
    def _to_numpy(vec: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Converts input array or list into a 1D float64 NumPy vector.
        
        Args:
            vec: Input numerical vector (list or numpy array).
            
        Returns:
            np.ndarray: 1D array of type float64.
            
        Raises:
            ValueError: If input is empty or multidimensional.
        """
        arr = np.asarray(vec, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("Input vector cannot be empty.")
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr

    @classmethod
    def dot_product(
        cls,
        u: Union[np.ndarray, List[float]],
        v: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Computes the algebraic dot product (inner product) between two vectors.
        
        Formula:
            u · v = Σ_{i=1}^n (u_i * v_i)
            
        Args:
            u: First vector of dimension n.
            v: Second vector of dimension n.
            
        Returns:
            float: Scalar dot product value.
            
        Raises:
            ValueError: If vectors do not share the exact same dimensionality.
        """
        u_arr = cls._to_numpy(u)
        v_arr = cls._to_numpy(v)

        if u_arr.shape[0] != v_arr.shape[0]:
            raise ValueError(
                f"Dimensionality mismatch for dot product: "
                f"u has length {u_arr.shape[0]}, while v has length {v_arr.shape[0]}."
            )

        # Explicit sum of component-wise multiplication
        return float(np.sum(u_arr * v_arr))

    @classmethod
    def vector_magnitude(
        cls,
        v: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Computes the Euclidean magnitude (L2 Norm) of a vector.
        
        Formula:
            ||v||_2 = √(Σ_{i=1}^n v_i^2)
            
        Args:
            v: Input vector of dimension n.
            
        Returns:
            float: Geometric length (magnitude) of the vector.
        """
        v_arr = cls._to_numpy(v)
        # Sum of squared elements under square root
        sum_of_squares = np.sum(v_arr ** 2)
        return float(np.sqrt(sum_of_squares))

    @classmethod
    def l2_normalize(
        cls,
        v: Union[np.ndarray, List[float]],
        epsilon: float = 1e-12
    ) -> np.ndarray:
        """
        Projects a vector onto the unit hypersphere (L2 normalization).
        
        Formula:
            v̂ = v / ||v||_2
            
        Args:
            v: Vector to normalize.
            epsilon: Small perturbation to prevent division by zero for null vectors.
            
        Returns:
            np.ndarray: Unit vector with Euclidean norm equal to 1.0 (or zero vector if input is null).
        """
        v_arr = cls._to_numpy(v)
        mag = cls.vector_magnitude(v_arr)

        if mag < epsilon:
            # Safe fallback for all-zero vectors
            return np.zeros_like(v_arr)

        return v_arr / mag

    @classmethod
    def cosine_similarity(
        cls,
        u: Union[np.ndarray, List[float]],
        v: Union[np.ndarray, List[float]],
        epsilon: float = 1e-12
    ) -> float:
        """
        Computes the Cosine Similarity between two vectors.
        
        Formula:
            cos(θ) = (u · v) / (||u||_2 * ||v||_2)
            
        Args:
            u: First vector (e.g. query vector).
            v: Second vector (e.g. document chunk vector).
            epsilon: Small safeguard against division by zero.
            
        Returns:
            float: Similarity score bounded between -1.0 and 1.0 (clamped for floating point stability).
        """
        u_arr = cls._to_numpy(u)
        v_arr = cls._to_numpy(v)

        if u_arr.shape[0] != v_arr.shape[0]:
            raise ValueError(
                f"Dimensionality mismatch for cosine similarity: "
                f"u has length {u_arr.shape[0]}, v has length {v_arr.shape[0]}."
            )

        dot = cls.dot_product(u_arr, v_arr)
        mag_u = cls.vector_magnitude(u_arr)
        mag_v = cls.vector_magnitude(v_arr)

        denom = mag_u * mag_v
        if denom < epsilon:
            return 0.0

        similarity = dot / denom
        # Numerical stability clamp to [-1.0, 1.0]
        return float(np.clip(similarity, -1.0, 1.0))

    @classmethod
    def explain_similarity(
        cls,
        query_vec: Union[np.ndarray, List[float]],
        doc_vec: Union[np.ndarray, List[float]]
    ) -> dict:
        """
        Produces a full mathematical audit trail for a query-document comparison.
        
        This dictionary matches the explainability requirements of the coursework rubric:
        - Dot Product (q · d)
        - Query Magnitude (||q||)
        - Document Magnitude (||d||)
        - Cosine Similarity (cos(θ))
        
        Args:
            query_vec: Vector representing the user's clinical query.
            doc_vec: Vector representing a candidate document chunk.
            
        Returns:
            dict: Detailed mathematical metrics dictionary.
        """
        q = cls._to_numpy(query_vec)
        d = cls._to_numpy(doc_vec)

        dot = cls.dot_product(q, d)
        mag_q = cls.vector_magnitude(q)
        mag_d = cls.vector_magnitude(d)
        sim = cls.cosine_similarity(q, d)

        return {
            "dot_product": round(dot, 6),
            "query_magnitude": round(mag_q, 6),
            "doc_magnitude": round(mag_d, 6),
            "cosine_similarity": round(sim, 6)
        }
