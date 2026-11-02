"""
Sparse TF-IDF Retrieval Engine for MediExplain AI.
DS205.3 - Data Science with Python (Coursework Component - Member 2)

This module implements a transparent, explainable Term Frequency - Inverse Document
Frequency (TF-IDF) retriever from foundational text mathematics principles.

Why Sparse TF-IDF in Medical RAG?
Dense semantic embeddings excel at conceptual paraphrasing (e.g., "renal" ≈ "kidney"),
but struggle with exact clinical numbers, units, and laboratory acronyms (e.g.,
distinguishing "HbA1c 6.4%" vs "HbA1c 6.5%", or exact markers like "ALT", "eGFR").
Sparse TF-IDF acts as the exact-match lexical anchor in our hybrid system.

Mathematical Formulation:
1. Term Frequency (Normalized Augmented / Sublinear):
   TF(t, d) = f_{t,d} / |d|
   where f_{t,d} is the count of term t in document d, and |d| is the document length.

2. Inverse Document Frequency (Smooth IDF):
   IDF(t, D) = ln((1 + |D|) / (1 + df_t)) + 1
   where |D| is the total number of documents, and df_t is the number of documents containing t.

3. TF-IDF Weight:
   w_{t, d} = TF(t, d) * IDF(t, D)

4. Sparse Cosine Similarity:
   cos(θ) = (q · d) / (||q||_2 * ||d||_2)  [Computed via VectorMathEngine]
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.mathematics.vector_math import VectorMathEngine


class SparseTFIDFRetriever:
    """
    Explainable Sparse TF-IDF Vector Space Retriever.
    
    Constructs an explicit vocabulary, computes transparent TF and IDF weights,
    and calculates document relevance scores using VectorMathEngine.
    """

    def __init__(self, min_df: int = 1, lowercase: bool = True):
        """
        Initializes the Sparse TF-IDF retriever.
        
        Args:
            min_df: Minimum document frequency for a term to be included in vocabulary.
            lowercase: Whether to convert text to lowercase before tokenization.
        """
        self.min_df = min_df
        self.lowercase = lowercase
        
        # Vocabulary mapping: term -> index
        self.vocabulary_: Dict[str, int] = {}
        # Inverse vocabulary mapping: index -> term
        self.feature_names_: List[str] = []
        # Precomputed IDF weights for each term in vocabulary
        self.idf_weights_: Optional[np.ndarray] = None
        # Stored document chunks
        self.chunks_: List[Dict[str, Any]] = []
        # Precomputed TF-IDF matrix for all indexed document chunks (Shape: [N_docs, N_vocab])
        self.doc_vectors_: Optional[np.ndarray] = None

    def tokenize(self, text: str) -> List[str]:
        """
        Clinical tokenizer preserving medical tokens, abbreviations, and numerical values.
        
        Examples preserved: 'hba1c', 'mg/dl', '6.5%', 'egfr', 'ldl', '70-99'.
        
        Args:
            text: Input clinical string.
            
        Returns:
            List[str]: List of sanitized tokens.
        """
        if not text:
            return []

        if self.lowercase:
            text = text.lower()

        # Token pattern: matches words with alphanumeric chars, hyphens, and medical signs
        # E.g., 'hba1c', '126', 'mg/dl', '6.5'
        tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[/\-\.][a-zA-Z0-9]+)*\b", text)
        return tokens

    def compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Computes normalized Term Frequency for a list of tokens:
        TF(t, d) = count(t, d) / total_tokens(d)
        
        Args:
            tokens: Tokenized document or query.
            
        Returns:
            Dict[str, float]: Mapping of term to its relative frequency.
        """
        if not tokens:
            return {}

        doc_len = float(len(tokens))
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        tf_dict = {term: count / doc_len for term, count in counts.items()}
        return tf_dict

    def fit(self, chunks: List[Dict[str, Any]]) -> "SparseTFIDFRetriever":
        """
        Builds the vocabulary and calculates IDF weights across all document chunks.
        
        Args:
            chunks: List of chunk dictionaries containing at least the 'text' key.
            
        Returns:
            self: Fitted retriever instance.
            
        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("Cannot fit SparseTFIDFRetriever on an empty chunk list.")

        self.chunks_ = chunks
        num_docs = len(chunks)

        # 1. Collect document frequency (DF) for each unique term
        doc_freq: Dict[str, int] = {}
        tokenized_corpus: List[List[str]] = []

        for chunk in chunks:
            text = chunk.get("text", "")
            tokens = self.tokenize(text)
            tokenized_corpus.append(tokens)

            # Unique terms in this single document
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        # 2. Build vocabulary based on min_df threshold
        filtered_terms = [
            term for term, df in sorted(doc_freq.items())
            if df >= self.min_df
        ]

        self.vocabulary_ = {term: idx for idx, term in enumerate(filtered_terms)}
        self.feature_names_ = filtered_terms
        vocab_size = len(self.feature_names_)

        if vocab_size == 0:
            raise ValueError("Vocabulary is empty after applying min_df filter.")

        # 3. Calculate smooth IDF vector:
        # IDF(t, D) = ln((1 + |D|) / (1 + df_t)) + 1
        self.idf_weights_ = np.zeros(vocab_size, dtype=np.float64)
        for term, idx in self.vocabulary_.items():
            df = doc_freq[term]
            self.idf_weights_[idx] = math.log((1.0 + num_docs) / (1.0 + df)) + 1.0

        # 4. Transform and vectorize all document chunks
        self.doc_vectors_ = np.zeros((num_docs, vocab_size), dtype=np.float64)
        for doc_idx, tokens in enumerate(tokenized_corpus):
            tf_dict = self.compute_tf(tokens)
            for term, tf_val in tf_dict.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    self.doc_vectors_[doc_idx, term_idx] = tf_val * self.idf_weights_[term_idx]

        return self

    def transform(self, text: str) -> np.ndarray:
        """
        Transforms an unseen query or text string into a sparse TF-IDF vector.
        
        Args:
            text: Query string.
            
        Returns:
            np.ndarray: TF-IDF vector of length len(vocabulary_).
        """
        if self.idf_weights_ is None or not self.vocabulary_:
            raise RuntimeError("SparseTFIDFRetriever must be fitted before calling transform().")

        vocab_size = len(self.feature_names_)
        query_vec = np.zeros(vocab_size, dtype=np.float64)

        tokens = self.tokenize(text)
        if not tokens:
            return query_vec

        tf_dict = self.compute_tf(tokens)
        for term, tf_val in tf_dict.items():
            if term in self.vocabulary_:
                term_idx = self.vocabulary_[term]
                query_vec[term_idx] = tf_val * self.idf_weights_[term_idx]

        return query_vec

    def get_top_terms_for_query(self, query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Returns the most heavily weighted terms in the query according to TF-IDF.
        Useful for explainability logs during agent execution.
        
        Args:
            query: Query text.
            top_n: Number of significant terms to return.
            
        Returns:
            List[Tuple[str, float]]: (term, weight) pairs sorted by weight descending.
        """
        q_vec = self.transform(query)
        nonzero_indices = np.nonzero(q_vec)[0]

        scored_terms = [
            (self.feature_names_[idx], float(q_vec[idx]))
            for idx in nonzero_indices
        ]
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        return scored_terms[:top_n]

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Searches the indexed chunks using explicit cosine similarity computed by VectorMathEngine.
        
        Args:
            query: Clinical query string (e.g. 'HbA1c 8.2% and creatinine 1.8').
            top_k: Number of highest ranking chunks to return.
            
        Returns:
            List[Dict[str, Any]]: Top scoring chunks with similarity score and math explainability metrics.
        """
        if self.doc_vectors_ is None or not self.chunks_:
            raise RuntimeError("Retriever has not been fitted with document chunks.")

        query_vec = self.transform(query)
        q_magnitude = VectorMathEngine.vector_magnitude(query_vec)

        scored_results: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(self.chunks_):
            doc_vec = self.doc_vectors_[idx]
            d_magnitude = VectorMathEngine.vector_magnitude(doc_vec)

            # Compute similarity using VectorMathEngine
            similarity = VectorMathEngine.cosine_similarity(query_vec, doc_vec)
            dot_prod = VectorMathEngine.dot_product(query_vec, doc_vec)

            result = {
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "unknown"),
                "page": chunk.get("page", 0),
                "tfidf_score": round(similarity, 6),
                "dot_product": round(dot_prod, 6),
                "query_magnitude": round(q_magnitude, 6),
                "doc_magnitude": round(d_magnitude, 6),
            }
            scored_results.append(result)

        # Sort descending by TF-IDF similarity score
        scored_results.sort(key=lambda x: x["tfidf_score"], reverse=True)
        return scored_results[:top_k]

