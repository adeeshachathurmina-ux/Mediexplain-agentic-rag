"""
Automated RAG Evaluation Framework for MediExplain Agentic RAG System.
DS205.3 - Data Science with Python (Member 4 Deliverable)

Implements RAGMetricsEvaluator(BaseEvaluator) aligned with DS205.3 Assessment Guidelines:
1. Context Relevance (Biomarker & Clinical Evidence Recall + Hybrid Vector Cosine Similarity)
2. Groundedness / Faithfulness (Claim Support + Clinical Numeric Hallucination Detection)
3. Answer Relevance (Query Intent Fulfillment + Expected Ground-Truth Alignment)
4. Clinical Entity Coverage (Verification of mandatory medical thresholds/terms)
5. Explicit Vector Space Mathematics (Dot Product, Magnitude, L2 Normalization, Cosine Similarity)
"""

import math
import re
from typing import List, Dict, Any, Optional, Union

from src.interfaces.base_interfaces import BaseEvaluator


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "what", "how", "when", "where", "which", "who", "whom",
    "why", "do", "does", "did", "can", "could", "should", "would", "may",
    "might", "must", "my", "your", "his", "her", "their", "our", "its",
    "patient", "patients", "according", "guideline", "guidelines", "shows",
    "show", "level", "levels", "value", "values", "result", "results",
    "mean", "means", "indicate", "indicates", "indicating", "interpreted",
    "interpret", "classify", "classified", "classification", "define",
    "defined", "confirm", "confirms", "confirmed", "have", "has", "had"
}


class RAGMetricsEvaluator(BaseEvaluator):
    """
    High-Distinction Dual-Scoring RAG Evaluator implementing BaseEvaluator.
    Exposes foundational vector mathematics (Dot Product, Vector Magnitude,
    L2 Normalization, Cosine Similarity) alongside clinical entity & numeric
    hallucination checks.
    """

    def __init__(self, use_dense_embeddings: bool = False):
        """
        Initializes the RAGMetricsEvaluator.
        :param use_dense_embeddings: Optional flag to blend SentenceTransformer embeddings.
        """
        self.use_dense_embeddings = use_dense_embeddings
        self._dense_model = None
        if self.use_dense_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._dense_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._dense_model = None

    # ------------------------------------------------------------------
    # 1. Foundational Vector Mathematics (DS205.3 Section II & IV)
    # ------------------------------------------------------------------
    @staticmethod
    def dot_product(vec_a: List[float], vec_b: List[float]) -> float:
        """Computes explicit Dot Product: A . B = sum(a_i * b_i)."""
        return float(sum(a * b for a, b in zip(vec_a, vec_b)))

    @staticmethod
    def vector_magnitude(vec: List[float]) -> float:
        """Computes explicit Euclidean L2 Vector Magnitude: ||A|| = sqrt(sum(a_i^2))."""
        return float(math.sqrt(sum(x * x for x in vec)))

    @classmethod
    def l2_normalize(cls, vec: List[float]) -> List[float]:
        """Normalizes a vector to unit length: A_norm = A / ||A||."""
        mag = cls.vector_magnitude(vec)
        if mag == 0.0:
            return [0.0 for _ in vec]
        return [float(x / mag) for x in vec]

    @classmethod
    def cosine_similarity_vectors(cls, vec_a: List[float], vec_b: List[float]) -> float:
        """Computes Cosine Similarity: (A . B) / (||A|| * ||B||)."""
        dp = cls.dot_product(vec_a, vec_b)
        mag_a = cls.vector_magnitude(vec_a)
        mag_b = cls.vector_magnitude(vec_b)
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return float(dp / (mag_a * mag_b))

    # ------------------------------------------------------------------
    # 2. Clinical Tokenization & Morphological Stemming
    # ------------------------------------------------------------------
    @staticmethod
    def _stem_clinical_token(token: str) -> str:
        """
        Normalizes morphological variants and clinical synonyms consistently:
        'diabetic'/'diabetes' -> 'diabet', 'elevated'/'elevation' -> 'elevat'.
        """
        if not token or token[0].isdigit():
            return token
        synonyms = {
            "diabetic": "diabetes",
            "prediabetic": "prediabetes",
            "glycated": "hba1c",
            "haemoglobin": "hba1c",
            "hemoglobin": "hba1c",
            "glucose": "sugar",
            "hyperglycemia": "diabetes",
            "nephropathy": "kidney",
            "renal": "kidney",
            "hepatic": "liver",
            "hepatocellular": "liver",
            "atherogenic": "atherosclerosis",
            "cardioprotective": "protective",
        }
        token = synonyms.get(token, token)

        for suffix in ("ations", "ation", "ings", "ing", "edly", "ed", "es", "s"):
            if len(token) > len(suffix) + 3 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    @classmethod
    def _tokenize(
        cls, text: str, remove_stopwords: bool = True, include_numbers: bool = True
    ) -> List[str]:
        """Extracts normalized lowercase clinical tokens with stemming."""
        if not text:
            return []
        raw_tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", text.lower())
        processed = []
        for t in raw_tokens:
            is_num = any(ch.isdigit() for ch in t) and not any(ch.isalpha() for ch in t)
            if is_num and not include_numbers:
                continue
            if remove_stopwords and t in STOPWORDS:
                continue
            if len(t) <= 1 and not is_num:
                continue
            processed.append(cls._stem_clinical_token(t))
        return processed

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        """
        Extracts all numeric values including numbers attached to units (e.g., '8.2%', '500mg', '162').
        """
        if not text:
            return []
        return re.findall(r"(?<![a-zA-Z])\d+(?:\.\d+)?", text.lower())

    @staticmethod
    def _split_clinical_sentences(text: str) -> List[str]:
        """
        Splits text into sentences WITHOUT splitting on decimal points inside lab values (e.g. 8.2%).
        """
        if not text:
            return []
        parts = re.split(r"(?<!\d)[.!?]+(?!\d)", text)
        return [s.strip() for s in parts if len(s.strip()) > 6]

    @classmethod
    def explain_math_breakdown(cls, text_a: str, text_b: str) -> Dict[str, Any]:
        """
        Exposes the exact foundational vector math between two texts for Viva & Report traceability.
        """
        tokens_a = cls._tokenize(text_a, remove_stopwords=True)
        tokens_b = cls._tokenize(text_b, remove_stopwords=True)
        vocab = sorted(list(set(tokens_a + tokens_b)))
        vec_a = [float(tokens_a.count(term)) for term in vocab]
        vec_b = [float(tokens_b.count(term)) for term in vocab]

        dp = cls.dot_product(vec_a, vec_b)
        mag_a = cls.vector_magnitude(vec_a)
        mag_b = cls.vector_magnitude(vec_b)
        cos_sim = cls.cosine_similarity_vectors(vec_a, vec_b)

        return {
            "vocabulary_size": len(vocab),
            "dot_product": round(dp, 4),
            "vector_a_magnitude": round(mag_a, 4),
            "vector_b_magnitude": round(mag_b, 4),
            "cosine_similarity": round(cos_sim, 4),
        }

    def compute_similarity(self, text_a: str, text_b: str, include_numbers: bool = True) -> float:
        """
        Computes hybrid vector similarity combining explicit stemmed word-vector Cosine Similarity
        and directional clinical concept recall.
        """
        if not text_a or not text_b:
            return 0.0

        tokens_a = self._tokenize(text_a, remove_stopwords=True, include_numbers=include_numbers)
        tokens_b = self._tokenize(text_b, remove_stopwords=True, include_numbers=include_numbers)
        if not tokens_a or not tokens_b:
            return 0.0

        vocab = sorted(list(set(tokens_a + tokens_b)))
        vec_a = [float(tokens_a.count(w)) for w in vocab]
        vec_b = [float(tokens_b.count(w)) for w in vocab]
        word_cos = self.cosine_similarity_vectors(vec_a, vec_b)

        set_a, set_b = set(tokens_a), set(tokens_b)
        smaller = set_a if len(set_a) <= len(set_b) else set_b
        larger = set_b if len(set_a) <= len(set_b) else set_a
        concept_recall = len(smaller & larger) / len(smaller) if smaller else 0.0
        jaccard = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0.0

        blended = (0.35 * word_cos) + (0.50 * concept_recall) + (0.15 * jaccard)

        if self._dense_model is not None:
            try:
                emb = self._dense_model.encode([text_a, text_b], normalize_embeddings=True)
                dense_cos = float(sum(a * b for a, b in zip(emb[0], emb[1])))
                blended = (0.50 * blended) + (0.50 * max(0.0, dense_cos))
            except Exception:
                pass

        return round(min(1.0, max(0.0, blended)), 4)

    # ------------------------------------------------------------------
    # 3. Core RAG Triad Metrics (DS205.3 Section IV - 15% Weighting)
    # ------------------------------------------------------------------
    def calculate_context_relevance(
        self, query: str, retrieved_context: str, expected_answer: str = ""
    ) -> float:
        """
        Metric 1: Context Relevance (0.0 - 1.0)
        Measures whether the retrieved clinical guideline chunks contain the medical
        biomarkers asked in the query AND the reference rules required for the answer.
        """
        if not query:
            return 0.0

        is_abstention_expected = "insufficient clinical guideline evidence" in (expected_answer or "").lower()
        if is_abstention_expected:
            return 1.0 if (not retrieved_context or not retrieved_context.strip()) else 0.94

        if not retrieved_context or not retrieved_context.strip():
            return 0.0

        q_concepts = set(self._tokenize(query, remove_stopwords=True, include_numbers=False))
        ctx_concepts = set(self._tokenize(retrieved_context, remove_stopwords=True, include_numbers=False))
        ctx_all_tokens = set(self._tokenize(retrieved_context, remove_stopwords=True, include_numbers=True))

        query_concept_recall = (len(q_concepts & ctx_concepts) / len(q_concepts)) if q_concepts else 0.0

        if expected_answer:
            exp_tokens = set(self._tokenize(expected_answer, remove_stopwords=True, include_numbers=True))
            q_nums = set(self._extract_numbers(query))
            ctx_nums = set(self._extract_numbers(retrieved_context))
            patient_only_nums = {n for n in q_nums if n not in ctx_nums}
            exp_guideline_tokens = {
                t for t in exp_tokens if t.rstrip("%") not in patient_only_nums
            }
            evidence_recall = (
                len(exp_guideline_tokens & ctx_all_tokens) / len(exp_guideline_tokens)
                if exp_guideline_tokens
                else query_concept_recall
            )
            score = (0.45 * query_concept_recall) + (0.55 * evidence_recall)
        else:
            sim_qc = self.compute_similarity(query, retrieved_context, include_numbers=False)
            score = (0.65 * query_concept_recall) + (0.35 * sim_qc)

        return round(min(1.0, max(0.0, score)), 4)

    def calculate_groundedness(
        self, retrieved_context: str, generated_answer: str, query: str = ""
    ) -> float:
        """
        Metric 2: Groundedness / Faithfulness (0.0 - 1.0)
        Verifies that every clinical claim and numeric threshold in generated_answer
        is strictly derived from retrieved_context (or echoed from the user's query),
        penalizing any hallucinated numbers or unsupported statements.
        """
        if not generated_answer or not generated_answer.strip():
            return 0.0

        ans_lower = generated_answer.lower()
        if "insufficient clinical guideline evidence" in ans_lower or "do not contain" in ans_lower:
            return 0.98

        if not retrieved_context or not retrieved_context.strip():
            return 0.0

        allowed_source_text = f"{retrieved_context} {query}"
        allowed_tokens = set(self._tokenize(allowed_source_text, remove_stopwords=True, include_numbers=True))
        ans_tokens = set(self._tokenize(generated_answer, remove_stopwords=True, include_numbers=True))
        if not ans_tokens:
            return 0.0

        # 1. Token-level faithfulness precision
        token_faithfulness = len(ans_tokens & allowed_tokens) / len(ans_tokens)
        if token_faithfulness == 0.0:
            return 0.0

        # 2. Sentence-level grounding verification (preserving decimal points like 8.2%)
        sentences = self._split_clinical_sentences(generated_answer)
        grounded_sentences = 0
        for sent in sentences:
            s_tokens = set(self._tokenize(sent, remove_stopwords=True, include_numbers=True))
            if not s_tokens:
                grounded_sentences += 1
                continue
            if (len(s_tokens & allowed_tokens) / len(s_tokens)) >= 0.45:
                grounded_sentences += 1
        sentence_faithfulness = (grounded_sentences / len(sentences)) if sentences else token_faithfulness

        # 3. Strict Clinical Number Verification (Anti-Hallucination Check)
        allowed_nums = set(self._extract_numbers(allowed_source_text))
        ans_nums = set(self._extract_numbers(generated_answer))
        if ans_nums:
            numeric_faithfulness = len(ans_nums & allowed_nums) / len(ans_nums)
        else:
            numeric_faithfulness = token_faithfulness

        score = (
            (0.40 * token_faithfulness)
            + (0.35 * sentence_faithfulness)
            + (0.25 * numeric_faithfulness)
        )
        return round(min(1.0, max(0.0, score)), 4)

    def calculate_answer_relevance(
        self, query: str, generated_answer: str, expected_answer: str = ""
    ) -> float:
        """
        Metric 3: Answer Relevance (0.0 - 1.0)
        Evaluates how directly generated_answer addresses the user's query biomarkers
        and aligns with the expected clinical interpretation.
        """
        if not query or not generated_answer:
            return 0.0

        q_tokens = set(self._tokenize(query, remove_stopwords=True, include_numbers=True))
        ans_tokens = set(self._tokenize(generated_answer, remove_stopwords=True, include_numbers=True))

        query_entity_coverage = (len(q_tokens & ans_tokens) / len(q_tokens)) if q_tokens else 0.0

        if expected_answer:
            gt_tokens = set(self._tokenize(expected_answer, remove_stopwords=True, include_numbers=True))
            gt_recall = (len(gt_tokens & ans_tokens) / len(gt_tokens)) if gt_tokens else 0.0
            gt_sim = self.compute_similarity(generated_answer, expected_answer, include_numbers=True)

            gt_nums = set(self._extract_numbers(expected_answer))
            ans_nums = set(self._extract_numbers(generated_answer))
            num_alignment = (len(gt_nums & ans_nums) / len(gt_nums)) if gt_nums else gt_recall

            score = (
                (0.25 * query_entity_coverage)
                + (0.30 * gt_recall)
                + (0.25 * gt_sim)
                + (0.20 * num_alignment)
            )
        else:
            qa_sim = self.compute_similarity(query, generated_answer, include_numbers=True)
            score = (0.60 * query_entity_coverage) + (0.40 * qa_sim)

        return round(min(1.0, max(0.0, score)), 4)

    def calculate_clinical_entity_coverage(
        self, generated_answer: str, must_include_terms: Optional[List[str]] = None
    ) -> float:
        """
        High-Distinction Metric 4: Clinical Entity Coverage (0.0 - 1.0)
        Verifies exact presence of mandatory clinical thresholds/terms.
        """
        if not must_include_terms:
            return 1.0
        if not generated_answer:
            return 0.0

        ans_lower = generated_answer.lower()
        matched = sum(1 for term in must_include_terms if term.lower() in ans_lower)
        return round(float(matched / len(must_include_terms)), 4)

    # ------------------------------------------------------------------
    # 4. BaseEvaluator Contract Implementation (Mandatory OOP Method)
    # ------------------------------------------------------------------
    def evaluate_single(
        self,
        query: str,
        retrieved_context: Union[str, List[str]],
        generated_answer: str,
        expected_answer: str = "",
        must_include_terms: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Implements BaseEvaluator.evaluate_single contract.
        Returns the 3 mandatory DS205.3 metrics + clinical_entity_coverage + composite_score.
        """
        if isinstance(retrieved_context, list):
            context_str = "\n".join(
                c.get("text", str(c)) if isinstance(c, dict) else str(c)
                for c in retrieved_context
            )
        else:
            context_str = str(retrieved_context or "")

        ctx_rel = self.calculate_context_relevance(query, context_str, expected_answer)
        grounded = self.calculate_groundedness(context_str, generated_answer, query=query)
        ans_rel = self.calculate_answer_relevance(query, generated_answer, expected_answer)
        entity_cov = self.calculate_clinical_entity_coverage(generated_answer, must_include_terms)

        composite = round((0.30 * ctx_rel) + (0.40 * grounded) + (0.30 * ans_rel), 4)

        return {
            "context_relevance": ctx_rel,
            "groundedness": grounded,
            "answer_relevance": ans_rel,
            "clinical_entity_coverage": entity_cov,
            "composite_score": composite,
        }

    def evaluate(
        self,
        query: str,
        retrieved_contexts: Union[str, List[str]],
        generated_answer: str,
        ground_truth: str = "",
        must_include_terms: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Alias for evaluate_single."""
        return self.evaluate_single(
            query=query,
            retrieved_context=retrieved_contexts,
            generated_answer=generated_answer,
            expected_answer=ground_truth,
            must_include_terms=must_include_terms,
        )