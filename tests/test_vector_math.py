"""
Unit & Demonstration Tests for Member 2: VectorMathEngine
DS205.3 - Data Science with Python

Run with:
    python tests/test_vector_math.py
"""

import sys
import os
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mathematics.vector_math import VectorMathEngine


def test_vector_math_operations():
    print("=" * 65)
    print("[*] RUNNING VECTOR MATH VERIFICATION SUITE (MEMBER 2)")
    print("=" * 65)

    # 1. Test Vector Magnitude (Pythagorean Triple: 3, 4 -> 5)
    v1 = [3.0, 4.0]
    mag1 = VectorMathEngine.vector_magnitude(v1)
    print(f"\n[Test 1] Vector Magnitude of {v1}:")
    print(f"  Calculated ||v||: {mag1:.4f} (Expected: 5.0000)")
    assert np.isclose(mag1, 5.0), "Vector magnitude test failed!"

    # 2. Test Dot Product
    u = [1.0, 2.0, 3.0]
    v = [4.0, -5.0, 6.0]
    # 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
    dot = VectorMathEngine.dot_product(u, v)
    print(f"\n[Test 2] Dot Product of {u} . {v}:")
    print(f"  Calculated u . v: {dot:.4f} (Expected: 12.0000)")
    assert np.isclose(dot, 12.0), "Dot product test failed!"

    # 3. Test L2 Normalization
    unit_v = VectorMathEngine.l2_normalize(v1)
    norm_mag = VectorMathEngine.vector_magnitude(unit_v)
    print(f"\n[Test 3] L2 Normalization of {v1}:")
    print(f"  Unit Vector: {unit_v}")
    print(f"  Magnitude of Unit Vector: {norm_mag:.4f} (Expected: 1.0000)")
    assert np.isclose(norm_mag, 1.0), "L2 normalization magnitude must be 1.0!"

    # 4. Test Cosine Similarity
    # Identical vectors -> 1.0
    cos_identical = VectorMathEngine.cosine_similarity([2.0, 5.0], [4.0, 10.0])
    # Orthogonal vectors -> 0.0
    cos_orthogonal = VectorMathEngine.cosine_similarity([1.0, 0.0], [0.0, 1.0])
    # Opposite vectors -> -1.0
    cos_opposite = VectorMathEngine.cosine_similarity([3.0, 0.0], [-5.0, 0.0])

    print("\n[Test 4] Cosine Similarity Cases:")
    print(f"  Collinear (Parallel)      : {cos_identical:.4f} (Expected: 1.0000)")
    print(f"  Orthogonal (Perpendicular): {cos_orthogonal:.4f} (Expected: 0.0000)")
    print(f"  Opposite (Anti-parallel)  : {cos_opposite:.4f} (Expected: -1.0000)")
    assert np.isclose(cos_identical, 1.0)
    assert np.isclose(cos_orthogonal, 0.0)
    assert np.isclose(cos_opposite, -1.0)

    # 5. Test Explainability Audit Trail
    # Clinical Example: Simulated 4D embedding for HbA1c query vs diabetes chunk
    q_vec = [0.85, 0.12, 0.50, 0.05]
    d_vec = [0.80, 0.10, 0.55, 0.02]
    audit = VectorMathEngine.explain_similarity(q_vec, d_vec)

    print("\n[Test 5] Mathematical Explainability Audit Trail (Clinical Demo):")
    for key, val in audit.items():
        print(f"  -> {key:20s}: {val}")

    print("\n" + "=" * 65)
    print("[SUCCESS] ALL VECTOR MATH TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_vector_math_operations()
