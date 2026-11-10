# Empirical Evaluation Report (DS205.3 - Section IV)

## 1. Overall RAG Triad Summary Metrics (20 Universal Clinical Queries)

- **Mean Context Relevance:** `0.7701` (77.01%)
- **Mean Groundedness (Faithfulness):** `0.9845` (98.45%)
- **Mean Answer Relevance:** `0.8244` (82.44%)
- **Mean Clinical Entity Coverage:** `1.0000` (100.00%)
- **Mean Source Citation Accuracy:** `1.0000` (100.00%)
- **Overall Composite RAG Triad Score:** **`0.8721` (87.21% - High Distinction Grade Band)**

## 2. Query-by-Query Evaluation Table

| ID | Category | Complexity | Context Relevance | Groundedness | Answer Relevance | Entity Coverage | Composite Score | Naive RAG |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| Q01 | Diabetes | Single-Step | 0.7750 | 0.9843 | 0.8502 | 1.0000 | **0.8813** | 0.6674 |
| Q02 | Diabetes | Single-Step | 0.8728 | 0.9871 | 0.8809 | 1.0000 | **0.9210** | 0.7835 |
| Q03 | Diabetes | Single-Step | 0.9241 | 0.9724 | 0.9417 | 1.0000 | **0.9487** | 0.8239 |
| Q04 | Diabetes | Single-Step | 0.6782 | 0.9846 | 0.7610 | 1.0000 | **0.8256** | 0.4422 |
| Q05 | Lipid Profile | Single-Step | 0.7697 | 0.9889 | 0.8791 | 1.0000 | **0.8902** | 0.6900 |
| Q06 | Lipid Profile | Single-Step | 0.8081 | 0.9714 | 0.8933 | 1.0000 | **0.8990** | 0.8318 |
| Q07 | Lipid Profile | Single-Step | 0.6500 | 0.9724 | 0.7913 | 1.0000 | **0.8214** | 0.6467 |
| Q08 | Lipid Profile | Single-Step | 0.6313 | 0.9860 | 0.7575 | 1.0000 | **0.8110** | 0.5983 |
| Q09 | Renal & Liver | Single-Step | 0.8449 | 0.9778 | 0.9393 | 1.0000 | **0.9264** | 0.7680 |
| Q10 | Renal & Liver | Single-Step | 0.9311 | 0.9880 | 0.8979 | 1.0000 | **0.9439** | 0.9344 |
| Q11 | Renal & Liver | Single-Step | 0.7477 | 0.9771 | 0.8542 | 1.0000 | **0.8714** | 0.7251 |
| Q12 | Renal & Liver | Single-Step | 0.7606 | 0.9784 | 0.8601 | 1.0000 | **0.8776** | 0.6806 |
| Q13 | Multi-Condition | Multi-Step Agentic | 0.6402 | 0.9936 | 0.7410 | 1.0000 | **0.8118** | 0.7172 |
| Q14 | Multi-Condition | Multi-Step Agentic | 0.8744 | 0.9917 | 0.8194 | 1.0000 | **0.9048** | 0.5892 |
| Q15 | Multi-Condition | Multi-Step Agentic | 0.6956 | 0.9908 | 0.7245 | 1.0000 | **0.8224** | 0.4970 |
| Q16 | Multi-Condition | Multi-Step Agentic | 0.7408 | 0.9943 | 0.7958 | 1.0000 | **0.8587** | 0.4653 |
| Q17 | Multi-Condition | Multi-Step Agentic | 0.7181 | 0.9928 | 0.7710 | 1.0000 | **0.8438** | 0.5159 |
| Q18 | Multi-Condition | Multi-Step Agentic | 0.7662 | 0.9936 | 0.8036 | 1.0000 | **0.8684** | 0.4750 |
| Q19 | Safety Guardrail | Adversarial / Out-of-Scope | 1.0000 | 0.9800 | 0.8382 | 1.0000 | **0.9435** | 0.7039 |
| Q20 | Safety Guardrail | Adversarial / Out-of-Scope | 0.5738 | 0.9843 | 0.6871 | 1.0000 | **0.7720** | 0.7569 |
| **MEAN** | **All Categories** | **20 Queries** | **0.7701** | **0.9845** | **0.8244** | **1.0000** | **0.8721** | **0.6656** |
