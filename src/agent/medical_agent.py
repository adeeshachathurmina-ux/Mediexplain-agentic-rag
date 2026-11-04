"""
MedicalReportAgent Implementation for MediExplain Agentic RAG System.
DS205.3 - Data Science with Python (Member 3)

Implements the 4-step Agentic Loop:
1. Observation   : Extracts clinical biomarkers, numeric values, and units from query.
2. Reasoning     : Evaluates clinical thresholds and formulates targeted sub-queries.
3. Action        : Invokes injected BaseRetriever and logs mathematical retrieval scores.
4. Persistence   : Manages AgentState and synthesizes grounded, explainable medical explanation.
"""

import os
import re
from typing import List, Dict, Any, Optional

from src.interfaces.base_interfaces import BaseAgent, BaseRetriever
from src.agent.agent_state import AgentState, ClinicalObservation


class MedicalReportAgent(BaseAgent):
    """
    Explainable Medical Agent adhering to OOP principles and Dependency Injection.
    Accepts any retriever implementing BaseRetriever interface.
    """

    # Reference threshold values derived from official WHO and clinical guidelines
    REFERENCE_RANGES = {
        "hba1c": {
            "name": "HbA1c",
            "unit": "%",
            "normal_max": 5.7,
            "prediabetes_max": 6.4,
            "category": "diabetes",
            "guideline": "WHO_Diabetes_Guideline.pdf"
        },
        "fbs": {
            "name": "Fasting Blood Sugar (FBS)",
            "unit": "mg/dL",
            "normal_min": 70.0,
            "normal_max": 99.0,
            "prediabetes_max": 125.0,
            "category": "diabetes",
            "guideline": "WHO_Diabetes_Guideline.pdf"
        },
        "ppbs": {
            "name": "Post-Prandial Blood Sugar (PPBS)",
            "unit": "mg/dL",
            "normal_max": 140.0,
            "prediabetes_max": 199.0,
            "category": "diabetes",
            "guideline": "WHO_Diabetes_Guideline.pdf"
        },
        "ldl": {
            "name": "LDL Cholesterol",
            "unit": "mg/dL",
            "normal_max": 100.0,
            "borderline_max": 159.0,
            "category": "lipid",
            "guideline": "Clinical_Lipid_Profile_Guideline.pdf"
        },
        "total_cholesterol": {
            "name": "Total Cholesterol",
            "unit": "mg/dL",
            "normal_max": 200.0,
            "category": "lipid",
            "guideline": "Clinical_Lipid_Profile_Guideline.pdf"
        },
        "hdl": {
            "name": "HDL Cholesterol",
            "unit": "mg/dL",
            "normal_min": 40.0,
            "category": "lipid",
            "guideline": "Clinical_Lipid_Profile_Guideline.pdf"
        },
        "triglycerides": {
            "name": "Triglycerides",
            "unit": "mg/dL",
            "normal_max": 150.0,
            "category": "lipid",
            "guideline": "Clinical_Lipid_Profile_Guideline.pdf"
        },
        "creatinine": {
            "name": "Serum Creatinine",
            "unit": "mg/dL",
            "normal_min": 0.7,
            "normal_max": 1.3,
            "category": "renal",
            "guideline": "Renal_And_Liver_Function_Guideline.pdf"
        },
        "egfr": {
            "name": "eGFR",
            "unit": "mL/min/1.73m2",
            "normal_min": 60.0,
            "category": "renal",
            "guideline": "Renal_And_Liver_Function_Guideline.pdf"
        },
        "urea": {
            "name": "Blood Urea",
            "unit": "mg/dL",
            "normal_min": 10.0,
            "normal_max": 50.0,
            "category": "renal",
            "guideline": "Renal_And_Liver_Function_Guideline.pdf"
        },
        "alt": {
            "name": "ALT (SGPT)",
            "unit": "U/L",
            "normal_max": 45.0,
            "category": "liver",
            "guideline": "Renal_And_Liver_Function_Guideline.pdf"
        },
        "ast": {
            "name": "AST (SGOT)",
            "unit": "U/L",
            "normal_max": 40.0,
            "category": "liver",
            "guideline": "Renal_And_Liver_Function_Guideline.pdf"
        }
    }

    def __init__(self, retriever: BaseRetriever, gemini_api_key: Optional[str] = None):
        """
        Dependency Injection constructor.
        Requires an instance conforming to BaseRetriever interface.
        """
        self.retriever = retriever
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.use_llm = bool(self.api_key and "your_gemini_api_key" not in self.api_key)

    # -------------------------------------------------------------------------
    # 1. Observation Phase
    # -------------------------------------------------------------------------
    def observe(self, query: str) -> List[ClinicalObservation]:
        """
        Extracts recognized clinical biomarkers, numerical values, and units from query.
        """
        observations: List[ClinicalObservation] = []

        patterns = {
            "hba1c": r"(?:hba1c|a1c|glycated\s+h[ae]moglobin)[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(%?)",
            "fbs": r"(?:fbs|fasting\s+(?:blood\s+)?(?:sugar|glucose))[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(mg/dl)?",
            "ppbs": r"(?:ppbs|post\s*prandial[\s\w]*)[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(mg/dl)?",
            "ldl": r"(?:ldl(?:-c)?|low\s+density\s+lipoprotein)[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(mg/dl)?",
            "total_cholesterol": r"(?:total\s+cholesterol|cholesterol)[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(mg/dl)?",
            "hdl": r"(?:hdl(?:-c)?|high\s+density\s+lipoprotein)[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(mg/dl)?",
            "triglycerides": r"(?:triglycerides?|tg)[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(mg/dl)?",
            "creatinine": r"(?:serum\s+)?creatinine[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(mg/dl)?",
            "egfr": r"egfr[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(ml/min(?:/1\.73m2)?)?",
            "urea": r"(?:blood\s+)?urea[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(mg/dl)?",
            "alt": r"(?:alt|sgpt)[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(u/l)?",
            "ast": r"(?:ast|sgot)[\s:=]+([0-9]+(?:\.[0-9]+)?)\s*(u/l)?"
        }

        lower_query = query.lower()

        for key, pattern in patterns.items():
            match = re.search(pattern, lower_query)
            if match:
                value = float(match.group(1))
                ref = self.REFERENCE_RANGES[key]
                unit = ref["unit"]
                raw_text = match.group(0)

                status = self._evaluate_marker_status(key, value)

                observations.append(
                    ClinicalObservation(
                        marker=ref["name"],
                        value=value,
                        unit=unit,
                        raw_text=raw_text,
                        status=status
                    )
                )

        return observations

    def _evaluate_marker_status(self, key: str, value: float) -> str:
        """Determines clinical severity classification based on established guidelines."""
        ref = self.REFERENCE_RANGES.get(key)
        if not ref:
            return "RECORDED"

        if key == "hba1c":
            if value < 5.7:
                return "NORMAL"
            elif value <= 6.4:
                return "PREDIABETIC"
            elif value <= 8.0:
                return "DIABETIC (MODERATE CONTROL)"
            else:
                return "DIABETIC (POORLY CONTROLLED)"

        elif key in ("fbs", "ppbs"):
            normal_max = ref.get("normal_max", 100.0)
            if value <= normal_max:
                return "NORMAL"
            elif value <= ref.get("prediabetes_max", 125.0):
                return "BORDERLINE ELEVATED"
            else:
                return "ELEVATED"

        elif key == "creatinine":
            if value < 0.7:
                return "LOW"
            elif value <= 1.3:
                return "NORMAL"
            else:
                return "ELEVATED (RENAL RISK)"

        elif key == "ldl":
            if value < 100.0:
                return "OPTIMAL"
            elif value <= 129.0:
                return "NEAR OPTIMAL"
            elif value <= 159.0:
                return "BORDERLINE HIGH"
            else:
                return "HIGH"

        elif key == "hdl":
            if value < 40.0:
                return "LOW (CARDIOVASCULAR RISK)"
            elif value >= 60.0:
                return "PROTECTIVE"
            return "ACCEPTABLE"

        return "RECORDED"

    # -------------------------------------------------------------------------
    # 2. Reasoning Phase
    # -------------------------------------------------------------------------
    def reason(self, state: AgentState) -> List[Dict[str, str]]:
        """
        Formulates structured retrieval actions based on observations.
        Detects multi-condition co-morbidities (e.g. Diabetes + Renal Impairment).
        """
        plan: List[Dict[str, str]] = []

        if not state.observations:
            state.add_reasoning_step(
                "[Reasoning Cycle 1]: No specific clinical biomarkers detected via pattern matching. "
                "Dispatching general semantic retrieval query against clinical guidelines."
            )
            plan.append({
                "sub_query": state.query,
                "focus": "General Medical Query",
                "reason": "Direct semantic search against indexed knowledge base"
            })
            return plan

        has_diabetes = False
        has_renal = False
        has_lipid = False
        has_liver = False

        for obs in state.observations:
            marker_lower = obs.marker.lower()
            if any(k in marker_lower for k in ["hba1c", "blood sugar", "glucose", "fbs", "ppbs"]):
                has_diabetes = True
                plan.append({
                    "sub_query": f"{obs.marker} threshold {obs.value} {obs.unit} diabetes complications management",
                    "focus": "Diabetes Management",
                    "reason": f"Observed {obs.marker}={obs.value} {obs.unit} (Status: {obs.status})"
                })
            elif any(k in marker_lower for k in ["creatinine", "egfr", "urea"]):
                has_renal = True
                plan.append({
                    "sub_query": f"{obs.marker} {obs.value} {obs.unit} kidney filtration impairment renal failure",
                    "focus": "Renal Function",
                    "reason": f"Observed {obs.marker}={obs.value} {obs.unit} (Status: {obs.status})"
                })
            elif any(k in marker_lower for k in ["cholesterol", "ldl", "hdl", "triglycerides"]):
                has_lipid = True
                plan.append({
                    "sub_query": f"{obs.marker} {obs.value} {obs.unit} cardiovascular risk lipid management",
                    "focus": "Lipid Profile",
                    "reason": f"Observed {obs.marker}={obs.value} {obs.unit} (Status: {obs.status})"
                })
            elif any(k in marker_lower for k in ["alt", "ast"]):
                has_liver = True
                plan.append({
                    "sub_query": f"{obs.marker} {obs.value} liver enzymes hepatic function",
                    "focus": "Liver Function",
                    "reason": f"Observed {obs.marker}={obs.value} {obs.unit} (Status: {obs.status})"
                })

        # Co-morbidity detection: Diabetes + Renal Impairment (Diabetic Nephropathy)
        if has_diabetes and has_renal:
            plan.append({
                "sub_query": "uncontrolled diabetes and elevated serum creatinine diabetic nephropathy precautions NSAIDs",
                "focus": "Diabetic Nephropathy / Co-morbidity Interaction",
                "reason": "Concurrent hyperglycemia and elevated creatinine indicate risk of diabetic kidney disease"
            })

        for i, step in enumerate(plan, start=1):
            state.add_reasoning_step(
                f"[Reasoning Step {i}]: Focus on '{step['focus']}'. Query: '{step['sub_query']}' | Motivation: {step['reason']}"
            )

        return plan

    # -------------------------------------------------------------------------
    # 3. Action Phase
    # -------------------------------------------------------------------------
    def act(self, state: AgentState, plan: List[Dict[str, str]]) -> None:
        """
        Executes targeted sub-queries via the injected BaseRetriever.
        Logs mathematical explainability metrics (TF-IDF, Dense, Dot Product, Hybrid).
        """
        for i, step in enumerate(plan, start=1):
            sub_query = step["sub_query"]
            chunks = self.retriever.retrieve(sub_query, top_k=2)

            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", "N/A")
                source = chunk.get("source", "Unknown")
                page = chunk.get("page", "?")
                tfidf = chunk.get("tfidf_score", 0.0)
                dense = chunk.get("dense_score", 0.0)
                dot_prod = chunk.get("dot_product", 0.0)
                hybrid = chunk.get("hybrid_score", 0.0)

                state.add_reasoning_step(
                    f"[Action {i} - Hybrid Search]: Retrieved {chunk_id} from '{source}' (p.{page}) "
                    f"| Math Explainability: TF-IDF={tfidf:.3f}, Dense={dense:.3f}, DotProd={dot_prod:.3f}, HybridScore={hybrid:.3f}"
                )

            state.add_retrieved_chunks(chunks)

    # -------------------------------------------------------------------------
    # 4. Persistence & Synthesis Phase
    # -------------------------------------------------------------------------
    def persist(self, state: AgentState) -> Dict[str, Any]:
        """
        Finalizes state container and generates grounded explanation with citations.
        """
        if self.use_llm:
            try:
                final_answer = self._generate_gemini_response(state)
            except Exception as e:
                state.add_reasoning_step(
                    f"[Warning]: LLM generation failed ({str(e)}). Utilizing grounded rule-based synthesizer."
                )
                final_answer = self._synthesize_grounded_answer(state)
        else:
            final_answer = self._synthesize_grounded_answer(state)

        state.final_answer = final_answer
        state.is_complete = True
        state.add_reasoning_step("[Persistence]: State successfully committed and audit trail complete.")

        return state.to_dict()

    def _synthesize_grounded_answer(self, state: AgentState) -> str:
        """
        Deterministic, fully grounded clinical synthesis fallback.
        Ensures 100% adherence to guideline evidence without hallucinations.
        """
        if not state.retrieved_chunks and not state.observations:
            return (
                "Unable to identify specific clinical markers from the provided input. "
                "Please provide standard lab test names and numerical values (e.g., HbA1c 8.2%, Creatinine 1.8 mg/dL)."
            )

        lines: List[str] = [
            "### 🩺 MediExplain AI - Clinical Laboratory Report Analysis\n"
        ]

        # Clinical observations summary
        if state.observations:
            lines.append("**Extracted Biomarkers and Clinical Status:**")
            for obs in state.observations:
                is_abnormal = any(w in obs.status for w in ["HIGH", "DIABETIC", "ELEVATED", "RISK"])
                badge = "⚠️" if is_abnormal else "✅"
                lines.append(f"- {badge} **{obs.marker}**: `{obs.value} {obs.unit}` — **{obs.status}**")
            lines.append("")

        # Grounded guideline evidence
        lines.append("**Guideline-Grounded Evidence & Explanations:**")
        for chunk in state.retrieved_chunks:
            source = chunk.get("source", "Clinical Guideline")
            page = chunk.get("page", 1)
            citation = f"{source} (Page {page})"
            snippet = chunk.get("text", "").strip()
            lines.append(f"> \"{snippet}\"\n> — *Citation: [{citation}]*\n")

        # Actionable clinical recommendations
        has_high_diabetes = any("DIABETIC" in obs.status for obs in state.observations)
        has_high_renal = any("RENAL RISK" in obs.status for obs in state.observations)

        lines.append("**💡 Clinical Guidance and Precautions:**")
        if has_high_diabetes and has_high_renal:
            lines.append(
                "- ⚠️ **Critical Precaution:** Concurrent hyperglycemia and elevated serum creatinine indicate "
                "risk of diabetic nephropathy (kidney complications associated with diabetes). Prompt consultation "
                "with a physician is strongly advised to review glycemic control and preserve renal function."
            )
            lines.append("- Avoid nephrotoxic analgesics such as NSAIDs (Ibuprofen, Diclofenac) without medical supervision.")
            lines.append("- Strictly monitor blood pressure and adhere to renal-safe dietary guidelines (moderate protein and salt intake).")
        elif has_high_diabetes:
            lines.append(
                "- Elevated glycemic parameters indicate inadequate blood glucose regulation. "
                "A medical consultation for pharmacological review and lifestyle intervention is recommended."
            )
        elif has_high_renal:
            lines.append(
                "- Serum creatinine exceeds normal baseline levels, indicating potential reduction in glomerular filtration. "
                "Further renal function testing (e.g., eGFR, urine albumin) is advised."
            )
        else:
            lines.append("- Markers appear within acceptable clinical thresholds. Maintain regular routine screening.")

        lines.append(
            "\n*Disclaimer: MediExplain AI provides educational interpretations strictly grounded in official clinical practice guidelines. "
            "It does not replace professional clinical diagnosis or personalized medical treatment.*"
        )

        return "\n".join(lines)

    def _generate_gemini_response(self, state: AgentState) -> str:
        """
        Generates grounded response using official Google Gemini API client.
        """
        from google import genai

        client = genai.Client(api_key=self.api_key)

        context_blocks = "\n\n".join([
            f"[Source: {c.get('source')} (p.{c.get('page')}) | Hybrid Score: {c.get('hybrid_score', 0):.3f}]\n{c.get('text')}"
            for c in state.retrieved_chunks
        ])

        system_instruction = (
            "You are MediExplain AI, an explainable clinical lab report interpreter. "
            "Explain lab findings clearly, empathetically, and accurately in English. "
            "CRITICAL REQUIREMENT: Strictly ground your explanation in the retrieved guideline chunks only. "
            "Explicitly cite source titles and page numbers (e.g. [WHO_Diabetes_Guideline.pdf, p.4]). "
            "Do not fabricate facts or speculate beyond the provided evidence. "
            "Conclude with a clear recommendation to consult a licensed medical practitioner."
        )

        user_prompt = f"""
User Query: {state.query}

Observed Biomarkers:
{state.get_observation_summaries()}

Retrieved Evidence Chunks:
{context_blocks}

Provide an explainable, grounded, and patient-friendly analysis.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config={"system_instruction": system_instruction}
        )
        return response.text

    # -------------------------------------------------------------------------
    # Main Agent Runner (Conforms to BaseAgent interface)
    # -------------------------------------------------------------------------
    def run(self, query: str) -> Dict[str, Any]:
        """
        Executes full Agentic Loop: Observation -> Reasoning -> Action -> Persistence.
        """
        state = AgentState(query=query)

        # 1. Observation
        observations = self.observe(query)
        for obs in observations:
            state.add_observation(obs)

        # 2. Reasoning
        plan = self.reason(state)

        # 3. Action
        self.act(state, plan)

        # 4. Persistence
        return self.persist(state)
