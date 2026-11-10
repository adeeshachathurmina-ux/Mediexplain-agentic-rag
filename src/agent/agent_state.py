"""
AgentState Model for MediExplain Agentic RAG System.
DS205.3 - Data Science with Python (Member 3)

This module maintains the execution state and audit trail across the Agentic Loop:
Observation -> Reasoning -> Action -> Persistence.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ClinicalObservation:
    """Represents a single clinical marker observed in user input or report."""
    marker: str                 # e.g., 'HbA1c', 'Creatinine', 'LDL'
    value: float                # e.g., 8.2, 1.8, 175.0
    unit: str                   # e.g., '%', 'mg/dL'
    raw_text: str               # The original matched string segment
    status: str = "UNKNOWN"     # 'NORMAL', 'ELEVATED', 'CRITICAL', etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker": self.marker,
            "value": self.value,
            "unit": self.unit,
            "raw_text": self.raw_text,
            "status": self.status
        }


@dataclass
class AgentState:
    """
    State container for the Medical Agent.
    Ensures state persistence and complete traceability between cycles.
    """
    query: str
    observations: List[ClinicalObservation] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    is_complete: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_observation(self, observation: ClinicalObservation) -> None:
        """Appends a new clinical observation if not already tracked."""
        if not any(obs.marker.lower() == observation.marker.lower() for obs in self.observations):
            self.observations.append(observation)

    def add_reasoning_step(self, step: str) -> None:
        """Logs a step in the Agent's Chain of Thought (CoT)."""
        self.reasoning_steps.append(step)

    def add_retrieved_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Appends newly retrieved chunks, deduplicating by chunk_id."""
        existing_ids = {c.get("chunk_id") for c in self.retrieved_chunks}
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id not in existing_ids:
                self.retrieved_chunks.append(chunk)
                existing_ids.add(chunk_id)

    def get_observation_summaries(self) -> List[str]:
        """Returns human-readable strings of all observed clinical markers."""
        return [
            f"{obs.marker}: {obs.value} {obs.unit} (Status: {obs.status})"
            for obs in self.observations
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the state for final response and evaluation."""
        return {
            "query": self.query,
            "observations": self.get_observation_summaries(),
            "reasoning_steps": self.reasoning_steps,
            "retrieved_chunks": self.retrieved_chunks,
            "final_answer": self.final_answer,
            "is_complete": self.is_complete
        }

