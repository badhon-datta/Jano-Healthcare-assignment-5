"""
Domain models and data contracts for the Resume Shortlisting System.
All external data is validated here before it enters the pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Tier(str, Enum):
    A = "A"  # Fast-track
    B = "B"  # Technical Screen
    C = "C"  # Needs Evaluation


@dataclass
class JobDescription:
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    responsibilities: list[str]
    min_years_experience: int = 0
    raw_text: str = ""

    def __post_init__(self):
        if not self.title:
            raise ValueError("Job title cannot be empty")
        if not self.required_skills:
            raise ValueError("Job description must have at least one required skill")


@dataclass
class ParsedResume:
    candidate_name: str
    email: str
    skills: list[str]
    years_experience: int
    achievements: list[str]          # quantified impact bullets
    ownership_signals: list[str]     # "led", "architected", "founded", etc.
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    raw_text: str = ""

    def __post_init__(self):
        if not self.candidate_name:
            raise ValueError("Candidate name cannot be empty")


@dataclass
class DimensionalScores:
    """
    Four-dimensional scoring rubric as specified in the PRD.
    All scores are 0.0 – 1.0.
    """
    exact_match: float        # literal skill overlap
    similarity: float         # semantic / adjacent skill match
    achievement: float        # quantified impact signals
    ownership: float          # leadership / ownership language

    # Human-readable explanation for each dimension (explainability)
    exact_match_reason: str = ""
    similarity_reason: str = ""
    achievement_reason: str = ""
    ownership_reason: str = ""

    def __post_init__(self):
        for attr in ("exact_match", "similarity", "achievement", "ownership"):
            v = getattr(self, attr)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{attr} score must be between 0 and 1, got {v}")

    @property
    def composite(self) -> float:
        """Weighted composite score used for tier classification."""
        return round(
            self.exact_match * 0.35
            + self.similarity * 0.25
            + self.achievement * 0.20
            + self.ownership * 0.20,
            4,
        )


@dataclass
class EvaluationResult:
    candidate_name: str
    email: str
    scores: DimensionalScores
    tier: Tier
    tier_reasoning: str
    recommended_interview_track: str
    raw_resume_text: str = ""

    @property
    def summary(self) -> str:
        return (
            f"{self.candidate_name} | Tier {self.tier.value} | "
            f"Composite: {self.scores.composite:.2f}"
        )
