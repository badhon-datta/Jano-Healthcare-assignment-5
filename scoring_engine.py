"""
Scoring Engine — evaluates a ParsedResume against a JobDescription and
produces four-dimensional scores with human-readable explanations.

Dimension breakdown:
  Exact Match  (0.35 weight) — literal skill overlap
  Similarity   (0.25 weight) — semantic / adjacent skills (e.g. Kafka ↔ Kinesis)
  Achievement  (0.20 weight) — quantified impact signals
  Ownership    (0.20 weight) — leadership / ownership language
"""

from __future__ import annotations
import json
import re
import anthropic

from src.models import DimensionalScores, JobDescription, ParsedResume


SCORE_PROMPT = """\
You are an expert technical recruiter. Score a candidate's resume against a job description.

JOB DESCRIPTION:
  Title: {job_title}
  Required Skills: {required_skills}
  Preferred Skills: {preferred_skills}
  Responsibilities: {responsibilities}
  Min Years Experience: {min_years}

CANDIDATE:
  Skills: {candidate_skills}
  Years Experience: {years_experience}
  Achievements: {achievements}
  Ownership Signals: {ownership_signals}

Score each dimension 0.0 to 1.0 and explain WHY in one concise sentence.

Scoring guidance:
- exact_match: fraction of required_skills the candidate explicitly lists
- similarity: how well adjacent/equivalent technologies bridge missing exact skills
  (e.g. AWS Kinesis experience is relevant for a Kafka role; score it 0.6–0.8, not 0)
- achievement: strength and quantity of quantified impact bullets
- ownership: strength of leadership / ownership language signals

Respond ONLY with valid JSON, no markdown:
{{
  "exact_match": <float 0-1>,
  "exact_match_reason": "<one sentence>",
  "similarity": <float 0-1>,
  "similarity_reason": "<one sentence>",
  "achievement": <float 0-1>,
  "achievement_reason": "<one sentence>",
  "ownership": <float 0-1>,
  "ownership_reason": "<one sentence>"
}}
"""


class ScoringEngine:
    def __init__(self, client: anthropic.Anthropic):
        self._client = client

    def score(self, resume: ParsedResume, jd: JobDescription) -> DimensionalScores:
        """Score a resume against a job description across four dimensions."""
        prompt = SCORE_PROMPT.format(
            job_title=jd.title,
            required_skills=", ".join(jd.required_skills),
            preferred_skills=", ".join(jd.preferred_skills),
            responsibilities="; ".join(jd.responsibilities),
            min_years=jd.min_years_experience,
            candidate_skills=", ".join(resume.skills) or "none listed",
            years_experience=resume.years_experience,
            achievements="; ".join(resume.achievements) or "none listed",
            ownership_signals="; ".join(resume.ownership_signals) or "none listed",
        )

        response = self._client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        raw = self._strip_markdown_fences(raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON during scoring: {e}\nRaw: {raw}") from e

        return DimensionalScores(
            exact_match=self._clamp(data.get("exact_match", 0.0)),
            exact_match_reason=data.get("exact_match_reason", ""),
            similarity=self._clamp(data.get("similarity", 0.0)),
            similarity_reason=data.get("similarity_reason", ""),
            achievement=self._clamp(data.get("achievement", 0.0)),
            achievement_reason=data.get("achievement_reason", ""),
            ownership=self._clamp(data.get("ownership", 0.0)),
            ownership_reason=data.get("ownership_reason", ""),
        )

    @staticmethod
    def _clamp(v) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()
