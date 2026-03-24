"""
Pipeline — orchestrates Parser → ScoringEngine → TierClassifier.
This is the single entry-point that callers (CLI, API, tests) use.
"""

from __future__ import annotations
import anthropic

from src.models import EvaluationResult, JobDescription
from src.parser import ResumeParser
from src.scoring_engine import ScoringEngine
from src.classifier import TierClassifier


class EvaluationPipeline:
    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        """
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._parser = ResumeParser(self._client)
        self._scorer = ScoringEngine(self._client)
        self._classifier = TierClassifier()

    def evaluate(self, resume_text: str, jd: JobDescription) -> EvaluationResult:
        """
        Run the full evaluation pipeline.

        Args:
            resume_text: Raw resume text (plain text extracted from PDF or typed).
            jd:          Structured JobDescription object.

        Returns:
            EvaluationResult with scores, tier, and interview recommendation.
        """
        parsed = self._parser.parse(resume_text)
        scores = self._scorer.score(parsed, jd)
        result = self._classifier.classify(parsed, scores)
        return result

    def evaluate_batch(
        self, resumes: list[str], jd: JobDescription
    ) -> list[EvaluationResult]:
        """Evaluate multiple resumes against the same JD."""
        results = []
        for resume_text in resumes:
            try:
                results.append(self.evaluate(resume_text, jd))
            except Exception as exc:
                # Log and continue so one bad resume doesn't break the batch
                print(f"[WARN] Skipping resume due to error: {exc}")
        results.sort(key=lambda r: r.scores.composite, reverse=True)
        return results
