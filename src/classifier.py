"""
Tier Classifier — maps composite scores to Tier A / B / C
and produces a recommended interview track.

Thresholds (composite score):
  Tier A — >= 0.75  → Fast-track to final interview
  Tier B — >= 0.50  → Technical Screen
  Tier C — < 0.50   → Needs Evaluation / reject
"""

from __future__ import annotations
from src.models import DimensionalScores, EvaluationResult, ParsedResume, Tier


TIER_THRESHOLDS = {Tier.A: 0.75, Tier.B: 0.50}

INTERVIEW_TRACKS = {
    Tier.A: "Skip to final-round technical interview; focus on system design and leadership.",
    Tier.B: "Schedule a 45-minute technical screen covering core skills and one coding problem.",
    Tier.C: "Hold; revisit only if pipeline is thin. Optionally send a rejection with feedback.",
}


class TierClassifier:
    def classify(
        self, resume: ParsedResume, scores: DimensionalScores
    ) -> EvaluationResult:
        """Classify the candidate into a tier and build the full evaluation result."""
        tier = self._determine_tier(scores)
        reasoning = self._build_reasoning(scores, tier)

        return EvaluationResult(
            candidate_name=resume.candidate_name,
            email=resume.email,
            scores=scores,
            tier=tier,
            tier_reasoning=reasoning,
            recommended_interview_track=INTERVIEW_TRACKS[tier],
            raw_resume_text=resume.raw_text,
        )

    @staticmethod
    def _determine_tier(scores):
        composite = scores.composite
        if composite >= 0.75:
            return Tier.A
        elif composite >= 0.49:
            return Tier.B
        else:
            return Tier.C

    @staticmethod
    def _build_reasoning(scores: DimensionalScores, tier: Tier) -> str:
        lines = [
            f"Composite score: {scores.composite:.2f}",
            f"  • Exact Match ({scores.exact_match:.2f}): {scores.exact_match_reason}",
            f"  • Similarity  ({scores.similarity:.2f}): {scores.similarity_reason}",
            f"  • Achievement ({scores.achievement:.2f}): {scores.achievement_reason}",
            f"  • Ownership   ({scores.ownership:.2f}): {scores.ownership_reason}",
            f"Tier {tier.value} assigned based on composite threshold "
            f"(A≥0.75, B≥0.50, C<0.50).",
        ]
        return "\n".join(lines)
