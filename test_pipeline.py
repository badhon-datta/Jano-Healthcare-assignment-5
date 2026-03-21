"""
Tests — covers the hardest business logic without hitting the LLM.
Run with:  pytest tests/ -v
"""

from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch

from src.models import (
    DimensionalScores,
    EvaluationResult,
    JobDescription,
    ParsedResume,
    Tier,
)
from src.classifier import TierClassifier
from src.parser import ResumeParser
from src.scoring_engine import ScoringEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_jd() -> JobDescription:
    return JobDescription(
        title="Backend Engineer",
        required_skills=["Python", "Kafka", "PostgreSQL"],
        preferred_skills=["Kubernetes"],
        responsibilities=["Build data pipelines", "Own streaming infra"],
        min_years_experience=3,
    )


@pytest.fixture
def strong_resume() -> ParsedResume:
    return ParsedResume(
        candidate_name="Jane Doe",
        email="jane@example.com",
        skills=["Python", "Kafka", "PostgreSQL", "Docker", "Kubernetes"],
        years_experience=5,
        achievements=["Reduced latency by 40%", "Scaled system to 2M events/day"],
        ownership_signals=["Architected the data pipeline", "Led team of 4"],
    )


@pytest.fixture
def weak_resume() -> ParsedResume:
    return ParsedResume(
        candidate_name="Bob Smith",
        email="bob@example.com",
        skills=["Excel", "Tableau"],
        years_experience=1,
        achievements=[],
        ownership_signals=[],
    )


@pytest.fixture
def high_scores() -> DimensionalScores:
    return DimensionalScores(
        exact_match=0.9, exact_match_reason="All required skills present",
        similarity=0.8, similarity_reason="Strong adjacent experience",
        achievement=0.85, achievement_reason="Multiple quantified wins",
        ownership=0.9, ownership_reason="Led and architected multiple systems",
    )


@pytest.fixture
def low_scores() -> DimensionalScores:
    return DimensionalScores(
        exact_match=0.1, exact_match_reason="Very few required skills",
        similarity=0.1, similarity_reason="Unrelated domain",
        achievement=0.0, achievement_reason="No quantified achievements",
        ownership=0.0, ownership_reason="No ownership signals",
    )


@pytest.fixture
def mid_scores() -> DimensionalScores:
    return DimensionalScores(
        exact_match=0.5, exact_match_reason="Some skills match",
        similarity=0.55, similarity_reason="Adjacent technology experience",
        achievement=0.4, achievement_reason="One quantified win",
        ownership=0.5, ownership_reason="Some ownership signals",
    )


# ---------------------------------------------------------------------------
# DimensionalScores — unit tests
# ---------------------------------------------------------------------------

class TestDimensionalScores:
    def test_composite_weighted_correctly(self):
        s = DimensionalScores(
            exact_match=1.0, similarity=1.0, achievement=1.0, ownership=1.0,
            exact_match_reason="", similarity_reason="", achievement_reason="", ownership_reason="",
        )
        assert s.composite == 1.0

    def test_composite_zero_when_all_zero(self):
        s = DimensionalScores(
            exact_match=0.0, similarity=0.0, achievement=0.0, ownership=0.0,
            exact_match_reason="", similarity_reason="", achievement_reason="", ownership_reason="",
        )
        assert s.composite == 0.0

    def test_composite_partial(self):
        s = DimensionalScores(
            exact_match=1.0, similarity=0.0, achievement=0.0, ownership=0.0,
            exact_match_reason="", similarity_reason="", achievement_reason="", ownership_reason="",
        )
        # exact_match weight is 0.35
        assert abs(s.composite - 0.35) < 0.001

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            DimensionalScores(
                exact_match=1.5, similarity=0.0, achievement=0.0, ownership=0.0,
            )

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            DimensionalScores(
                exact_match=-0.1, similarity=0.0, achievement=0.0, ownership=0.0,
            )


# ---------------------------------------------------------------------------
# TierClassifier — business rule tests
# ---------------------------------------------------------------------------

class TestTierClassifier:
    def setup_method(self):
        self.classifier = TierClassifier()

    def test_high_scores_yield_tier_a(self, strong_resume, high_scores):
        result = self.classifier.classify(strong_resume, high_scores)
        assert result.tier == Tier.A

    def test_low_scores_yield_tier_c(self, weak_resume, low_scores):
        result = self.classifier.classify(weak_resume, low_scores)
        assert result.tier == Tier.C

    def test_mid_scores_yield_tier_b(self, strong_resume, mid_scores):
        result = self.classifier.classify(strong_resume, mid_scores)
        assert result.tier == Tier.B

    def test_result_contains_reasoning(self, strong_resume, high_scores):
        result = self.classifier.classify(strong_resume, high_scores)
        assert "Composite score" in result.tier_reasoning
        assert "Exact Match" in result.tier_reasoning

    def test_result_contains_interview_track(self, strong_resume, high_scores):
        result = self.classifier.classify(strong_resume, high_scores)
        assert result.recommended_interview_track != ""

    def test_tier_a_threshold_boundary(self, strong_resume):
        """Score exactly at 0.75 should be Tier A."""
        # Build scores that produce composite ≈ 0.75
        # 0.75 * 0.35 + 0.75 * 0.25 + 0.75 * 0.20 + 0.75 * 0.20 = 0.75
        scores = DimensionalScores(
            exact_match=0.75, similarity=0.75, achievement=0.75, ownership=0.75,
            exact_match_reason="boundary", similarity_reason="boundary",
            achievement_reason="boundary", ownership_reason="boundary",
        )
        result = self.classifier.classify(strong_resume, scores)
        assert result.tier == Tier.A

    def test_just_below_tier_a_is_tier_b(self, strong_resume):
        scores = DimensionalScores(
            exact_match=0.74, similarity=0.74, achievement=0.74, ownership=0.74,
            exact_match_reason="", similarity_reason="", achievement_reason="", ownership_reason="",
        )
        result = self.classifier.classify(strong_resume, scores)
        assert result.tier == Tier.B


# ---------------------------------------------------------------------------
# JobDescription — validation tests
# ---------------------------------------------------------------------------

class TestJobDescription:
    def test_empty_title_raises(self):
        with pytest.raises(ValueError):
            JobDescription(title="", required_skills=["Python"], preferred_skills=[], responsibilities=[])

    def test_empty_required_skills_raises(self):
        with pytest.raises(ValueError):
            JobDescription(title="Engineer", required_skills=[], preferred_skills=[], responsibilities=[])


# ---------------------------------------------------------------------------
# ParsedResume — validation tests
# ---------------------------------------------------------------------------

class TestParsedResume:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ParsedResume(
                candidate_name="", email="a@b.com", skills=[], years_experience=0,
                achievements=[], ownership_signals=[],
            )


# ---------------------------------------------------------------------------
# ResumeParser — mock LLM tests
# ---------------------------------------------------------------------------

class TestResumeParser:
    def _make_parser(self, llm_response: str) -> ResumeParser:
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=llm_response)]
        client.messages.create.return_value = msg
        return ResumeParser(client)

    def test_valid_response_parses_correctly(self):
        json_response = """{
            "candidate_name": "Jane Doe",
            "email": "jane@example.com",
            "skills": ["Python", "Kafka"],
            "years_experience": 5,
            "achievements": ["Reduced latency by 40%"],
            "ownership_signals": ["Led team of 4"],
            "github_url": null,
            "linkedin_url": null
        }"""
        parser = self._make_parser(json_response)
        result = parser.parse("some resume text")
        assert result.candidate_name == "Jane Doe"
        assert "Kafka" in result.skills
        assert result.years_experience == 5

    def test_strips_markdown_fences(self):
        json_response = "```json\n{\"candidate_name\": \"Test User\", \"email\": \"t@t.com\", \"skills\": [], \"years_experience\": 0, \"achievements\": [], \"ownership_signals\": [], \"github_url\": null, \"linkedin_url\": null}\n```"
        parser = self._make_parser(json_response)
        result = parser.parse("some text")
        assert result.candidate_name == "Test User"

    def test_invalid_json_raises(self):
        parser = self._make_parser("not json at all")
        with pytest.raises(ValueError, match="invalid JSON"):
            parser.parse("some text")

    def test_empty_resume_raises(self):
        parser = self._make_parser("{}")
        with pytest.raises(ValueError):
            parser.parse("")


# ---------------------------------------------------------------------------
# ScoringEngine — mock LLM tests
# ---------------------------------------------------------------------------

class TestScoringEngine:
    def _make_engine(self, llm_response: str) -> ScoringEngine:
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=llm_response)]
        client.messages.create.return_value = msg
        return ScoringEngine(client)

    def test_valid_scores_parsed(self, strong_resume, sample_jd):
        json_response = """{
            "exact_match": 0.9,
            "exact_match_reason": "All required skills present",
            "similarity": 0.7,
            "similarity_reason": "Strong adjacent experience",
            "achievement": 0.8,
            "achievement_reason": "Multiple wins",
            "ownership": 0.85,
            "ownership_reason": "Led multiple systems"
        }"""
        engine = self._make_engine(json_response)
        scores = engine.score(strong_resume, sample_jd)
        assert scores.exact_match == 0.9
        assert 0.0 <= scores.composite <= 1.0

    def test_out_of_range_scores_are_clamped(self, strong_resume, sample_jd):
        json_response = """{
            "exact_match": 1.5,
            "exact_match_reason": "overflow test",
            "similarity": -0.2,
            "similarity_reason": "underflow test",
            "achievement": 0.5,
            "achievement_reason": "ok",
            "ownership": 0.5,
            "ownership_reason": "ok"
        }"""
        engine = self._make_engine(json_response)
        scores = engine.score(strong_resume, sample_jd)
        assert scores.exact_match == 1.0
        assert scores.similarity == 0.0
