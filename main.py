"""
CLI — evaluate one or more resumes against a job description.

Usage:
    python main.py --resume sample_data/resume1.txt --jd sample_data/jd.json
    python main.py --resume sample_data/resume1.txt sample_data/resume2.txt --jd sample_data/jd.json
    python main.py --demo   # runs built-in demo without any files
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from src.models import JobDescription
from src.pipeline import EvaluationPipeline


# ---------------------------------------------------------------------------
# Demo data (used when --demo flag is passed)
# ---------------------------------------------------------------------------

DEMO_JD = JobDescription(
    title="Backend Engineer — Data Platform",
    required_skills=["Python", "Kafka", "PostgreSQL", "REST APIs", "Docker"],
    preferred_skills=["Kubernetes", "Spark", "Airflow"],
    responsibilities=[
        "Design and maintain high-throughput data pipelines",
        "Own the reliability of our streaming infrastructure",
        "Collaborate with ML team to productionise models",
    ],
    min_years_experience=3,
)

DEMO_RESUME_A = """
Jane Doe | jane@example.com | github.com/janedoe

EXPERIENCE
Senior Data Engineer — Acme Corp (2020–2024)
  - Architected a Kafka-based event streaming platform processing 2M events/day
  - Led migration from monolith to microservices, reducing deploy time by 60%
  - Owned PostgreSQL cluster optimisation; cut query latency by 45%
  - Mentored team of 4 junior engineers

Software Engineer — StartupXYZ (2018–2020)
  - Built REST APIs in Python/FastAPI serving 50k daily active users
  - Containerised services with Docker; deployed via Kubernetes

SKILLS: Python, Kafka, PostgreSQL, FastAPI, Docker, Kubernetes, Redis, Airflow
"""

DEMO_RESUME_B = """
Bob Smith | bob@example.com

EXPERIENCE
Data Analyst — MegaCorp (2022–2024)
  - Created dashboards in Tableau
  - Wrote SQL queries for reporting
  - Assisted senior engineers with data migration tasks

Junior Developer — Freelance (2021–2022)
  - Built small web scrapers in Python

SKILLS: SQL, Python (basic), Excel, Tableau
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jd(path: str) -> JobDescription:
    data = json.loads(Path(path).read_text())
    return JobDescription(
        title=data["title"],
        required_skills=data["required_skills"],
        preferred_skills=data.get("preferred_skills", []),
        responsibilities=data.get("responsibilities", []),
        min_years_experience=data.get("min_years_experience", 0),
    )


def print_result(result) -> None:
    print("\n" + "=" * 60)
    print(f"  CANDIDATE: {result.candidate_name}  ({result.email})")
    print(f"  TIER:      {result.tier.value}  (composite {result.scores.composite:.2f})")
    print("=" * 60)
    print(result.scores_detail if hasattr(result, "scores_detail") else result.tier_reasoning)
    print(f"\n  RECOMMENDED TRACK:\n  {result.recommended_interview_track}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Resume Shortlisting System")
    parser.add_argument("--resume", nargs="+", help="Path(s) to resume .txt file(s)")
    parser.add_argument("--jd", help="Path to job description .json file")
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo data")
    args = parser.parse_args()

    pipeline = EvaluationPipeline()  # reads ANTHROPIC_API_KEY from env

    if args.demo:
        print("\n[DEMO MODE] Running two sample resumes against a Backend Engineer JD...\n")
        results = pipeline.evaluate_batch([DEMO_RESUME_A, DEMO_RESUME_B], DEMO_JD)
        for r in results:
            print_result(r)
        return

    if not args.resume or not args.jd:
        parser.print_help()
        sys.exit(1)

    jd = load_jd(args.jd)
    resume_texts = [Path(p).read_text() for p in args.resume]
    results = pipeline.evaluate_batch(resume_texts, jd)
    for r in results:
        print_result(r)


if __name__ == "__main__":
    main()
