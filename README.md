# AI Resume Shortlisting & Interview Assistant System
### Assignment 5 — Jano Health SWE Internship Take-Home

---

## What this implements

**Option A: Evaluation & Scoring Engine (Core)**

A Python backend that ingests raw resume text, compares it against a Job Description, and produces four-dimensional scores with human-readable explanations, a tier classification (A/B/C), and an interview-track recommendation.

---

## System Architecture

```
Raw Resume Text
      │
      ▼
┌─────────────┐     Anthropic Claude      ┌───────────────┐
│ ResumeParser│ ──────────────────────►  │  ParsedResume │
└─────────────┘                           └───────┬───────┘
                                                  │
                                                  ▼
                               JobDescription ──► ScoringEngine ──► DimensionalScores
                                                                           │
                                                                           ▼
                                                                    TierClassifier
                                                                           │
                                                                           ▼
                                                                   EvaluationResult
                                                           (tier, scores, reasoning, track)
```

### Components

| Module | Responsibility |
|--------|---------------|
| `src/models.py` | Domain types and data contracts (`JobDescription`, `ParsedResume`, `DimensionalScores`, `EvaluationResult`) |
| `src/parser.py` | `ResumeParser` — LLM-powered extraction of structured fields from raw text |
| `src/scoring_engine.py` | `ScoringEngine` — LLM-powered four-dimensional scoring with explainability |
| `src/classifier.py` | `TierClassifier` — rule-based tier assignment from composite score |
| `src/pipeline.py` | `EvaluationPipeline` — orchestrates the three stages; single entry-point for callers |
| `main.py` | CLI for local evaluation |
| `tests/test_pipeline.py` | Unit tests covering scoring logic, tier boundaries, validation, and LLM parsing |

---

## Scoring Model

Each resume is scored across four dimensions (all 0.0–1.0):

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Exact Match | 35% | Literal overlap between candidate skills and required JD skills |
| Similarity | 25% | Semantic / adjacent skill relevance (e.g. AWS Kinesis ≈ Kafka) |
| Achievement | 20% | Quantified impact in experience bullets (%, $, scale) |
| Ownership | 20% | Leadership and ownership language (led, architected, founded, drove) |

**Composite score** = weighted sum. Tier thresholds:
- **Tier A** (≥ 0.75) — Fast-track to final interview
- **Tier B** (≥ 0.50) — Technical screen
- **Tier C** (< 0.50) — Needs evaluation

Every score comes with a one-sentence explanation so reviewers understand *why* a candidate was ranked the way they were.

---

## Data Strategy

Resumes are accepted as plain text strings. In production, PDF text extraction (e.g. `pdfplumber`) would run before the parser. The `ResumeParser` sends the raw text to Claude and asks for a strict JSON response, then validates and deserialises it into a `ParsedResume` dataclass. All fields have defaults so a partially-extractable resume never crashes the pipeline.

---

## AI Strategy

- **Model**: `claude-opus-4-6` for both parsing and scoring
- **Prompts**: Each LLM call has a tightly-scoped system instruction and a `Respond ONLY with valid JSON` constraint to keep outputs parseable
- **Semantic similarity**: delegated to the LLM scoring step — Claude is asked to reason about technology adjacency (e.g. Kafka ↔ Kinesis, AWS ↔ GCP) rather than doing vector embedding, which is appropriate for the volume of a take-home exercise
- **Explainability**: every dimension score is accompanied by a natural-language reason, surfaced in the final output

---

## Scalability Notes (10,000+ resumes/day)

For production scale the following changes would be made:

1. **Async batch processing** — replace sequential calls with `asyncio` + `anthropic.AsyncAnthropic`; fan-out requests in parallel with a concurrency limiter (`asyncio.Semaphore`)
2. **Queue-based architecture** — resumes land in an SQS / Kafka queue; a pool of workers pulls and evaluates; results written to PostgreSQL
3. **Caching** — identical JDs don't need fresh LLM scoring; cache parsed JD embeddings
4. **PDF extraction** — `pdfplumber` or AWS Textract runs before the parser so the LLM only sees clean text
5. **Observability** — structured logs (candidate ID, latency, token usage, tier) to Datadog / CloudWatch

---

## Setup & Running

### Prerequisites
- Python 3.11+
- An Anthropic API key

### Install
```bash
git clone <repo-url>
cd resume-shortlister
pip install -r requirements.txt
```

### Set API key
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Run the demo (no files needed)
```bash
python main.py --demo
```

### Evaluate your own resume
```bash
# Paste resume into a .txt file, fill in sample_data/jd.json, then:
python main.py --resume sample_data/resume1.txt --jd sample_data/jd.json
```

### Evaluate multiple resumes (ranked output)
```bash
python main.py --resume resume_a.txt resume_b.txt resume_c.txt --jd sample_data/jd.json
```

### Run tests
```bash
pytest tests/ -v
```
Tests use mocked LLM responses — no API key required.

---

## Sample Output

```
============================================================
  CANDIDATE: Jane Doe  (jane@example.com)
  TIER:      A  (composite 0.87)
============================================================
Composite score: 0.87
  • Exact Match (0.95): All five required skills are explicitly listed.
  • Similarity  (0.80): Kubernetes and Airflow experience directly supports preferred skills.
  • Achievement (0.85): Two strong quantified wins (2M events/day, 60% deploy time reduction).
  • Ownership   (0.90): Architected, led, owned, and drove are all present.
Tier A assigned based on composite threshold (A≥0.75, B≥0.50, C<0.50).

  RECOMMENDED TRACK:
  Skip to final-round technical interview; focus on system design and leadership.
```

---

## Assumptions & Trade-offs

- **Text input only** — PDF parsing is out of scope; callers pass pre-extracted text. A production system would add `pdfplumber` or Textract as a pre-processing step.
- **LLM for similarity** — instead of vector embeddings, semantic similarity scoring is delegated to Claude. This is accurate but costs tokens; at scale, a lightweight embedding model (e.g. `text-embedding-3-small`) with cosine similarity would be cheaper.
- **Single JD per batch** — the pipeline evaluates N resumes against one JD. Multi-JD routing would require an additional dispatcher layer.
- **No persistence** — results are returned in memory and printed. A production version would write to a database and expose a REST API.
- **Tier thresholds are hard-coded** — in production these would be configurable per role/team.
