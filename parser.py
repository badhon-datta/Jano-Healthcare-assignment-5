"""
Resume Parser — converts raw PDF text or plain text into a structured ParsedResume.
Uses Claude to extract fields reliably even from messy, unstructured layouts.
"""

from __future__ import annotations
import json
import re
import anthropic

from src.models import ParsedResume


PARSE_PROMPT = """\
You are a precise resume parser. Extract the following fields from the resume text below.
Respond ONLY with a valid JSON object — no markdown, no explanation.

JSON schema:
{{
  "candidate_name": "string",
  "email": "string (empty string if not found)",
  "skills": ["list of technical skills, tools, languages, frameworks"],
  "years_experience": 0,
  "achievements": ["quantified impact bullets, e.g. 'reduced latency by 40%'"],
  "ownership_signals": ["phrases showing leadership/ownership, e.g. 'led team of 5', 'architected the data pipeline'"],
  "github_url": null,
  "linkedin_url": null
}}

Rules:
- skills: include ALL technical mentions (languages, frameworks, tools, cloud platforms, databases)
- achievements: only include bullets with measurable outcomes (numbers, %, $, scale)
- ownership_signals: look for: led, owned, architected, designed, founded, drove, spearheaded, built from scratch
- years_experience: count from earliest role to most recent; if unclear, return 0

Resume text:
---
{resume_text}
---
"""


class ResumeParser:
    def __init__(self, client: anthropic.Anthropic):
        self._client = client

    def parse(self, raw_text: str) -> ParsedResume:
        """Parse raw resume text into a structured ParsedResume object."""
        if not raw_text or not raw_text.strip():
            raise ValueError("Resume text cannot be empty")

        raw_text = raw_text.strip()

        response = self._client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": PARSE_PROMPT.format(resume_text=raw_text),
                }
            ],
        )

        raw_json = response.content[0].text.strip()
        raw_json = self._strip_markdown_fences(raw_json)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw_json}") from e

        return ParsedResume(
            candidate_name=self._require_str(data, "candidate_name"),
            email=data.get("email", ""),
            skills=[s for s in data.get("skills", []) if isinstance(s, str)],
            years_experience=int(data.get("years_experience", 0)),
            achievements=[a for a in data.get("achievements", []) if isinstance(a, str)],
            ownership_signals=[o for o in data.get("ownership_signals", []) if isinstance(o, str)],
            github_url=data.get("github_url") or None,
            linkedin_url=data.get("linkedin_url") or None,
            raw_text=raw_text,
        )

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove ```json ... ``` wrappers if present."""
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _require_str(data: dict, key: str) -> str:
        val = data.get(key, "")
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Required field '{key}' missing or empty in parsed resume")
        return val.strip()
