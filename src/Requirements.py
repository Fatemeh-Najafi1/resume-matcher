# src/Requirements.py
"""Turn a job posting into a list of discrete requirements.

This replaces the fixed skill vocabulary as the basis of the analysis. That
list held 130 terms and found only 10 in a JPMorgan quant posting — everything
the role actually asked for ("financial time-series modelling", "production
research systems") was invisible because nobody had added it to a text file.
Requirements are read out of the posting itself instead.

Two extraction paths:
  * LLM — better at collapsing a rambling paragraph into atomic requirements.
  * Rule-based — no API key, no network, deterministic. Also the fallback when
    the LLM call fails, so the feature never hard-depends on a provider.
"""

import re

from .LLMAnalyzer import call_json

SYSTEM_PROMPT = """You extract hiring requirements from job postings.
You report only what the posting states. You never invent a requirement, and
never generalise one requirement into a broader one.
Return ONLY valid JSON. No markdown, no commentary."""

EXTRACTION_PROMPT = """Read this job posting and list the distinct things it asks of a candidate.

Return a JSON object of exactly this shape:

{{
  "requirements": [
    {{
      "text": "<one requirement, as a short self-contained phrase>",
      "kind": "required" | "preferred",
      "category": "skill" | "experience" | "education" | "responsibility"
    }}
  ]
}}

Rules:
- One idea per entry. Split "experience with PyTorch and TensorFlow" into two.
- Keep the posting's own wording where you can; do not paraphrase into jargon.
- "kind" is "required" only if the posting states it as a must / minimum.
- Ignore benefits, salary, company boilerplate and equal-opportunity text.
- 8 to 25 entries. Return ONLY the JSON object.

JOB POSTING:
{job}"""


# Lines that read like a requirement rather than prose about the company.
_SIGNALS = (
    "experience", "proficien", "familiar", "knowledge of", "skilled",
    "ability to", "able to", "expertise", "background in", "degree",
    "years", "must ", "should ", "required", "preferred", "strong ",
    "demonstrated", "understanding of", "track record", "you will",
    "responsible for", "hands-on", "working with", "fluent",
    "qualification", "we are looking for", "you have", "competen",
)

# Lines that are about the employer, not the candidate.
_REJECT = (
    "equal opportunity", "accommodation", "affirmative action", "benefits",
    "salary range", "compensation", "we offer", "about us", "our mission",
    "apply now", "click here", "privacy", "e-verify", "base pay",
)

_MIN_WORDS = 4
_MAX_WORDS = 45


def _split_units(job_text):
    """Break the posting into candidate lines: bullets, then sentences."""
    units = []
    for line in job_text.splitlines():
        line = line.strip(" \t•·-–—*+>").strip()
        if not line:
            continue
        # A long line is prose; split it into sentences so one requirement
        # per unit stays achievable.
        if len(line.split()) > _MAX_WORDS:
            units.extend(s.strip() for s in re.split(r"(?<=[.;!?])\s+", line))
        else:
            units.append(line)
    return [u for u in units if u]


def extract_rule_based(job_text):
    """Deterministic extraction — no API key required."""
    seen = set()
    out = []
    for unit in _split_units(job_text):
        lowered = unit.lower()
        words = unit.split()

        if not (_MIN_WORDS <= len(words) <= _MAX_WORDS):
            continue
        if any(bad in lowered for bad in _REJECT):
            continue
        if not any(sig in lowered for sig in _SIGNALS):
            continue
        # Section headings ("REQUIREMENTS") aren't requirements themselves.
        if unit.isupper() or unit.endswith(":"):
            continue

        key = re.sub(r"\W+", " ", lowered).strip()
        if key in seen:
            continue
        seen.add(key)

        preferred = any(w in lowered for w in ("preferred", "nice to have", "bonus", "plus", "desirable"))
        out.append({
            "text": unit.rstrip(".,;"),
            "kind": "preferred" if preferred else "required",
            "category": "skill",
            "source": "rules",
        })
    return out


def extract_with_llm(job_text, provider, api_key):
    """LLM extraction. Returns [] on any failure so the caller can fall back."""
    result = call_json(
        provider, api_key, SYSTEM_PROMPT,
        EXTRACTION_PROMPT.format(job=job_text[:6000]),
    )
    if not result.get("success"):
        return []

    items = result.get("requirements") or result.get("data") or []
    out = []
    for item in items:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if len(text.split()) < 2:
            continue
        out.append({
            "text": text,
            "kind": "preferred" if item.get("kind") == "preferred" else "required",
            "category": item.get("category") or "skill",
            "source": "llm",
        })
    return out


def extract_requirements(job_text, provider=None, api_key=None):
    """Preferred path first, rules as the safety net.

    Returns (requirements, source) so the UI can be honest about which ran —
    rule-based output is noisier and the user should know when they're seeing it.
    """
    if provider and api_key:
        llm = extract_with_llm(job_text, provider, api_key)
        if len(llm) >= 5:
            return llm, "llm"

    return extract_rule_based(job_text), "rules"
