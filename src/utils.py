# src/utils.py
"""Shared helpers.

Score bands live here rather than inside Comparator and Generator, because
they used to be hardcoded separately in both — on two different scales — and
could disagree about the same score. One definition, read from config.yaml.
"""

import re
from functools import lru_cache

from .config import SCORE_BANDS, SKILLS

_VERDICTS = {
    "strong": "Excellent match — you're ready to apply.",
    "good": "Good match — close a few gaps to strengthen the application.",
    "partial": "Partial match — align your resume more closely with this posting.",
    "low": "Low match — significant gaps for this role.",
}


def score_band(score: float) -> str:
    """Bucket a cosine similarity into strong / good / partial / low."""
    score = float(score)
    if score >= SCORE_BANDS["strong"]:
        return "strong"
    if score >= SCORE_BANDS["good"]:
        return "good"
    if score >= SCORE_BANDS["partial"]:
        return "partial"
    return "low"


def score_verdict(score: float) -> str:
    """One-line plain-English reading of a score."""
    return _VERDICTS[score_band(score)]


@lru_cache(maxsize=1)
def skill_patterns():
    """Compile the skill vocabulary once, shared by Cleaner and Matcher.

    Lookarounds keep a skill from firing inside a longer word ("java" inside
    "javascript") while still matching when punctuation follows.
    """
    return tuple(
        (s.lower(), re.compile(r"(?<![a-z0-9])" + re.escape(s.lower()) + r"(?![a-z0-9])"))
        for s in SKILLS
    )


def find_skills(text):
    """Every known skill literally present in `text`."""
    lowered = text.lower()
    return sorted(skill for skill, pattern in skill_patterns() if pattern.search(lowered))


# Degree levels, so "requires a Master's" can be checked against what the
# resume actually claims instead of being judged by vector similarity.
_DEGREE_LEVELS = (
    (3, ("phd", "ph.d", "doctorate", "doctoral")),
    (2, ("master", "msc", "m.sc", "m.s.", "mba", "graduate degree", "advanced degree")),
    (1, ("bachelor", "bsc", "b.sc", "b.s.", "undergraduate", "b.eng", "beng")),
)


def degree_level(text):
    """Highest degree level named in the text; 0 if none."""
    lowered = text.lower()
    for level, markers in _DEGREE_LEVELS:
        if any(m in lowered for m in markers):
            return level
    return 0
