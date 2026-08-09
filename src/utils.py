# src/utils.py
"""Shared helpers.

Score bands live here rather than inside Comparator and Generator, because
they used to be hardcoded separately in both — on two different scales — and
could disagree about the same score. One definition, read from config.yaml.
"""

import re
from functools import lru_cache

from .config import SCORE_BANDS, SKILL_ALIASES, SKILLS

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

    The alias map's keys are part of the vocabulary too. Without that, writing
    `gradient boosting: [xgboost, lightgbm]` in skill_aliases.yaml had no
    effect: "gradient boosting" was not a recognised skill, so the gate never
    engaged and the alias never got a chance to fire.
    """
    vocabulary = {s.lower() for s in SKILLS} | set(SKILL_ALIASES)
    return tuple(
        (s, re.compile(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])"))
        for s in sorted(vocabulary)
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


@lru_cache(maxsize=512)
def _term_pattern(term):
    return re.compile(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])")


def evidence_for(skill, text):
    """Which term in `text` evidences `skill` — the skill itself or an alias.

    Returns the matching term, or "" if nothing does. Aliases are one-way:
    Git evidences "version control", but a posting asking for Git is not
    satisfied by a resume that only says "version control".
    """
    lowered = text.lower()
    skill = skill.lower()
    if _term_pattern(skill).search(lowered):
        return skill
    for alias in SKILL_ALIASES.get(skill, ()):
        if _term_pattern(alias).search(lowered):
            return alias
    return ""


# Requirements phrased as personal qualities rather than checkable facts. A
# resume cannot really settle "attention to detail", and the labelled eval set
# shows why chasing them is futile: on that category two careful people
# disagree, so the model is being scored against a coin flip. Better to hand
# these to the reader with the closest evidence than to guess a verdict.
_JUDGEMENT_PATTERNS = tuple(re.compile(p) for p in (
    r"\b(strong|excellent|good|outstanding|exceptional|solid|great)\s+\w*\s*skills?\b",
    r"\battention to detail\b",
    r"\bdetail[- ]oriented\b",
    r"\bfast[- ]paced\b",
    r"\bself[- ](starter|directed|motivated)\b",
    r"\bwork(s|ing)? (independently|autonomously)\b",
    r"\b(passion|passionate|enthusiasm|curiosity|commitment|dedicated)\b",
    r"\bteam player\b",
    r"\bwork ethic\b",
    r"\bability to (thrive|adapt|multitask|juggle)\b",
    r"\b(adaptab|flexib)\w*\b",
    r"\bproblem[- ]solving\b",
    r"\bgrowth mindset\b",
    r"\bproactive\b",
    r"\bhighly motivated\b",
    r"\b(excellent|strong|clear|effective)\s+(written|verbal|oral|interpersonal|communication)\b",
    r"\bwritten and verbal\b",
    r"\binterpersonal\b",
    r"\bcross[- ]functional(ly)?\b",
    r"\bteam (player|fit|spirit)\b",
))


def is_judgement_call(requirement):
    """True if the requirement asks about a personal quality, not a fact."""
    lowered = requirement.lower()
    return any(p.search(lowered) for p in _JUDGEMENT_PATTERNS)
