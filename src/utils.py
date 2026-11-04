# src/utils.py
"""Shared helpers.

Score bands live here rather than inside Comparator and Generator, because
they used to be hardcoded separately in both — on two different scales — and
could disagree about the same score. One definition, read from config.yaml.
"""

import re
import unicodedata
from functools import lru_cache

from .config import SCORE_BANDS, SKILL_ALIASES, SKILLS

# ── Script-neutral text normalisation ────────────────────────────────────────
#
# The skill vocabulary is matched literally, so every literal match depends on
# both sides being spelled the same way. For English that means lowercasing.
# For Arabic it means rather more, because the same word has several legal
# written forms:
#
#   * alef carries optional hamza/madda — أ إ آ ٱ all stand for ا
#   * final yeh is written ي or ى more or less interchangeably
#   * teh marbuta ة is often typed as ه
#   * short vowels (harakat) are optional diacritics, usually omitted
#   * tatweel ـ is a typographic stretch with no phonetic value
#
# A resume written one way and a posting written the other would never match
# without this, and the failure would be silent — exactly the class of bug the
# chunking fix in Embedder was about.

_ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟـ]")

_ARABIC_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي",
    "ة": "ه",
    # Arabic-Indic and Eastern Arabic-Indic digits -> ASCII, so "٥ سنوات"
    # and "5 years" parse through the same code path.
    **{chr(0x0660 + d): str(d) for d in range(10)},
    **{chr(0x06F0 + d): str(d) for d in range(10)},
})


def normalise(text):
    """Fold a string to the form every literal lookup in this project uses.

    Lowercases, strips Arabic diacritics and tatweel, folds alef/yeh/teh-marbuta
    variants, and maps Arabic-Indic digits to ASCII. Latin text is unaffected
    beyond lowercasing, so this is safe to apply everywhere.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _ARABIC_DIACRITICS.sub("", text)
    return text.translate(_ARABIC_FOLD)


# U+0600..U+06FF is the main Arabic block; U+0750.. and U+FB50.. cover
# supplements and presentation forms that turn up in copied-and-pasted text.
_ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def has_arabic(text):
    """True if the text contains Arabic script.

    Deliberately a script test rather than language detection. The question
    the caller is asking is "can the English-only encoder represent this?",
    and the answer turns on the alphabet, not on which language it is. It also
    degrades the right way: a probabilistic detector that guesses wrong on a
    two-word Arabic line fails silently, while this cannot.
    """
    return bool(_ARABIC_SCRIPT.search(text or ""))


def _bounded(term):
    """Compile `term` so it matches as a whole word in any script.

    The old pattern used `(?<![a-z0-9])`, which keeps "java" from firing inside
    "javascript" but does nothing in Arabic: an Arabic letter is not in [a-z0-9],
    so the lookaround passed and a term matched happily in the middle of a
    longer word. `\\w` is Unicode-aware by default in Python 3, so it holds the
    boundary for both scripts.
    """
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")

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
def aliases_normalised():
    """SKILL_ALIASES with both sides folded through normalise().

    config.py lowercases the YAML keys, which is enough for Latin text and
    does nothing at all for Arabic: "تعلم الآلة" kept its madda alef while
    find_skills() returned the folded "تعلم الاله", so every Arabic alias
    lookup missed and the gate reported "not found in your resume" about a
    skill that was sitting right there.

    Folding here rather than in config.py keeps config free of imports from
    this module, which would be circular.
    """
    return {normalise(concept): {normalise(t) for t in terms}
            for concept, terms in SKILL_ALIASES.items()}


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
    vocabulary = {normalise(s) for s in SKILLS} | set(aliases_normalised())
    return tuple(
        (skill, _bounded(skill)) for skill in sorted(v for v in vocabulary if v)
    )


def find_skills(text):
    """Every known skill literally present in `text`."""
    folded = normalise(text)
    return sorted(skill for skill, pattern in skill_patterns() if pattern.search(folded))


# Degree levels, so "requires a Master's" can be checked against what the
# resume actually claims instead of being judged by vector similarity.
# Markers are normalise()d before comparison, so the Arabic entries here are
# written in their plain form and will still match the hamza'd, tatweel'd and
# teh-marbuta variants a real document uses.
#
# The Arabic entries are not an optimisation. Without them `degree_required`
# returned 0 for every Arabic requirement, the degree gate never fired, and
# the requirement fell through to cosine — which cheerfully returned "covered"
# for a master's demand against a resume with no master's. Silent, confident
# and wrong, which is the worst of the three.
_DEGREE_LEVELS = (
    (3, ("phd", "ph.d", "doctorate", "doctoral",
         "دكتوراه", "الدكتوراه")),
    (2, ("master", "msc", "m.sc", "m.s.", "mba", "graduate degree", "advanced degree",
         "ماجستير", "الماجستير")),
    (1, ("bachelor", "bsc", "b.sc", "b.s.", "undergraduate", "b.eng", "beng",
         "بكالوريوس", "البكالوريوس", "ليسانس")),
)


def degree_levels(text):
    """Every degree level named in the text, as a set. Empty if none."""
    folded = normalise(text)
    return {level for level, markers in _DEGREE_LEVELS
            if any(normalise(m) in folded for m in markers)}


def degree_level(text):
    """Highest degree level named in the text; 0 if none.

    Right for a resume, where naming a master's and a bachelor's means the
    master's is held. Wrong for a requirement, where "Master's or PhD" is
    satisfied by the lower of the two — see `degree_required`.
    """
    levels = degree_levels(text)
    return max(levels) if levels else 0


def degree_required(text):
    """The LOWEST degree level a requirement will accept; 0 if it names none.

    Postings overwhelmingly phrase this as a floor plus a wish — "Master's
    degree required, PhD preferred", "Bachelor's or higher". Reading the
    highest level named turned every one of those into a demand for the
    wish, and told a perfectly qualified master's-holder they were
    contradicted by a posting that would have accepted them.
    """
    levels = degree_levels(text)
    return min(levels) if levels else 0


@lru_cache(maxsize=512)
def _term_pattern(term):
    return _bounded(normalise(term))


def evidence_for(skill, text):
    """Which term in `text` evidences `skill` — the skill itself or an alias.

    Returns the matching term, or "" if nothing does. Aliases are one-way:
    Git evidences "version control", but a posting asking for Git is not
    satisfied by a resume that only says "version control".
    """
    folded = normalise(text)
    skill = normalise(skill)
    if _term_pattern(skill).search(folded):
        return skill
    for alias in aliases_normalised().get(skill, ()):
        if _term_pattern(alias).search(folded):
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
    folded = normalise(requirement)
    return any(p.search(folded) for p in _JUDGEMENT_PATTERNS)
