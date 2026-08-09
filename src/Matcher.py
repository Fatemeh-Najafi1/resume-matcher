# src/Matcher.py
"""Match each job requirement against each resume bullet.

Why this exists: a single cosine over two whole documents produces one number
that can't be acted on. It says "52.3%" without saying which of your experience
counted, or which requirement went unanswered.

Comparing at requirement-vs-bullet granularity gives:
  * a coverage figure that means something ("11 of 18 requirements covered")
  * the specific bullet that is your best evidence for each requirement
  * gaps expressed as real requirements instead of stray keywords

It also sidesteps the truncation problem rather than mitigating it: a single
requirement and a single bullet both sit far inside the model's 256-token
window, so nothing is silently dropped.

Thresholds here are measured, not chosen. `python -m eval.run_eval` scores this
matcher against 90 labelled pairs; the current settings give precision 0.94 /
recall 0.73 in-sample and 0.88 / 0.80 under 5-fold cross-validation. Two
alternatives were tried and rejected on that benchmark:

  * a cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) reached
    only recall 0.68 -- it is trained for search relevance, and this is not a
    relevance question;
  * an NLI entailment model fixed the synonym cases the cross-encoder missed
    (1/8 -> 6/8) but did not beat a properly calibrated bi-encoder overall.

The bi-encoder already here wins once its threshold is calibrated, so no third
model was added. That conclusion is reproducible: run the eval.
"""

import re

import numpy as np

from . import Reranker
from .config import MATCH_THRESHOLDS, RERANKER
from .utils import degree_level, evidence_for, find_skills, is_judgement_call

# Lines that are structure, not content.
_HEADING = re.compile(
    r"^(summary|education|experience|projects?|technical skills|skills|"
    r"certifications?|activities|languages?|interests|references|"
    r"work experience|professional experience|publications)\b[:\s]*$",
    re.IGNORECASE,
)

_MIN_BULLET_WORDS = 5

# "at least 1 year", "3+ years", "minimum of two years" — a duration demand.
_DURATION = re.compile(
    r"\b(?:\d+\+?|one|two|three|four|five|six|seven|eight|nine|ten)\s*\+?\s*(?:year|yr)s?\b"
)


def split_bullets(resume_text):
    """Split a resume into the individual claims it makes.

    Section headings and contact lines are dropped; everything else becomes a
    unit that a requirement can be matched against.
    """
    bullets = []
    for raw in resume_text.splitlines():
        line = raw.strip(" \t•·-–—*+>").strip()
        if not line or _HEADING.match(line):
            continue
        if len(line.split()) < _MIN_BULLET_WORDS:
            continue
        # A "Languages: Python, JavaScript, SQL" line is a real claim, but a
        # bare date or a URL is not.
        if re.fullmatch(r"[\d\s/–—-]+", line):
            continue
        bullets.append(line)
    return bullets


def hard_gate(requirement_text, resume_text):
    """Decide a requirement literally, when it can be decided literally.

    Embedding similarity answers "is this similar work?" well and "do they know
    this specific thing?" badly. Left to cosine alone, this matcher rated
    "Relevant experience using PyTorch" a partial match against a line reading
    "Flask, Streamlit, Pandas, NumPy, scikit-learn" — the vector neighbourhood
    of ML tooling is dense, and PyTorch simply isn't there. It likewise gave a
    third-year undergraduate partial credit for "Advanced degree (Masters, PhD)".

    So when a requirement names concrete technologies or a degree level, that
    decides it. Cosine only rules on requirements too general to check —
    "experience solving complex problems".

    Returns (status, reason, present_skills), or (None, "", []) to defer to
    similarity.
    """
    # 1. Degree requirements are ordinal, not fuzzy.
    needed = degree_level(requirement_text)
    if needed:
        held = degree_level(resume_text)
        names = {0: "none", 1: "a bachelor's", 2: "a master's", 3: "a PhD"}
        if held >= needed:
            return "covered", f"your resume shows {names[held]}", []
        return "missing", f"requires {names[needed]}; your resume shows {names[held]}", []

    # 2. Named technologies: presence is checkable, so check it.
    named = find_skills(requirement_text)
    if not named:
        return None, "", []

    # A skill counts as present if the resume names it OR names something that
    # evidences it: Git for "version control", NLP for "natural language
    # processing". Without this the gate rejected a bullet reading "Built NLP
    # tools" against a requirement spelling the phrase out in full.
    present, absent, via = [], [], {}
    for skill in named:
        match = evidence_for(skill, resume_text)
        if match:
            present.append(skill)
            if match != skill:
                via[skill] = match
        else:
            absent.append(skill)

    def describe(skills):
        return ", ".join(f"{s} (via {via[s]})" if s in via else s for s in skills)

    if not present:
        return "missing", "not found in your resume: " + ", ".join(named), []
    if absent:
        return "partial", "found " + describe(present) + "; missing " + ", ".join(absent), present

    # "At least 1 year of machine learning research" is not satisfied by having
    # touched scikit-learn. Naming the skill is necessary but not sufficient
    # when the posting also demands a duration, so cap these at partial rather
    # than claiming the years are proven. Same principle as the degree check.
    if _DURATION.search(requirement_text.lower()):
        return ("partial",
                "found " + describe(present) + ", but this asks for a specific amount of "
                "experience, which a skills list cannot prove", present)

    return "covered", "found " + describe(present), present


def _cosine_matrix(a, b):
    """Row-wise cosine similarity between two matrices of vectors."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1e-9
    b_norm[b_norm == 0] = 1e-9
    return (a / a_norm) @ (b / b_norm).T


def _rerank_scores(req_texts, bullets, cosine_scores):
    """Cross-encoder scores for the top-K bullets of each requirement.

    Returns a matrix shaped like `cosine_scores` with -inf everywhere outside
    the shortlist, or None if reranking is off or unavailable.
    """
    if not RERANKER.get("model"):
        return None

    k = min(Reranker.TOP_K, len(bullets))
    shortlist = [list(np.argsort(-cosine_scores[i])[:k]) for i in range(len(req_texts))]

    flat = [(req_texts[i], bullets[j]) for i, cols in enumerate(shortlist) for j in cols]
    scored = Reranker.score_pairs(flat)
    if scored is None:
        return None

    out = np.full_like(cosine_scores, -np.inf, dtype=float)
    pos = 0
    for i, cols in enumerate(shortlist):
        for j in cols:
            out[i][j] = scored[pos]
            pos += 1
    return out


def match(requirements, bullets, embedder, resume_text=None):
    """Score every requirement against every bullet.

    Returns a dict with per-requirement evidence and an overall coverage
    figure. `coverage` counts a partial match as half — a requirement your
    resume gestures at is not the same as one it answers, and not the same as
    one it ignores.
    """
    covered_at = MATCH_THRESHOLDS["covered"]
    partial_at = MATCH_THRESHOLDS["partial"]

    if not requirements:
        return {"requirements": [], "bullets": bullets, "coverage": 0.0,
                "counts": {"covered": 0, "partial": 0, "missing": 0}}

    req_texts = [r["text"] for r in requirements]
    if not bullets:
        rows = [dict(r, status="missing", score=0.0, evidence="", evidence_index=-1)
                for r in requirements]
        return {"requirements": rows, "bullets": [], "coverage": 0.0,
                "counts": {"covered": 0, "partial": 0, "missing": len(rows)}}

    req_vecs = embedder.embed_many(req_texts)
    bullet_vecs = embedder.embed_many(bullets)
    scores = _cosine_matrix(req_vecs, bullet_vecs)

    # Optionally re-score the shortlist with a cross-encoder. The bi-encoder
    # stays responsible for shortlisting (it is cheap and a good filter); the
    # cross-encoder decides, because it is the better judge. Falls back
    # silently to cosine if the model is unavailable.
    rerank = _rerank_scores(req_texts, bullets, scores)
    if rerank is not None:
        decision_scores, covered_at, partial_at = rerank, RERANKER["covered"], RERANKER["partial"]
        scorer = "cross-encoder"
    else:
        decision_scores, scorer = scores, "bi-encoder"

    full_resume = resume_text if resume_text is not None else "\n".join(bullets)

    rows = []
    counts = {"covered": 0, "partial": 0, "missing": 0, "judgement": 0}
    for i, requirement in enumerate(requirements):
        best = int(np.argmax(decision_scores[i]))
        score = float(decision_scores[i][best])

        # A literal verdict wins where one exists; similarity fills the rest.
        status, reason, present = hard_gate(requirement["text"], full_resume)
        decided_by = "literal" if status else "similarity"
        if status is None:
            if is_judgement_call(requirement["text"]):
                # Not answerable from a resume, and pretending otherwise is how
                # a tool loses trust. Surface the closest evidence and let the
                # reader decide.
                status, decided_by = "judgement", "judgement"
                reason = "a personal quality rather than a checkable fact — your call"
            elif score >= covered_at:
                status = "covered"
            elif score >= partial_at:
                status = "partial"
            else:
                status = "missing"
        counts[status] = counts.get(status, 0) + 1

        # Evidence must support the verdict it's shown under. When the verdict
        # came from finding a named skill, cite a bullet that actually contains
        # that skill — otherwise "found python" gets illustrated with whichever
        # unrelated line happened to score highest.
        evidence_index = best if status != "missing" else -1
        if status == "judgement":
            evidence_index = best   # always show the closest line, verdict withheld
        if present:
            supporting = [k for k, b in enumerate(bullets) if set(find_skills(b)) & set(present)]
            if supporting:
                evidence_index = max(supporting, key=lambda k: scores[i][k])

        rows.append(dict(
            requirement,
            status=status,
            score=score,
            cosine=float(scores[i][best]),
            scorer=scorer,
            reason=reason,
            decided_by=decided_by,
            evidence=bullets[evidence_index] if evidence_index >= 0 else "",
            evidence_index=evidence_index,
        ))

    # Judgement calls are excluded from the denominator. Counting them either
    # way would be a guess dressed up as a measurement, and it would make the
    # headline number depend on how many soft skills the posting happened to
    # list rather than on the candidate.
    decidable = len(requirements) - counts["judgement"]
    coverage = ((counts["covered"] + 0.5 * counts["partial"]) / decidable) if decidable else 0.0

    # Strongest evidence first inside each status group, so the reader sees the
    # clearest wins and the starkest gaps rather than posting order.
    order = {"missing": 0, "partial": 1, "covered": 2, "judgement": 3}
    rows.sort(key=lambda r: (order[r["status"]], -r["score"]))

    return {"requirements": rows, "bullets": bullets, "scorer": scorer,
            "coverage": coverage, "counts": counts, "decidable": decidable}


def unmatched_bullets(result):
    """Resume bullets that were nobody's best evidence.

    Not necessarily dead weight, but for a targeted application these are the
    lines earning the least against this particular posting.
    """
    used = {r["evidence_index"] for r in result["requirements"] if r["evidence_index"] >= 0}
    return [b for i, b in enumerate(result["bullets"]) if i not in used]
