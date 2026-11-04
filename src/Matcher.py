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
matcher against the labelled pairs in data/eval/; `python -m eval.holdout`
re-scores it on a set that was written before any of it was run and never
tuned against.

An earlier version of this docstring recorded the opposite conclusion — that a
cross-encoder had been tried and rejected on recall 0.68. That was true of the
first eval set, which was built only from the strong senior resume in the repo
and therefore over-represented pairs where a match exists. Pairs 91–124 added
junior-vs-senior cases and the ranking reversed: the cross-encoder now wins on
the same benchmark, 5-fold cross-validated, and ships. The lesson was about
eval-set construction rather than about either model, and it is written up in
docs/ENGINEERING-LOG.md.
"""

import re

import numpy as np

from . import Reranker, Verdict
from .config import MATCH_THRESHOLDS, RERANKER
from .utils import evidence_for, find_skills, has_arabic, is_judgement_call

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

    The verdicts come from src/Verdict.py, which keeps "your resume is silent
    on this" apart from "your resume rules this out". See that module for why
    the distinction is the point rather than a detail.

    Returns (status, reason, present_skills), or (None, "", []) to defer to
    similarity.
    """
    # 1. Degree requirements are ordinal, not fuzzy — and a resume that names
    #    no degree is unanswered, not unqualified.
    degree = Verdict.resolve_degree(requirement_text, resume_text)
    if degree:
        return degree

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

    # Nothing found is silence, not disproof. A resume that never mentions
    # Kafka says nothing about whether the candidate has used Kafka, so this
    # is reported as a question to ask rather than a reason to reject.
    if not present:
        return (Verdict.UNSUPPORTED,
                "not found in your resume: " + ", ".join(named) +
                " — add it if you have it", [])

    if absent:
        return (Verdict.PARTIAL,
                "found " + describe(present) + "; not found: " + ", ".join(absent),
                present)

    # "At least 5 years of machine learning research" is not satisfied by
    # having touched scikit-learn. Naming a tool never proves a duration, so
    # the best case here is partial — unless the resume states a figure of its
    # own that falls short, which is a real shortfall and gets said plainly.
    duration = Verdict.resolve_duration(requirement_text, resume_text, present, describe)
    if duration:
        return duration

    return Verdict.COVERED, "found " + describe(present), present


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
    the shortlist, or None if reranking is off, unavailable, or inapplicable.

    Reranking is skipped entirely when either side contains Arabic. The
    configured reranker (ms-marco) is English-only, so on Arabic it emits
    confident nonsense — and because the reranker *overrides* cosine where it
    runs, leaving it on would have thrown away the multilingual encoder's work
    and replaced it with noise. Routing the embedder without routing the
    reranker was a real bug: the bilingual numbers improved sharply once the
    cross-encoder stopped overruling a model that could actually read the
    text.
    """
    if not RERANKER.get("model"):
        return None
    if any(has_arabic(t) for t in req_texts) or any(has_arabic(b) for b in bullets):
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
                "counts": {v: 0 for v in Verdict.ORDER}}

    req_texts = [r["text"] for r in requirements]
    if not bullets:
        rows = [dict(r, status=Verdict.UNSUPPORTED, score=0.0, evidence="",
                     evidence_index=-1) for r in requirements]
        counts = {v: 0 for v in Verdict.ORDER}
        counts[Verdict.UNSUPPORTED] = len(rows)
        return {"requirements": rows, "bullets": [], "coverage": 0.0, "counts": counts}

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
    counts = {v: 0 for v in Verdict.ORDER}
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
                status, decided_by = Verdict.JUDGEMENT, "judgement"
                reason = "a personal quality rather than a checkable fact — your call"
            elif score >= covered_at:
                status = Verdict.COVERED
            elif score >= partial_at:
                status = Verdict.PARTIAL
            else:
                # Similarity can fail to find evidence; it can never show that
                # the resume rules the requirement out. So the floor of the
                # similarity path is `unsupported`, never `contradicted`.
                status = Verdict.UNSUPPORTED
        counts[status] = counts.get(status, 0) + 1

        # Evidence must support the verdict it's shown under. When the verdict
        # came from finding a named skill, cite a bullet that actually contains
        # that skill — otherwise "found python" gets illustrated with whichever
        # unrelated line happened to score highest.
        evidence_index = -1 if status == Verdict.UNSUPPORTED else best
        if status == Verdict.JUDGEMENT:
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
    decidable = len(requirements) - counts[Verdict.JUDGEMENT]
    coverage = Verdict.coverage(counts, decidable)

    # Strongest evidence first inside each status group, so the reader sees the
    # clearest wins and the starkest gaps rather than posting order.
    order = {status: i for i, status in enumerate(Verdict.ORDER)}
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
