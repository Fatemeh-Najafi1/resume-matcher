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
"""

import re

import numpy as np

from .config import MATCH_THRESHOLDS
from .utils import degree_level, find_skills

# Lines that are structure, not content.
_HEADING = re.compile(
    r"^(summary|education|experience|projects?|technical skills|skills|"
    r"certifications?|activities|languages?|interests|references|"
    r"work experience|professional experience|publications)\b[:\s]*$",
    re.IGNORECASE,
)

_MIN_BULLET_WORDS = 5


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

    in_resume = set(find_skills(resume_text))
    present = [s for s in named if s in in_resume]
    absent = [s for s in named if s not in in_resume]

    if not present:
        return "missing", "not found in your resume: " + ", ".join(named), []
    if absent:
        return "partial", "found " + ", ".join(present) + "; missing " + ", ".join(absent), present
    return "covered", "found " + ", ".join(present), present


def _cosine_matrix(a, b):
    """Row-wise cosine similarity between two matrices of vectors."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1e-9
    b_norm[b_norm == 0] = 1e-9
    return (a / a_norm) @ (b / b_norm).T


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

    full_resume = resume_text if resume_text is not None else "\n".join(bullets)

    rows = []
    counts = {"covered": 0, "partial": 0, "missing": 0}
    for i, requirement in enumerate(requirements):
        best = int(np.argmax(scores[i]))
        score = float(scores[i][best])

        # A literal verdict wins where one exists; similarity fills the rest.
        status, reason, present = hard_gate(requirement["text"], full_resume)
        decided_by = "literal" if status else "similarity"
        if status is None:
            if score >= covered_at:
                status = "covered"
            elif score >= partial_at:
                status = "partial"
            else:
                status = "missing"
        counts[status] += 1

        # Evidence must support the verdict it's shown under. When the verdict
        # came from finding a named skill, cite a bullet that actually contains
        # that skill — otherwise "found python" gets illustrated with whichever
        # unrelated line happened to score highest.
        evidence_index = best if status != "missing" else -1
        if present:
            supporting = [k for k, b in enumerate(bullets) if set(find_skills(b)) & set(present)]
            if supporting:
                evidence_index = max(supporting, key=lambda k: scores[i][k])

        rows.append(dict(
            requirement,
            status=status,
            score=score,
            reason=reason,
            decided_by=decided_by,
            evidence=bullets[evidence_index] if evidence_index >= 0 else "",
            evidence_index=evidence_index,
        ))

    coverage = (counts["covered"] + 0.5 * counts["partial"]) / len(requirements)

    # Strongest evidence first inside each status group, so the reader sees the
    # clearest wins and the starkest gaps rather than posting order.
    order = {"missing": 0, "partial": 1, "covered": 2}
    rows.sort(key=lambda r: (order[r["status"]], -r["score"]))

    return {"requirements": rows, "bullets": bullets,
            "coverage": coverage, "counts": counts}


def unmatched_bullets(result):
    """Resume bullets that were nobody's best evidence.

    Not necessarily dead weight, but for a targeted application these are the
    lines earning the least against this particular posting.
    """
    used = {r["evidence_index"] for r in result["requirements"] if r["evidence_index"] >= 0}
    return [b for i, b in enumerate(result["bullets"]) if i not in used]
