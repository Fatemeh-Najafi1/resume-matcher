# eval/scorers.py
"""Every scorer the matcher could use, behind one interface, with a clock on it.

The brief this was built against asks for "a clear baseline against
alternatives such as skills mapping, semantic search or ranking methods,
including the impact on cost and response time". So the ladder here is
deliberately built from the bottom up, and every rung is measured on the same
pairs with the same timer:

    keyword overlap   the floor. Jaccard over content words. No model, no
                      download, microseconds. If something cannot beat this it
                      has no business being in the pipeline.
    BM25              proper lexical ranking, still no model. The honest
                      version of what a "keyword scanner" does.
    skills mapping    the literal gate alone — vocabulary, aliases, degree and
                      duration rules. Deterministic, and the only scorer here
                      that can produce an auditable reason.
    semantic search   bi-encoder cosine. One ~90MB model.
    ranking           cross-encoder over the bi-encoder's shortlist. A second
                      model, and far more compute per pair.
    entailment        NLI. Asks "does this bullet mean the candidate has X",
                      which is a different question from "is this relevant".

Reporting cost next to accuracy is the part usually left out, and it changes
conclusions: a rung that buys two points of F1 for forty times the latency is
not obviously the right choice for a free-tier deployment, and a reader cannot
see that from an accuracy table alone.
"""

import math
import re
import time
from collections import Counter

import numpy as np

from src.Matcher import hard_gate
from src import Verdict
from src.utils import normalise

_WORD = re.compile(r"\w+", re.UNICODE)

# Words carrying no signal about whether a bullet evidences a requirement.
# Kept tiny and explicit rather than pulled from NLTK, so the lexical baselines
# stay dependency-free and reproducible.
_STOP = frozenset("""
a an the and or of in on at to for with by from as is are was were be been
experience experiences working work works ability able strong good excellent
knowledge familiar proficiency proficient skills skill years year
""".split())


def _tokens(text):
    return [w for w in _WORD.findall(normalise(text)) if w not in _STOP and len(w) > 1]


# ── Rung 1: keyword overlap ──────────────────────────────────────────────────

def score_keyword_overlap(pairs):
    """Jaccard overlap of content words. The floor every other rung must clear."""
    out = []
    for p in pairs:
        a, b = set(_tokens(p["requirement"])), set(_tokens(p["bullet"]))
        out.append(len(a & b) / len(a | b) if a | b else 0.0)
    return out


# ── Rung 2: BM25 ─────────────────────────────────────────────────────────────

def score_bm25(pairs, k1=1.5, b=0.75):
    """BM25 of the requirement against the bullet.

    Implemented here rather than pulled in as a dependency: it is twenty lines,
    it keeps the baseline reproducible without a package install, and a
    baseline nobody can run is not a baseline.

    The corpus is every bullet in the evaluation set, so the IDF term knows
    which words are actually discriminative in this domain — "experience" is
    everywhere and earns nothing, "Debezium" is rare and earns a lot.
    """
    corpus = [_tokens(p["bullet"]) for p in pairs]
    n = len(corpus)
    if not n:
        return []

    avg_len = sum(len(d) for d in corpus) / n
    doc_freq = Counter()
    for doc in corpus:
        doc_freq.update(set(doc))

    def idf(term):
        df = doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    out = []
    for p, doc in zip(pairs, corpus):
        counts = Counter(doc)
        length = len(doc) or 1
        score = 0.0
        for term in _tokens(p["requirement"]):
            tf = counts.get(term, 0)
            if not tf:
                continue
            score += idf(term) * (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * length / avg_len))
        out.append(score)
    return out


# ── Rung 3: skills mapping (the literal gate alone) ──────────────────────────

def score_skills_mapping(pairs):
    """The literal gate with no model behind it.

    Returns 1.0 / 0.5 / 0.0 for covered / partial / everything else, and 0.5
    where the gate declines to rule — a deliberate "no opinion" that lands
    exactly on the decision boundary rather than pretending to a verdict.

    This is the rung most commercial tools actually are, and it is worth
    measuring honestly: it is fast, free and fully explainable, and the gap
    between it and the model rungs is the only thing that justifies shipping a
    model at all.
    """
    out = []
    for p in pairs:
        status, _, _ = hard_gate(p["requirement"], p["bullet"])
        out.append({
            Verdict.COVERED: 1.0,
            Verdict.PARTIAL: 0.5,
            Verdict.UNSUPPORTED: 0.0,
            Verdict.CONTRADICTED: 0.0,
        }.get(status, 0.5))
    return out


# ── Rung 4: semantic search ──────────────────────────────────────────────────

def score_bi_encoder(pairs, embedder):
    """Cosine between two independently-embedded texts."""
    reqs = embedder.embed_many([p["requirement"] for p in pairs])
    bullets = embedder.embed_many([p["bullet"] for p in pairs])
    out = []
    for r, b in zip(reqs, bullets):
        rn, bn = np.linalg.norm(r), np.linalg.norm(b)
        out.append(float(np.dot(r, b) / (rn * bn)) if rn and bn else 0.0)
    return out


# ── Rung 5: ranking ──────────────────────────────────────────────────────────

def score_cross_encoder(pairs, model_name):
    """Read requirement and bullet together and score the pair."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    return [float(s) for s in
            model.predict([(p["requirement"], p["bullet"]) for p in pairs])]


# ── Rung 6: entailment ───────────────────────────────────────────────────────

def score_nli(pairs, model_name):
    """P(the bullet entails that the candidate has the requirement)."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    inputs = [(p["bullet"], f"The candidate has {p['requirement'].lower()}.")
              for p in pairs]
    logits = np.asarray(model.predict(inputs))

    labels = getattr(getattr(model, "config", None), "id2label", None) or {}
    entail = next((i for i, name in labels.items()
                   if str(name).lower().startswith("entail")), logits.shape[1] - 1)

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return [float(x) for x in probs[:, int(entail)]]


# ── The ladder, with a clock ─────────────────────────────────────────────────

#: Approximate on-disk size of each model, for the cost column. Measured from
#: the HuggingFace cache rather than quoted, where the cache is present.
MODEL_SIZES = {
    "keyword overlap": 0,
    "BM25": 0,
    "skills mapping": 0,
}


def timed(fn, *args, **kwargs):
    """Run a scorer and report wall-clock seconds alongside its scores.

    Timing includes model loading on the first call, which is the honest
    number for a cold Streamlit container but not for a warm one, so callers
    that care about steady-state throughput should call twice and take the
    second.
    """
    start = time.perf_counter()
    scores = fn(*args, **kwargs)
    return scores, time.perf_counter() - start


def model_disk_bytes(repo_id):
    """Bytes the model occupies in the local HuggingFace cache, or 0.

    Measured, not quoted. Returns 0 when the model is not cached, which is
    honest — an uncached model's download cost is a different number from its
    disk footprint and should not be conflated.
    """
    from pathlib import Path
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    folder = cache / ("models--" + repo_id.replace("/", "--"))
    if not folder.exists():
        return 0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
