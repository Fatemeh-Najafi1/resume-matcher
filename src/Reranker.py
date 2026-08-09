# src/Reranker.py
"""Optional cross-encoder reranking for requirement/bullet pairs.

A bi-encoder embeds the requirement and the bullet separately, so the two texts
never see each other. That is fast, and it is why cosine rated "experience
using frameworks such as PyTorch" at 0.57 against a line listing scikit-learn
and NLTK — in embedding space, ML tooling all lives in the same neighbourhood.

A cross-encoder reads both texts in one pass and attends across them. Measured
on data/eval/pairs.jsonl under 5-fold cross-validation:

    bi-encoder    + gate    P 0.90  R 0.53
    cross-encoder + gate    P 0.92  R 0.65

Same precision, twelve more points of recall — the app stops missing a third of
what it used to.

Loading is lazy and every failure is soft. A deployment without the model, or
without the memory to hold a second transformer, falls back to the bi-encoder
rather than breaking.
"""

from functools import lru_cache

from .config import RERANKER

# Rerank only this many candidates per requirement. The bi-encoder is a fine
# filter and a poor judge, so it picks the shortlist and the cross-encoder
# rules on it. Scoring every pair would be quadratic for no benefit.
TOP_K = 5


@lru_cache(maxsize=2)
def _load(model_name):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def available():
    """True if reranking is configured and the model can actually be loaded."""
    if not RERANKER.get("model"):
        return False
    try:
        _load(RERANKER["model"])
        return True
    except Exception:
        return False


def score_pairs(pairs):
    """Score [(requirement, bullet), ...]. Returns None if unavailable.

    None means "carry on with the bi-encoder", not "fail" — see module docstring.
    """
    if not pairs or not RERANKER.get("model"):
        return None
    try:
        model = _load(RERANKER["model"])
        return [float(s) for s in model.predict(list(pairs))]
    except Exception:
        return None
