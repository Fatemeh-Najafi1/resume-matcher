# eval/attempts.py
"""Re-run the changes that were tried and rejected.

Run:  python -m eval.attempts            # every attempt
      python -m eval.attempts --only 2   # just one
      python -m eval.attempts --list

Why this exists
---------------
docs/EVALUATION.md reports six candidate improvements, all of which lost. A
reader has no reason to take that on trust, and "we tried X and it didn't
help" is the single easiest claim in machine learning to make carelessly. So
each one is a function here that produces its number on demand.

Everything runs on the DEVELOPMENT set. These are tuning experiments, and
pointing them at the held-out set is what the protocol in docs/EVALUATION.md
exists to prevent.

Attempt 6 is the odd one out. Its code was removed from `src/` after it lost,
because dead branches in a matcher are worse than a missing experiment. The
rule is reimplemented here instead, wrapping `hard_gate` from the outside, so
the result stays reproducible without the production path carrying a flag
nobody should set.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import scorers                                    # noqa: E402
from eval.holdout import (SETS, choose_threshold, load,      # noqa: E402
                          predict, prf)
from src.Embedder import Embedder                           # noqa: E402

BASELINE = "baseline: gate 82 @ 93.9%  P 0.90 R 0.57 F1 0.70"


def _dev():
    pairs = load(SETS["dev"][0])
    return pairs, [p["label"] for p in pairs]


def _report(tag, pairs, labels, scores):
    t = choose_threshold(pairs, scores, labels)
    preds, gated = predict(pairs, scores, t)
    ruled = [(p, l) for p, l, g in zip(preds, labels, gated) if g]
    acc = sum(1 for p, l in ruled if p == l) / len(ruled) if ruled else 0.0
    m = prf(preds, labels)
    print(f"  {tag:<44} gate {len(ruled):>3} @ {acc * 100:5.1f}%   "
          f"P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f}")
    return m


# ── 1 ────────────────────────────────────────────────────────────────────────

def attempt_1_fusion():
    """Are lexical and dense errors complementary enough to fuse?"""
    print("\n1. Fuse BM25 with the bi-encoder\n")
    pairs, labels = _dev()
    lex = scorers.score_bm25(pairs)
    dense = scorers.score_bi_encoder(pairs, Embedder())
    p_lex, _ = predict(pairs, lex, choose_threshold(pairs, lex, labels))
    p_den, _ = predict(pairs, dense, choose_threshold(pairs, dense, labels))

    both = sum(1 for a, b, l in zip(p_lex, p_den, labels) if a != l and b != l)
    lex_only = sum(1 for a, b, l in zip(p_lex, p_den, labels) if a != l and b == l)
    den_only = sum(1 for a, b, l in zip(p_lex, p_den, labels) if a == l and b != l)
    print(f"  BM25 wrong alone   {lex_only:>3}")
    print(f"  dense wrong alone  {den_only:>3}")
    print(f"  both wrong         {both:>3}")
    print(f"\n  complementary {lex_only + den_only} vs {both} shared -> "
          f"{'worth fusing' if lex_only + den_only > both else 'NOT worth fusing'}")


# ── 2 ────────────────────────────────────────────────────────────────────────

def attempt_2_bigger_encoder():
    """Does a far stronger encoder fix the synonym and paraphrase failures?"""
    print("\n2. A much larger encoder\n")
    pairs, labels = _dev()
    for tag, model in (
        ("all-MiniLM-L6-v2 (shipped, ~90MB)", "sentence-transformers/all-MiniLM-L6-v2"),
        ("bge-large-en-v1.5 (~1.3GB)", "BAAI/bge-large-en-v1.5"),
    ):
        try:
            scores = scorers.score_bi_encoder(pairs, Embedder(model_name=model))
            t = choose_threshold(pairs, scores, labels)
            preds, _ = predict(pairs, scores, t)
            cats = {}
            for p, pr, l in zip(pairs, preds, labels):
                cats.setdefault(p["category"], []).append(pr == l)
            syn, par = cats.get("synonym", []), cats.get("paraphrase", [])
            m = prf(preds, labels)
            print(f"  {tag:<38} F1 {m['f1']:.2f}  "
                  f"synonym {sum(syn)}/{len(syn)}  paraphrase {sum(par)}/{len(par)}")
        except Exception as e:
            print(f"  {tag:<38} unavailable ({type(e).__name__})")


# ── 3 ────────────────────────────────────────────────────────────────────────

def attempt_3_esco():
    """Does ESCO contain the technologies this project needs? (needs network)"""
    print("\n3. ESCO as an external vocabulary\n")
    import json
    import urllib.parse
    import urllib.request
    for term in ("python", "javascript", "kubernetes", "docker",
                 "terraform", "machine learning"):
        url = ("https://ec.europa.eu/esco/api/search?"
               + urllib.parse.urlencode({"text": term, "language": "en",
                                         "type": "skill", "limit": 1}))
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                total = json.loads(r.read()).get("total", 0)
            note = "  <- absent" if total == 0 else ""
            print(f"  {term:<20} {total:>5} hits{note}")
        except Exception as e:
            print(f"  {term:<20} lookup failed ({type(e).__name__})")
    print("\n  ESCO is an occupational taxonomy; it does not contain software.")


# ── 4 and 5 ──────────────────────────────────────────────────────────────────

def attempt_4_5_vocabulary():
    """Generated vocabulary on/off, and the alias ablation that cleared them."""
    print("\n4 & 5. External vocabulary, and which half of it was to blame\n")
    import importlib
    import src.config

    pairs, labels = _dev()
    scores = scorers.score_bi_encoder(pairs, Embedder())   # vocabulary-independent

    def with_config(use_generated, aliases_on, tag):
        src.config.USE_GENERATED_VOCABULARY = use_generated
        src.config.SKILLS = src.config.load_skills()
        if not aliases_on:
            saved = src.config.GENERATED_ALIASES_PATH
            src.config.GENERATED_ALIASES_PATH = Path("/nonexistent")
        src.config.SKILL_ALIASES = src.config.load_aliases()
        if not aliases_on:
            src.config.GENERATED_ALIASES_PATH = saved
        import src.utils
        importlib.reload(src.utils)
        import src.Verdict, src.Matcher, eval.holdout
        importlib.reload(src.Verdict); importlib.reload(src.Matcher)
        importlib.reload(eval.holdout)
        globals()["predict"] = eval.holdout.predict
        globals()["choose_threshold"] = eval.holdout.choose_threshold
        _report(tag, pairs, labels, scores)

    with_config(False, True, "hand-written vocabulary (shipped)")
    with_config(True, True, "+ generated vocabulary + generated aliases")
    with_config(True, False, "+ generated vocabulary, aliases OFF")
    with_config(False, True, "back to shipped (sanity check)")
    print(f"\n  {BASELINE}")
    print("  Turning the aliases off changes nothing: the vocabulary was the cause.")


# ── 6 ────────────────────────────────────────────────────────────────────────

def attempt_6_concept_split():
    """Let only named technologies drive a literal veto.

    Reimplemented here rather than in src/, because it lost and a matcher
    carrying a branch nobody should enable is worse than an experiment living
    in the evaluation folder.
    """
    print("\n6. Only named technologies may veto; concepts defer\n")
    import re

    import src.Matcher as M
    from src.utils import aliases_normalised, find_skills, normalise
    from src import Verdict

    _CONCEPTS = {"frontend", "front end", "backend", "back end", "devops",
                 "fullstack", "full stack", "microservices", "serverless"}

    def specific(term):
        term = normalise(term).strip()
        if term in _CONCEPTS:
            return False
        if any(c.isdigit() or c in ".+#_" for c in term):
            return True
        try:
            from nltk.corpus import wordnet
            return any(not wordnet.synsets(tok)
                       for tok in re.split(r"[\s/-]+", term) if tok)
        except Exception:
            return False

    original = M.hard_gate

    def patched(requirement, resume):
        status, reason, present = original(requirement, resume)
        if status == Verdict.UNSUPPORTED and not present:
            named = find_skills(requirement)
            if named and not any(specific(s) or aliases_normalised().get(s)
                                 for s in named):
                return None, "", []       # defer to similarity
        return status, reason, present

    pairs, labels = _dev()
    scores = scorers.score_bi_encoder(pairs, Embedder())
    _report("baseline (gate vetoes on any absent term)", pairs, labels, scores)

    M.hard_gate = patched
    import eval.holdout as H
    H.hard_gate = patched
    try:
        _report("concept split (concepts defer to similarity)", pairs, labels, scores)
    finally:
        M.hard_gate = original
        H.hard_gate = original
    print("\n  Gate accuracy recovers and F1 still falls: the accuracy was bought")
    print("  by handing decisions to a path thirty points worse.")


ATTEMPTS = {
    1: ("fuse BM25 with the bi-encoder", attempt_1_fusion),
    2: ("a much larger encoder", attempt_2_bigger_encoder),
    3: ("ESCO as an external vocabulary", attempt_3_esco),
    4: ("external vocabulary, and the alias ablation", attempt_4_5_vocabulary),
    6: ("only named technologies may veto", attempt_6_concept_split),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, choices=sorted(ATTEMPTS))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, (name, _) in sorted(ATTEMPTS.items()):
            print(f"  {n}. {name}")
        return

    print("=" * 78)
    print("  Changes that were tried and rejected. Development set only.")
    print("  " + BASELINE)
    print("=" * 78)

    for n, (_, fn) in sorted(ATTEMPTS.items()):
        if args.only and n != args.only:
            continue
        try:
            fn()
        except Exception as e:
            print(f"\n  attempt {n} failed to run: {type(e).__name__}: {str(e)[:90]}")

    print("\n" + "=" * 78)
    print("  None of these shipped. See docs/EVALUATION.md, 'What did not work'.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
