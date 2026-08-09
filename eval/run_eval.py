"""Measure how well the matcher decides whether a resume bullet evidences a requirement.

Run:  python -m eval.run_eval            # compare every scorer
      python -m eval.run_eval --errors   # also print what each one got wrong

Why this exists: every threshold in this project was originally picked by
eyeballing a single resume against a single posting. That is not a defensible
way to choose 0.55. This script turns the question into a measurable one, so a
change can be shown to help rather than asserted to.

Read the caveat at the top of data/eval/pairs.jsonl before quoting any number:
the set is a curated collection of hard cases, so absolute scores are
pessimistic compared with real-world traffic. It is built for COMPARING
scorers, which is what the ranking below does.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.Embedder import Embedder                       # noqa: E402
from src.Matcher import hard_gate                       # noqa: E402

PAIRS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "pairs.jsonl"


def load_pairs():
    pairs = []
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_comment" in row:      # header note, not a data row
                continue
            pairs.append(row)
    return pairs


# ── Metrics ──────────────────────────────────────────────────────────────────

def prf(predictions, labels):
    """Precision, recall and F1 for the positive class."""
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def sweep(scores, labels, steps=81):
    """Best threshold by F1, and the precision/recall curve behind it."""
    lo, hi = float(min(scores)), float(max(scores))
    if hi <= lo:
        hi = lo + 1e-6
    curve = []
    for t in np.linspace(lo, hi, steps):
        m = prf([1 if s >= t else 0 for s in scores], labels)
        curve.append((float(t), m))
    best = max(curve, key=lambda item: item[1]["f1"])
    return best, curve


def precision_at(curve, floor):
    """Highest-recall operating point whose precision is at least `floor`.

    This tool's whole promise is not overstating what a candidate has, so the
    interesting question is not "best F1" but "how much recall can I buy while
    keeping precision high".
    """
    ok = [(t, m) for t, m in curve if m["precision"] >= floor]
    return max(ok, key=lambda item: item[1]["recall"]) if ok else None


# ── Scorers under test ───────────────────────────────────────────────────────

def score_bi_encoder(pairs, embedder):
    """Current production behaviour: cosine between two separate embeddings."""
    reqs = embedder.embed_many([p["requirement"] for p in pairs])
    bullets = embedder.embed_many([p["bullet"] for p in pairs])
    out = []
    for r, b in zip(reqs, bullets):
        rn, bn = np.linalg.norm(r), np.linalg.norm(b)
        out.append(float(np.dot(r, b) / (rn * bn)) if rn and bn else 0.0)
    return out


def score_cross_encoder(pairs, model_name):
    """Read requirement and bullet TOGETHER and score the pair.

    A bi-encoder never lets the two texts see each other, which is why it can
    rate "experience using PyTorch" highly against a line listing other ML
    frameworks. A cross-encoder attends across both.
    """
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    return [float(s) for s in model.predict([(p["requirement"], p["bullet"]) for p in pairs])]


def score_nli(pairs, model_name):
    """Score the pair as an entailment question rather than a relevance one.

    Error analysis on the ms-marco cross-encoder showed it failing almost every
    synonym case: Docker did not evidence "containerisation", Git did not
    evidence "version control", BERT did not evidence "transformer
    architectures". That is unsurprising in hindsight — ms-marco is trained on
    "is this passage relevant to this search query", and relevance is not the
    question being asked here.

    "Does this bullet mean the candidate has X" is textual entailment. So the
    bullet becomes the premise and the requirement becomes a hypothesis, and we
    score P(entailment).
    """
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    inputs = [(p["bullet"], f"The candidate has {p['requirement'].lower()}.") for p in pairs]
    logits = np.asarray(model.predict(inputs))

    # These models emit 3 classes; find which column is entailment.
    labels = getattr(getattr(model, "config", None), "id2label", None) or {}
    entail_idx = next((i for i, name in labels.items()
                       if str(name).lower().startswith("entail")), logits.shape[1] - 1)

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return [float(p) for p in probs[:, int(entail_idx)]]


def predict_gated(pairs, base_scores, threshold):
    """Cosine, but overridden by the literal check where one applies.

    This is what the app actually ships today.
    """
    preds, decided = [], []
    for pair, score in zip(pairs, base_scores):
        status, _, _ = hard_gate(pair["requirement"], pair["bullet"])
        if status is not None:
            preds.append(1 if status == "covered" else 0)
            decided.append("literal")
        else:
            preds.append(1 if score >= threshold else 0)
            decided.append("similarity")
    return preds, decided


# ── Reporting ────────────────────────────────────────────────────────────────

def row(name, m, extra=""):
    return (f"  {name:<34} P {m['precision']:.2f}  R {m['recall']:.2f}  "
            f"F1 {m['f1']:.2f}  acc {m['accuracy']:.2f}   "
            f"(fp {m['fp']}, fn {m['fn']}) {extra}")


def by_category(pairs, preds, labels):
    cats = {}
    for pair, p, l in zip(pairs, preds, labels):
        cats.setdefault(pair["category"], []).append((p, l))
    out = []
    for cat, rows in sorted(cats.items()):
        correct = sum(1 for p, l in rows if p == l)
        out.append(f"    {cat:<20} {correct}/{len(rows)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", action="store_true", help="print misclassified pairs")
    ap.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--nli-model", default="cross-encoder/nli-MiniLM2-L6-H768")
    args = ap.parse_args()

    pairs = load_pairs()
    labels = [p["label"] for p in pairs]
    print(f"\n{len(pairs)} labelled pairs  ({sum(labels)} positive, {len(labels)-sum(labels)} negative)\n")

    results = {}

    # 1. Bi-encoder alone
    bi = score_bi_encoder(pairs, Embedder())
    (bi_t, bi_m), bi_curve = sweep(bi, labels)
    results["bi-encoder (best F1)"] = (bi_m, [1 if s >= bi_t else 0 for s in bi], f"@ {bi_t:.2f}")

    # 2. Bi-encoder at whatever threshold is currently in config.yaml, so this
    #    row tracks the shipped app instead of drifting away from it.
    from src.config import MATCH_THRESHOLDS
    shipped_t = MATCH_THRESHOLDS["covered"]
    results[f"bi-encoder @ config {shipped_t}"] = (
        prf([1 if s >= shipped_t else 0 for s in bi], labels),
        [1 if s >= shipped_t else 0 for s in bi], "")

    # 3. What the app actually does: cosine + literal gate
    gated_preds, decided = predict_gated(pairs, bi, shipped_t)
    results["bi-encoder + literal gate"] = (prf(gated_preds, labels), gated_preds,
                                            f"({decided.count('literal')} decided literally)")

    # 4. Cross-encoder
    try:
        ce = score_cross_encoder(pairs, args.cross_model)
        (ce_t, ce_m), ce_curve = sweep(ce, labels)
        results["cross-encoder (best F1)"] = (ce_m, [1 if s >= ce_t else 0 for s in ce], f"@ {ce_t:.2f}")

        ce_gated, ce_decided = predict_gated(pairs, ce, ce_t)
        results["cross-encoder + literal gate"] = (prf(ce_gated, labels), ce_gated, "")
    except Exception as e:
        ce = ce_curve = None
        print(f"  (cross-encoder unavailable: {type(e).__name__}: {str(e)[:120]})\n")

    # 5. NLI entailment
    try:
        nli = score_nli(pairs, args.nli_model)
        (nli_t, nli_m), nli_curve = sweep(nli, labels)
        results["NLI entailment (best F1)"] = (nli_m, [1 if s >= nli_t else 0 for s in nli], f"@ {nli_t:.2f}")

        nli_gated, _ = predict_gated(pairs, nli, nli_t)
        results["NLI + literal gate"] = (prf(nli_gated, labels), nli_gated, "")
    except Exception as e:
        nli_curve = None
        print(f"  (NLI model unavailable: {type(e).__name__}: {str(e)[:120]})\n")

    print("Scorers, ranked by F1:\n")
    for name, (m, _, extra) in sorted(results.items(), key=lambda kv: -kv[1][0]["f1"]):
        print(row(name, m, extra))

    print("\nHigh-precision operating points (the mode this tool should run in):")
    for label, curve in (("bi-encoder", bi_curve), ("cross-encoder", ce_curve), ("NLI", nli_curve)):
        if curve is None:
            continue
        hit = precision_at(curve, 0.90)
        if hit:
            t, m = hit
            print(f"  {label:<16} P>=0.90 at threshold {t:.2f} -> recall {m['recall']:.2f}")
        else:
            print(f"  {label:<16} never reaches precision 0.90 at any threshold")

    # The operating point that should actually drive config.yaml. Best-F1 is the
    # wrong objective for this tool: a false positive tells someone they cover a
    # requirement they do not, which is the exact failure the project exists to
    # avoid. So maximise recall subject to precision staying high.
    print("\nShippable operating points (max recall with precision >= 0.90, gate applied):")
    gated_candidates = {"bi-encoder": bi}
    if ce is not None:
        gated_candidates["cross-encoder"] = ce
    if nli_curve is not None:
        gated_candidates["NLI"] = nli
    for name, scores in gated_candidates.items():
        lo, hi = float(min(scores)), float(max(scores))
        best = None
        for t in np.linspace(lo, hi, 121):
            preds, _ = predict_gated(pairs, scores, float(t))
            m = prf(preds, labels)
            if m["precision"] >= 0.90 and (best is None or m["recall"] > best[1]["recall"]):
                best = (float(t), m)
        if best:
            t, m = best
            print(f"  {name + ' + gate':<24} threshold {t:>7.2f}  ->  "
                  f"P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}  (fp {m['fp']})")
        else:
            print(f"  {name + ' + gate':<24} cannot reach precision 0.90")

    # Every number above picks its threshold on the same pairs it reports, which
    # flatters it. Cross-validation tunes on one part of the set and measures on
    # a part it has not seen, which is the number worth quoting.
    print("\n5-fold cross-validated (threshold tuned on train folds, measured on held-out):")
    for name, scores in gated_candidates.items():
        folds = [[] for _ in range(5)]
        for i in range(len(pairs)):
            folds[i % 5].append(i)

        agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for k in range(5):
            test_idx = set(folds[k])
            train_idx = [i for i in range(len(pairs)) if i not in test_idx]

            # choose the threshold using only the training folds
            train_pairs = [pairs[i] for i in train_idx]
            train_scores = [scores[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            lo, hi = float(min(scores)), float(max(scores))
            chosen, best_r = hi, -1.0
            for t in np.linspace(lo, hi, 121):
                preds, _ = predict_gated(train_pairs, train_scores, float(t))
                m = prf(preds, train_labels)
                if m["precision"] >= 0.90 and m["recall"] > best_r:
                    chosen, best_r = float(t), m["recall"]

            test_pairs = [pairs[i] for i in sorted(test_idx)]
            test_scores = [scores[i] for i in sorted(test_idx)]
            test_labels = [labels[i] for i in sorted(test_idx)]
            preds, _ = predict_gated(test_pairs, test_scores, chosen)
            m = prf(preds, test_labels)
            for key in agg:
                agg[key] += m[key]

        p = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 0.0
        r = agg["tp"] / (agg["tp"] + agg["fn"]) if agg["tp"] + agg["fn"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  {name + ' + gate':<24} P {p:.2f}  R {r:.2f}  F1 {f1:.2f}   "
              f"(fp {agg['fp']}, fn {agg['fn']})")

    best_name = max(results.items(), key=lambda kv: kv[1][0]["f1"])[0]
    print(f"\nPer-category accuracy for the winner ({best_name}):")
    for line in by_category(pairs, results[best_name][1], labels):
        print(line)

    if args.errors:
        print(f"\nMistakes made by {best_name}:")
        for pair, p in zip(pairs, results[best_name][1]):
            if p != pair["label"]:
                kind = "FALSE POSITIVE" if p == 1 else "FALSE NEGATIVE"
                print(f"\n  [{kind}] #{pair['id']} ({pair['category']})")
                print(f"    req    : {pair['requirement']}")
                print(f"    bullet : {pair['bullet'][:100]}")
                print(f"    why    : {pair['note']}")
    print()


if __name__ == "__main__":
    main()
