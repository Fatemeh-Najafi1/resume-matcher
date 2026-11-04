# eval/holdout.py
"""Score the matcher on data it was never tuned against, and say what it costs.

Run:
    python -m eval.holdout                  # the held-out English set
    python -m eval.holdout --set bilingual  # the Arabic/English set
    python -m eval.holdout --set dev        # the development set, for contrast
    python -m eval.holdout --all            # every set, one table each
    python -m eval.holdout --quick          # skip the two heavyweight models

Why this exists, and why it is a separate file from run_eval.py:

`run_eval.py` tunes. It sweeps thresholds, compares scorers, and prints what
the matcher got wrong so the next fix can be chosen. Every number it reports
has seen the data it is reported on, which is exactly what you want while
developing and exactly what you must not quote afterwards.

This file measures. The split is enforced by procedure rather than by code:

  * Thresholds and confidence calibration are fitted on the DEV set only
    (data/eval/pairs.jsonl), inside this script, on every run.
  * They are then applied unchanged to the held-out sets.
  * No setting in config.yaml, no alias, no vocabulary entry and no rule may
    be chosen by looking at a held-out number. If one ever is, the held-out
    set has become a dev set and has to be replaced.

That protocol is the whole point. The README used to carry an admission that
its headline numbers were contaminated — the alias map and the skill
vocabulary were both repaired after inspecting which pairs the matcher failed
on, which is textbook test-set leakage. The fix for leakage is not a
disclaimer, it is a second set that the repair never touched.

The leak was real and it was large. Dev and held-out numbers are printed side
by side so the size of the gap is visible rather than described.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics, scorers                                   # noqa: E402
from src.Embedder import Embedder                                   # noqa: E402
from src.Matcher import hard_gate                                   # noqa: E402
from src.utils import is_judgement_call                             # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "eval"

SETS = {
    "dev": (DATA / "pairs.jsonl", "development set (tuned against — contaminated)"),
    "holdout": (DATA / "holdout.jsonl", "held-out English set (never tuned against)"),
    "bilingual": (DATA / "pairs_bilingual.jsonl", "Arabic/English set (never tuned against)"),
}

CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PRECISION_FLOOR = 0.90


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_comment" in row:
                continue
            rows.append(row)
    return rows


# ── Prediction ───────────────────────────────────────────────────────────────

def predict(pairs, scores, threshold):
    """The shipped decision rule: literal gate first, similarity otherwise.

    Returns (predictions, gate_decided) so the caller can tell which half of
    the system produced each answer — they have very different error profiles
    and deserve separate confidence estimates.
    """
    preds, gated = [], []
    for pair, score in zip(pairs, scores):
        status, _, _ = hard_gate(pair["requirement"], pair["bullet"])
        if status is not None:
            preds.append(1 if status == "covered" else 0)
            gated.append(True)
        else:
            preds.append(1 if score >= threshold else 0)
            gated.append(False)
    return preds, gated


def prf(preds, labels):
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(labels) if labels else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def choose_threshold(pairs, scores, labels):
    """Highest-recall threshold that holds precision at the floor, on DEV only.

    Precision is the constraint rather than the objective because a false
    "covered" tells someone they meet a requirement they do not — the failure
    this project exists to prevent. Maximising F1 instead would trade that
    away for recall the user did not ask for.
    """
    lo, hi = float(min(scores)), float(max(scores))
    if hi <= lo:
        return hi
    best, best_recall = hi, -1.0
    for t in np.linspace(lo, hi, 161):
        preds, _ = predict(pairs, scores, float(t))
        m = prf(preds, labels)
        if m["precision"] >= PRECISION_FLOOR and m["recall"] > best_recall:
            best, best_recall = float(t), m["recall"]
    return best


# ── Confidence ───────────────────────────────────────────────────────────────

def fit_confidence(dev_scores, dev_preds, dev_gated, dev_labels, threshold):
    """Learn P(this decision is correct) from the DEV set.

    Two features, both cheap and both meaningful:

      * how far the raw score sits from the decision threshold — a pair
        scoring just over the line is a coin flip, one scoring far above it is
        not;
      * whether the literal gate decided it, since the gate is deterministic
        and its errors come from a different source (vocabulary gaps) than the
        model's.

    Platt scaling over those two. Fitted here and never refitted on held-out
    data, so the confidences the held-out run reports are genuine predictions
    rather than a description of the data they are measured on.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([
        np.abs(np.asarray(dev_scores, dtype=float) - threshold),
        np.asarray(dev_gated, dtype=float),
    ])
    y = np.asarray([int(p == l) for p, l in zip(dev_preds, dev_labels)])

    if len(set(y.tolist())) < 2:
        # Everything right or everything wrong on dev: no signal to fit, so
        # report a flat confidence rather than inventing one.
        return lambda s, g: np.full(len(s), float(y.mean()))

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    def confidence(raw_scores, gated):
        feats = np.column_stack([
            np.abs(np.asarray(raw_scores, dtype=float) - threshold),
            np.asarray(gated, dtype=float),
        ])
        return model.predict_proba(feats)[:, 1]

    return confidence


# ── Reporting ────────────────────────────────────────────────────────────────

def report_set(name, pairs, ladder, dev_pairs, dev_ladder, quick=False):
    path_label = SETS[name][1]
    labels = [p["label"] for p in pairs]
    dev_labels = [p["label"] for p in dev_pairs]

    print(f"\n{'=' * 78}")
    print(f"  {name.upper()}  —  {path_label}")
    print(f"  {len(pairs)} pairs ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
    print(f"{'=' * 78}\n")

    print("  Accuracy and cost, every rung measured on the same pairs")
    print(f"  {'scorer':<22} {'P':>5} {'R':>5} {'F1':>6}   {'thresh':>8}  "
          f"{'ms/pair':>8}  {'disk':>8}")
    print("  " + "-" * 72)

    rows = {}
    for scorer_name, (scores, seconds, disk) in ladder.items():
        if quick and scorer_name in ("ranking (cross-encoder)", "entailment (NLI)"):
            continue
        dev_scores = dev_ladder[scorer_name][0]
        threshold = choose_threshold(dev_pairs, dev_scores, dev_labels)
        preds, gated = predict(pairs, scores, threshold)
        m = prf(preds, labels)
        rows[scorer_name] = (m, preds, gated, scores, threshold)

        per_pair_ms = seconds / max(1, len(pairs)) * 1000
        disk_text = f"{disk / 1e6:.0f}MB" if disk else "—"
        print(f"  {scorer_name:<22} {m['precision']:>5.2f} {m['recall']:>5.2f} "
              f"{m['f1']:>6.2f}   {threshold:>8.2f}  {per_pair_ms:>8.1f}  {disk_text:>8}")

    # Selective prediction, for the rung the app actually ships.
    #
    # Which rung that is depends on the language. Matcher disables the
    # cross-encoder entirely when either side contains Arabic, because the
    # configured reranker is English-only and would overrule a multilingual
    # encoder that can actually read the text. Reporting the cross-encoder
    # here anyway would be measuring a path the app never takes — which this
    # function did until the Arabic routing landed, and the resulting "AURC
    # not better than chance" line was describing a scorer nobody runs.
    if "semantic (multilingual)" in rows:
        shipped = "semantic (multilingual)"
    elif "ranking (cross-encoder)" in rows:
        shipped = "ranking (cross-encoder)"
    else:
        shipped = "semantic search"
    if shipped not in rows:
        return rows

    m, preds, gated, scores, threshold = rows[shipped]
    dev_scores = dev_ladder[shipped][0]
    dev_preds, dev_gated = predict(dev_pairs, dev_scores, threshold)
    confidence_of = fit_confidence(dev_scores, dev_preds, dev_gated, dev_labels, threshold)

    confidences = confidence_of(scores, gated)
    correct = [p == l for p, l in zip(preds, labels)]

    print(f"\n  Selective prediction for the shipped scorer ({shipped})")
    print("  Confidence calibrated on the dev set, applied unchanged here.\n")
    for line in metrics.format_curve(confidences, correct):
        print(line)

    area = metrics.aurc(confidences, correct)
    chance = metrics.aurc_baseline(correct)
    ece, table = metrics.abstain_ece(confidences, correct)
    print(f"\n    AURC {area:.3f}  (shuffled-confidence baseline {chance:.3f}, "
          f"{'informative' if area < chance - 0.005 else 'NOT better than chance'})")
    print(f"    Abstain ECE {ece:.3f}  "
          f"({'well calibrated' if ece < 0.10 else 'poorly calibrated — do not show as a probability'})")

    for ceiling in (0.05, 0.10):
        cov = metrics.coverage_at_risk(confidences, correct, ceiling)
        print(f"    to stay under {ceiling * 100:.0f}% error it can answer "
              f"{cov * 100:.0f}% of pairs")

    # Where the errors actually are.
    print("\n  Per-category accuracy (shipped scorer)")
    cats = {}
    for pair, p, l in zip(pairs, preds, labels):
        cats.setdefault(pair.get("category", "?"), []).append(p == l)
    for cat, results in sorted(cats.items()):
        print(f"    {cat:<20} {sum(results)}/{len(results)}")

    if any("direction" in p for p in pairs):
        print("\n  Per-language-direction accuracy")
        dirs = {}
        for pair, p, l in zip(pairs, preds, labels):
            dirs.setdefault(pair.get("direction", "?"), []).append(p == l)
        for d, results in sorted(dirs.items()):
            print(f"    {d:<20} {sum(results)}/{len(results)}")

    answerable = [i for i, p in enumerate(pairs) if not is_judgement_call(p["requirement"])]
    if len(answerable) < len(pairs):
        sub = prf([preds[i] for i in answerable], [labels[i] for i in answerable])
        print(f"\n  Excluding {len(pairs) - len(answerable)} personal-quality pairs the app "
              f"declines to judge:")
        print(f"    P {sub['precision']:.2f}  R {sub['recall']:.2f}  F1 {sub['f1']:.2f}")

    return rows


def build_ladder(pairs, embedder, quick=False, multilingual=False):
    """Run every scorer over `pairs`, timing each."""
    ladder = {}

    s, t = scorers.timed(scorers.score_keyword_overlap, pairs)
    ladder["keyword overlap"] = (s, t, 0)

    s, t = scorers.timed(scorers.score_bm25, pairs)
    ladder["BM25"] = (s, t, 0)

    s, t = scorers.timed(scorers.score_skills_mapping, pairs)
    ladder["skills mapping"] = (s, t, 0)

    s, t = scorers.timed(scorers.score_bi_encoder, pairs, embedder)
    ladder["semantic search"] = (s, t, scorers.model_disk_bytes(embedder.model_name))

    if multilingual:
        multi = Embedder(model_name=MULTILINGUAL_MODEL)
        s, t = scorers.timed(scorers.score_bi_encoder, pairs, multi)
        ladder["semantic (multilingual)"] = (s, t, scorers.model_disk_bytes(MULTILINGUAL_MODEL))

    if not quick:
        try:
            s, t = scorers.timed(scorers.score_cross_encoder, pairs, CROSS_MODEL)
            ladder["ranking (cross-encoder)"] = (s, t, scorers.model_disk_bytes(CROSS_MODEL))
        except Exception as e:
            print(f"  (cross-encoder unavailable: {type(e).__name__})")
        try:
            s, t = scorers.timed(scorers.score_nli, pairs, NLI_MODEL)
            ladder["entailment (NLI)"] = (s, t, scorers.model_disk_bytes(NLI_MODEL))
        except Exception as e:
            print(f"  (NLI unavailable: {type(e).__name__})")

    return ladder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="holdout", choices=list(SETS))
    ap.add_argument("--all", action="store_true", help="run every set")
    ap.add_argument("--quick", action="store_true", help="skip the two heavyweight models")
    args = ap.parse_args()

    embedder = Embedder()
    dev_pairs = load(SETS["dev"][0])

    targets = list(SETS) if args.all else [args.which]

    print("\nBuilding the dev-set ladder (thresholds and calibration come from here only)...")
    dev_ladder = build_ladder(dev_pairs, embedder, quick=args.quick)

    for name in targets:
        pairs = load(SETS[name][0])
        bilingual = name == "bilingual"
        ladder = (dev_ladder if name == "dev"
                  else build_ladder(pairs, embedder, quick=args.quick,
                                    multilingual=bilingual))
        if bilingual:
            # The multilingual encoder needs a dev-set column too, or there is
            # no threshold to carry over to it.
            multi = Embedder(model_name=MULTILINGUAL_MODEL)
            s, t = scorers.timed(scorers.score_bi_encoder, dev_pairs, multi)
            dev_ladder["semantic (multilingual)"] = (
                s, t, scorers.model_disk_bytes(MULTILINGUAL_MODEL))
        report_set(name, pairs, ladder, dev_pairs, dev_ladder, quick=args.quick)

    print("\n" + "=" * 78)
    print("  Read the dev row as an upper bound and the held-out rows as the")
    print("  estimate. The gap between them is what tuning on a set buys you,")
    print("  and it is the number most projects in this space never publish.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
