# eval/metrics.py
"""Metrics for a system that is allowed to say "I don't know".

A single precision/recall pair describes a classifier that answers everything.
This one does not: it withholds a verdict on personal qualities, and it reports
`unsupported` rather than guessing when the resume is silent. Scoring it as if
it always answered measures the wrong system.

The right frame is **selective prediction** — accuracy as a function of how
much you are willing to answer. The vocabulary here is standard:

  coverage        the fraction of decisions the system is willing to make
  risk            the error rate among the decisions it did make
  risk-coverage   risk plotted against coverage, from answering nothing to
                  answering everything
  AURC            area under that curve; lower is better. A system whose
                  confidence is informative puts its errors at the low-
                  confidence end, so risk stays flat as coverage grows and
                  then climbs. A system whose confidence means nothing traces
                  a flat line at its overall error rate.

The headline claim this enables is strictly stronger than a single F1: not
"precision 0.91", but "answering the 70% it is most sure of, it is wrong 4% of
the time". The second is harder to overfit and tells a user what to expect.

AURC is reported alongside a **random-confidence baseline**: the AURC the same
predictions would score if their confidences were shuffled. The gap between
them is the part attributable to the confidence signal rather than to the
underlying accuracy, and without it an AURC number is unreadable.

Abstain ECE asks a different question — not "are the confident answers right?"
but "when it says 80% sure, is it right 80% of the time?". A system can rank
its answers perfectly and still be systematically overconfident about all of
them, which matters here because the confidence is shown to a user.

Reference points for the vocabulary: Wen et al., "Know Your Limits: A Survey of
Abstention in Large Language Models" (TACL 2025), and the risk-coverage
framing standard in the selective-prediction literature since El-Yaniv &
Wiener (2010). Cited for the method, not as evidence about this system.
"""

import numpy as np


def risk_coverage_curve(confidences, correct):
    """Risk at every coverage level, answering most-confident-first.

    `confidences` is any score where higher means "more likely to be right";
    `correct` is 1/True where the decision was right. Returns a list of
    (coverage, risk) from answering one decision to answering all of them.

    **Ties are averaged, not ordered.** A naive sort leaves decisions that
    share a confidence in whatever order they arrived, so the curve — and the
    AURC computed from it — silently depends on the order of the input file.
    That is not a hypothetical: the literal gate hands many decisions the same
    confidence, and the first version of this function scored a completely
    uninformative confidence signal at AURC 0.04 purely because the correct
    rows happened to be listed first. The test that caught it is in
    tests/test_metrics.py.

    So within a block of equal confidences the errors are spread evenly, which
    is exactly the expected curve over random tie-breaking, computed
    deterministically instead of sampled.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(correct)
    if n == 0:
        return []

    order = np.argsort(-confidences, kind="stable")
    sorted_conf = confidences[order]
    errors = (~correct[order]).astype(float)

    # Replace each tied block with its mean error rate, so every step inside
    # the block contributes the block's average rather than its arrangement.
    start = 0
    for i in range(1, n + 1):
        if i == n or sorted_conf[i] != sorted_conf[start]:
            errors[start:i] = errors[start:i].mean()
            start = i

    cumulative_errors = np.cumsum(errors)
    answered = np.arange(1, n + 1)
    return list(zip(answered / n, cumulative_errors / answered))


def aurc(confidences, correct):
    """Area under the risk-coverage curve. Lower is better."""
    curve = risk_coverage_curve(confidences, correct)
    return float(np.mean([risk for _, risk in curve])) if curve else 0.0


def aurc_baseline(correct, trials=200, seed=0):
    """AURC the same predictions would get with meaningless confidences.

    This is the number an AURC must beat to show the confidence signal is
    doing any work. Averaged over shuffles because a single shuffle is noisy.
    """
    correct = np.asarray(correct, dtype=bool)
    if len(correct) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    return float(np.mean([
        aurc(rng.permutation(len(correct)), correct) for _ in range(trials)
    ]))


def risk_at_coverage(confidences, correct, target):
    """Error rate when answering only the `target` fraction it is surest of.

    Returns (actual_coverage, risk). The actual coverage can differ slightly
    from the target because coverage moves in steps of one decision.
    """
    curve = risk_coverage_curve(confidences, correct)
    if not curve:
        return 0.0, 0.0
    reachable = [(c, r) for c, r in curve if c <= target + 1e-9]
    return reachable[-1] if reachable else curve[0]


def coverage_at_risk(confidences, correct, ceiling):
    """How much it can answer while staying under `ceiling` error rate.

    The operating question for this tool, stated the way a user would ask it:
    "if I want to be wrong at most 5% of the time, how many of my requirements
    can you actually rule on?" Returns 0.0 if no coverage level qualifies.
    """
    curve = risk_coverage_curve(confidences, correct)
    ok = [c for c, r in curve if r <= ceiling + 1e-9]
    return max(ok) if ok else 0.0


def abstain_ece(confidences, correct, bins=10):
    """Expected calibration error of the confidence signal.

    Bins decisions by stated confidence and compares, within each bin, the
    average confidence against the fraction actually correct. Returns the
    size-weighted mean gap: 0.0 is perfect, and anything above ~0.1 means the
    number should not be shown to a user as a probability.

    Also returns the per-bin table, because a single ECE hides direction — a
    system can be overconfident at the top and underconfident at the bottom and
    still post a flattering average.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(correct) == 0:
        return 0.0, []

    edges = np.linspace(0.0, 1.0, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            in_bin |= confidences == 0.0
        count = int(in_bin.sum())
        if not count:
            continue
        mean_confidence = float(confidences[in_bin].mean())
        accuracy = float(correct[in_bin].mean())
        total += (count / len(correct)) * abs(accuracy - mean_confidence)
        table.append({
            "range": f"{lo:.1f}-{hi:.1f}", "n": count,
            "confidence": mean_confidence, "accuracy": accuracy,
            "gap": accuracy - mean_confidence,
        })
    return total, table


def format_curve(confidences, correct, points=(0.3, 0.5, 0.7, 0.9, 1.0)):
    """The risk-coverage curve as a few readable rows."""
    lines = []
    for target in points:
        cov, risk = risk_at_coverage(confidences, correct, target)
        lines.append(f"    answer {cov * 100:5.1f}%  ->  wrong {risk * 100:5.1f}% "
                     f"of the time  (accuracy {100 - risk * 100:.1f}%)")
    return lines
