# tests/test_metrics.py
"""Selective-prediction metrics, checked against cases with known answers.

These matter more than they look. A risk-coverage curve that is subtly wrong
produces a plausible number nobody can sanity-check by eye, and the whole
argument of this project rests on those numbers being right.
"""

import numpy as np

from eval import metrics


class TestRiskCoverage:

    def test_perfect_confidence_puts_all_errors_last(self):
        """If confidence is perfectly informative, risk stays 0 then climbs."""
        correct = [True] * 8 + [False] * 2
        confidences = list(range(10, 0, -1))      # already in the right order
        curve = metrics.risk_coverage_curve(confidences, correct)

        coverage, risk = curve[7]                 # answering the 8 best
        assert coverage == 0.8
        assert risk == 0.0
        assert curve[-1][1] == 0.2                # all 10 answered -> 20% wrong

    def test_useless_confidence_traces_a_flat_line(self):
        correct = [True, False] * 10
        confidences = [1.0] * 20                  # no information at all
        curve = metrics.risk_coverage_curve(confidences, correct)
        assert abs(curve[-1][1] - 0.5) < 1e-9

    def test_aurc_beats_chance_only_when_confidence_informs(self):
        correct = [True] * 15 + [False] * 5
        informative = list(range(20, 0, -1))
        assert metrics.aurc(informative, correct) < metrics.aurc_baseline(correct)

        shuffled = [1.0] * 20
        assert metrics.aurc(shuffled, correct) >= metrics.aurc_baseline(correct) - 0.05

    def test_coverage_at_risk_is_monotone_in_the_ceiling(self):
        rng = np.random.default_rng(0)
        confidences = rng.random(100)
        correct = rng.random(100) < confidences   # better confidence, more often right
        loose = metrics.coverage_at_risk(confidences, correct, 0.3)
        tight = metrics.coverage_at_risk(confidences, correct, 0.1)
        assert loose >= tight

    def test_empty_input_does_not_explode(self):
        assert metrics.risk_coverage_curve([], []) == []
        assert metrics.aurc([], []) == 0.0


class TestAbstainECE:

    def test_perfect_calibration_scores_zero(self):
        """Stated 100% confidence on things that are always right."""
        confidences = [1.0] * 50
        correct = [True] * 50
        ece, _ = metrics.abstain_ece(confidences, correct)
        assert ece < 1e-9

    def test_overconfidence_is_caught(self):
        """Claims 90% sure, is right half the time."""
        confidences = [0.9] * 100
        correct = [True] * 50 + [False] * 50
        ece, table = metrics.abstain_ece(confidences, correct)
        assert abs(ece - 0.4) < 0.01
        assert table[0]["gap"] < 0            # negative gap == overconfident

    def test_bins_cover_every_point(self):
        rng = np.random.default_rng(1)
        confidences = rng.random(200)
        correct = rng.random(200) < 0.5
        _, table = metrics.abstain_ece(confidences, correct)
        assert sum(row["n"] for row in table) == 200
