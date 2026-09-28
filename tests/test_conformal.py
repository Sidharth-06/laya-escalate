"""Tests for the conformal layer.

The important test here is ``test_risk_bound_holds_empirically``: it checks that
the conformal threshold actually delivers the coverage-free guarantee it claims,
by running many independent calibration/test draws and averaging the realised
risk. If that test passes, the artifact's headline claim is mechanically sound.
"""

from __future__ import annotations

import numpy as np
import pytest

from laya_escalate.conformal import (
    aurc,
    conformal_threshold,
    coverage_at_risk,
    risk_coverage_curve,
)


def _draw(rng: np.random.Generator, n: int) -> tuple:
    """Exchangeable (score, wrong) pairs with score informative about wrong."""
    scores = rng.random(n)
    p_wrong = 0.45 * (1.0 - scores) ** 2
    wrong = rng.random(n) < p_wrong
    return scores, wrong


def test_risk_bound_holds_empirically():
    """E[wrong-and-acted rate] must be <= alpha, averaged over many trials."""
    alpha = 0.10
    n_calib = 200
    trials = 3000
    rng = np.random.default_rng(1234)

    losses = []
    for _ in range(trials):
        cal_scores, cal_wrong = _draw(rng, n_calib)
        res = conformal_threshold(cal_scores, cal_wrong, alpha=alpha)

        test_scores, test_wrong = _draw(rng, 1)
        acted = test_scores >= res.threshold
        losses.append(float(acted[0] and test_wrong[0]))

    realised = float(np.mean(losses))
    assert realised <= alpha * 1.05, (
        f"conformal bound violated: realised risk {realised:.4f} > alpha {alpha}"
    )
    # Sanity: the bound should be non-trivial, not achieved by escalating all.
    assert realised > 0.0


def test_threshold_is_smallest_satisfying_value():
    scores = np.array([0.1, 0.4, 0.5, 0.9])
    wrong = np.array([True, True, False, False])
    res = conformal_threshold(scores, wrong, alpha=0.5)
    # Ascending scan must pick the first feasible threshold.
    for lam in np.unique(scores):
        if lam < res.threshold:
            acted = scores >= lam
            assert (acted & wrong).sum() / scores.size + 1 / (scores.size + 1) > 0.5 + 1e-12


def test_vacuous_bound_escalates_everything():
    scores = np.array([0.2, 0.8, 0.9])
    wrong = np.array([False, False, False])
    res = conformal_threshold(scores, wrong, alpha=1e-6)
    assert res.vacuous
    assert res.coverage == 0.0


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        conformal_threshold([0.5, 0.6], [False, True], alpha=0.0)
    with pytest.raises(ValueError):
        conformal_threshold([0.5, 0.6], [False, True], alpha=1.0)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        conformal_threshold([0.5, 0.6], [False], alpha=0.5)


def test_aurc_orders_perfect_worse_than_random():
    n = 500
    rng = np.random.default_rng(7)
    correct = np.array([True] * 250 + [False] * 250)
    rng.shuffle(correct)

    perfect = np.where(correct, 1.0, 0.0)
    inverted = np.where(correct, 0.0, 1.0)
    random_scores = rng.random(n)

    assert aurc(perfect, correct) <= aurc(random_scores, correct) <= aurc(inverted, correct)

    # The minimum AURC for a ranker is not zero when the base error rate is
    # non-zero: once every correct decision has been admitted, further coverage
    # necessarily adds errors. Compute that floor analytically.
    n_correct = int(correct.sum())
    k = np.arange(1, n + 1)
    floor = float(np.mean(np.maximum(k - n_correct, 0) / k))
    assert aurc(perfect, correct) == pytest.approx(floor, rel=1e-6)
    assert floor > 0.0


def test_risk_coverage_curve_endpoints():
    scores = np.array([0.1, 0.5, 0.9, 0.95])
    correct = np.array([False, False, True, True])
    thresholds, coverage, selective = risk_coverage_curve(scores, correct)
    # Acting on everything must show up as full coverage somewhere.
    assert coverage.max() == pytest.approx(1.0)
    # The most confident single decision is correct, so risk there is zero.
    top = int(np.argmin(np.abs(coverage - 0.25)))
    assert selective[top] == pytest.approx(0.0)
    # Thresholds are ascending, which callers rely on to take the first
    # feasible one as the max-coverage choice.
    assert np.all(np.diff(thresholds) >= 0)


def test_coverage_at_risk_infeasible_returns_zero():
    scores = np.array([0.9, 0.9])
    correct = np.array([False, False])
    assert coverage_at_risk(scores, correct, target_risk=0.01) == 0.0


def test_coverage_at_risk_monotone_in_target():
    rng = np.random.default_rng(11)
    scores = rng.random(400)
    correct = rng.random(400) < (0.3 + 0.6 * scores)
    low = coverage_at_risk(scores, correct, 0.02)
    high = coverage_at_risk(scores, correct, 0.20)
    assert high >= low
