"""Tests for the frontier module.

The key test is ``test_head_beats_confidence_when_it_has_extra_signal``: it
constructs a case where correctness depends on label-space geometry rather than
on confidence, and asserts the comparison harness detects the head's advantage.
That is the artifact's core claim reduced to a unit test.
"""

from __future__ import annotations

import numpy as np
import pytest

from laya_escalate.conformal import coverage_at_risk
from laya_escalate.frontier import (
    Policy,
    compare_to_confidence,
    evaluate,
    fit_conformal_policy,
    fit_target_risk_policy,
    format_report,
)


def test_conformal_policy_reports_guarantee():
    rng = np.random.default_rng(0)
    scores = rng.random(300)
    wrong = rng.random(300) < 0.2
    policy = fit_conformal_policy(scores, wrong, alpha=0.1)
    assert "conformal risk control" in policy.guarantee
    assert 0.0 <= policy.threshold <= 1.0


def test_target_risk_policy_meets_target_on_fitting_data():
    rng = np.random.default_rng(1)
    scores = rng.random(1000)
    correct = rng.random(1000) < (0.4 + 0.5 * scores)
    policy = fit_target_risk_policy(scores, correct, target_risk=0.05)
    acted = policy.apply(scores)
    if acted.any():
        assert (~correct[acted]).mean() <= 0.05 + 1e-9


def test_head_beats_confidence_when_it_has_extra_signal():
    """Construct correctness driven by cardinality, invisible to confidence."""
    rng = np.random.default_rng(2)
    n = 4000
    # Every decision reports the same confidence; only cardinality matters.
    confidence = np.full(n, 0.80)
    n_options = rng.choice([2, 60], size=n, p=[0.5, 0.5])
    p_correct = np.where(n_options == 2, 0.99, 0.55)
    correct = rng.random(n) < p_correct

    # The head can see cardinality, so give it a score that separates the cases.
    head_scores = np.where(n_options == 2, 0.9, 0.3) + rng.normal(0, 0.01, n)

    rows = compare_to_confidence(head_scores, confidence, correct, targets=[0.05])
    row = rows[0]
    assert row.head_coverage > row.confidence_coverage
    assert row.coverage_gain > 0.2
    # Confidence is constant, so it cannot select anything useful.
    assert coverage_at_risk(confidence, correct, 0.05) <= 1.0

    report = evaluate(head_scores, confidence, correct, targets=[0.05])
    assert report.aurc_improvement > 0.0


def test_confidence_alone_is_sufficient_when_it_is_sufficient():
    """If confidence fully determines correctness, the head cannot beat it."""
    rng = np.random.default_rng(3)
    n = 3000
    confidence = rng.random(n)
    correct = confidence > 0.5
    rows = compare_to_confidence(confidence, confidence, correct, targets=[0.02])
    assert rows[0].coverage_gain == pytest.approx(0.0, abs=1e-9)


def test_evaluate_includes_conformal_payload():
    rng = np.random.default_rng(4)
    n = 500
    scores = rng.random(n)
    correct = rng.random(n) < 0.7
    report = evaluate(
        scores, scores, correct, targets=[0.05], conformal_alpha=0.1, wrong=~correct
    )
    assert report.conformal is not None
    assert report.conformal["alpha"] == 0.1
    assert report.n == n


def test_evaluate_measures_the_supplied_policy_out_of_sample():
    """A policy fitted elsewhere must be *measured*, not refitted.

    Refitting on the evaluated rows makes the bound hold by construction, which
    is why the source is recorded: a reader has to be able to tell a real
    held-out number from a restatement of the estimator.
    """
    rng = np.random.default_rng(7)
    n = 600
    cal_scores = rng.random(n)
    cal_correct = rng.random(n) < cal_scores
    policy = fit_conformal_policy(cal_scores, ~cal_correct, alpha=0.1)

    scores = rng.random(n)
    correct = rng.random(n) < scores
    report = evaluate(
        scores, scores, correct, targets=[0.05],
        conformal_policy=policy, wrong=~correct,
    )
    assert report.conformal is not None
    assert report.conformal["threshold"] == pytest.approx(policy.threshold)
    assert report.conformal["source"] == (
        "calibration-split threshold, measured on held-out rows"
    )
    acted = scores >= policy.threshold
    assert report.conformal["coverage"] == pytest.approx(float(acted.mean()))
    assert report.conformal["empirical_unconditional_risk"] == pytest.approx(
        float((acted & ~correct).mean())
    )


def test_evaluate_flags_a_bound_that_did_not_hold():
    rng = np.random.default_rng(8)
    scores = rng.random(400)
    correct = rng.random(400) < 0.5
    policy = Policy(threshold=0.0, alpha=0.001)  # act on everything, cap at 0.1%
    report = evaluate(
        scores, scores, correct, targets=[0.05],
        conformal_policy=policy, wrong=~correct,
    )
    text = format_report(report)
    assert "EXCEEDS the bound" in text


def test_format_report_renders_expected_sections():
    rng = np.random.default_rng(5)
    scores = rng.random(200)
    correct = rng.random(200) < 0.6
    report = evaluate(scores, scores, correct, targets=[0.05, 0.10],
                      conformal_alpha=0.1, wrong=~correct)
    text = format_report(report)
    assert "held-out evaluation" in text
    assert "AURC" in text
    assert "conformal@alpha=0.100" in text
    assert "coverage at matched selective risk" in text


def test_report_serializes_to_json_safe_types():
    import json

    rng = np.random.default_rng(6)
    scores = rng.random(100)
    correct = rng.random(100) < 0.5
    report = evaluate(scores, scores, correct, targets=[0.05])
    json.dumps(report.to_dict())
