"""Tests for the calibration metrics.

The per-regime bias statistic is the artifact's headline number, so it gets
tested against hand-computed values rather than just checked for finiteness.
"""

from __future__ import annotations

import numpy as np
import pytest

from laya_escalate.calibration import (
    BUCKET_LABELS,
    assign_buckets,
    brier,
    bucket_calibration,
    bucket_label,
    ece,
    fit_confidence_temperature,
    fit_regime_temperature,
    format_bucket_table,
    nll,
    reliability_curve,
    rescaled_confidence,
    rescaled_confidence_many,
)


def _regime_data(a_by_k, n_per_k, seed):
    """Scores whose reliability genuinely bends with the option count.

    ``p_correct = sigmoid(a_k * z)`` with a different ``a_k`` per regime. Raw
    sigmoid(z) is then miscalibrated by a known amount in each regime, which is
    the situation a finer temperature grid is supposed to repair.
    """
    rng = np.random.default_rng(seed)
    ks = np.repeat(list(a_by_k), n_per_k)
    a_k = np.array([a_by_k[int(k)] for k in ks])
    z = rng.normal(0.0, 1.0, size=ks.size)
    p = 1.0 / (1.0 + np.exp(-a_k * z))
    correct = rng.random(ks.size) < p
    return 1.0 / (1.0 + np.exp(-z)), correct, ks


def test_regime_temperature_recovers_differing_slopes():
    a_by_k = {4: 1.6, 12: 1.0, 64: 0.4}
    scores, correct, ks = _regime_data(a_by_k, 4000, seed=0)
    fit = fit_regime_temperature(scores, correct, ks, share_intercept=True)
    assert fit.slopes[4] > fit.slopes[12] > fit.slopes[64]
    assert fit.slopes[64] == pytest.approx(0.4, abs=0.12)
    assert fit.slope_spread > 2.0
    assert len(set(fit.intercepts.values())) == 1, "shared intercept must be one value"


def test_regime_temperature_improves_held_out_calibration():
    """The whole point of the baseline: it must actually repair the scores.

    The repair is *per-regime* and largely cancels in a pooled average -- the
    regimes are miscalibrated in different directions. Asserting on pooled ECE
    would understate a working fit, so this checks the worst regime and the
    likelihood, and separately checks that a regime which was already honest is
    left roughly alone rather than damaged.
    """
    a_by_k = {4: 1.6, 12: 1.0, 64: 0.4}  # k=12 is already calibrated
    train_scores, train_correct, train_ks = _regime_data(a_by_k, 4000, seed=1)
    test_scores, test_correct, test_ks = _regime_data(a_by_k, 2000, seed=2)

    fit = fit_regime_temperature(train_scores, train_correct, train_ks)
    fitted = fit.apply(test_scores, test_ks)

    worst_raw, worst_fit = 0.0, 0.0
    for k in a_by_k:
        m = test_ks == k
        raw = ece(test_scores[m], test_correct[m])
        after = ece(fitted[m], test_correct[m])
        worst_raw = max(worst_raw, raw)
        worst_fit = max(worst_fit, after)
    assert worst_fit < 0.5 * worst_raw
    assert nll(fitted, test_correct) < nll(test_scores, test_correct)

    m = test_ks == 12
    assert ece(fitted[m], test_correct[m]) == pytest.approx(
        ece(test_scores[m], test_correct[m]), abs=0.01
    ), "an already-calibrated regime should be left alone, not damaged"


def test_regime_temperature_unshared_intercept_is_per_regime():
    a_by_k = {4: 1.6, 12: 1.0, 64: 0.4}
    scores, correct, ks = _regime_data(a_by_k, 2000, seed=3)
    shared = fit_regime_temperature(scores, correct, ks, share_intercept=True)
    full = fit_regime_temperature(scores, correct, ks, share_intercept=False)
    assert len(set(shared.intercepts.values())) == 1
    assert len(set(full.intercepts.values())) == len(set(ks.tolist()))


def test_regime_temperature_rejects_unfitted_regime():
    scores, correct, ks = _regime_data({4: 1.0, 12: 1.0}, 500, seed=4)
    fit = fit_regime_temperature(scores, correct, ks)
    with pytest.raises(ValueError, match="no temperature fitted"):
        fit.apply(np.array([0.9, 0.8]), np.array([4, 64]))


def test_ece_zero_for_perfectly_calibrated_constant():
    scores = np.full(100, 0.7)
    correct = np.array([True] * 70 + [False] * 30)
    assert ece(scores, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_equals_bias_for_single_bin():
    scores = np.full(100, 0.9)
    correct = np.zeros(100, dtype=bool)
    # Entirely overconfident by 0.9, and all in one bin.
    assert ece(scores, correct) == pytest.approx(0.9)


def test_ece_ignores_empty_bins_and_stays_in_range():
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    correct = rng.random(500) < scores
    value = ece(scores, correct, n_bins=15)
    assert 0.0 <= value <= 1.0


def test_bucket_labels_cover_all_option_counts():
    assert bucket_label(2) == "<=4"
    assert bucket_label(4) == "<=4"
    assert bucket_label(5) == "5-10"
    assert bucket_label(77) == ">20"
    assert bucket_label(100000) == ">20"


def test_assign_buckets_matches_labels():
    labels = assign_buckets([2, 6, 15, 77])
    assert list(labels) == ["<=4", "5-10", "11-20", ">20"]


def test_bucket_calibration_detects_regime_bias():
    """A scorer that is overconfident only at high cardinality must show it."""
    n = 400
    n_options = np.array([4] * (n // 2) + [77] * (n // 2))
    scores = np.full(n, 0.9)
    correct = np.concatenate(
        [np.random.default_rng(1).random(n // 2) < 0.9,
         np.random.default_rng(2).random(n // 2) < 0.4]
    )
    rows = {r.bucket: r for r in bucket_calibration(scores, correct, n_options)}
    assert rows["<=4"].bias < 0.1
    assert rows[">20"].bias > 0.4
    # Same stated confidence, wildly different realised accuracy.
    assert rows["<=4"].accuracy > rows[">20"].accuracy


def test_small_buckets_are_dropped_not_reported():
    n_options = np.array([4] * 100 + [77] * 5)
    scores = np.full(105, 0.8)
    correct = np.ones(105, dtype=bool)
    rows = bucket_calibration(scores, correct, n_options, min_bucket_size=25)
    reported = {r.bucket for r in rows}
    assert "<=4" in reported
    assert ">20" not in reported


def test_bias_sign_is_overconfidence_positive():
    scores = np.full(50, 0.95)
    correct = np.array([True] * 10 + [False] * 40)
    rows = bucket_calibration(scores, correct, [4] * 50, min_bucket_size=10)
    assert rows[0].bias > 0


def test_reliability_curve_marks_empty_bins_nan():
    scores = np.full(20, 0.5)
    correct = np.ones(20, dtype=bool)
    predicted, observed, counts = reliability_curve(scores, correct, n_bins=10)
    assert np.isnan(predicted[0])
    assert counts[0] == 0
    assert not np.isnan(predicted[5])


def test_brier_and_nll_bounds():
    scores = np.array([0.0, 1.0])
    correct = np.array([False, True])
    assert brier(scores, correct) == pytest.approx(0.0)
    # Clipping keeps NLL finite even for a confident miss.
    assert np.isfinite(nll(np.array([0.0, 1.0]), np.array([True, False])))


def test_format_bucket_table_renders_both_scorers():
    n = 200
    n_options = [4] * 100 + [77] * 100
    scores_conf = np.full(n, 0.9)
    scores_head = np.where(np.array(n_options) > 20, 0.4, 0.9)
    correct = np.array([True] * 90 + [False] * 10 + [True] * 40 + [False] * 60)

    conf_rows = bucket_calibration(scores_conf, correct, n_options, min_bucket_size=25)
    head_rows = bucket_calibration(scores_head, correct, n_options, min_bucket_size=25)
    text = format_bucket_table(conf_rows, head_rows)
    assert "ECE conf" in text
    assert "ECE head" in text
    assert "bias conf" in text
    assert BUCKET_LABELS[0] in text


# --------------------------------------------------------------------------
# The documented field: `confidence` is normalised entropy, not the top
# probability, so it cannot be repaired by a top-1 correction. These tests pin
# the definition, the power-transform relationship, and the fitting contract.
# --------------------------------------------------------------------------


def test_confidence_is_not_the_top_probability():
    """The distinction that makes an audit of p_top1 an audit of the wrong thing."""
    p = np.array([0.9, 0.05, 0.05])
    conf = rescaled_confidence(p, temperature=1.0, applied=1.0)
    assert conf != pytest.approx(0.9)
    # 1 - H(p)/log(k), computed by hand for this distribution.
    expected = 1.0 - (-(p * np.log(p)).sum()) / np.log(3)
    assert conf == pytest.approx(expected)


def test_confidence_is_unchanged_when_temperature_matches_the_applied_one():
    p = np.array([0.7, 0.2, 0.1])
    identity = 1.0 - (-(p * np.log(p)).sum()) / np.log(3)
    assert rescaled_confidence(p, temperature=2.5, applied=2.5) == pytest.approx(identity)


def test_confidence_follows_the_same_power_transform_as_top1():
    """Rescaling a softmax is a power transform on what was published."""
    p = np.array([0.6, 0.3, 0.1])
    q = p ** 1.5
    q = q / q.sum()
    expected = 1.0 - (-(q * np.log(q)).sum()) / np.log(3)
    assert rescaled_confidence(p, temperature=1.0, applied=1.5) == pytest.approx(expected)


def test_higher_temperature_lowers_stated_confidence():
    p = np.array([0.9, 0.05, 0.05])
    vals = [rescaled_confidence(p, t, applied=1.0) for t in (1.0, 2.0, 4.0, 8.0)]
    assert vals == sorted(vals, reverse=True)
    assert vals[-1] < vals[0]


def test_confidence_at_the_degenerate_ends():
    assert rescaled_confidence(np.ones(5) / 5.0, 2.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert rescaled_confidence(np.array([1.0, 0.0, 0.0, 0.0]), 2.0, 1.0) == pytest.approx(1.0)
    # A single option has no entropy to normalise by.
    assert np.isnan(rescaled_confidence(np.array([1.0]), 1.0, 1.0))


def test_confidence_many_accepts_per_option_count_values():
    rows = [np.array([0.8, 0.2]), np.array([0.7, 0.2, 0.05, 0.05])]
    ks = np.array([2, 4])
    out = rescaled_confidence_many(
        rows, {2: 1.0, 4: 2.0}, ks, applied={2: 1.0, 4: 4.0}
    )
    assert out[0] == pytest.approx(rescaled_confidence(rows[0], 1.0, 1.0))
    assert out[1] == pytest.approx(rescaled_confidence(rows[1], 2.0, 4.0))


def _flat_field(p_row, n, accuracy, seed):
    """Deterministic distribution, independently drawn labels.

    Confidence is then a known constant while accuracy is whatever was asked for,
    so the required temperature can be worked out on paper.
    """
    p = np.tile(np.asarray(p_row, dtype=np.float64), (n, 1))
    correct = np.random.default_rng(seed).random(n) < accuracy
    return p, correct


def test_fit_confidence_temperature_flattens_an_overconfident_field():
    p, correct = _flat_field([0.8, 0.2], 4000, 0.10, seed=6)
    ks = np.full(p.shape[0], 2)
    stated = rescaled_confidence_many(list(p), 1.0, ks, 1.0)
    assert stated.mean() == pytest.approx(1.0 - 0.5004 / np.log(2), abs=2e-3)

    t = fit_confidence_temperature(list(p), correct, ks, applied=1.0)
    assert t > 1.0
    fixed = rescaled_confidence_many(list(p), t, ks, 1.0)
    assert fixed.mean() == pytest.approx(float(correct.mean()), abs=1e-3)


def test_fit_confidence_temperature_sharpens_an_underconfident_field():
    """The other direction: a sign error here would pass an overconficence-only test."""
    p, correct = _flat_field([0.8, 0.2], 4000, 0.50, seed=7)
    ks = np.full(p.shape[0], 2)
    t = fit_confidence_temperature(list(p), correct, ks, applied=1.0)
    assert t < 1.0
    fixed = rescaled_confidence_many(list(p), t, ks, 1.0)
    assert fixed.mean() == pytest.approx(float(correct.mean()), abs=1e-3)


def test_fit_confidence_temperature_respects_min_options():
    """Fitting inside choice:11+ must ignore rows from smaller buckets."""
    rows = [np.array([0.8, 0.2])] * 2000
    ks = np.array([4] * 1000 + [20] * 1000)
    correct = np.zeros(2000, dtype=bool)
    correct[:1000] = True  # k=4 perfect, k=20 never right
    confined = fit_confidence_temperature(
        rows, correct, ks, applied=1.0, min_options=11
    )
    everything = fit_confidence_temperature(rows, correct, ks, applied=1.0)
    assert confined != pytest.approx(everything)
