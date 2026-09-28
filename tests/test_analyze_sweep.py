"""Tests for the sweep analysis helpers.

These functions produce the numbers that go in the write-up, so a silent bug in
them is expensive: it would look exactly like a result. The one that matters most
is ``paired_trend_test`` -- it is what turns "the numbers look like they drift"
into a claim, and it has to keep the paired structure of a sweep (the same items
asked at every ``k``) rather than pretend the rows are independent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_sweep import (  # noqa: E402
    bucket_of,
    build_matrix,
    global_threshold_consequence,
    load_sweep,
    paired_bootstrap_ci,
    paired_trend_test,
    recalibration_adequacy,
    spearman,
    threshold_at_risk,
)

KS = [12, 20, 32, 64]


def test_bucket_of_matches_layas_temperature_buckets():
    assert bucket_of(2) == "choice:2"
    assert bucket_of(4) == "choice:3-5"
    assert bucket_of(10) == "choice:6-10"
    assert bucket_of(11) == "choice:11+"
    assert bucket_of(77) == "choice:11+"


def test_spearman_is_exactly_one_for_a_perfectly_monotone_pair():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, x[::-1]) == pytest.approx(-1.0)
    # Ties must not break self-correlation.
    tied = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    assert spearman(tied, tied) == pytest.approx(1.0)


def test_build_matrix_lays_out_items_by_k_and_skips_other_columns():
    obs_k = np.array([4, 4, 12, 12])
    items = np.array([0, 1, 0, 1])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    m = build_matrix(values, items, obs_k, [4, 12])
    np.testing.assert_array_equal(m, np.array([[1.0, 3.0], [2.0, 4.0]]))
    only_12 = build_matrix(values, items, obs_k, [12])
    np.testing.assert_array_equal(only_12, np.array([[3.0], [4.0]]))


def test_build_matrix_rejects_a_ragged_grid():
    obs_k = np.array([4, 12])
    items = np.array([0, 0])
    with pytest.raises(ValueError):
        build_matrix(np.array([1.0, 2.0]), items, obs_k, [4, 12, 64])


def test_paired_trend_test_detects_a_real_trend():
    rng = np.random.default_rng(0)
    n_items = 150
    matrix = np.stack(
        [rng.random(n_items) < (0.99 - 0.004 * (k - 12)) for k in KS], axis=1
    ).astype(np.float64)
    res = paired_trend_test(matrix, KS, n_perm=300, direction="decreasing")
    assert res["p_value"] < 0.05
    assert res["spearman"] < 0.0


def test_paired_trend_test_does_not_invent_a_trend():
    rng = np.random.default_rng(1)
    matrix = (rng.random((150, len(KS))) < 0.8).astype(np.float64)
    res = paired_trend_test(matrix, KS, n_perm=300, direction="decreasing")
    assert res["p_value"] > 0.05


def test_paired_trend_test_is_two_sided_when_asked():
    """A rising trend must be detectable without asserting the direction."""
    rng = np.random.default_rng(2)
    n_items = 150
    matrix = np.stack(
        [rng.random(n_items) < (0.5 + 0.006 * (k - 12)) for k in KS], axis=1
    ).astype(np.float64)
    up = paired_trend_test(matrix, KS, n_perm=300, direction="two-sided")
    down = paired_trend_test(matrix, KS, n_perm=300, direction="decreasing")
    assert up["p_value"] < 0.05
    assert down["p_value"] > 0.5, "a rising trend should look unsurprising to 'decreasing'"


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(3)
    matrix = rng.random((80, len(KS)))
    ci = paired_bootstrap_ci(matrix, n_boot=400, seed=0)
    for i in range(len(KS)):
        assert ci[i, 0] <= matrix[:, i].mean() <= ci[i, 1]


def test_recalibration_adequacy_null_flags_no_shift_when_one_map_is_right():
    """Same score->accuracy curve at every k: the slope spread is just noise.

    The spread statistic is only readable against its own null -- a per-k slope
    fitted on a few hundred rows is noisy. Here one map generates the data, so the
    observed spread must sit inside what the shared-map null produces.
    """
    rng = np.random.default_rng(4)
    n = 400
    ks = rng.choice(KS, size=n)
    score = rng.random(n)
    correct = rng.random(n) < score
    res = recalibration_adequacy(score, correct, ks, min_k=11, n_boot=60)
    null = res["slope_spread_null"]
    assert res["slope_spread"] <= null["slope_spread"]["null_hi"], (
        "noise-only spread flagged as a shift"
    )
    assert null["slope_spread"]["p_value"] > 0.05
    assert res["worst_abs_bias_shared"] <= null["worst_abs_bias_shared"]["null_hi"], (
        "noise-only bias flagged as a shift"
    )


def test_recalibration_adequacy_exposes_a_per_regime_shift():
    """Score distribution identical across k, but reliability is not.

    At k=64 correctness stops tracking the score at all and accuracy collapses;
    everywhere else the score is honest. One shared map then cannot serve the
    bucket, while per-k maps can -- which is the contrast the diagnostic exists to
    measure. The collapse is deliberately large: a shift of a few accuracy points
    is genuinely inside the null (see the test above), and the diagnostic is
    supposed to stay quiet for it.
    """
    rng = np.random.default_rng(5)
    n = 800
    ks = rng.choice(KS, size=n)
    score = rng.uniform(0.05, 0.95, size=n)
    z = np.log(score / (1 - score))
    p_correct = np.where(ks >= 64, 0.30, 1.0 / (1.0 + np.exp(-z)))
    correct = rng.random(n) < p_correct

    res = recalibration_adequacy(score, correct, ks, min_k=11, n_boot=60)
    null = res["slope_spread_null"]["worst_abs_bias_shared"]
    assert res["worst_abs_bias_shared"] > 0.10, "one map should not fit this"
    assert res["worst_abs_bias_per_k_map"] < 0.01, "per-k maps should"
    assert res["worst_abs_bias_shared"] > res["worst_abs_bias_per_k_map"]
    assert res["worst_abs_bias_shared"] > null["null_hi"], (
        "shift is real but sits inside the null band"
    )


def test_threshold_at_risk_returns_max_when_target_is_unreachable():
    scores = np.full(50, 1.0)
    correct = np.zeros(50, dtype=bool)
    lam = threshold_at_risk(scores, correct, 0.01)
    consequence = global_threshold_consequence(scores, correct, np.full(50, 12), 0.01)
    assert lam > 1.0
    assert consequence["pooled_coverage"] == 0.0
    assert consequence["per_k"][0]["n_acted"] == 0


def test_global_threshold_consequence_reports_every_k():
    rng = np.random.default_rng(6)
    n = 300
    ks = np.repeat(KS, n // len(KS))
    scores = rng.random(n)
    correct = rng.random(n) < scores
    res = global_threshold_consequence(scores, correct, ks, 0.05)
    assert [int(r["k"]) for r in res["per_k"]] == KS
    assert 0.0 <= res["pooled_coverage"] <= 1.0


def _write_sweep(path: Path, n_options: np.ndarray) -> None:
    n = n_options.size
    np.savez_compressed(
        path,
        x=np.zeros((n, 19), dtype=np.float32),
        correct=np.ones(n, dtype=bool),
        p_top1=np.full(n, 0.9),
        confidence=np.full(n, 0.8),
        act_probability=np.ones(n),
        n_options=n_options.astype(int),
        latency_ms=np.ones(n),
        probs=np.array([np.full(int(k), 1.0 / int(k)) for k in n_options], dtype=object),
        qtype=np.array(["choice"] * n),
    )


def test_load_sweep_recovers_item_grouping_from_a_k_block_layout(tmp_path):
    """Artifacts written before collect_sweep.py saved item_index still pair."""
    n_items, ks = 4, [4, 12, 64]
    path = tmp_path / "sweep.npz"
    _write_sweep(path, np.repeat(ks, n_items))
    data = load_sweep(path)
    assert data["grouping_recovered"] is True
    np.testing.assert_array_equal(data["item_index"], np.arange(n_items * len(ks)) % n_items)


def test_load_sweep_refuses_to_guess_when_blocks_are_ragged(tmp_path):
    path = tmp_path / "ragged.npz"
    _write_sweep(path, np.array([4, 4, 12, 12, 12]))
    with pytest.raises(ValueError):
        load_sweep(path)

