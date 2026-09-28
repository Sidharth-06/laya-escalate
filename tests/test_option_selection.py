"""Tests for distractor selection and the within-``k`` adequacy analysis.

Two things here carry the weight of a published claim:

``similarity_ranking`` / ``build_options``
    the ``similar`` condition. If the ranking silently stopped selecting
    confusable intents the experiment would report a null that was an artifact of
    the selector, not of the model.
``confusability_adequacy``
    the within-``k`` statistic. It takes *probabilities* and logits them itself;
    feeding it an already-logit'd array fails silently in the worst way, because
    the second logit saturates the values and every slope comes out absurd. That
    guard is tested explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_sweep import (  # noqa: E402
    confusability_adequacy,
    group_recalibration_adequacy,
    recalibration_adequacy,
)
from collect_sweep import (  # noqa: E402
    build_options,
    char_ngram_vectors,
    cosine_similarity,
    similarity_ranking,
)

LABELS = [
    "pto_request",
    "pto_request_status",
    "pto_used",
    "pto_balance",
    "international_fees",
    "international_visa",
    "weather",
    "tell_joke",
]


# --- name similarity -------------------------------------------------------


def test_char_ngram_vectors_are_built_over_padded_character_grams():
    vectors = char_ngram_vectors(LABELS)
    assert len(vectors) == len(LABELS)
    assert all(vectors), "every label must produce at least one n-gram"
    # A shared morpheme is a shared term, which is the whole point of using
    # character n-grams rather than whole snake_case tokens.
    assert "pto" in vectors[0]


def test_cosine_similarity_is_bounded_symmetric_and_self_identical():
    vectors = char_ngram_vectors(LABELS)
    assert cosine_similarity(vectors[0], vectors[0]) == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(vectors[0], vectors[6]) == pytest.approx(
        cosine_similarity(vectors[6], vectors[0]), abs=1e-9
    )
    # Argument order must not matter even though the implementation swaps to
    # iterate the smaller dict.
    assert -1e-9 <= cosine_similarity(vectors[6], vectors[0]) <= 1.0 + 1e-9


def test_similarity_ranking_puts_the_shared_morpheme_first_and_excludes_gold():
    vectors = char_ngram_vectors(LABELS)
    ranked = similarity_ranking(0, range(len(LABELS)), vectors)
    assert 0 not in ranked, "the gold label is not its own distractor"
    assert ranked[0] == 1, "pto_request_status shares the most with pto_request"
    scores = [cosine_similarity(vectors[0], vectors[i]) for i in ranked]
    assert scores == sorted(scores, reverse=True)


def test_similarity_ranking_ties_break_on_index_deterministically():
    vectors = char_ngram_vectors(LABELS)
    first = similarity_ranking(0, range(len(LABELS)), vectors)
    second = similarity_ranking(0, range(len(LABELS)), vectors)
    assert first == second


# --- option construction ---------------------------------------------------


def test_build_options_returns_the_gold_plus_k_minus_one_distractors():
    rng = np.random.default_rng(0)
    labels, indices = build_options(0, LABELS, list(range(len(LABELS))), 4, rng)
    assert len(labels) == 4
    assert len(set(labels)) == 4, "no option may repeat"
    assert LABELS[0] in labels
    assert sorted(indices) == sorted(LABELS.index(name) for name in labels)
    assert 0 in indices


def test_build_options_similar_mode_takes_the_top_ranked_candidates():
    vectors = char_ngram_vectors(LABELS)
    ranked = similarity_ranking(0, range(len(LABELS)), vectors)
    rng = np.random.default_rng(0)
    labels, indices = build_options(0, LABELS, ranked, 3, rng, mode="similar")
    assert set(indices) == {0, ranked[0], ranked[1]}
    assert set(labels) == {LABELS[i] for i in {0, ranked[0], ranked[1]}}


def test_build_options_random_mode_uses_the_whole_candidate_pool():
    """Over many draws a uniform pool should reach every candidate."""
    pool = list(range(len(LABELS)))
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        _, indices = build_options(0, LABELS, pool, 2, rng, mode="random")
        seen.update(indices)
    assert seen == set(pool)


def test_build_options_refuses_a_pool_smaller_than_k():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        build_options(0, LABELS, [0, 1], 4, rng)


# --- within-k adequacy -----------------------------------------------------


def _synthetic_group_data(n_per_group=400, seed=0, a_by_group=(1.0, 1.0, 1.0, 1.0)):
    """Scores drawn from one latent logit, with a per-group link the map must fit."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.4, size=n_per_group * len(a_by_group))
    group = np.repeat(np.arange(len(a_by_group)), n_per_group)
    a = np.asarray(a_by_group, dtype=np.float64)[group]
    p_true = 1.0 / (1.0 + np.exp(-(a * z)))
    correct = rng.random(z.size) < p_true
    scores = 1.0 / (1.0 + np.exp(-z))
    return scores, correct, group


def test_group_recalibration_adequacy_flags_a_planted_shift():
    scores, correct, group = _synthetic_group_data(a_by_group=(0.4, 0.7, 1.4, 2.4))
    out = group_recalibration_adequacy(
        np.log(scores / (1 - scores)), correct, group, [0.0, 1.0, 2.0, 3.0], n_boot=40
    )
    assert out["worst_abs_bias_shared"] > out["slope_spread_null"]["worst_abs_bias_shared"]["null_hi"]
    assert out["worst_abs_bias_shared"] > out["worst_abs_bias_group_map"]


def test_group_recalibration_adequacy_is_quiet_when_one_map_is_right():
    scores, correct, group = _synthetic_group_data(a_by_group=(1.0, 1.0, 1.0, 1.0))
    out = group_recalibration_adequacy(
        np.log(scores / (1 - scores)), correct, group, [0.0, 1.0, 2.0, 3.0], n_boot=40
    )
    assert out["worst_abs_bias_shared"] <= out["slope_spread_null"]["worst_abs_bias_shared"]["null_hi"]


def test_recalibration_adequacy_keeps_its_historical_k_facing_keys():
    """The wrapper must still expose the names the reported tables use."""
    rng = np.random.default_rng(0)
    ks = np.repeat([12, 20, 32, 64], 300)
    scores = np.clip(rng.beta(9, 1.4, size=ks.size), 1e-4, 1 - 1e-4)
    correct = rng.random(ks.size) < (scores - 0.02)
    out = recalibration_adequacy(scores, correct, ks, min_k=11, n_boot=30)
    for key in (
        "per_k_slopes",
        "per_k_shared_intercept",
        "per_k_slopes_independent",
        "worst_abs_bias_per_k_slope",
        "worst_abs_bias_per_k_map",
        "worst_abs_bias_shared",
        "slope_spread",
        "slope_spread_null",
    ):
        assert key in out, f"missing historical key {key}"
    assert [row["k"] for row in out["rows"]] == [12.0, 20.0, 32.0, 64.0]
    assert set(out["per_k_slopes"]) == {"12.0", "20.0", "32.0", "64.0"}


def test_confusability_adequacy_groups_on_similarity_not_score():
    """Two similarity halves with the SAME score distribution, different accuracy.

    This mirrors the real data: at a fixed ``k`` the published score barely moves
    with confusability while accuracy falls by several points. A single map has
    one slope and one intercept to serve both halves, so it must split the
    difference and leave a residual bias of opposite sign in each -- which is the
    quantity the analysis reports.
    """
    rng = np.random.default_rng(3)
    n = 2400
    z = rng.normal(0.0, 1.0, size=n)
    similarity = np.concatenate([np.full(n // 2, 0.01), np.full(n // 2, 0.30)])
    low = np.arange(0, n // 2)
    high = np.arange(n // 2, n)
    correct = np.zeros(n, dtype=bool)
    # Same score distribution in both halves; accuracy differs by the link.
    correct[low] = rng.random(low.size) < 1 / (1 + np.exp(-(1.6 * z[low] + 1.5)))
    correct[high] = rng.random(high.size) < 1 / (1 + np.exp(-(1.0 * z[high] - 0.5)))
    assert abs(correct[low].mean() - correct[high].mean()) > 0.2, "fixture must shift accuracy"
    scores = 1.0 / (1.0 + np.exp(-z))
    out = confusability_adequacy(scores, correct, similarity, n_groups=2, n_boot=40)
    assert out["worst_abs_bias_shared"] > 0.02
    assert out["worst_abs_bias_shared"] > out["slope_spread_null"]["worst_abs_bias_shared"]["null_hi"]
    # The residual must be the shared map's failure, not a bad per-group fit.
    assert out["worst_abs_bias_group_map"] < out["worst_abs_bias_shared"]


def test_confusability_adequacy_rejects_an_already_logit_ed_input():
    """Guards the exact bug that produced absurd slopes in the first run."""
    rng = np.random.default_rng(4)
    n = 800
    logits = rng.normal(4.0, 1.0, size=n)
    correct = rng.random(n) < 0.9
    similarity = rng.uniform(0.0, 0.1, size=n)
    with pytest.raises(ValueError, match="probabilities"):
        confusability_adequacy(logits, correct, similarity, n_groups=2, n_boot=10)
    # ...and the same data on the probability scale is accepted.
    out = confusability_adequacy(
        1.0 / (1.0 + np.exp(-logits)), correct, similarity, n_groups=2, n_boot=10
    )
    assert out["n"] == n


def test_confusability_adequacy_refuses_when_similarity_is_absent():
    rng = np.random.default_rng(5)
    n = 400
    with pytest.raises(ValueError, match="distractor similarity"):
        confusability_adequacy(
            rng.uniform(0.9, 1.0, size=n),
            rng.random(n) < 0.9,
            np.full(n, np.nan),
            n_groups=4,
            n_boot=10,
        )
