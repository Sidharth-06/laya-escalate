"""Tests for the featurizer.

Features are the head's only input, so silent NaNs or a mis-scaled entropy term
would quietly destroy the artifact. These tests pin the invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from laya_escalate.features import (
    FEATURE_NAMES,
    N_FEATURES,
    DecisionRecord,
    FeatureContext,
    featurize,
    featurize_many,
    tokens_per_option,
)


def test_feature_vector_shape_and_finiteness():
    rec = DecisionRecord(probs=[0.7, 0.2, 0.1], qtype="choice", correct=True)
    vec = featurize(rec)
    assert vec.shape == (N_FEATURES,)
    assert np.all(np.isfinite(vec))
    assert len(FEATURE_NAMES) == N_FEATURES


def test_probs_are_normalized_on_construction():
    rec = DecisionRecord(probs=[7.0, 3.0], qtype="noul")
    assert rec.probs.sum() == pytest.approx(1.0)
    assert rec.n_options == 2


def test_degenerate_distribution_has_no_nans():
    rec = DecisionRecord(probs=[1.0, 0.0, 0.0, 0.0], qtype="choice")
    vec = featurize(rec)
    assert np.all(np.isfinite(vec))
    assert vec[FEATURE_NAMES.index("entropy_norm")] == pytest.approx(0.0, abs=1e-9)
    assert vec[FEATURE_NAMES.index("p_top2")] == pytest.approx(0.0)


def test_uniform_distribution_has_max_entropy():
    rec = DecisionRecord(probs=[0.25] * 4, qtype="choice")
    vec = featurize(rec)
    assert vec[FEATURE_NAMES.index("entropy_norm")] == pytest.approx(1.0, abs=1e-9)
    assert vec[FEATURE_NAMES.index("margin")] == pytest.approx(0.0, abs=1e-9)


def test_n_options_mismatch_rejected():
    with pytest.raises(ValueError):
        DecisionRecord(probs=[0.5, 0.5], qtype="choice", n_options=3)


def test_invalid_qtype_rejected():
    with pytest.raises(ValueError):
        DecisionRecord(probs=[0.5, 0.5], qtype="regression")


def test_negative_probs_rejected():
    with pytest.raises(ValueError):
        DecisionRecord(probs=[1.5, -0.5], qtype="choice")


def test_high_cardinality_flag_tracks_threshold():
    small = featurize(DecisionRecord(probs=[0.5] * 20, qtype="choice"))
    big = featurize(DecisionRecord(probs=[1 / 21] * 21, qtype="choice"))
    idx = FEATURE_NAMES.index("is_high_cardinality")
    assert small[idx] == 0.0
    assert big[idx] == 1.0


def test_tokens_per_option_decreases_with_cardinality():
    assert tokens_per_option(4, 192) > tokens_per_option(20, 192) > tokens_per_option(77, 192)
    # The documented Banking77 case: ~3-4 tokens per label.
    assert 2.0 <= tokens_per_option(77, 256) <= 5.0


def test_qtype_onehot_is_exclusive():
    idx = [FEATURE_NAMES.index("qtype_choice"), FEATURE_NAMES.index("qtype_score"),
           FEATURE_NAMES.index("qtype_noul")]
    for qtype in ("choice", "score", "noul"):
        vec = featurize(DecisionRecord(probs=[0.6, 0.4], qtype=qtype))
        assert vec[idx].sum() == pytest.approx(1.0)


def test_ordinal_spread_present_only_for_score():
    spread = featurize(DecisionRecord(probs=[0.1, 0.2, 0.4, 0.3], qtype="score"))
    flat = featurize(DecisionRecord(probs=[0.1, 0.2, 0.4, 0.3], qtype="choice"))
    idx = FEATURE_NAMES.index("ordinal_spread")
    assert spread[idx] > 0.0
    assert flat[idx] == 0.0


def test_context_pressure_clamped_to_one():
    rec = DecisionRecord(
        probs=[0.5, 0.5], qtype="noul", state_tokens=5000, max_len=512
    )
    vec = featurize(rec)
    assert vec[FEATURE_NAMES.index("context_pressure")] == pytest.approx(1.0)


def test_defaults_come_from_checkpoint_config():
    rec = DecisionRecord(probs=[0.5, 0.5], qtype="noul")
    en = featurize(rec, FeatureContext(checkpoint="english"))
    ml = featurize(rec, FeatureContext(checkpoint="multilingual"))
    idx = FEATURE_NAMES.index("log1p_tokens_per_option")
    assert ml[idx] > en[idx]


def test_featurize_many_stacks_and_handles_empty():
    records = [DecisionRecord(probs=[0.5, 0.5], qtype="noul") for _ in range(5)]
    assert featurize_many(records).shape == (5, N_FEATURES)
    assert featurize_many([]).shape == (0, N_FEATURES)


def test_confidence_interaction_feature_is_present():
    idx = FEATURE_NAMES.index("p_top1_x_log_n_options")
    low_card = featurize(DecisionRecord(probs=[0.9, 0.1], qtype="choice"))
    high_card = featurize(DecisionRecord(probs=[0.9] + [0.1 / 19] * 19, qtype="choice"))
    assert high_card[idx] > low_card[idx]
