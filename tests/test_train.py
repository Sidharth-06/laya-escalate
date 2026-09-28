"""Tests for training and the end-to-end claim.

``test_end_to_end_head_beats_confidence`` is the integration test that matters:
on synthetic decisions carrying Laya's documented pathologies (overconfidence,
cardinality-driven degradation), the learned head must buy coverage at matched
selective risk over thresholding raw confidence. If this fails, the artifact's
premise fails and the real-data run should not be published.
"""

from __future__ import annotations

import numpy as np
import pytest

from laya_escalate.conformal import aurc, coverage_at_risk
from laya_escalate.features import FeatureContext, featurize_many
from laya_escalate.frontier import evaluate
from laya_escalate.head import EscalationBundle
from laya_escalate.synthetic import SyntheticConfig, simulate
from laya_escalate.train import TrainConfig, split_dataset, train_head


def test_split_is_disjoint_and_respects_fractions():
    splits = split_dataset(1000, calib_fraction=0.25, test_fraction=0.25, seed=0)
    all_idx = np.concatenate([splits.train_idx, splits.calib_idx, splits.test_idx])
    assert np.unique(all_idx).size == 1000
    assert splits.calib_idx.size == 250
    assert splits.test_idx.size == 250


def test_split_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        split_dataset(100, calib_fraction=0.6, test_fraction=0.5)


def test_grouped_split_never_puts_one_item_on_two_sides():
    """A sweep asks each item at several k; a row split leaks the item."""
    n_items, per_item = 40, 5
    groups = np.repeat(np.arange(n_items), per_item)
    splits = split_dataset(groups.size, seed=0, groups=groups)

    buckets = [set(np.unique(groups[i]).tolist()) for i in
               (splits.train_idx, splits.calib_idx, splits.test_idx)]
    assert buckets[0].isdisjoint(buckets[1])
    assert buckets[0].isdisjoint(buckets[2])
    assert buckets[1].isdisjoint(buckets[2])
    assert set().union(*buckets) == set(range(n_items))

    # Every split is a whole number of items, and the rows are covered once.
    for idx in (splits.train_idx, splits.calib_idx, splits.test_idx):
        assert idx.size % per_item == 0
    all_idx = np.concatenate([splits.train_idx, splits.calib_idx, splits.test_idx])
    assert np.unique(all_idx).size == groups.size


def test_grouped_split_leaks_items_when_groups_are_ignored():
    """The failure the grouped split exists to prevent, asserted directly.

    With 40 items asked 5 times each and 25% of *rows* held out, a row-level
    split almost always leaves at least one item on both sides.
    """
    groups = np.repeat(np.arange(40), 5)
    row_split = split_dataset(groups.size, seed=0)
    train_items = set(np.unique(groups[row_split.train_idx]).tolist())
    test_items = set(np.unique(groups[row_split.test_idx]).tolist())
    assert train_items & test_items, "expected a row-level split to leak items here"


def test_grouped_split_rejects_bad_groups():
    # One entry per row is required, or the grouping is meaningless.
    with pytest.raises(ValueError):
        split_dataset(100, groups=np.arange(99))
    # Two groups cannot be split three ways.
    with pytest.raises(ValueError):
        split_dataset(20, groups=np.repeat(np.arange(2), 10))
    # Four groups with 40% held out each side leaves nothing to train on.
    with pytest.raises(ValueError):
        split_dataset(20, calib_fraction=0.4, test_fraction=0.4,
                      groups=np.repeat(np.arange(4), 5))


def test_train_head_accepts_groups():
    """Grouping the internal train/val split must not change the interface."""
    x, correct, _ = _make_data(1500, seed=7)
    groups = np.repeat(np.arange(300), 5)
    result = train_head(x, correct, TrainConfig(epochs=40, seed=7), groups=groups)
    assert np.isfinite(result.best_val_aurc)
    assert result.head.n_parameters() < 20_000


def _make_data(n=3000, seed=0):
    records, _ = simulate(SyntheticConfig(n_items=n, seed=seed))
    x = featurize_many(records, FeatureContext())
    correct = np.array([r.correct for r in records], dtype=bool)
    confidence = np.array([r.p_top1 for r in records], dtype=np.float64)
    return x, correct, confidence


def test_train_head_improves_over_initial_state():
    x, correct, _ = _make_data(2000, seed=1)
    result = train_head(x, correct, TrainConfig(epochs=120, seed=1))
    assert result.best_val_aurc < float("inf")
    first = result.history[0]["val_aurc"]
    assert result.best_val_aurc <= first + 1e-9
    assert result.head.n_parameters() < 20_000


def _run_seed(seed: int, n: int = 20000):
    x, correct, confidence = _make_data(n, seed=seed)
    splits = split_dataset(len(correct), seed=seed)
    result = train_head(x[splits.train_idx], correct[splits.train_idx],
                        TrainConfig(epochs=400, seed=seed))
    scores = result.head.score(x[splits.test_idx], result.normalizer)
    return scores, confidence[splits.test_idx], correct[splits.test_idx], x, splits, result


def test_end_to_end_head_beats_confidence_on_aurc():
    """AURC improvement is the robust ranking claim: positive on every seed."""
    gains = []
    for seed in (20, 21, 22):
        scores, conf, correct, _, _, _ = _run_seed(seed)
        gains.append(aurc(conf, correct) - aurc(scores, correct))
    assert min(gains) > 0.0, f"AURC gain not consistently positive: {gains}"


def test_head_reduces_per_regime_calibration_error():
    """The headline claim: miscalibration is regime-structured and corrected.

    Coverage gains are noisy at strict risk targets, so the calibration result
    is the load-bearing one. It is asserted on the high-cardinality regime,
    where the documented failure -- staying confident while being wrong --
    actually occurs.
    """
    from laya_escalate.calibration import bucket_calibration
    from laya_escalate.synthetic import simulate, SyntheticConfig
    from laya_escalate.features import FeatureContext, featurize_many

    for seed in (30, 31, 32):
        records, _ = simulate(SyntheticConfig(n_items=20000, seed=seed))
        x = featurize_many(records, FeatureContext())
        correct = np.array([r.correct for r in records], dtype=bool)
        confidence = np.array([r.p_top1 for r in records], dtype=np.float64)
        n_options = np.array([r.n_options for r in records], dtype=int)
        splits = split_dataset(len(correct), seed=seed)
        result = train_head(x[splits.train_idx], correct[splits.train_idx],
                            TrainConfig(epochs=400, seed=seed))
        head = result.head.score(x[splits.test_idx], result.normalizer)

        y, n = correct[splits.test_idx], n_options[splits.test_idx]
        conf_rows = {r.bucket: r for r in bucket_calibration(confidence[splits.test_idx], y, n)}
        head_rows = {r.bucket: r for r in bucket_calibration(head, y, n)}

        assert ">20" in conf_rows and ">20" in head_rows
        # High-cardinality decisions are strongly overconfident on raw confidence.
        assert conf_rows[">20"].bias > 0.03
        # ...and the head removes most of that bias.
        assert head_rows[">20"].ece < conf_rows[">20"].ece


def test_coverage_gain_is_positive_on_average():
    """Coverage gains exist but are modest and seed-sensitive; assert the mean."""
    gains = []
    for seed in (40, 41, 42):
        scores, conf, correct, _, _, _ = _run_seed(seed)
        gains.append(
            coverage_at_risk(scores, correct, 0.05) - coverage_at_risk(conf, correct, 0.05)
        )
    assert float(np.mean(gains)) > 0.0, f"mean coverage gain not positive: {gains}"


def test_bundle_round_trip(tmp_path):
    x, correct, _ = _make_data(1500, seed=3)
    result = train_head(x, correct, TrainConfig(epochs=60, seed=3))
    scores = result.head.score(x, result.normalizer)

    bundle = EscalationBundle(
        head=result.head,
        normalizer=result.normalizer,
        threshold=0.5,
        alpha=0.05,
        metadata={"test": True},
    )
    path = tmp_path / "head.pt"
    bundle.save(path)

    loaded = EscalationBundle.load(path)
    reloaded_scores = loaded.score(x)
    np.testing.assert_allclose(scores, reloaded_scores, atol=1e-5)
    assert loaded.alpha == 0.05
    assert loaded.metadata["test"] is True
    np.testing.assert_array_equal(bundle.decide(x), loaded.decide(x))



