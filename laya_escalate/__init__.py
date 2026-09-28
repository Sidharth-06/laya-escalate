"""laya-escalate: calibration auditing for System One decision models.

A System One decision model (Laya, Jev, or an open reproduction) answers typed
questions about a state in a single forward pass. It is fast and cheap, and it is
sometimes wrong. This package measures *where* its published confidence is wrong,
and supplies the candidate repairs -- a rescaled temperature
(:func:`fit_option_temperatures`, :func:`fit_shared_temperature`,
:func:`fit_regime_temperature`, :func:`fit_confidence_temperature`) and a small
learned head -- so the cheapest sufficient one can be identified.

On Laya's ``choice:11+`` bucket the answer was neither a finer grid nor a learned
head. The bucket's one shipped temperature is out of the range Laya validates and
gets clamped; the documented ``confidence`` overstates accuracy by +0.009 at 12
options and +0.156 at 64. One correctly fitted temperature (1.59) removes most of
that ramp, and a seven-float affine recalibration of the published top-1
probability -- one logistic slope per option count on ``logit(p_top1)`` plus one
shared offset -- beats both the per-option-count temperature grid and the
5,543-parameter head on the same rows. See the README for the numbers, and
:mod:`laya_escalate.calibration` for the diagnostics that establish them.

Quickstart::

    from laya_escalate import (
        DecisionRecord, FeatureContext, featurize_many,
        train_head, split_dataset, fit_conformal_policy, evaluate, format_report,
    )
"""

from .calibration import (
    BUCKET_LABELS,
    BucketCalibration,
    OptionTemperatures,
    RegimeTemperature,
    assign_buckets,
    brier,
    bucket_calibration,
    ece,
    fit_confidence_temperature,
    fit_option_temperatures,
    fit_regime_temperature,
    fit_shared_temperature,
    format_bucket_table,
    nll,
    reliability_curve,
    rescaled_confidence,
    rescaled_confidence_many,
    summarize,
    top1_from_probs,
    top1_many,
)
from .conformal import (
    ConformalResult,
    aurc,
    conformal_threshold,
    coverage_at_risk,
    risk_coverage_curve,
)
from .features import (
    DEFAULT_HEAD_MAX_LEN,
    DEFAULT_MAX_LEN,
    FEATURE_NAMES,
    N_FEATURES,
    DecisionRecord,
    FeatureContext,
    featurize,
    featurize_many,
    tokens_per_option,
)
from .frontier import (
    EvaluationReport,
    FrontierRow,
    Policy,
    ScorerComparison,
    compare_scorers,
    compare_to_confidence,
    evaluate,
    fit_conformal_policy,
    fit_target_risk_policy,
    format_report,
    format_scorer_comparison,
)
from .head import EscalationBundle, EscalationHead, Normalizer
from .train import Splits, TrainConfig, TrainResult, split_dataset, train_head

__version__ = "0.1.0"

__all__ = [
    "BUCKET_LABELS",
    "BucketCalibration",
    "ConformalResult",
    "DecisionRecord",
    "EscalationBundle",
    "EscalationHead",
    "EvaluationReport",
    "FEATURE_NAMES",
    "FeatureContext",
    "FrontierRow",
    "N_FEATURES",
    "Normalizer",
    "OptionTemperatures",
    "Policy",
    "RegimeTemperature",
    "ScorerComparison",
    "Splits",
    "TrainConfig",
    "TrainResult",
    "assign_buckets",
    "aurc",
    "brier",
    "bucket_calibration",
    "compare_scorers",
    "compare_to_confidence",
    "conformal_threshold",
    "coverage_at_risk",
    "ece",
    "evaluate",
    "featurize",
    "featurize_many",
    "fit_conformal_policy",
    "fit_target_risk_policy",
    "fit_confidence_temperature",
    "fit_option_temperatures",
    "fit_regime_temperature",
    "fit_shared_temperature",
    "format_bucket_table",
    "format_report",
    "format_scorer_comparison",
    "nll",
    "reliability_curve",
    "rescaled_confidence",
    "rescaled_confidence_many",
    "risk_coverage_curve",
    "split_dataset",
    "summarize",
    "tokens_per_option",
    "top1_from_probs",
    "top1_many",
    "train_head",
    "DEFAULT_HEAD_MAX_LEN",
    "DEFAULT_MAX_LEN",
]
