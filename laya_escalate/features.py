"""Feature extraction for the escalation head.

A System One decision model (Laya, Jev, and the open reproductions) returns a
probability distribution over a *caller-supplied* set of options. The escalation
head needs to decide, per decision, whether that distribution can be trusted
enough to act on, or whether the decision should be handed to a slower/stronger
model.

The central empirical claim this module supports is that the model's own
top-line confidence is *not* a sufficient statistic for its correctness. In
particular Laya's calibration and accuracy degrade with the number of options
in a question, because every option shares a fixed ``head_max_len`` token
budget (the model card's own Banking77 result: 77 options get ~3-4 tokens each
and accuracy collapses to 0.425). That means the *shape of the label space*
carries information about correctness that confidence alone discards.

So the features here fall into three families:

1. Confidence shape      -- how peaked is the distribution, what is the margin.
2. Label-space geometry  -- how many options, how many tokens each one got.
3. Context pressure      -- how full was the encoder's context window.

Family 2 is the part that a plain confidence threshold cannot see, and it is
the reason a learned head beats confidence thresholding at matched coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

# Encoder context windows and option-header budgets, taken from the Laya model
# card. Used to derive the "tokens per option" signal, which is the observable
# proxy for the option-starvation failure mode.
DEFAULT_HEAD_MAX_LEN = {"english": 192, "multilingual": 256, "typed-decisions": 256}
DEFAULT_MAX_LEN = {"english": 512, "multilingual": 1024, "typed-decisions": 1024}

# Fixed cost of the question header, subtracted before dividing the remaining
# budget across options.
HEAD_OVERHEAD_TOKENS = 16

# Above this many options, the model card documents sharply degraded accuracy.
HIGH_CARDINALITY_THRESHOLD = 20

QUESTION_TYPES = ("choice", "score", "noul")

FEATURE_NAMES = (
    "p_top1",
    "p_top2",
    "margin",
    "logit_p_top1",
    "logit_margin",
    "entropy_norm",
    "log1p_n_options",
    "log1p_tokens_per_option",
    "context_pressure",
    "top3_mass",
    "concentration_l2",
    "effective_options",
    "p_top1_x_log_n_options",
    "is_high_cardinality",
    "ordinal_spread",
    "qtype_choice",
    "qtype_score",
    "qtype_noul",
    "act_probability",
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class DecisionRecord:
    """One decision made by a System One model about one typed question.

    Attributes
    ----------
    probs:
        Full probability distribution over the options of this question. Must be
        non-negative and sum to ~1.
    qtype:
        One of ``choice``, ``score``, ``noul``.
    correct:
        Ground-truth correctness of the model's answer (argmax == label). This is
        only known at fit/evaluation time; at serving time the head predicts it.
    n_options:
        Cardinality of the label space.
    head_max_len / max_len:
        The budget and context window configured for the request. Defaults track
        the Laya card; pass explicitly when you tune them.
    state_tokens:
        Number of tokens the state (document/transcript) occupied, if known.
        Used only to derive ``context_pressure``; may be ``None``.
    metadata:
        Free-form passthrough, ignored by the featurizer.
    """

    probs: Sequence[float]
    qtype: str = "choice"
    correct: bool = False
    n_options: Optional[int] = None
    head_max_len: Optional[int] = None
    max_len: Optional[int] = None
    state_tokens: Optional[int] = None
    act_probability: Optional[float] = None
    """Output of the decision model's own act/escalate head, when it exposes one.

    Laya returns ``answers[qid]["action"]["act_probability"]``. This is a
    *competing* escalation signal that ships with the model, so it is used two
    ways: as a feature the head may draw on, and as a baseline the head must
    beat to justify its existence. ``None`` means the model does not expose one.
    """
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.probs = np.asarray(self.probs, dtype=np.float64)
        if self.probs.ndim != 1 or self.probs.size < 2:
            raise ValueError("probs must be a 1-D vector with at least 2 entries")
        if self.qtype not in QUESTION_TYPES:
            raise ValueError(f"qtype must be one of {QUESTION_TYPES}, got {self.qtype!r}")
        if self.n_options is None:
            self.n_options = int(self.probs.size)
        if self.n_options != self.probs.size:
            raise ValueError(
                f"n_options ({self.n_options}) does not match probs length ({self.probs.size})"
            )
        if not np.all(np.isfinite(self.probs)):
            raise ValueError("probs contains non-finite values")
        if np.any(self.probs < 0):
            raise ValueError("probs contains negative values")
        total = self.probs.sum()
        if total <= 0:
            raise ValueError("probs sums to zero")
        self.probs = self.probs / total

    @property
    def p_top1(self) -> float:
        return float(np.max(self.probs))

    @property
    def argmax(self) -> int:
        return int(np.argmax(self.probs))


@dataclass
class FeatureContext:
    """Global request configuration needed to derive label-space features."""

    checkpoint: str = "english"
    head_max_len: Optional[int] = None
    max_len: Optional[int] = None

    def resolved(self) -> "FeatureContext":
        checkpoint = self.checkpoint if self.checkpoint in DEFAULT_HEAD_MAX_LEN else "english"
        return FeatureContext(
            checkpoint=checkpoint,
            head_max_len=self.head_max_len
            if self.head_max_len is not None
            else DEFAULT_HEAD_MAX_LEN[checkpoint],
            max_len=self.max_len if self.max_len is not None else DEFAULT_MAX_LEN[checkpoint],
        )


def tokens_per_option(n_options: int, head_max_len: int) -> float:
    """Approximate token budget left for each option label.

    Mirrors the failure mode described in the Laya model card: options share a
    fixed header budget, so cardinality and per-option budget trade off directly.
    """
    n_options = max(int(n_options), 1)
    remaining = max(head_max_len - HEAD_OVERHEAD_TOKENS, 0)
    return remaining / n_options


def _logit(p: float) -> float:
    """Log-odds of a probability, clipped to keep the feature finite.

    Calibration models are conventionally fitted in log-odds space (Platt
    scaling, temperature scaling) because the relationship between a model's
    stated confidence and its actual correctness is close to linear there. The
    same reasoning applies here: exposing ``logit(p_top1)`` lets the head learn
    the confidence-to-correctness mapping with far less capacity and far less
    data than it would need from the raw probability, leaving its capacity for
    the label-space signal that confidence cannot express.
    """
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def _normalized_entropy(probs: np.ndarray) -> float:
    n = probs.size
    if n <= 1:
        return 0.0
    safe = np.clip(probs, 1e-12, None)
    h = float(-np.sum(safe * np.log(safe)))
    return h / float(np.log(n))


def featurize(record: DecisionRecord, context: Optional[FeatureContext] = None) -> np.ndarray:
    """Map a decision record to a fixed-length feature vector.

    Returns an array of shape ``(N_FEATURES,)`` whose order matches
    :data:`FEATURE_NAMES`. All values are finite; degenerate distributions
    (one option at probability 1.0) are handled without producing NaNs.
    """
    ctx = (context or FeatureContext()).resolved()
    probs = record.probs
    n = probs.size

    head_max_len = record.head_max_len or ctx.head_max_len
    max_len = record.max_len or ctx.max_len
    tpo = tokens_per_option(n, head_max_len)

    sorted_probs = np.sort(probs)[::-1]
    p_top1 = float(sorted_probs[0])
    p_top2 = float(sorted_probs[1]) if n >= 2 else 0.0

    top3 = float(np.sum(sorted_probs[:3]))
    l2 = float(np.sum(probs**2))
    effective_options = 1.0 / l2 if l2 > 0 else float(n)

    if record.state_tokens is not None and max_len:
        context_pressure = min(float(record.state_tokens) / float(max_len), 1.0)
    else:
        # Unknown state length: assume the state filled whatever the header did
        # not consume, which is the documented worst case for option budget.
        context_pressure = 1.0 if max_len else 0.0

    if record.qtype == "score":
        idx = np.arange(n, dtype=np.float64)
        mean = float(np.sum(idx * probs))
        var = float(np.sum(((idx - mean) ** 2) * probs))
        ordinal_spread = float(np.sqrt(var)) / max(n - 1, 1)
    else:
        ordinal_spread = 0.0

    log_n = float(np.log1p(n))

    features = np.array(
        [
            p_top1,
            p_top2,
            p_top1 - p_top2,
            np.clip(_logit(p_top1), -12.0, 12.0),
            np.clip(_logit(p_top1) - _logit(p_top2), -12.0, 12.0),
            _normalized_entropy(probs),
            log_n,
            float(np.log1p(max(tpo, 0.0))),
            context_pressure,
            top3,
            l2,
            effective_options,
            p_top1 * log_n,
            1.0 if n > HIGH_CARDINALITY_THRESHOLD else 0.0,
            ordinal_spread,
            1.0 if record.qtype == "choice" else 0.0,
            1.0 if record.qtype == "score" else 0.0,
            1.0 if record.qtype == "noul" else 0.0,
            float(record.act_probability) if record.act_probability is not None else 0.0,
        ],
        dtype=np.float32,
    )
    if features.shape != (N_FEATURES,):
        raise AssertionError("feature vector has unexpected shape")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def featurize_many(
    records: Sequence[DecisionRecord], context: Optional[FeatureContext] = None
) -> np.ndarray:
    """Featurize a batch of records into an ``(n_records, N_FEATURES)`` array."""
    if not records:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.stack([featurize(r, context) for r in records]).astype(np.float32)
