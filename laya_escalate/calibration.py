"""Calibration metrics, broken down by label-space regime.

This module exists because the per-regime number is the artifact's real finding.

Laya's model card reports a single global ECE and fixes it with "one temperature
per (question type, option count)". That implies the miscalibration is
*regime-structured*: the model is not uniformly overconfident, it is
differentially overconfident depending on how many options it was asked about.
A global ECE number hides exactly that structure, and hiding it is dangerous,
because a downstream confidence threshold then ships an error rate that varies
wildly across label-space regimes while looking calibrated in aggregate.

So the metrics here are deliberately conditioned on the option-count bucket:
``bias`` (mean stated confidence minus observed accuracy, with sign) is the
statistic that makes the failure legible. A positive bias means overconfident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Option-count buckets chosen to line up with the corpora the Laya card reports
# on: 2 (noul), 4 (AG News), 6 (DAIR Emotion), ~60 (MASSIVE), 77 (Banking77).
DEFAULT_BUCKET_EDGES: Tuple[Tuple[int, int], ...] = ((2, 4), (5, 10), (11, 20), (21, 10**9))
BUCKET_LABELS: Tuple[str, ...] = ("<=4", "5-10", "11-20", ">20")


def bucket_label(n_options: int, edges: Sequence[Tuple[int, int]] = DEFAULT_BUCKET_EDGES) -> str:
    """Map an option count onto its regime bucket."""
    for (lo, hi), label in zip(edges, BUCKET_LABELS):
        if lo <= n_options <= hi:
            return label
    return BUCKET_LABELS[-1]


def assign_buckets(
    n_options: Sequence[int], edges: Sequence[Tuple[int, int]] = DEFAULT_BUCKET_EDGES
) -> np.ndarray:
    return np.array([bucket_label(int(n), edges) for n in n_options], dtype=object)


def ece(scores: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> float:
    """Expected calibration error over equal-width probability bins."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=bool).reshape(-1)
    if scores.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(scores[mask].mean() - correct[mask].mean())
    return float(total)


def brier(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Brier score: mean squared error of the probability estimate."""
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool).astype(np.float64)
    return float(np.mean((scores - correct) ** 2))


def nll(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Negative log-likelihood, clipped to avoid infinities."""
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-9, 1 - 1e-9)
    correct = np.asarray(correct, dtype=bool).astype(np.float64)
    return float(-np.mean(correct * np.log(scores) + (1 - correct) * np.log(1 - scores)))


def reliability_curve(
    scores: Sequence[float], correct: Sequence[bool], n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mean_predicted, observed_frequency, count)`` per bin.

    Empty bins are returned as NaN so downstream plotting can skip them rather
    than plotting a misleading zero.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=bool).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)
    predicted = np.full(n_bins, np.nan)
    observed = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if mask.any():
            predicted[b] = float(scores[mask].mean())
            observed[b] = float(correct[mask].mean())
    return predicted, observed, counts


@dataclass
class BucketCalibration:
    """Calibration of a scorer restricted to one label-space regime."""

    bucket: str
    n: int
    accuracy: float
    mean_score: float
    bias: float
    """Mean stated confidence minus observed accuracy. Positive = overconfident."""
    ece: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def bucket_calibration(
    scores: Sequence[float],
    correct: Sequence[bool],
    n_options: Sequence[int],
    edges: Sequence[Tuple[int, int]] = DEFAULT_BUCKET_EDGES,
    min_bucket_size: int = 25,
) -> List[BucketCalibration]:
    """Per-regime calibration table, skipping buckets that are too small to trust.

    ``min_bucket_size`` matters: a regime with a handful of decisions produces a
    bias estimate dominated by noise, and publishing it would be worse than
    omitting it. Buckets below the threshold are dropped rather than reported.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=bool).reshape(-1)
    labels = assign_buckets(n_options, edges)

    rows: List[BucketCalibration] = []
    for label in BUCKET_LABELS:
        mask = labels == label
        if mask.sum() < min_bucket_size:
            continue
        rows.append(
            BucketCalibration(
                bucket=label,
                n=int(mask.sum()),
                accuracy=float(correct[mask].mean()),
                mean_score=float(scores[mask].mean()),
                bias=float(scores[mask].mean() - correct[mask].mean()),
                ece=ece(scores[mask], correct[mask]),
            )
        )
    return rows


def summarize(scores: Sequence[float], correct: Sequence[bool]) -> Dict[str, float]:
    """Global calibration summary for a single scorer."""
    return {
        "ece": ece(scores, correct),
        "brier": brier(scores, correct),
        "nll": nll(scores, correct),
        "mean_score": float(np.mean(scores)),
        "accuracy": float(np.mean(np.asarray(correct, dtype=bool))),
    }


def _logit(scores: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class RegimeTemperature:
    """A temperature grid finer than the one the checkpoint ships.

    This is the honest competitor to a learned head. Laya already rescales its
    logits by one temperature per ``(question type, option-count)`` bucket, so
    the obvious fix for a bucket that spans 11..64 options is *more buckets* --
    one temperature per option count. If that repairs the miscalibration, no
    learned correction is needed and this artifact has no contribution.

    ``share_intercept`` distinguishes the two forms:

    ``True``
        one slope per regime, one shared intercept. Deployable: every regime's
        calibrated score is on the same scale, so a single global threshold
        still means the same thing everywhere.
    ``False``
        a slope and an intercept per regime. More capacity, but each regime can
        absorb its own bias into its own intercept, so a "global" threshold is
        only global by convention. This is a ceiling, not a policy.
    """

    slopes: Dict[int, float]
    intercepts: Dict[int, float]
    share_intercept: bool
    l2: float = 0.0

    def apply(self, scores: Sequence[float], n_options: Sequence[int]) -> np.ndarray:
        """Rescale to calibrated probabilities, one slope per option count."""
        z = _logit(scores)
        ks = np.asarray(n_options, dtype=int).reshape(-1)
        if z.size != ks.size:
            raise ValueError("scores and n_options must have the same length")
        out = np.empty(z.size, dtype=np.float64)
        for k in np.unique(ks):
            key = int(k)
            if key not in self.slopes:
                raise ValueError(
                    f"no temperature fitted for n_options={key}; "
                    f"fitted regimes are {sorted(self.slopes)}"
                )
            out[ks == k] = 1.0 / (
                1.0 + np.exp(-(self.slopes[key] * z[ks == k] + self.intercepts[key]))
            )
        return out

    @property
    def slope_spread(self) -> float:
        """Max/min ratio of the fitted slopes -- how much sharpness varies."""
        values = np.abs(np.array(list(self.slopes.values()), dtype=np.float64))
        return float(values.max() / max(float(values.min()), 1e-9))


def fit_regime_temperature(
    scores: Sequence[float],
    correct: Sequence[bool],
    n_options: Sequence[int],
    share_intercept: bool = True,
    l2: float = 0.0,
) -> RegimeTemperature:
    """Fit one logistic slope per option count, on data you are not evaluating on.

    Fit this on the split that trained the underlying model (or a disjoint
    calibration split) and apply it to held-out rows. Fitting it on the rows you
    report would make it look better than any deployed version of it.
    """
    import torch

    z = _logit(scores)
    ks = np.asarray(n_options, dtype=int).reshape(-1)
    y = np.asarray(correct, dtype=bool).astype(np.float64).reshape(-1)
    if not (z.size == ks.size == y.size):
        raise ValueError("scores, correct and n_options must have the same length")
    if z.size == 0:
        raise ValueError("cannot fit on an empty split")

    ks_unique = sorted(set(int(v) for v in ks))
    index = {k: i for i, k in enumerate(ks_unique)}
    g = np.array([index[int(v)] for v in ks], dtype=np.int64)

    zt = torch.as_tensor(z, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    gt = torch.as_tensor(g, dtype=torch.int64)
    a = torch.zeros(len(ks_unique), dtype=torch.float64, requires_grad=True)
    n_b = 1 if share_intercept else len(ks_unique)
    b = torch.zeros(n_b, dtype=torch.float64, requires_grad=True)
    lossf = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=300)

    def closure() -> "torch.Tensor":
        opt.zero_grad()
        logits = a[gt] * zt + (b[0] if share_intercept else b[gt])
        loss = lossf(logits, yt) + l2 * (a**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        slopes = a.detach().numpy().copy()
        bis = b.detach().numpy().copy()
    return RegimeTemperature(
        slopes={k: float(slopes[i]) for i, k in enumerate(ks_unique)},
        intercepts=(
            {k: float(bis[0]) for k in ks_unique}
            if share_intercept
            else {k: float(bis[i]) for i, k in enumerate(ks_unique)}
        ),
        share_intercept=share_intercept,
        l2=l2,
    )


def top1_from_probs(probs: Sequence[float], temperature: float = 1.0, applied: float = 1.0) -> float:
    """Top probability after rescaling a published distribution to a new temperature.

    Laya publishes ``p_i = softmax(z_i / T_applied)``. Dividing the raw logits by a
    different temperature instead is exactly a power transform on what was
    published, because ``z_i / T_applied = log p_i + const``:

        softmax(z_i / T_new) = softmax((T_applied / T_new) * log p_i)
                             = p_i ** (T_applied / T_new), renormalised

    So the published ``probabilities`` are enough to apply a temperature you did
    not get to choose at inference time. No access to raw logits is needed -- which
    is what makes this fix deployable against Laya's own output.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 1e-12, None)
    if p.size == 0:
        return float("nan")
    exponent = applied / temperature
    scaled = np.power(p, exponent)
    total = scaled.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float(scaled.max() / total)


def top1_many(
    probs: Sequence[Sequence[float]], temperature: float, n_options: Sequence[int],
    applied: float = 1.0,
) -> np.ndarray:
    """Apply one temperature to each row's published distribution, keeping its top-1."""
    ks = np.asarray(n_options, dtype=int).reshape(-1)
    if len(probs) != ks.size:
        raise ValueError("probs and n_options must have the same length")
    return np.array(
        [
            top1_from_probs(np.asarray(row, dtype=np.float64)[: int(k)], temperature, applied)
            for row, k in zip(probs, ks)
        ],
        dtype=np.float64,
    )


def rescaled_confidence(
    probs: Sequence[float], temperature: float, applied: float = 1.0
) -> float:
    """Laya's own ``confidence`` field, recomputed at a different temperature.

    Laya publishes ``confidence = 1 - H(p) / log(k)`` (normalised entropy) *and*
    ``probabilities``. The entropy form is not a monotone function of the top
    probability, so it does not inherit a top-1 correction: it has to be
    recomputed from a rescaled distribution. That is what this does, using the
    same power transform as :func:`top1_from_probs`.

    Answering to Laya's own definition of the field matters here, because a
    deployment thresholds on ``confidence`` -- not on the top probability -- so a
    repair that only fixes ``p_top1`` fixes a signal nobody was reading.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 1e-300, None)
    total_options = p.size
    if total_options < 2:
        return float("nan")
    scaled = np.power(p, applied / temperature)
    total = scaled.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    q = scaled / total
    # Restoring confidence to a flatter temperature multiplies the exponent down,
    # which can underflow the tail of a very peaked distribution to exactly zero.
    # Those entries contribute no entropy, so clipping them instead of taking
    # log(0) keeps the value finite without changing it materially.
    q = np.clip(q, 1e-300, None)
    ent = float(-(q * np.log(q)).sum())
    return float(1.0 - ent / np.log(total_options))


def rescaled_confidence_many(
    probs: Sequence[Sequence[float]],
    temperature: Union[float, Mapping[int, float]],
    n_options: Sequence[int],
    applied: Union[float, Mapping[int, float]] = 1.0,
) -> np.ndarray:
    """Apply a temperature to each row's published distribution and re-derive confidence."""
    ks = np.asarray(n_options, dtype=int).reshape(-1)
    if len(probs) != ks.size:
        raise ValueError("probs and n_options must have the same length")
    scalar_temp = not isinstance(temperature, Mapping)
    scalar_applied = not isinstance(applied, Mapping)
    out = np.empty(ks.size, dtype=np.float64)
    for i, (row, k) in enumerate(zip(probs, ks)):
        key = int(k)
        t = float(temperature) if scalar_temp else float(temperature[key])
        a = float(applied) if scalar_applied else float(applied[key])
        out[i] = rescaled_confidence(np.asarray(row, dtype=np.float64)[:key], t, a)
    return out


def fit_confidence_temperature(
    probs: Sequence[Sequence[float]],
    correct: Sequence[bool],
    n_options: Sequence[int],
    applied: Union[float, Mapping[int, float]] = 1.0,
    min_options: Optional[int] = None,
    lo: float = 0.2,
    hi: float = 8.0,
    steps: int = 40,
) -> float:
    """Temperature that makes mean stated confidence equal mean accuracy.

    A deliberately different objective from :func:`fit_option_temperatures`, which
    minimises log-loss of the top probability. Fitting the *documented field* to
    the observed accuracy is the calibration a practitioner would ship, and it is
    also a check on the other fit: if the two objectives land on the same value,
    the temperature is identified by the data rather than by the loss.

    Stated confidence falls as temperature rises (more flattening), so the search
    is a plain bisection between ``lo`` and ``hi``. Pass ``min_options=11`` to fit
    inside the ``choice:11+`` bucket rather than across buckets.
    """
    ks = np.asarray(n_options, dtype=int).reshape(-1)
    y = np.asarray(correct, dtype=bool).reshape(-1)
    if not (len(probs) == ks.size == y.size):
        raise ValueError("probs, correct and n_options must have the same length")
    idx = np.flatnonzero(ks >= min_options) if min_options is not None else np.arange(ks.size)
    if idx.size == 0:
        raise ValueError(f"no rows with n_options >= {min_options}")

    rows = [probs[int(i)] for i in idx]
    sub_k = ks[idx]
    want = float(y[idx].mean())
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        got = float(rescaled_confidence_many(rows, mid, sub_k, applied).mean())
        if got > want:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


@dataclass
class OptionTemperatures:
    """One temperature per option count, applied to the published distribution.

    This is the literal form of the finding. Laya already rescales its logits by a
    temperature; the defect is that its grid is too coarse, so a question with 12
    options and one with 64 share a single value. ``exponents`` is that grid made
    fine enough to be one value per option count.

    Why an exponent and not the temperature itself: rescaling a softmax is a power
    transform on what was published.

        softmax(z / T_new) = p ** (T_applied / T_new), renormalised

    so ``exponent[k] = T_applied(k) / T_new(k)`` is what you apply to Laya's own
    ``probabilities`` output, and it is the quantity that is identified by the
    data. ``temperatures`` converts back to the config vocabulary, which needs
    ``applied`` -- the temperature that produced the input -- because that differs
    by bucket (this checkpoint applied ``0.5`` above 10 options but ``1.76`` at 3-5,
    so an exponent is *not* comparable across buckets while a temperature is).
    """

    exponents: Dict[int, float]
    applied: Dict[int, float]
    l2: float = 0.0

    def apply(
        self, probs: Sequence[Sequence[float]], n_options: Sequence[int]
    ) -> np.ndarray:
        """Calibrated top-1 probabilities, one row at a time."""
        ks = np.asarray(n_options, dtype=int).reshape(-1)
        out = np.empty(ks.size, dtype=np.float64)
        for k in np.unique(ks):
            key = int(k)
            if key not in self.exponents:
                raise ValueError(
                    f"no temperature fitted for n_options={key}; "
                    f"fitted option counts are {sorted(self.exponents)}"
                )
            m = ks == key
            # exponent[k] = applied / T_new, so T_new = applied / exponent and the
            # power to raise the published probabilities to is just the exponent.
            out[m] = top1_many(
                [probs[i] for i in np.flatnonzero(m)],
                1.0 / self.exponents[key],
                ks[m],
                applied=1.0,
            )
        return out

    @property
    def temperatures(self) -> Dict[int, float]:
        """The fitted grid as absolute temperatures on the raw logits."""
        return {k: self.applied[k] / g for k, g in self.exponents.items()}

    @property
    def temperature_spread(self) -> float:
        """Max/min ratio of the fitted temperatures."""
        values = np.abs(np.array(list(self.temperatures.values()), dtype=np.float64))
        return float(values.max() / max(float(values.min()), 1e-9))


def _padded_log_probs(
    probs: Sequence[Sequence[float]], idx: np.ndarray
) -> np.ndarray:
    """``log p`` for the given rows, zero-padded to the widest row.

    Rows of a choice question can have different lengths, so they cannot be
    stacked directly. The pad value is an effectively zero probability: it never
    wins the max and contributes nothing to the denominator, so it cannot affect
    the top-1 probability the fit is scored on.
    """
    width = max(len(np.asarray(probs[int(i)]).reshape(-1)) for i in idx)
    out = np.full((idx.size, width), 1e-30, dtype=np.float64)
    for r, i in enumerate(idx):
        row = np.clip(np.asarray(probs[int(i)], dtype=np.float64).reshape(-1)[:width], 1e-12, None)
        out[r, : row.size] = row
    return np.log(out)


def _fit_exponent(log_p: np.ndarray, y: np.ndarray, l2: float) -> float:
    """One positive power exponent minimising BCE of the rescaled top-1 probability.

    ``log`` of the exponent is the free parameter, which keeps it positive without
    a constraint and makes the penalty scale-free. The penalty matters: a fully
    separable group would otherwise send the exponent to infinity.
    """
    import torch

    lp = torch.as_tensor(np.asarray(log_p, dtype=np.float64), dtype=torch.float64)
    yt = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    log_g = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_g], lr=0.5, max_iter=500)

    def closure() -> "torch.Tensor":
        opt.zero_grad()
        g = torch.exp(log_g)
        # q_i is proportional to p_i ** g; a positive exponent preserves the
        # ordering, so the top-1 stays the top-1 and the calibrated top
        # probability is exp(max - logsumexp) of the rescaled log-values.
        scaled = g * lp
        top1 = torch.exp(scaled.max(dim=-1).values - torch.logsumexp(scaled, dim=-1))
        loss = torch.nn.functional.binary_cross_entropy(
            top1.clamp(1e-12, 1 - 1e-12), yt
        ) + l2 * (log_g**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(torch.exp(log_g))


def fit_shared_temperature(
    probs: Sequence[Sequence[float]],
    correct: Sequence[bool],
    n_options: Sequence[int],
    applied: Union[float, Mapping[int, float]] = 1.0,
    l2: float = 1e-3,
    min_options: Optional[int] = None,
) -> OptionTemperatures:
    """Fit ONE temperature across a whole set of option counts.

    This is the control that decides whether a finer temperature grid is
    justified at all: if a single value serves every option count as well as one
    per option count, the grid was never the problem and the finding is a single
    wrong constant. ``min_options`` restricts the fit (``11`` answers "does one
    temperature suffice for the ``choice:11+`` bucket?").

    Returns an :class:`OptionTemperatures` whose ``exponents`` are the same value
    replicated per option count, so ``temperature_spread`` is 1.0 by construction
    and the two fits are directly comparable through the same ``apply``.
    """
    ks = np.asarray(n_options, dtype=int).reshape(-1)
    y = np.asarray(correct, dtype=bool).astype(np.float64).reshape(-1)
    if not (len(probs) == ks.size == y.size):
        raise ValueError("probs, correct and n_options must have the same length")
    if ks.size == 0:
        raise ValueError("cannot fit on an empty split")

    option_counts = sorted(set(int(v) for v in ks))
    if isinstance(applied, Mapping):
        applied_by_k = {k: float(applied[k]) for k in option_counts}
    else:
        applied_by_k = {k: float(applied) for k in option_counts}

    idx = np.flatnonzero(ks >= min_options) if min_options is not None else np.arange(ks.size)
    if idx.size == 0:
        raise ValueError(f"no rows with n_options >= {min_options}")

    g = _fit_exponent(_padded_log_probs(probs, idx), y[idx], l2)
    return OptionTemperatures(
        exponents={k: g for k in option_counts}, applied=applied_by_k, l2=l2
    )


def fit_option_temperatures(
    probs: Sequence[Sequence[float]],
    correct: Sequence[bool],
    n_options: Sequence[int],
    applied: Union[float, Mapping[int, float]] = 1.0,
    l2: float = 1e-3,
) -> OptionTemperatures:
    """Fit one temperature per option count, on data you are not evaluating on.

    ``applied`` is the temperature that produced the input probabilities: either a
    scalar, or a mapping from option count to temperature when the checkpoint
    used different buckets (it usually did). It is needed only to express the
    result as an absolute temperature; the fit itself is on the exponent.

    The fit is separable -- one option count's temperature affects only that option
    count's rows -- so this is a handful of independent one-parameter fits rather
    than one joint one. Compare it against :func:`fit_shared_temperature` before
    concluding the grid needs to be finer.
    """
    ks = np.asarray(n_options, dtype=int).reshape(-1)
    y = np.asarray(correct, dtype=bool).astype(np.float64).reshape(-1)
    if not (len(probs) == ks.size == y.size):
        raise ValueError("probs, correct and n_options must have the same length")
    if ks.size == 0:
        raise ValueError("cannot fit on an empty split")

    option_counts = sorted(set(int(v) for v in ks))
    if isinstance(applied, Mapping):
        applied_by_k = {k: float(applied[k]) for k in option_counts}
    else:
        applied_by_k = {k: float(applied) for k in option_counts}

    exponents: Dict[int, float] = {}
    for k in option_counts:
        idx = np.flatnonzero(ks == k)
        exponents[k] = _fit_exponent(_padded_log_probs(probs, idx), y[idx], l2)
    return OptionTemperatures(exponents=exponents, applied=applied_by_k, l2=l2)


def format_bucket_table(
    confidence_rows: Sequence[BucketCalibration],
    head_rows: Sequence[BucketCalibration],
    title: str = "calibration by label-space regime",
) -> str:
    """Render the side-by-side per-regime calibration table."""
    by_bucket = {r.bucket: r for r in confidence_rows}
    head_by_bucket = {r.bucket: r for r in head_rows}

    lines = [f"== {title} =="]
    lines.append(
        f"{'bucket':>8} {'n':>6} {'acc':>7} "
        f"{'ECE conf':>9} {'ECE head':>9} {'bias conf':>10} {'bias head':>10}"
    )
    for label in BUCKET_LABELS:
        c = by_bucket.get(label)
        h = head_by_bucket.get(label)
        if c is None:
            continue
        head_ece = f"{h.ece:>9.4f}" if h else f"{'--':>9}"
        head_bias = f"{h.bias:>+10.4f}" if h else f"{'--':>10}"
        lines.append(
            f"{label:>8} {c.n:>6} {c.accuracy:>7.3f} "
            f"{c.ece:>9.4f} {head_ece} {c.bias:>+10.4f} {head_bias}"
        )
    return "\n".join(lines)
