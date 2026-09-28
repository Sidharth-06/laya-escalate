"""Conformal risk control for the escalation threshold.

The point of this module is to turn a tuned threshold into a *guarantee*.

The head emits a score ``s in [0, 1]`` per decision, where ``s`` is its estimate
of ``P(correct)``. The policy is: act if ``s >= lam``, otherwise escalate. We
want to pick ``lam`` so that the rate at which we silently act on a wrong
decision is bounded.

What is guaranteed
------------------
Using conformal risk control (Angelopoulos, Bates, Candes, Jordan & Lei,
"Conformal Risk Control", ICLR 2024), with a loss

    L_i(lam) = 1{ s_i >= lam and decision_i is wrong }

which is monotone non-increasing in ``lam`` and bounded in ``[0, 1]``, selecting

    lam_hat = inf{ lam : mean_i L_i(lam) + 1/(n+1) <= alpha }

yields

    E[ L_{n+1}(lam_hat) ] <= alpha

i.e. the **unconditional** rate of wrong-but-automatically-executed decisions is
at most ``alpha``, with no distributional assumptions beyond exchangeability of
the calibration and test decisions. No model assumptions, no asymptotics, no
Gaussian approximation.

What is *not* guaranteed
------------------------
Conformal risk control bounds the marginal rate over all traffic. It does **not**
bound the *conditional* risk (error rate among the acted-on subset) without
substantially stronger assumptions, and the tail behaviour on rare option-count
regimes is not covered by a marginal guarantee. We therefore also report the
empirical selective risk and coverage curves, and the docs are explicit that the
conditional number is measured, not proven.

This distinction matters and is the honest version of the claim. A vendor saying
"<=1% error" about the conditional rate is making an assumption-free claim they
cannot support; a marginal bound plus a measured conditional curve is defensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class ConformalResult:
    """Outcome of a conformal risk control fit."""

    threshold: float
    alpha: float
    n_calibration: int
    empirical_unconditional_risk: float
    empirical_selective_risk: float
    coverage: float
    vacuous: bool = False
    """True when ``alpha`` is below the achievable floor (``1/(n+1)``), which
    forces the policy to escalate everything. The bound then holds trivially."""


def conformal_threshold(
    scores: Sequence[float],
    wrong: Sequence[bool],
    alpha: float,
    bound: float = 1.0,
) -> ConformalResult:
    """Select the smallest acting threshold satisfying the risk bound.

    Parameters
    ----------
    scores:
        Head scores in ``[0, 1]`` on the calibration split. Higher = more likely
        correct, so a higher threshold means acting less.
    wrong:
        Boolean ground truth, ``True`` where the underlying decision model was
        incorrect.
    alpha:
        Target upper bound on the unconditional rate of wrong acted-on decisions.
    bound:
        Upper bound of the loss, kept as a parameter for clarity (it is 1).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    wrong = np.asarray(wrong, dtype=bool).reshape(-1)
    if scores.size != wrong.size:
        raise ValueError("scores and wrong must have the same length")
    n = scores.size
    if n == 0:
        raise ValueError("calibration set is empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")

    floor = bound / (n + 1)
    vacuous = alpha < floor

    # L_i(lam) is a step function that only changes at observed scores, so the
    # smallest satisfying lam is always one of the observed scores (or just
    # above the maximum, meaning "act on nothing").
    candidates = np.unique(scores)
    candidates = np.concatenate([candidates, [float(candidates.max()) + 1e-12]])

    chosen = float(candidates[-1])
    for lam in candidates:
        acted = scores >= lam
        empirical = float(np.sum(acted & wrong)) / n
        if empirical + floor <= alpha:
            chosen = float(lam)
            break

    acted = scores >= chosen
    n_acted = int(np.sum(acted))
    uncond = float(np.sum(acted & wrong)) / n
    selective = float(np.sum(acted & wrong)) / n_acted if n_acted else 0.0

    return ConformalResult(
        threshold=chosen,
        alpha=alpha,
        n_calibration=n,
        empirical_unconditional_risk=uncond,
        empirical_selective_risk=selective,
        coverage=n_acted / n,
        vacuous=vacuous,
    )


def risk_coverage_curve(
    scores: Sequence[float],
    correct: Sequence[bool],
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Selective risk as a function of coverage, sweeping the threshold.

    Returns ``(thresholds, coverage, selective_risk)``. For each threshold we
    act on every decision with ``score >= threshold``; coverage is the fraction
    acted on and selective risk is the error rate among those.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=bool).reshape(-1)
    if scores.size != correct.size:
        raise ValueError("scores and correct must have the same length")

    if thresholds is None:
        thresholds = np.unique(np.concatenate([[0.0], scores, [1.0 + 1e-12]]))
    thresholds = np.asarray(thresholds, dtype=np.float64)

    coverage = np.zeros(thresholds.size)
    selective = np.full(thresholds.size, np.nan)
    n = scores.size
    for i, lam in enumerate(thresholds):
        acted = scores >= lam
        n_acted = int(np.sum(acted))
        coverage[i] = n_acted / n if n else 0.0
        if n_acted:
            selective[i] = float(np.sum(acted & ~correct)) / n_acted
    return thresholds, coverage, selective


def aurc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve (lower is better).

    Standard selective-prediction metric: order decisions by descending
    confidence and integrate the cumulative error rate over the coverage
    fraction. A perfectly ordered scorer gives the smallest possible AURC.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=bool).reshape(-1)
    n = scores.size
    if n == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    errors = (~correct[order]).astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, n + 1)
    return float(np.mean(cumulative_risk))


def coverage_at_risk(
    scores: Sequence[float],
    correct: Sequence[bool],
    target_risk: float,
) -> float:
    """Maximum coverage achievable while keeping selective risk <= target_risk.

    Returns 0.0 when even acting on the single most confident decision is not
    accurate enough.
    """
    thresholds, coverage, selective = risk_coverage_curve(scores, correct)
    feasible = np.isfinite(selective) & (selective <= target_risk)
    if not np.any(feasible):
        return 0.0
    return float(np.max(coverage[feasible]))
