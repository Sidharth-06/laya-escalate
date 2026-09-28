"""The escalation frontier: turning scores into a policy and measuring it.

This module is where the artifact earns its headline. It produces:

* a policy (threshold) fitted under a *guarantee* (conformal risk control),
* the measured risk/coverage frontier, and
* the comparison that matters: the learned head versus plain confidence
  thresholding at matched selective risk.

The last item is the whole claim. Any confidence threshold can trade coverage
for accuracy; the question is whether the head buys more coverage at the same
error rate. If it does not, this artifact is not worth shipping and the numbers
here will say so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .calibration import (
    BucketCalibration,
    bucket_calibration,
    format_bucket_table,
    summarize,
)
from .conformal import (
    ConformalResult,
    aurc,
    conformal_threshold,
    coverage_at_risk,
    risk_coverage_curve,
)


@dataclass
class Policy:
    """A fitted escalation policy."""

    threshold: float
    alpha: Optional[float] = None
    target_risk: Optional[float] = None
    vacuous: bool = False
    guarantee: str = "none"
    """Human-readable statement of what, if anything, is guaranteed."""

    def apply(self, scores: Sequence[float]) -> np.ndarray:
        """Boolean mask, True where a decision may be acted on."""
        return np.asarray(scores, dtype=np.float64) >= self.threshold


def fit_conformal_policy(
    scores: Sequence[float],
    wrong: Sequence[bool],
    alpha: float,
) -> Policy:
    """Fit a threshold with a distribution-free marginal risk bound."""
    res: ConformalResult = conformal_threshold(scores, wrong, alpha)
    return Policy(
        threshold=res.threshold,
        alpha=alpha,
        vacuous=res.vacuous,
        guarantee=(
            f"E[wrong-and-auto-executed rate] <= {alpha:.3f} "
            f"(conformal risk control, n={res.n_calibration})"
        ),
    )


def fit_target_risk_policy(
    scores: Sequence[float],
    correct: Sequence[bool],
    target_risk: float,
) -> Policy:
    """Fit the max-coverage threshold whose *measured* selective risk meets a target.

    This is the pragmatic variant used when you do not need a formal bound. It
    is reported alongside the conformal policy so the two can be compared; the
    conformal one is the defensible claim, this one is the aggressive one.
    """
    thresholds, coverage, selective = risk_coverage_curve(scores, correct)
    feasible = np.isfinite(selective) & (selective <= target_risk)
    if not np.any(feasible):
        return Policy(
            threshold=float(np.max(thresholds)),
            target_risk=target_risk,
            guarantee="none (target risk infeasible)",
        )
    best = np.argmax(np.where(feasible, coverage, -np.inf))
    return Policy(
        threshold=float(thresholds[best]),
        target_risk=target_risk,
        guarantee=f"measured selective risk <= {target_risk:.3f} on the fitting split",
    )


@dataclass
class FrontierRow:
    target_risk: float
    head_coverage: float
    confidence_coverage: float
    coverage_gain: float
    head_selective_risk: float
    confidence_selective_risk: float


def compare_to_confidence(
    head_scores: Sequence[float],
    confidence: Sequence[float],
    correct: Sequence[bool],
    targets: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
) -> List[FrontierRow]:
    """Compare head-driven and confidence-driven escalation at matched risk.

    For each target selective risk, compute the maximum coverage each scorer
    achieves. ``coverage_gain`` is the head's advantage in automation rate at
    the same error rate -- the number that goes in the model card.
    """
    correct = np.asarray(correct, dtype=bool)
    rows: List[FrontierRow] = []
    for target in targets:
        head_cov = coverage_at_risk(head_scores, correct, target)
        conf_cov = coverage_at_risk(confidence, correct, target)

        head_mask = np.asarray(head_scores, dtype=np.float64) >= _threshold_for_risk(
            head_scores, correct, target
        )
        conf_mask = np.asarray(confidence, dtype=np.float64) >= _threshold_for_risk(
            confidence, correct, target
        )
        head_risk = float(np.mean(~correct[head_mask])) if head_mask.any() else float("nan")
        conf_risk = float(np.mean(~correct[conf_mask])) if conf_mask.any() else float("nan")

        rows.append(
            FrontierRow(
                target_risk=float(target),
                head_coverage=head_cov,
                confidence_coverage=conf_cov,
                coverage_gain=head_cov - conf_cov,
                head_selective_risk=head_risk,
                confidence_selective_risk=conf_risk,
            )
        )
    return rows


def _threshold_for_risk(
    scores: Sequence[float], correct: Sequence[bool], target_risk: float
) -> float:
    """Smallest threshold whose measured selective risk meets the target."""
    thresholds, _, selective = risk_coverage_curve(scores, correct)
    feasible = np.isfinite(selective) & (selective <= target_risk)
    if not np.any(feasible):
        return float(np.max(thresholds))
    # thresholds are ascending, so the first feasible one has the most coverage.
    idx = int(np.argmax(feasible))
    return float(thresholds[idx])


@dataclass
class ScorerComparison:
    """Coverage and AURC for several competing scorers on one held-out split.

    Exists because a decision model ships more than one escalation signal -- a
    reported ``confidence`` and, in Laya's case, an ``act_probability`` from its
    own act/escalate head. The head has to be measured against all of them, not
    only against confidence, or the artifact claims credit for work the model
    already did.
    """

    targets: List[float]
    coverage: Dict[str, List[float]]
    aurc: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {"targets": self.targets, "coverage": self.coverage, "aurc": self.aurc}


def compare_scorers(
    scorers: Mapping[str, Sequence[float]],
    correct: Sequence[bool],
    targets: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
) -> ScorerComparison:
    """Evaluate several scorers side by side at matched selective risk."""
    correct = np.asarray(correct, dtype=bool)
    coverage: Dict[str, List[float]] = {}
    aurc_by_name: Dict[str, float] = {}
    for name, scores in scorers.items():
        coverage[name] = [coverage_at_risk(scores, correct, t) for t in targets]
        aurc_by_name[name] = aurc(scores, correct)
    return ScorerComparison(
        targets=[float(t) for t in targets],
        coverage=coverage,
        aurc=aurc_by_name,
    )


def format_scorer_comparison(cmp: ScorerComparison, title: str = "scorer comparison") -> str:
    """Render a side-by-side coverage table across scorers."""
    names = list(cmp.coverage.keys())
    lines = [f"== {title} =="]
    header = f"{'target':>8}" + "".join(f"{n:>14}" for n in names)
    lines.append(header)
    for i, target in enumerate(cmp.targets):
        row = f"{target:>8.3f}" + "".join(f"{cmp.coverage[n][i]:>14.3f}" for n in names)
        lines.append(row)
    lines.append("")
    lines.append(f"{'AURC':>8}" + "".join(f"{cmp.aurc[n]:>14.4f}" for n in names))
    return "\n".join(lines)


@dataclass
class EvaluationReport:
    """Everything a model card needs, computed from one held-out split."""

    n: int
    accuracy: float
    head_aurc: float
    confidence_aurc: float
    aurc_improvement: float
    frontier: List[FrontierRow] = field(default_factory=list)
    conformal: Optional[Dict[str, object]] = None
    confidence_summary: Optional[Dict[str, float]] = None
    head_summary: Optional[Dict[str, float]] = None
    calibration_confidence: List[BucketCalibration] = field(default_factory=list)
    calibration_head: List[BucketCalibration] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["frontier"] = [asdict(r) for r in self.frontier]
        out["calibration_confidence"] = [r.to_dict() for r in self.calibration_confidence]
        out["calibration_head"] = [r.to_dict() for r in self.calibration_head]
        return out

    @property
    def worst_confidence_bias(self) -> float:
        """Largest absolute over/under-confidence across regimes."""
        if not self.calibration_confidence:
            return float("nan")
        return max(abs(r.bias) for r in self.calibration_confidence)


def measure_fixed_threshold(
    scores: Sequence[float],
    wrong: Sequence[bool],
    threshold: float,
) -> Dict[str, float]:
    """Measure an already-fitted threshold on rows it was not fitted on.

    Fitting the conformal threshold on the evaluated rows reports an in-sample
    number: the bound holds there by construction, so it verifies nothing. The
    deployed policy is fitted on the calibration split, and the only honest
    question is what it does to rows it has never seen -- which is this.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    wrong = np.asarray(wrong, dtype=bool).reshape(-1)
    if scores.size != wrong.size:
        raise ValueError("scores and wrong must have the same length")
    n = scores.size
    acted = scores >= float(threshold)
    n_acted = int(np.sum(acted))
    hits = int(np.sum(acted & wrong))
    return {
        "threshold": float(threshold),
        "coverage": n_acted / n if n else 0.0,
        "empirical_unconditional_risk": hits / n if n else 0.0,
        "empirical_selective_risk": hits / n_acted if n_acted else 0.0,
    }


def evaluate(
    head_scores: Sequence[float],
    confidence: Sequence[float],
    correct: Sequence[bool],
    targets: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
    conformal_alpha: Optional[float] = None,
    wrong: Optional[Sequence[bool]] = None,
    n_options: Optional[Sequence[int]] = None,
    conformal_policy: Optional[Policy] = None,
) -> EvaluationReport:
    """Produce the full held-out report: AURC, the frontier, and the bound check.

    Pass ``n_options`` to additionally get the per-regime calibration table,
    which is where miscalibration actually lives.

    Pass ``conformal_policy`` -- a policy fitted on the *calibration* split -- to
    have the conformal row report what that threshold does to these held-out
    rows. That is the deployed artefact, and the number worth quoting. Falling
    back to ``conformal_alpha`` refits the threshold on the evaluated rows, which
    holds the bound trivially; the payload labels which of the two happened so a
    reader cannot mistake one for the other.
    """
    correct = np.asarray(correct, dtype=bool)
    head_aurc = aurc(head_scores, correct)
    conf_aurc = aurc(confidence, correct)

    conformal_payload: Optional[Dict[str, float]] = None
    if conformal_policy is not None and wrong is not None:
        conformal_payload = {
            "alpha": conformal_policy.alpha,
            "source": "calibration-split threshold, measured on held-out rows",
            "vacuous": float(conformal_policy.vacuous),
            **measure_fixed_threshold(head_scores, wrong, conformal_policy.threshold),
        }
    elif conformal_alpha is not None and wrong is not None:
        res = conformal_threshold(head_scores, wrong, conformal_alpha)
        conformal_payload = {
            "alpha": res.alpha,
            "source": "refit on the evaluated rows (in-sample, verifies nothing)",
            "threshold": res.threshold,
            "coverage": res.coverage,
            "empirical_unconditional_risk": res.empirical_unconditional_risk,
            "empirical_selective_risk": res.empirical_selective_risk,
            "vacuous": float(res.vacuous),
        }

    confidence_rows: List[BucketCalibration] = []
    head_rows: List[BucketCalibration] = []
    confidence_summary: Optional[Dict[str, float]] = None
    head_summary: Optional[Dict[str, float]] = None
    if n_options is not None:
        confidence_rows = bucket_calibration(confidence, correct, n_options)
        head_rows = bucket_calibration(head_scores, correct, n_options)
        confidence_summary = summarize(confidence, correct)
        head_summary = summarize(head_scores, correct)

    return EvaluationReport(
        n=int(correct.size),
        accuracy=float(np.mean(correct)) if correct.size else float("nan"),
        head_aurc=head_aurc,
        confidence_aurc=conf_aurc,
        aurc_improvement=conf_aurc - head_aurc,
        frontier=compare_to_confidence(head_scores, confidence, correct, targets),
        conformal=conformal_payload,
        confidence_summary=confidence_summary,
        head_summary=head_summary,
        calibration_confidence=confidence_rows,
        calibration_head=head_rows,
    )


def format_report(report: EvaluationReport, title: str = "held-out evaluation") -> str:
    """Render a report as a plain-text table for the terminal."""
    lines: List[str] = []
    lines.append(f"== {title} ==")
    lines.append(f"decisions        : {report.n}")
    lines.append(f"base accuracy    : {report.accuracy:.4f}")
    lines.append(
        f"AURC  head={report.head_aurc:.4f}  confidence={report.confidence_aurc:.4f}  "
        f"gain={report.aurc_improvement:+.4f}"
    )
    lines.append("")
    lines.append("coverage at matched selective risk (higher is better)")
    lines.append(f"{'target':>8} {'head':>8} {'conf':>8} {'gain':>8}")
    for row in report.frontier:
        lines.append(
            f"{row.target_risk:>8.3f} {row.head_coverage:>8.3f} "
            f"{row.confidence_coverage:>8.3f} {row.coverage_gain:>+8.3f}"
        )
    if report.conformal:
        c = report.conformal
        lines.append("")
        alpha = float(c["alpha"])  # type: ignore[arg-type]
        measured_risk = float(c["empirical_unconditional_risk"])  # type: ignore[arg-type]
        lines.append(
            f"conformal@alpha={alpha:.3f}: threshold={float(c['threshold']):.4f} "  # type: ignore[arg-type]
            f"coverage={float(c['coverage']):.3f} "  # type: ignore[arg-type]
            f"measured uncond risk={measured_risk:.4f} "
            f"(bound={alpha:.4f})"
        )
        if "source" in c:
            lines.append(f"  threshold source: {c['source']}")
        if measured_risk > alpha:
            lines.append(
                "  note: measured risk EXCEEDS the bound on these rows. A marginal "
                "guarantee is an expectation over calibration draws and is not "
                "promised per-split; at this n it did not hold. Report it as is."
            )
        if c["vacuous"]:
            lines.append("  note: bound is vacuous at this alpha given n; escalate-all is safe.")

    if report.calibration_confidence:
        lines.append("")
        lines.append(format_bucket_table(report.calibration_confidence, report.calibration_head))
        if report.confidence_summary and report.head_summary:
            cs, hs = report.confidence_summary, report.head_summary
            lines.append("")
            lines.append(
                f"global ECE   confidence={cs['ece']:.4f}  head={hs['ece']:.4f}  "
                f"({cs['ece'] / max(hs['ece'], 1e-9):.1f}x better)"
            )
            lines.append(
                f"global Brier confidence={cs['brier']:.4f}  head={hs['brier']:.4f}"
            )
    return "\n".join(lines)
