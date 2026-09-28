"""Does the `choice:11+` temperature bucket hold together?

The question this answers
------------------------
Laya's checkpoint ships one temperature per ``(question type, option count)``
bucket -- ``{2, 3-5, 6-10, 11+}`` -- and the loader warns that it refuses to apply
some of them::

    RuntimeWarning: laya: this checkpoint ships temperatures outside [0.5, 5]
    which would distort confidence; clamping choice:11+=0.1006.

So every question with 11 or more options is scored with the single clamped value
``T = 0.5``, even though those questions range from 11 options to 64+ and each
option's share of the fixed ``head_max_len`` header budget therefore varies by
more than 5x. If accuracy and calibration both drift across that range, then no
single temperature can repair the bucket, and the published confidence is
systematically wrong for part of it.

What is measured
----------------
* the per-``k`` table: accuracy, the two published signals, bias, ECE, tokens per
  option, latency;
* whether accuracy and bias drift monotonically with ``k`` **inside** ``11+``,
  with a paired permutation test (the same items are asked at every ``k``, so the
  comparison is paired and the test permutes ``k`` within item);
* whether a single monotone recalibration of the published score can reproduce
  per-``k`` accuracy, by fitting one Platt map to the whole bucket and reading off
  the per-``k`` residuals, then comparing to per-``k`` maps;
* what a single global threshold does to per-``k`` risk -- the operational
  consequence, which is the thing a caller actually experiences.

Two terms are kept strictly apart, because conflating them is the easiest way to
publish a wrong number:

``confidence``
    ``1 - H(p) / log(k)``, a normalised entropy concentration score. It is what
    Laya's ``answers[qid]["confidence"]`` contains and what its own
    ``ece_score`` is computed on. It is *not* a probability of being correct.
``p_top1``
    the published top probability. This one is on a probability scale, so ECE and
    Brier are meaningful for it.

Usage::

    python scripts/analyze_sweep.py --sweep artifacts/sweep.npz
    python scripts/analyze_sweep.py --sweep artifacts/sweep.npz \
        --compare artifacts/sweep_t1.npz --compare-label "T=1.0"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_escalate.calibration import brier, ece, nll  # noqa: E402
from laya_escalate.conformal import aurc, coverage_at_risk, risk_coverage_curve  # noqa: E402
from laya_escalate.frontier import compare_scorers, format_scorer_comparison  # noqa: E402

# The buckets laya's checkpoint actually ships temperatures for.
TEMP_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("2", 2, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11+", 11, 10**9),
)

TARGET_RISKS: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)


def bucket_of(k: int) -> str:
    for name, lo, hi in TEMP_BUCKETS:
        if lo <= k <= hi:
            return f"choice:{name}"
    return "choice:11+"


def load_sweep(path: Path) -> Dict[str, object]:
    """Load a sweep, recovering the item grouping if the artifact predates it."""
    payload = np.load(path, allow_pickle=True)
    data: Dict[str, object] = {key: payload[key] for key in payload.files}
    k = data["n_options"].astype(int)  # type: ignore[union-attr]
    if "item_index" not in data:
        # Older artifacts were written k-block-major over one shared item order:
        # `for k in ks: for idx in order`. Position within the block is therefore
        # the item, and all blocks are the same length.
        blocks = {int(v): int((k == v).sum()) for v in np.unique(k)}
        sizes = set(blocks.values())
        if len(sizes) != 1:
            raise ValueError(
                f"cannot recover item grouping: k-blocks are ragged {blocks}; "
                "re-collect with collect_sweep.py, which now writes item_index"
            )
        per_block = sizes.pop()
        data["item_index"] = np.arange(k.size) % per_block
        data["grouping_recovered"] = True
    else:
        data["grouping_recovered"] = False
    data["blocks"] = {int(v): int((k == v).sum()) for v in np.unique(k)}
    return data


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, so Spearman survives the ties a 0/1 outcome creates."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i
        while j + 1 < sx.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra = _rankdata(np.asarray(a, dtype=np.float64))
    rb = _rankdata(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def build_matrix(
    values: np.ndarray, item_index: np.ndarray, obs_k: np.ndarray, columns: Sequence[int]
) -> np.ndarray:
    """``(n_items, len(columns))`` matrix, one cell per (item, k).

    ``obs_k`` is the per-observation option count and ``columns`` the ``k`` values
    to lay out; they are separate arguments because a caller usually wants a
    subset of the observed ``k`` values (one temperature bucket, say) while the
    observations still carry all of them.
    """
    items = np.unique(item_index)
    col = {int(v): i for i, v in enumerate(columns)}
    out = np.full((items.size, len(columns)), np.nan, dtype=np.float64)
    pos = {int(v): i for i, v in enumerate(items)}
    for value, item, k in zip(values, item_index, obs_k):
        key = int(k)
        if key not in col:
            continue
        out[pos[int(item)], col[key]] = float(value)
    if np.isnan(out).any():
        raise ValueError(
            "the sweep is not a complete item x k grid over "
            f"k={list(columns)}; too ragged to pair"
        )
    return out


def paired_trend_test(
    matrix: np.ndarray,
    ks: Sequence[int],
    n_perm: int = 5000,
    seed: int = 0,
    direction: str = "decreasing",
) -> Dict[str, float]:
    """Is the k-trend real, given that the same items appear at every k?

    The statistic is the slope across ``k``: ``w . mean_over_items``, with
    ``w = k - mean(k)``. The null is "``k`` does not matter", under which a row's
    cells are exchangeable, so the null distribution comes from permuting ``k``
    within each item. That keeps the paired structure -- same item, same
    distractors -- instead of pretending the 1000 decisions are independent.
    """
    ks_arr = np.asarray(ks, dtype=np.float64)
    w = ks_arr - ks_arr.mean()

    def stat(m: np.ndarray) -> float:
        return float(w @ m.mean(axis=0))

    observed = stat(matrix)
    rng = np.random.default_rng(seed)
    n_items, n_k = matrix.shape
    extreme = 0
    for _ in range(n_perm):
        idx = rng.random((n_items, n_k)).argsort(axis=1)
        s = stat(np.take_along_axis(matrix, idx, axis=1))
        if direction == "decreasing":
            extreme += int(s <= observed)
        else:
            extreme += int(abs(s) >= abs(observed))
    return {
        "slope_statistic": observed,
        "p_value": float((extreme + 1) / (n_perm + 1)),
        "spearman": spearman(ks_arr, matrix.mean(axis=0)),
        "n_permutations": int(n_perm),
        "direction": direction,
    }


def paired_bootstrap_ci(
    matrix: np.ndarray, n_boot: int = 4000, seed: int = 0, alpha: float = 0.05
) -> np.ndarray:
    """Per-column mean CI, resampling *items* so the pairing is respected."""
    rng = np.random.default_rng(seed)
    n_items = matrix.shape[0]
    draws = np.empty((n_boot, matrix.shape[1]), dtype=np.float64)
    for b in range(n_boot):
        rows = rng.integers(0, n_items, size=n_items)
        draws[b] = matrix[rows].mean(axis=0)
    return np.stack(
        [np.quantile(draws, alpha / 2, axis=0), np.quantile(draws, 1 - alpha / 2, axis=0)],
        axis=1,
    )


def platt(
    logits: np.ndarray,
    correct: np.ndarray,
    l2: float = 1e-3,
) -> Tuple[float, float, float]:
    """Fit ``sigmoid(a * z + b)`` to correctness. Returns ``(a, b, loss)``.

    ``a`` is the slope: a recalibration that only rescales the score's sharpness.
    That is the closest observable analogue of a temperature, and it is what a
    bucket-wide fix is limited to -- one value of ``a`` for every ``k`` in it.
    """
    import torch

    z = torch.as_tensor(np.asarray(logits, dtype=np.float64))
    y = torch.as_tensor(np.asarray(correct, dtype=np.float64))
    a = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    lossf = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=300)

    def closure() -> "torch.Tensor":
        opt.zero_grad()
        loss = lossf(a * z + b, y) + l2 * (a**2)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        loss = float(lossf(a * z + b, y))
    return float(a), float(b), loss


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _platt_per_group_slope(
    z: np.ndarray, y: np.ndarray, group: np.ndarray, n_groups: int, l2: float = 1e-3
) -> Tuple[np.ndarray, float, np.ndarray]:
    """One slope per group, one shared intercept.

    This is the operational form of "give each ``k`` its own temperature, keep the
    bucket's published confidence comparable across it": the slopes are what a
    temperature can move, the shared intercept is what a single threshold assumes.
    Unlike a fully per-group Platt map it cannot absorb a group's bias into its own
    intercept, so what it leaves behind is real residual structure.
    """
    import torch

    zt = torch.as_tensor(np.asarray(z, dtype=np.float64))
    yt = torch.as_tensor(np.asarray(y, dtype=np.float64))
    gt = torch.as_tensor(np.asarray(group, dtype=np.int64))
    a = torch.zeros(n_groups, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    lossf = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=300)

    def closure() -> "torch.Tensor":
        opt.zero_grad()
        loss = lossf(a[gt] * zt + b, yt) + l2 * (a**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        slopes = a.detach().numpy().copy()
        intercept = float(b.detach())
        calibrated = torch.sigmoid(a[gt] * zt + b).numpy()
    return slopes, intercept, calibrated



def shared_map_null(
    z: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    observed: Dict[str, float],
    n_boot: int = 200,
    seed: int = 0,
    l2: float = 1e-3,
) -> Dict[str, Dict[str, float]]:
    """What do these statistics look like when ONE map really is enough?

    Per-``k`` slopes are noisy and a linear-logit map is not the true link, so
    both the slope spread and the residual per-``k`` bias take non-zero values
    even when a single map generated the data. Simulating from the fitted shared
    map -- and refitting both models on each draw -- gives the distribution each
    observed statistic has to be read against. Reporting either without this
    invites reading noise as structure, which is the failure mode a calibration
    artifact can least afford.
    """
    a_shared, b_shared, _ = platt(z, y)
    p_shared = 1.0 / (1.0 + np.exp(-(a_shared * z + b_shared)))
    rng = np.random.default_rng(seed)
    spreads = np.empty(n_boot, dtype=np.float64)
    biases = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        y_star = (rng.random(p_shared.size) < p_shared).astype(np.float64)
        slopes, _, _ = _platt_per_group_slope(z, y_star, group_idx, n_groups, l2)
        spreads[i] = float(
            np.max(np.abs(slopes)) / max(float(np.min(np.abs(slopes))), 1e-9)
        )
        a_s, b_s, _ = platt(z, y_star)
        cal_star = 1.0 / (1.0 + np.exp(-(a_s * z + b_s)))
        biases[i] = max(
            abs(float(cal_star[group_idx == g].mean() - y_star[group_idx == g].mean()))
            for g in range(n_groups)
        )

    def summarize(draws: np.ndarray, observed_value: float) -> Dict[str, float]:
        lo, hi = np.quantile(draws, [0.025, 0.975])
        return {
            "observed": float(observed_value),
            "null_median": float(np.median(draws)),
            "null_lo": float(lo),
            "null_hi": float(hi),
            "p_value": float((np.sum(draws >= observed_value) + 1) / (n_boot + 1)),
        }

    return {
        "n_boot": int(n_boot),
        "slope_spread": summarize(spreads, observed["slope_spread"]),
        "worst_abs_bias_shared": summarize(biases, observed["worst_abs_bias_shared"]),
    }


def group_recalibration_adequacy(
    z: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    group_labels: Sequence[float],
    n_boot: int = 200,
) -> Dict[str, object]:
    """Can ONE recalibration of ``z`` serve every group?

    The grouping variable is a parameter: ``k`` for the bucket question, and
    distractor confusability for the within-``k`` question. Keeping one
    implementation is the point -- these numbers decide whether a fix is
    justified, and two copies of the fitting code that drift apart would show up
    as a result rather than as a bug.

    Fit one map to everything, then read each group's residual bias and ECE. A
    dataset a single rescaling can repair leaves residuals near zero everywhere.
    Anything left is structure one shared parameter cannot express, whatever value
    it is set to -- and it is judged against a simulated null from that same
    estimator, because per-group slopes are noisy even when one map is right.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    group_idx = np.asarray(group_idx, dtype=np.int64)
    n_groups = len(group_labels)

    a_shared, b_shared, loss_shared = platt(z, y)
    shared = 1.0 / (1.0 + np.exp(-(a_shared * z + b_shared)))

    per_group_slopes, shared_intercept, slope_cal = _platt_per_group_slope(
        z, y, group_idx, n_groups
    )

    rows: List[Dict[str, float]] = []
    for g, label in enumerate(group_labels):
        m = group_idx == g
        a_g, b_g, loss_g = platt(z[m], y[m])
        per_group = 1.0 / (1.0 + np.exp(-(a_g * z[m] + b_g)))
        rows.append(
            {
                "group": float(label),
                "n": float(m.sum()),
                "accuracy": float(y[m].mean()),
                "platt_slope_per_group": a_g,
                "platt_loss_per_group": loss_g,
                "residual_bias_shared": float(shared[m].mean() - y[m].mean()),
                "residual_bias_per_group_slope": float(slope_cal[m].mean() - y[m].mean()),
                "residual_bias_per_group_map": float(per_group.mean() - y[m].mean()),
                "ece_shared": float(ece(shared[m], y[m])),
                "ece_per_group_slope": float(ece(slope_cal[m], y[m])),
                "ece_per_group_map": float(ece(per_group, y[m])),
                "nll_shared": float(nll(shared[m], y[m])),
                "nll_per_group_map": float(loss_g),
            }
        )

    slopes = np.array([r["platt_slope_per_group"] for r in rows], dtype=np.float64)
    # The spread statistic comes from the deployable estimator (per-group slope,
    # shared intercept), and is judged against the null from that same estimator.
    deployable_spread = float(
        np.max(np.abs(per_group_slopes)) / max(float(np.min(np.abs(per_group_slopes))), 1e-9)
    )
    worst_shared = float(max(abs(r["residual_bias_shared"]) for r in rows))
    null = shared_map_null(
        z,
        y,
        group_idx,
        n_groups,
        {"slope_spread": deployable_spread, "worst_abs_bias_shared": worst_shared},
        n_boot=n_boot,
    )
    shared_bias = np.array([abs(r["residual_bias_shared"]) for r in rows], dtype=np.float64)
    slope_bias = np.array(
        [abs(r["residual_bias_per_group_slope"]) for r in rows], dtype=np.float64
    )
    per_group_bias = np.array(
        [abs(r["residual_bias_per_group_map"]) for r in rows], dtype=np.float64
    )
    return {
        "n": int(y.size),
        "n_groups": int(n_groups),
        "group_labels": [float(v) for v in group_labels],
        "shared_slope": a_shared,
        "shared_intercept": b_shared,
        "shared_loss": loss_shared,
        "group_slopes": {str(l): float(per_group_slopes[i]) for i, l in enumerate(group_labels)},
        "group_shared_intercept": shared_intercept,
        "group_slopes_independent": {str(l): float(slopes[i]) for i, l in enumerate(group_labels)},
        "slope_spread": deployable_spread,
        "slope_spread_null": null,
        "worst_abs_bias_shared": float(shared_bias.max()),
        "worst_abs_bias_group_slope": float(slope_bias.max()),
        "worst_abs_bias_group_map": float(per_group_bias.max()),
        "rows": rows,
    }


def recalibration_adequacy(
    scores: np.ndarray,
    correct: np.ndarray,
    ks: np.ndarray,
    min_k: int = 11,
    n_boot: int = 200,
) -> Dict[str, object]:
    """Can one recalibration of the published score serve the whole 11+ bucket?

    The ``k``-grouped case of :func:`group_recalibration_adequacy`. Keys keep their
    historical ``per_k`` spelling so the reported tables and the tests that pin
    them do not have to change.
    """
    bucket = ks >= min_k
    ks_unique = sorted(set(int(v) for v in ks[bucket]))
    k_of = {v: i for i, v in enumerate(ks_unique)}
    group_idx = np.array([k_of[int(v)] for v in ks[bucket]], dtype=np.int64)

    out = group_recalibration_adequacy(
        _logit(scores[bucket]),
        correct[bucket].astype(bool),
        group_idx,
        [float(v) for v in ks_unique],
        n_boot=n_boot,
    )
    # Restore the k-facing names the printed tables and tests use.
    out["min_k"] = int(min_k)
    out["n"] = int(bucket.sum())
    out["per_k_slopes"] = out.pop("group_slopes")
    out["per_k_shared_intercept"] = out.pop("group_shared_intercept")
    out["per_k_slopes_independent"] = out.pop("group_slopes_independent")
    out["worst_abs_bias_per_k_slope"] = out.pop("worst_abs_bias_group_slope")
    out["worst_abs_bias_per_k_map"] = out.pop("worst_abs_bias_group_map")
    for row in out["rows"]:  # type: ignore[union-attr]
        row["k"] = row.pop("group")
        row["platt_slope_per_k"] = row.pop("platt_slope_per_group")
        row["platt_loss_per_k"] = row.pop("platt_loss_per_group")
        row["residual_bias_per_k_slope"] = row.pop("residual_bias_per_group_slope")
        row["residual_bias_per_k_map"] = row.pop("residual_bias_per_group_map")
        row["ece_per_k_slope"] = row.pop("ece_per_group_slope")
        row["ece_per_k_map"] = row.pop("ece_per_group_map")
        row["nll_per_k_map"] = row.pop("nll_per_group_map")
    return out


def confusability_adequacy(
    scores: np.ndarray,
    correct: np.ndarray,
    similarity: np.ndarray,
    n_groups: int = 4,
    n_boot: int = 200,
) -> Dict[str, object]:
    """Can one recalibration serve every level of distractor confusability?

    This is the within-``k`` question. At a *fixed* option count the temperature
    applied is one number, so if the right rescaling depends on how confusable the
    alternatives are, no per-``k`` temperature -- however finely gridded -- can
    express it. Groups are quantile bins of the achieved distractor similarity, so
    they are equal-sized and the comparison is not confounded with sample size.

    ``scores`` are **probabilities**, matching :func:`recalibration_adequacy`, and
    are logit-transformed here. Passing an already-logit'd array is the obvious
    way to get this wrong and it fails silently -- the values land in ``[0, 1]``,
    the second logit saturates them, and every fitted slope comes out absurd. So
    the range is checked rather than assumed.
    """
    scored = np.isfinite(similarity)
    if scored.sum() < n_groups * 10:
        raise ValueError(
            f"only {int(scored.sum())} rows carry a finite distractor similarity; "
            "re-collect with collect_sweep.py, which records it per row"
        )
    prob = np.asarray(scores, dtype=np.float64)[scored]
    if prob.size and (prob.min() < 0.0 or prob.max() > 1.0):
        raise ValueError(
            "confusability_adequacy expects probabilities in [0, 1] and logits them "
            f"itself; got range [{prob.min():.4f}, {prob.max():.4f}], which looks like "
            "values that were already transformed"
        )
    sim = np.asarray(similarity, dtype=np.float64)[scored]
    edges = np.quantile(sim, np.linspace(0, 1, n_groups + 1)[1:-1])
    group_idx = np.digitize(sim, edges)
    labels = [float(np.round(np.mean(sim[group_idx == g]), 4)) for g in range(n_groups)]
    return group_recalibration_adequacy(
        _logit(prob),
        np.asarray(correct, dtype=bool)[scored],
        group_idx,
        labels,
        n_boot=n_boot,
    )


def threshold_at_risk(
    scores: Sequence[float], correct: Sequence[bool], target_risk: float
) -> float:
    """Smallest threshold whose measured selective risk still meets the target."""
    thresholds, _, selective = risk_coverage_curve(scores, correct)
    feasible = np.isfinite(selective) & (selective <= target_risk)
    if not np.any(feasible):
        return float(np.max(thresholds))
    return float(thresholds[int(np.argmax(feasible))])


def global_threshold_consequence(
    scores: np.ndarray, correct: np.ndarray, ks: np.ndarray, target: float
) -> Dict[str, object]:
    """One global threshold, many regimes: what the caller actually experiences.

    The threshold is chosen on the pooled traffic so that the *average* selective
    risk meets the target, which is exactly what a single-confidence-gate
    deployment does. The per-``k`` rows then show how unevenly that risk is
    distributed -- the discrepancy a marginal guarantee does not cover.
    """
    lam = threshold_at_risk(scores, correct, target)
    acted = np.asarray(scores, dtype=np.float64) >= lam
    rows: List[Dict[str, float]] = []
    for k in sorted(set(int(v) for v in ks)):
        m = ks == k
        sel = acted[m]
        n_acted = int(sel.sum())
        rows.append(
            {
                "k": float(k),
                "coverage": n_acted / max(int(m.sum()), 1),
                "selective_risk": float(np.mean(~correct[m][sel])) if n_acted else float("nan"),
                "n_acted": float(n_acted),
            }
        )
    risks = np.array([r["selective_risk"] for r in rows], dtype=np.float64)
    finite = risks[np.isfinite(risks)]
    # An all-NaN column set means nothing was acted on in any regime; there is no
    # spread to report, and nanmax over an empty slice would only warn.
    spread = (
        float(finite.max() / max(finite.min(), 1e-9)) if finite.size else float("nan")
    )
    return {
        "target_risk": float(target),
        "threshold": float(lam),
        "pooled_coverage": float(acted.mean()),
        "pooled_selective_risk": float(np.mean(~correct[acted])) if acted.any() else float("nan"),
        "per_k": rows,
        "risk_spread": spread,
    }


def load_meta(path: Path) -> Dict[str, object]:
    """The sidecar written by collect_sweep.py, if it is still around."""
    sidecar = path.with_suffix(".meta.json")
    return json.loads(sidecar.read_text()) if sidecar.exists() else {}


def per_k_table(
    data: Dict[str, object], ks: Sequence[int], head_max_len: int
) -> List[Dict[str, float]]:
    """One row per ``k``: accuracy, both published signals, bias, ECE, cost."""
    k_arr = data["n_options"].astype(int)  # type: ignore[union-attr]
    correct = data["correct"].astype(bool)  # type: ignore[union-attr]
    conf = data["confidence"].astype(np.float64)  # type: ignore[union-attr]
    pt1 = data["p_top1"].astype(np.float64)  # type: ignore[union-attr]
    act = data["act_probability"].astype(np.float64)  # type: ignore[union-attr]
    lat = data["latency_ms"].astype(np.float64)  # type: ignore[union-attr]

    rows: List[Dict[str, float]] = []
    for k in ks:
        m = k_arr == k
        if not m.any():
            continue
        rows.append(
            {
                "k": float(k),
                "bucket": bucket_of(int(k)),
                "n": float(m.sum()),
                "accuracy": float(correct[m].mean()),
                "mean_confidence": float(conf[m].mean()),
                "bias_confidence": float(conf[m].mean() - correct[m].mean()),
                "ece_confidence": float(ece(conf[m], correct[m])),
                "mean_p_top1": float(pt1[m].mean()),
                "bias_p_top1": float(pt1[m].mean() - correct[m].mean()),
                "ece_p_top1": float(ece(pt1[m], correct[m])),
                "brier_p_top1": float(brier(pt1[m], correct[m])),
                "nll_p_top1": float(nll(pt1[m], correct[m])),
                "mean_act_probability": float(np.nanmean(act[m])),
                # The clamp's most user-visible consequence, and the number the
                # write-up leads with: Laya rounds published probabilities to four
                # decimals, so at the clamped `choice:11+` temperature a large
                # share of decisions are published as *exactly* 1.0000 -- literal
                # certainty. It falls with k, which partly masks the problem: the
                # bucket looks less sure as it gets harder.
                "fraction_p_top1_at_one": float(np.mean(pt1[m] >= 1 - 5e-5)),
                "fraction_confidence_at_one": float(np.mean(conf[m] >= 1 - 5e-5)),
                "tokens_per_option": (head_max_len - 16) / float(k),
                "median_latency_ms": float(np.median(lat[m])),
            }
        )
    return rows


def _jsonable(obj: object) -> object:
    """numpy scalars/arrays -> plain python, so the payload can be dumped."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _match_rows(
    base: Dict[str, object], other: Dict[str, object]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index the rows the two sweeps have in common, keyed by (item, k).

    Returned as ``(base_idx, other_idx, k)``. Two sweeps of the identical grid
    intersect completely; a narrower sweep -- one ``k`` re-run with a different
    option rendering, say -- intersects on the blocks it covers, and the
    comparison is then restricted to those. Matching on the key rather than on
    position is what makes that safe.
    """
    b_item = base["item_index"].astype(int)  # type: ignore[union-attr]
    o_item = other["item_index"].astype(int)  # type: ignore[union-attr]
    b_k = base["n_options"].astype(int)  # type: ignore[union-attr]
    o_k = other["n_options"].astype(int)  # type: ignore[union-attr]

    b_pos = {(int(i), int(k)): n for n, (i, k) in enumerate(zip(b_item, b_k))}
    o_pos = {(int(i), int(k)): n for n, (i, k) in enumerate(zip(o_item, o_k))}
    common = sorted(set(b_pos) & set(o_pos))
    if not common:
        raise ValueError(
            "the two sweeps share no (item, k) pairs, so nothing can be compared; "
            "re-run collect_sweep.py with the same --seed on both sides"
        )
    base_idx = np.array([b_pos[c] for c in common], dtype=int)
    other_idx = np.array([o_pos[c] for c in common], dtype=int)
    ks = np.array([c[1] for c in common], dtype=int)
    return base_idx, other_idx, ks


def compare_sweeps(
    base: Dict[str, object],
    other: Dict[str, object],
    n_perm: int,
    min_k_within: int = 11,
    n_boot: int = 200,
) -> Dict[str, object]:
    """Paired per-``k`` comparison of two sweeps over their shared grid.

    Rows are matched on ``(item, k)``, so a sweep that only re-ran part of the
    grid is compared on exactly that part and nothing else is assumed.

    Note what a temperature cannot do: ``softmax(z / T)`` with ``T > 0`` is a
    positive rescaling, so it is monotone in ``z`` and ``argmax`` is invariant.
    Accuracy therefore cannot move when only the temperature changes -- if it
    does, the pairing has gone wrong somewhere. That makes a temperature sweep a
    *calibration* experiment, which is why it is worth running: it separates "the
    confidence is distorted by the temperature" from "the model is worse at high
    cardinality". The first moves bias and ECE; the second leaves a per-``k`` bias
    that survives the temperature change.
    """
    base_idx, other_idx, matched_k = _match_rows(base, other)
    b_correct = base["correct"].astype(bool)[base_idx]  # type: ignore[union-attr]
    o_correct = other["correct"].astype(bool)[other_idx]  # type: ignore[union-attr]
    b_k = base["n_options"].astype(int)[base_idx]  # type: ignore[union-attr]
    o_k = other["n_options"].astype(int)[other_idx]  # type: ignore[union-attr]
    if not np.array_equal(b_k, o_k):
        raise AssertionError("matched rows disagree on k; the key matching is broken")
    b_item = base["item_index"].astype(int)[base_idx]  # type: ignore[union-attr]
    ks = sorted(set(int(v) for v in matched_k))

    b_pt1 = base["p_top1"].astype(np.float64)[base_idx]  # type: ignore[union-attr]
    o_pt1 = other["p_top1"].astype(np.float64)[other_idx]  # type: ignore[union-attr]
    b_conf = base["confidence"].astype(np.float64)[base_idx]  # type: ignore[union-attr]
    o_conf = other["confidence"].astype(np.float64)[other_idx]  # type: ignore[union-attr]

    rows: List[Dict[str, float]] = []
    for k in ks:
        m = b_k == k
        hit = b_correct[m].astype(np.float64)
        rows.append(
            {
                "k": float(k),
                "accuracy_base": float(b_correct[m].mean()),
                "accuracy_other": float(o_correct[m].mean()),
                "accuracy_delta": float(o_correct[m].mean() - b_correct[m].mean()),
                "n_items_changed": float(np.sum(b_correct[m] != o_correct[m])),
                "mean_p_top1_base": float(b_pt1[m].mean()),
                "mean_p_top1_other": float(o_pt1[m].mean()),
                "bias_p_top1_base": float(b_pt1[m].mean() - hit.mean()),
                "bias_p_top1_other": float(o_pt1[m].mean() - hit.mean()),
                "ece_p_top1_base": float(ece(b_pt1[m], b_correct[m])),
                "ece_p_top1_other": float(ece(o_pt1[m], o_correct[m])),
                "nll_p_top1_base": float(nll(b_pt1[m], b_correct[m])),
                "nll_p_top1_other": float(nll(o_pt1[m], o_correct[m])),
                "fraction_p_top1_at_one_base": float(np.mean(b_pt1[m] >= 1 - 5e-5)),
                "fraction_p_top1_at_one_other": float(np.mean(o_pt1[m] >= 1 - 5e-5)),
                "mean_confidence_base": float(b_conf[m].mean()),
                "mean_confidence_other": float(o_conf[m].mean()),
            }
        )

    within = [k for k in ks if k >= min_k_within]
    bias_base = b_pt1 - b_correct.astype(np.float64)
    bias_other = o_pt1 - o_correct.astype(np.float64)

    # --- fixed-k case: is cardinality the whole story? -----------------------
    # If the two sweeps differ only in how the distractors were chosen at ONE k,
    # then every other handle a temperature could grab (option count, question
    # type, head_max_len, the temperature value itself) is held constant. Any
    # calibration difference that survives must come from something the
    # temperature cannot see, which is exactly what the head's features are for.
    within_k: Optional[Dict[str, object]] = None
    has_sim = (
        "distractor_similarity_mean" in base and "distractor_similarity_mean" in other
    )
    if len(ks) == 1 and has_sim:
        # Probabilities, not logits: each callee applies its own transform.
        pooled_p = np.concatenate([b_pt1, o_pt1])
        pooled_z = _logit(pooled_p)
        pooled_y = np.concatenate([b_correct, o_correct])
        condition = np.concatenate(
            [np.zeros(b_pt1.size, dtype=np.int64), np.ones(o_pt1.size, dtype=np.int64)]
        )
        pooled_sim = np.concatenate(
            [
                base["distractor_similarity_mean"].astype(np.float64)[base_idx],  # type: ignore[union-attr]
                other["distractor_similarity_mean"].astype(np.float64)[other_idx],  # type: ignore[union-attr]
            ]
        )
        by_condition = group_recalibration_adequacy(
            pooled_z, pooled_y, condition, [0.0, 1.0], n_boot=n_boot
        )
        by_similarity = confusability_adequacy(
            pooled_p, pooled_y, pooled_sim, n_groups=4, n_boot=n_boot
        )
        within_k = {
            "k": int(ks[0]),
            "n_pooled": int(pooled_y.size),
            "similarity_base_mean": float(np.nanmean(
                base["distractor_similarity_mean"].astype(np.float64)[base_idx]  # type: ignore[union-attr]
            )),
            "similarity_other_mean": float(np.nanmean(
                other["distractor_similarity_mean"].astype(np.float64)[other_idx]  # type: ignore[union-attr]
            )),
            "by_condition": by_condition,
            "by_confusability": by_similarity,
        }

    return {
        "n_matched": int(b_correct.size),
        "ks": ks,
        "overall_accuracy_base": float(b_correct.mean()),
        "overall_accuracy_other": float(o_correct.mean()),
        "overall_accuracy_delta": float(o_correct.mean() - b_correct.mean()),
        "n_items_changed": int(np.sum(b_correct != o_correct)),
        "overall_ece_p_top1_base": float(ece(b_pt1, b_correct)),
        "overall_ece_p_top1_other": float(ece(o_pt1, o_correct)),
        "overall_bias_p_top1_base": float(np.mean(bias_base)),
        "overall_bias_p_top1_other": float(np.mean(bias_other)),
        "overall_nll_p_top1_base": float(nll(b_pt1, b_correct)),
        "overall_nll_p_top1_other": float(nll(o_pt1, o_correct)),
        "rows": rows,
        # Bias rises with k, so the interesting tail is the two-sided one; asking
        # for "decreasing" here would report p ~ 1 for a real effect.
        "bias_trend_base": paired_trend_test(
            build_matrix(bias_base, b_item, b_k, ks), ks, n_perm=n_perm,
            direction="two-sided",
        ),
        "bias_trend_other": paired_trend_test(
            build_matrix(bias_other, b_item, b_k, ks), ks, n_perm=n_perm,
            direction="two-sided",
        ),
        "within": within,
        "within_bias_trend_base": paired_trend_test(
            build_matrix(bias_base, b_item, b_k, within), within, n_perm=n_perm,
            direction="two-sided",
        ),
        "within_bias_trend_other": paired_trend_test(
            build_matrix(bias_other, b_item, b_k, within), within, n_perm=n_perm,
            direction="two-sided",
        ),
        "within_k": within_k,
    }


def print_per_k(rows: Sequence[Dict[str, float]]) -> None:
    print(f"{'k':>4} {'bucket':>12} {'n':>4} {'acc':>7} {'conf':>7} {'biasC':>8} "
          f"{'ECE_C':>7} {'p_top1':>7} {'biasP':>8} {'ECE_P':>7} {'P==1':>6} "
          f"{'tok/opt':>8} {'ms':>6}")
    for r in rows:
        print(f"{int(r['k']):>4} {r['bucket']:>12} {int(r['n']):>4} {r['accuracy']:>7.3f} "
              f"{r['mean_confidence']:>7.3f} {r['bias_confidence']:>+8.4f} "
              f"{r['ece_confidence']:>7.4f} {r['mean_p_top1']:>7.3f} "
              f"{r['bias_p_top1']:>+8.4f} {r['ece_p_top1']:>7.4f} "
              f"{r['fraction_p_top1_at_one'] * 100:>5.1f}% "
              f"{r['tokens_per_option']:>8.1f} {r['median_latency_ms']:>6.0f}")
    print("  biasC/ECE_C are on laya's own `confidence` (entropy concentration, not a "
          "probability); biasP/ECE_P are on the published top probability.")
    print("  P==1 is the share of decisions whose published top probability is exactly "
          "1.0000 -- laya rounds to four decimals, so this is literal certainty.")
    print("  tok/opt is (head_max_len - 16) / k, the header budget's nominal share. It "
          "is NOT measured consumption: measure_option_tokens.py reports the real "
          "per-option token counts and whether anything was truncated.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("artifacts/sweep.npz"))
    parser.add_argument("--compare", type=Path, default=None,
                        help="second sweep of the identical grid, e.g. a "
                             "--temperature-override run")
    parser.add_argument("--compare-label", default="other")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--head-max-len", type=int, default=None,
                        help="defaults to the sidecar meta, else 512")
    parser.add_argument("--min-k-within-bucket", type=int, default=11)
    parser.add_argument("--n-boot", type=int, default=200,
                        help="parametric-bootstrap draws behind the slope-spread null")
    parser.add_argument("--out", type=Path, default=Path("artifacts/sweep_analysis.json"))
    args = parser.parse_args()

    data = load_sweep(args.sweep)
    meta = load_meta(args.sweep)
    head_max_len = int(args.head_max_len or meta.get("head_max_len") or 512)
    ks = sorted(set(int(v) for v in data["n_options"]))  # type: ignore[union-attr]
    item_index = data["item_index"].astype(int)  # type: ignore[union-attr]
    correct = data["correct"].astype(bool)  # type: ignore[union-attr]
    conf = data["confidence"].astype(np.float64)  # type: ignore[union-attr]
    pt1 = data["p_top1"].astype(np.float64)  # type: ignore[union-attr]
    act = data["act_probability"].astype(np.float64)  # type: ignore[union-attr]
    k_arr = data["n_options"].astype(int)  # type: ignore[union-attr]

    print(f"sweep            : {args.sweep}")
    print(f"dataset          : {meta.get('dataset', '?')} ({meta.get('split', 'test')})")
    print(f"decisions        : {correct.size}  items: {np.unique(item_index).size}  k: {ks}")
    print(f"head_max_len     : {head_max_len}  (held constant across k by design)")
    if meta.get("temperature_by_options"):
        print(f"temperatures     : {meta['temperature_by_options']}")
    if meta.get("temperature_by_options_raw"):
        print(f"temperatures raw : {meta['temperature_by_options_raw']}  <- what the checkpoint ships")
    if data.get("grouping_recovered"):
        print("note             : item_index was absent from the artifact; recovered from "
              "the k-block layout (`for k: for item:`).")
    print("note             : the option budget is held constant, but collect_sweep.py "
          "renders options with descriptions for k <= 10 and bare label names above, so "
          "the k=4 control differs in rendering as well as cardinality.")
    print()

    print("== per-k calibration and cost ==")
    table = per_k_table(data, ks, head_max_len)
    print_per_k(table)
    print()

    result: Dict[str, object] = {
        "sweep": str(args.sweep),
        "meta": meta,
        "n_decisions": int(correct.size),
        "n_items": int(np.unique(item_index).size),
        "ks": ks,
        "per_k": table,
    }

    # --- the hypothesis: does one temperature per bucket suffice? -------------
    within = [k for k in ks if k >= args.min_k_within_bucket]
    if len(within) >= 2:
        print(f"== within-bucket drift ({bucket_of(within[0])}, k = {within}) ==")
        hit = correct.astype(np.float64)
        acc_matrix = build_matrix(hit, item_index, k_arr, within)
        bias_matrix = build_matrix(pt1 - hit, item_index, k_arr, within)
        conf_bias_matrix = build_matrix(conf - hit, item_index, k_arr, within)

        acc_ci = paired_bootstrap_ci(acc_matrix, seed=0)
        print(f"{'k':>4} {'accuracy':>18} {'bias_p_top1':>12} {'bias_conf':>11}")
        for i, k in enumerate(within):
            print(f"{k:>4} {acc_matrix[:, i].mean():>8.4f} "
                  f"[{acc_ci[i, 0]:.3f},{acc_ci[i, 1]:.3f}] "
                  f"{bias_matrix[:, i].mean():>+12.4f} {conf_bias_matrix[:, i].mean():>+11.4f}")

        acc_trend = paired_trend_test(acc_matrix, within, n_perm=args.n_perm,
                                     direction="decreasing")
        bias_trend = paired_trend_test(bias_matrix, within, n_perm=args.n_perm,
                                      direction="two-sided")
        print()
        print(f"accuracy vs k : spearman={acc_trend['spearman']:+.3f}  "
              f"slope={acc_trend['slope_statistic']:+.5f}  "
              f"paired p={acc_trend['p_value']:.4f}")
        print(f"bias vs k     : spearman={bias_trend['spearman']:+.3f}  "
              f"slope={bias_trend['slope_statistic']:+.5f}  "
              f"paired p={bias_trend['p_value']:.4f}")
        print()
        print(f"== can one recalibration of the published score serve the bucket? ==")
        adequacy = recalibration_adequacy(pt1, correct, k_arr,
                                         min_k=args.min_k_within_bucket,
                                         n_boot=args.n_boot)
        print(f"{'k':>4} {'acc':>7} {'slope_k':>8} {'resid_1map':>11} "
              f"{'resid_slope':>12} {'resid_perk':>11} {'ECE_1map':>9} {'ECE_slope':>10}")
        for r in adequacy["rows"]:  # type: ignore[union-attr]
            print(f"{int(r['k']):>4} {r['accuracy']:>7.3f} {r['platt_slope_per_k']:>8.3f} "
                  f"{r['residual_bias_shared']:>+11.4f} {r['residual_bias_per_k_slope']:>+12.4f} "
                  f"{r['residual_bias_per_k_map']:>+11.4f} {r['ece_shared']:>9.4f} "
                  f"{r['ece_per_k_slope']:>10.4f}")
        null = adequacy["slope_spread_null"]  # type: ignore[index]
        spread = null["slope_spread"]  # type: ignore[index]
        bias_null = null["worst_abs_bias_shared"]  # type: ignore[index]
        print(f"  per-k slope spread = {spread['observed']:.2f}x. One shared map "
              f"reproduces {spread['null_median']:.2f}x "
              f"[{spread['null_lo']:.2f}, {spread['null_hi']:.2f}] by luck alone over "
              f"{null['n_boot']} draws, so p={spread['p_value']:.3f}.")
        print(f"  worst |per-k residual bias| under one map = {bias_null['observed']:.4f}. "
              f"The same statistic on data simulated from one map gives "
              f"{bias_null['null_median']:.4f} "
              f"[{bias_null['null_lo']:.4f}, {bias_null['null_hi']:.4f}], "
              f"so p={bias_null['p_value']:.3f}.")
        for label, stats in (("slope spread", spread), ("residual bias", bias_null)):
            verdict = (
                "NOT distinguishable from noise -> not evidence on its own"
                if stats["p_value"] > 0.05
                else "beyond what one shared map produces -> evidence the bucket needs "
                     "a finer correction"
            )
            print(f"  -> {label}: {verdict}.")
        print(f"  worst |residual bias|: one map={adequacy['worst_abs_bias_shared']:.4f}  "
              f"per-k slope+shared intercept={adequacy['worst_abs_bias_per_k_slope']:.4f}  "
              f"fully per-k={adequacy['worst_abs_bias_per_k_map']:.4f}")
        print("  ('fully per-k' can absorb a regime's bias into its own intercept -- it is a "
              "ceiling, not a deployable fix. The middle column is the deployable one that "
              "still leaves a single threshold meaning the same thing everywhere.)")
        print()
        result["within_bucket"] = {
            "ks": within,
            "accuracy_trend": acc_trend,
            "bias_trend": bias_trend,
            "accuracy_ci": {str(k): acc_ci[i].tolist() for i, k in enumerate(within)},
            "recalibration_adequacy": adequacy,
        }

    # --- what a single global gate does to per-k risk ------------------------
    print("== one global threshold, per-k outcome ==")
    gtc: Dict[str, object] = {}
    for name, scores in (("p_top1", pt1), ("confidence", conf)):
        for target in TARGET_RISKS:
            row = global_threshold_consequence(scores, correct, k_arr, target)
            gtc[f"{name}@{target}"] = row
            if target == 0.05:
                print(f"  {name:>10} @ target {target:.2f}: threshold={row['threshold']:.4f} "
                      f"pooled coverage={row['pooled_coverage']:.3f} "
                      f"pooled risk={row['pooled_selective_risk']:.4f} "
                      f"per-k risk spread={row['risk_spread']:.1f}x")
                for r in row["per_k"]:  # type: ignore[union-attr]
                    print(f"      k={int(r['k']):>3} coverage={r['coverage']:.3f} "
                          f"selective_risk={r['selective_risk']:.4f}")
    result["global_threshold"] = gtc
    print()

    # --- the model ships a second gate; the head has to beat both -----------
    scorers = {"confidence": conf, "p_top1": pt1, "act_probability": act}
    comparison = compare_scorers(scorers, correct, targets=TARGET_RISKS)
    print(format_scorer_comparison(comparison, title="published signals, matched selective risk"))
    for name, scores in scorers.items():
        print(f"  AURC {name:>16} = {aurc(scores, correct):.4f}")
    if float(np.nanstd(act)) < 1e-9:
        print(f"  note: act_probability is constant at {float(np.nanmean(act)):.4f} across all "
              f"{correct.size} decisions, so the model's own act head carries no escalation "
              f"signal on this corpus at any threshold.")
        result["act_probability_constant"] = float(np.nanmean(act))
    result["scorers"] = comparison.to_dict()
    print()

    # --- optional: the identical grid at a single temperature ----------------
    if args.compare is not None:
        other = load_sweep(args.compare)
        other_meta = load_meta(args.compare)
        cmp = compare_sweeps(data, other, args.n_perm,
                            min_k_within=args.min_k_within_bucket,
                            n_boot=args.n_boot)
        print(f"== paired comparison vs {args.compare} ({args.compare_label}) ==")
        print(f"  temperatures: {other_meta.get('temperature_by_options', '?')} "
              f"(override={other_meta.get('temperature_override', '?')})")
        print(f"  matched rows : {cmp['n_matched']} of {correct.size} "
              f"(k = {cmp['ks']})")
        print(f"  accuracy did not move at all: {cmp['n_items_changed']} of "
              f"{cmp['n_matched']} decisions changed ({cmp['overall_accuracy_base']:.4f} -> "
              f"{cmp['overall_accuracy_other']:.4f}). Expected when only a rescaling of "
              f"the logits changed: argmax, and therefore every accuracy number, is "
              f"invariant. The (item, k) pairing is exact.")
        print()
        print(f"{'k':>4} {'acc':>7} {'p_top1 base':>12} {'p_top1 new':>11} "
              f"{'bias base':>10} {'bias new':>10} {'ECE base':>9} {'ECE new':>8} "
              f"{'NLL base':>9} {'NLL new':>8} {'P==1 base':>10} {'P==1 new':>9}")
        for r in cmp["rows"]:  # type: ignore[union-attr]
            print(f"{int(r['k']):>4} {r['accuracy_base']:>7.3f} {r['mean_p_top1_base']:>12.3f} "
                  f"{r['mean_p_top1_other']:>11.3f} {r['bias_p_top1_base']:>+10.4f} "
                  f"{r['bias_p_top1_other']:>+10.4f} {r['ece_p_top1_base']:>9.4f} "
                  f"{r['ece_p_top1_other']:>8.4f} {r['nll_p_top1_base']:>9.4f} "
                  f"{r['nll_p_top1_other']:>8.4f} "
                  f"{r['fraction_p_top1_at_one_base'] * 100:>9.1f}% "
                  f"{r['fraction_p_top1_at_one_other'] * 100:>8.1f}%")
        print(f"  overall ECE {cmp['overall_ece_p_top1_base']:.4f} -> "
              f"{cmp['overall_ece_p_top1_other']:.4f} | bias "
              f"{cmp['overall_bias_p_top1_base']:+.4f} -> {cmp['overall_bias_p_top1_other']:+.4f} "
              f"| NLL {cmp['overall_nll_p_top1_base']:.4f} -> "
              f"{cmp['overall_nll_p_top1_other']:.4f}")
        print(f"  bias trend across k, base     : spearman="
              f"{cmp['bias_trend_base']['spearman']:+.3f} "  # type: ignore[index]
              f"p={cmp['bias_trend_base']['p_value']:.4f}")  # type: ignore[index]
        print(f"  bias trend across k, {args.compare_label:<9}: spearman="
              f"{cmp['bias_trend_other']['spearman']:+.3f} "  # type: ignore[index]
              f"p={cmp['bias_trend_other']['p_value']:.4f}")  # type: ignore[index]
        if len(cmp["within"]) >= 2:  # type: ignore[arg-type]
            print(f"  bias trend within {cmp['within']}, base    : spearman="
                  f"{cmp['within_bias_trend_base']['spearman']:+.3f} "  # type: ignore[index]
                  f"p={cmp['within_bias_trend_base']['p_value']:.4f}")  # type: ignore[index]
            print(f"  bias trend within {cmp['within']}, {args.compare_label:<9}: spearman="
                  f"{cmp['within_bias_trend_other']['spearman']:+.3f} "  # type: ignore[index]
                  f"p={cmp['within_bias_trend_other']['p_value']:.4f}")  # type: ignore[index]
        else:
            print("  (fewer than two matched k values sit in the high-cardinality "
                  "bucket, so there is no within-bucket trend to test here.)")
        print("  A drift that survives a change of temperature is not a temperature "
              "artifact; it is the model's own confidence curve bending with k. A drift "
              "that vanishes was the bucket temperature's doing.")

        wk = cmp.get("within_k")
        if wk:
            print()
            print(f"== within a FIXED k = {wk['k']}: is cardinality the whole story? ==")
            print(f"  pooled decisions: {wk['n_pooled']}  "
                  f"distractor similarity {wk['similarity_base_mean']:.3f} "
                  f"(base) -> {wk['similarity_other_mean']:.3f} ({args.compare_label})")
            print("  Everything a temperature can see is identical here: same k, same "
                  "question type, same head_max_len, same applied bucket temperature.")
            for name, key in (("by condition", "by_condition"),
                              ("by confusability quartile", "by_confusability")):
                block = wk[key]  # type: ignore[index]
                print()
                print(f"  one recalibration cannot serve every group ({name}):")
                print(f"    {'group':>8} {'n':>5} {'acc':>7} {'slope':>7} "
                      f"{'resid_1map':>11} {'resid_own':>10} {'ECE_1map':>9}")
                for r in block["rows"]:  # type: ignore[union-attr]
                    print(f"    {r['group']:>8.3f} {int(r['n']):>5} {r['accuracy']:>7.3f} "
                          f"{r['platt_slope_per_group']:>7.3f} "
                          f"{r['residual_bias_shared']:>+11.4f} "
                          f"{r['residual_bias_per_group_map']:>+10.4f} "
                          f"{r['ece_shared']:>9.4f}")
                null = block["slope_spread_null"]["worst_abs_bias_shared"]  # type: ignore[index]
                print(f"    worst |residual bias| under one map = "
                      f"{block['worst_abs_bias_shared']:.4f}; one map simulated on this "
                      f"data gives {null['null_median']:.4f} "
                      f"[{null['null_lo']:.4f}, {null['null_hi']:.4f}], "
                      f"p={null['p_value']:.3f}")
                print(f"    slope spread = {block['slope_spread']:.2f}x "
                      f"(p={block['slope_spread_null']['slope_spread']['p_value']:.3f})")  # type: ignore[index]
            verdict = (
                "within-k calibration MOVES -> a per-k temperature structurally cannot "
                "capture it, and a head with distribution-shape features has room to"
                if wk["by_confusability"]["worst_abs_bias_shared"]  # type: ignore[index]
                > wk["by_confusability"]["slope_spread_null"]["worst_abs_bias_shared"]["null_hi"]  # type: ignore[index]
                else "within-k calibration does not move beyond the null -> the head has "
                     "nothing a finer temperature does not already give it"
            )
            print(f"  -> {verdict}.")
        result["compare"] = {
            "path": str(args.compare),
            "label": args.compare_label,
            "meta": other_meta,
            "result": cmp,
        }
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2))
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




