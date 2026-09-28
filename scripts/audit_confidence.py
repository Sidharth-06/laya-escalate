"""Audit Laya's published ``confidence`` field across option counts.

The subject here is ``confidence`` -- ``1 - H(p) / log(k)``, the field Laya's
card tells you to threshold on -- not the top probability. They are different
signals and they are miscalibrated differently, so an audit of a proxy is not an
audit of the thing in use.

What is measured, and on which rows:

* Every number is reported per option count. Laya's temperature grid is
  ``{2, 3-5, 6-10, 11+}``, so a per-bucket table averages 12, 20, 32, 48 and 64
  options into one row and the averaging is what hides the defect.
* Temperatures are fitted on a training split and evaluated on disjoint held-out
  rows. Nothing here is fitted on the rows it is scored on.
* The last table is the consequence a deployment actually cares about: one
  threshold, and the error rate it delivers at each option count.

Usage::

    USE_TF=0 python scripts/audit_confidence.py --records artifacts/sweep_big.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_escalate import (  # noqa: E402
    ece,
    fit_confidence_temperature,
    rescaled_confidence_many,
    split_dataset,
)


def bucket_of(k: int) -> str:
    """Laya's own bucket for a choice question with ``k`` options."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"choice:{size}"


def applied_temperatures(records: Path, ks: List[int]) -> Dict[int, float]:
    """What the checkpoint actually used, read from the collector's sidecar."""
    sidecar = records.with_suffix(".meta.json")
    if not sidecar.exists():
        return {int(k): 1.0 for k in ks}
    table = json.loads(sidecar.read_text()).get("temperature_by_options") or {}
    return {int(k): float(table.get(bucket_of(int(k)), 1.0)) for k in ks}


def per_k_table(conf, correct, k) -> List[dict]:
    """Bias and ECE per option count. Bias is the statistic that shows the ramp."""
    rows = []
    for kk in sorted(set(int(v) for v in k)):
        m = k == kk
        rows.append(
            {
                "k": kk,
                "n": int(m.sum()),
                "accuracy": float(correct[m].mean()),
                "mean_confidence": float(conf[m].mean()),
                "bias": float(conf[m].mean() - correct[m].mean()),
                "ece": float(ece(conf[m], correct[m])),
            }
        )
    return rows


def print_table(label: str, rows: List[dict]) -> None:
    print(f"  {label}")
    print(f"      {'k':>4} {'n':>5} {'acc':>7} {'mean':>7} {'bias':>9} {'ECE':>8}")
    for r in rows:
        print(
            f"      {r['k']:>4} {r['n']:>5} {r['accuracy']:>7.4f} "
            f"{r['mean_confidence']:>7.4f} {r['bias']:>+9.4f} {r['ece']:>8.4f}"
        )


def threshold_at_risk(conf, correct, target: float = 0.05) -> float:
    """Lowest threshold whose accepted set still has ``target`` error rate.

    Acceptance is ``confidence >= tau``. Solved on the training rows and then
    applied unchanged to held-out rows, so the disparity it reveals across option
    counts is not an artifact of tuning per count.
    """
    order = np.argsort(-conf)
    c, y = conf[order], correct[order]
    errors = (~y).cumsum()
    # Risk among the first i accepted. Take the largest set that is still at or
    # below the target -- the most permissive threshold that is safe.
    for i in range(c.size, 0, -1):
        if errors[i - 1] / i <= target:
            return float(c[i - 1])
    return float("inf")


def risk_spread(conf, correct, k, tau: float) -> List[dict]:
    rows = []
    for kk in sorted(set(int(v) for v in k)):
        m = (k == kk) & (conf >= tau)
        if m.sum() == 0:
            rows.append({"k": kk, "accepted": 0, "coverage": 0.0, "risk": float("nan")})
            continue
        rows.append(
            {
                "k": kk,
                "accepted": int(m.sum()),
                "coverage": float(m.sum() / (k == kk).sum()),
                "risk": float(1.0 - correct[m].mean()),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("artifacts/sweep_big.npz"))
    parser.add_argument("--out", type=Path, default=None,
                        help="write the audit as JSON here")
    parser.add_argument("--target-risk", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.records.exists():
        print(f"no sweep at {args.records}; run scripts/collect_sweep.py first",
              file=sys.stderr)
        return 1

    d = np.load(args.records, allow_pickle=True)
    probs = d["probs"]
    correct = d["correct"].astype(bool)
    k = d["n_options"].astype(int)
    shipped = d["confidence"].astype(float)
    groups = d["item_index"].astype(int) if "item_index" in d else None

    ks = sorted(set(k.tolist()))
    applied = applied_temperatures(args.records, ks)
    splits = split_dataset(len(correct), 0.25, 0.25, seed=args.seed, groups=groups)
    tr, te = splits.train_idx, splits.test_idx

    # The audited bucket. `choice:11+` is where the checkpoint's temperature is
    # both out of range and shared by every option count from 11 upward.
    def hi(idx: np.ndarray) -> np.ndarray:
        return idx[k[idx] >= 11]

    tr_hi, te_hi = hi(tr), hi(te)
    k_te, y_te = k[te_hi], correct[te_hi]
    is_train = np.zeros(k.size, dtype=bool)
    is_train[tr] = True
    print(f"records          : {args.records}")
    print(f"audited bucket   : choice:11+  (option counts {sorted(set(k_te.tolist()))})")
    print(f"held-out rows    : {te_hi.size} decisions")
    print(f"applied temp     : {applied[ks[-1]]:.4f} "
          f"(the checkpoint's stored value for this bucket)")
    print()

    # Everything fitted below sees training rows only; everything scored below is
    # computed on held-out rows only.
    t_conf = fit_confidence_temperature(
        [probs[int(i)] for i in tr_hi], correct[tr_hi], k[tr_hi], applied=applied
    )
    fixed_all = rescaled_confidence_many(probs, t_conf, k, applied)
    t1_all = rescaled_confidence_many(probs, 1.0, k, applied)

    print("== the documented field, as shipped vs rescaled (fit on train, scored on held out) ==")
    shipped_rows = per_k_table(shipped[te_hi], y_te, k_te)
    print_table("as shipped", shipped_rows)
    rows_t1 = per_k_table(t1_all[te_hi], y_te, k_te)
    print_table("T=1.0 (no refit, 0 floats)", rows_t1)
    rows_fit = per_k_table(fixed_all[te_hi], y_te, k_te)
    print_table(f"T={t_conf:.3f} (1 float, fitted)", rows_fit)
    print()

    def pooled(rows):
        n = sum(r["n"] for r in rows)
        acc = sum(r["accuracy"] * r["n"] for r in rows) / n
        mean = sum(r["mean_confidence"] * r["n"] for r in rows) / n
        return acc, mean

    print(f"  {'':>26} {'pooled acc':>11} {'pooled mean':>12} {'pooled bias':>12} {'max |bias|':>11}")
    for label, rows in (
        ("as shipped", shipped_rows),
        ("T=1.0", rows_t1),
        (f"T={t_conf:.3f} (fitted)", rows_fit),
    ):
        acc, mean = pooled(rows)
        worst = max(abs(r["bias"]) for r in rows)
        print(
            f"  {label:>26} {acc:>11.4f} {mean:>12.4f} {mean - acc:>+12.4f} {worst:>11.4f}"
        )
    print("  (max |bias| is the worst option count, not an average: averaging across")
    print("   option counts is exactly what hides a monotone ramp.)")
    print()

    # Is a finer grid justified? Fit one temperature per option count with the
    # same objective and look at the spread. Noise across counts does not
    # constitute a gradient.
    print("== is the grid the problem? ==")
    # Each option count gets its own fit, on its own training rows. Fitting with a
    # lower bound instead would let rows above that bound contribute, which is a
    # different question from "does this count need its own temperature?".
    per_k_exact = {}
    for kk in sorted(set(k_te.tolist())):
        m = is_train & (k == kk)
        per_k_exact[kk] = fit_confidence_temperature(
            [probs[i] for i in np.flatnonzero(m)],
            correct[m],
            k[m],
            applied=applied[int(kk)],
        )
    vals = [per_k_exact[kk] for kk in sorted(per_k_exact)]
    print("  one temperature per option count, each fitted to its own rows:")
    print("    " + "   ".join(f"k={kk}:{v:.3f}" for kk, v in per_k_exact.items()))
    print(f"    spread {max(vals) / min(vals):.2f}x")
    print(f"  one temperature for the whole bucket: {t_conf:.3f}")
    spread = [r["bias"] for r in rows_fit]
    print(f"  after ONE temperature the per-count bias spans "
          f"{max(spread) - min(spread):.4f} (was "
          f"{max(r['bias'] for r in shipped_rows) - min(r['bias'] for r in shipped_rows):.4f} shipped)")
    print()

    # The consequence a deployment sees: one threshold, an error rate per count.
    # The consequence a deployment sees. Thresholding on confidence, one "safe"
    # threshold delivers a very different error rate depending on how many options
    # the question had. Each field gets its own threshold, fitted on train rows
    # for a pooled risk of `target_risk`, so the two columns are the same policy
    # applied to two different signals rather than one number reused.
    print(f"== one threshold per field, {args.target_risk:.0%} pooled risk on train ==")
    tau_ship = threshold_at_risk(shipped[tr_hi], correct[tr_hi], args.target_risk)
    tau_fix = threshold_at_risk(fixed_all[tr_hi], correct[tr_hi], args.target_risk)
    print(f"  threshold on shipped confidence : >= {tau_ship:.4f}")
    print(f"  threshold on rescaled confidence: >= {tau_fix:.4f}")
    ship_rows = risk_spread(shipped[te_hi], y_te, k_te, tau_ship)
    fixed_rows = risk_spread(fixed_all[te_hi], y_te, k_te, tau_fix)
    fixed_same_tau = risk_spread(fixed_all[te_hi], y_te, k_te, tau_ship)
    print(
        f"  {'k':>4} | {'ship cov':>8} {'ship risk':>9} | "
        f"{'T=%.2f cov' % t_conf:>10} {'risk':>8} | {'at ship tau':>11}"
    )
    for a, b, c_row in zip(ship_rows, fixed_rows, fixed_same_tau):
        print(
            f"  {a['k']:>4} | {a['coverage']:>8.3f} {a['risk']:>9.4f} | "
            f"{b['coverage']:>10.3f} {b['risk']:>8.4f} | {c_row['risk']:>11.4f}"
        )

    def spread_of(rows) -> tuple:
        risks = [r["risk"] for r in rows if r["accepted"] and np.isfinite(r["risk"])]
        if not risks:
            return float("nan"), float("nan")
        return min(risks), max(risks)

    for label, rows in (("shipped", ship_rows), ("rescaled", fixed_rows)):
        lo_r, hi_r = spread_of(rows)
        print(f"  {label:>8}: {lo_r:.4f} at its best count .. {hi_r:.4f} at its worst "
              f"({hi_r / max(lo_r, 1e-9):.1f}x spread)")
    acc_lo = min(r["accuracy"] for r in shipped_rows)
    acc_hi = max(r["accuracy"] for r in shipped_rows)
    print(f"  accuracy itself runs {acc_lo:.3f} .. {acc_hi:.3f} over the same counts, so")
    print("  a threshold cannot equalise risk across them at any confidence scale.")
    print("  Rescaling repairs the level of the field; it cannot repair the accuracy")
    print("  decline, and reporting the two as one defect would overstate the fix.")

    if args.out:
        payload = {
            "records": str(args.records),
            "bucket": "choice:11+",
            "held_out_rows": int(te_hi.size),
            "applied_temperature": float(applied[ks[-1]]),
            "temperature_fitted_to_confidence": float(t_conf),
            "per_k": {
                "shipped": shipped_rows,
                "t1": rows_t1,
                "fitted": rows_fit,
            },
            "per_option_count_temperatures": {str(a): b for a, b in per_k_exact.items()},
            "per_option_count_spread": float(max(vals) / min(vals)),
            "threshold_shipped": float(tau_ship),
            "threshold_rescaled": float(tau_fix),
            "risk_at_threshold": {
                "shipped": ship_rows,
                "rescaled": fixed_rows,
                "rescaled_at_shipped_threshold": fixed_same_tau,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
