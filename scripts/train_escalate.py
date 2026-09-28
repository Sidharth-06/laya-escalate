"""End-to-end: build features, train the head, fit the conformal bound, evaluate.

Usage
-----
Synthetic (default, no GPU, validates the machinery)::

    python scripts/train_escalate.py

Real Laya decisions collected by ``scripts/collect_laya.py``::

    python scripts/train_escalate.py --records artifacts/laya_records.npz

The conformal threshold is fitted on a calibration split the head never trained
on, and every reported number is computed on a further held-out split. That
discipline is the difference between a guarantee and a claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_escalate import (  # noqa: E402
    BUCKET_LABELS,
    EscalationBundle,
    FeatureContext,
    TrainConfig,
    bucket_calibration,
    compare_scorers,
    ece,
    evaluate,
    featurize_many,
    fit_conformal_policy,
    fit_option_temperatures,
    fit_regime_temperature,
    fit_shared_temperature,
    format_report,
    format_scorer_comparison,
    split_dataset,
    summarize,
    top1_from_probs,
    train_head,
)
from laya_escalate.synthetic import SyntheticConfig, simulate  # noqa: E402


def load_synthetic(args: argparse.Namespace) -> tuple:
    records, _ = simulate(
        SyntheticConfig(n_items=args.n_items, seed=args.seed, checkpoint=args.checkpoint)
    )
    x = featurize_many(records, FeatureContext(checkpoint=args.checkpoint))
    correct = np.array([r.correct for r in records], dtype=bool)
    confidence = np.array([r.p_top1 for r in records], dtype=np.float64)
    n_options = np.array([r.n_options for r in records], dtype=int)
    # The simulator's "confidence" *is* its top probability, so there is no
    # separate p_top1 signal, and it exposes no act head.
    return x, correct, confidence, None, n_options, None, None, None


def load_records(path: Path) -> tuple:
    """Read a collected npz. Every key past ``x``/``correct`` is optional.

    On a real Laya sweep ``confidence`` and ``p_top1`` are two different things:
    the first is the model's normalised entropy score, the second is the top
    probability. Both are baselines the head has to beat, so both are read.
    """
    payload = np.load(path, allow_pickle=True)
    x = payload["x"].astype(np.float32)
    correct = payload["correct"].astype(bool)
    confidence = payload["confidence"].astype(np.float64)
    p_top1 = payload["p_top1"].astype(np.float64) if "p_top1" in payload else None
    n_options = payload["n_options"].astype(int) if "n_options" in payload else None
    act = payload["act_probability"].astype(np.float64) if "act_probability" in payload else None
    groups = payload["item_index"].astype(int) if "item_index" in payload else None
    # The full published distribution, when the collector stored it. It is what a
    # per-option-count *temperature* needs: rescaling a softmax is a power
    # transform on the published probabilities, so the raw logits are not required.
    probs = payload["probs"] if "probs" in payload else None
    return x, correct, confidence, p_top1, n_options, act, groups, probs


def applied_temperatures(records_path: Path, ks: Sequence[int]) -> Dict[int, float]:
    """Per-option-count temperature that produced the recorded probabilities.

    Read from the sidecar ``collect_sweep.py`` writes, so the fitted temperatures
    are reported against what was actually applied rather than a guess. This
    matters: the checkpoint used ``0.5`` above 10 options but ``1.76`` at 3-5, and
    an exponent is only comparable across option counts once that is divided out.
    """
    sidecar = records_path.with_suffix(".meta.json")
    if not sidecar.exists():
        return {int(k): 1.0 for k in ks}
    table = json.loads(sidecar.read_text()).get("temperature_by_options") or {}
    if not table:
        return {int(k): 1.0 for k in ks}

    def bucket(k: int) -> str:
        size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        return f"choice:{size}"

    return {int(k): float(table.get(bucket(int(k)), 1.0)) for k in ks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=None,
                        help="npz from collect_laya.py; omit to run synthetic")
    parser.add_argument("--n-items", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="english")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="target bound on unconditional wrong-action rate")
    parser.add_argument("--targets", type=float, nargs="+",
                        default=[0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    if args.records:
        x, correct, confidence, p_top1, n_options, act, groups, probs = load_records(
            args.records
        )
        source = str(args.records)
    else:
        x, correct, confidence, p_top1, n_options, act, groups, probs = load_synthetic(args)
        source = "synthetic simulator (machinery validation only)"

    splits = split_dataset(
        len(correct),
        calib_fraction=0.25,
        test_fraction=0.25,
        seed=args.seed,
        groups=groups,
    )
    if groups is not None:
        n_items = int(np.unique(groups).size)
        print(f"split by item   : {n_items} items, none spanning two splits "
              f"(train {splits.train_idx.size} / calib {splits.calib_idx.size} / "
              f"test {splits.test_idx.size} decisions)")

    result = train_head(
        x[splits.train_idx],
        correct[splits.train_idx],
        TrainConfig(epochs=args.epochs, seed=args.seed),
        groups=None if groups is None else groups[splits.train_idx],
    )

    calib_scores = result.head.score(x[splits.calib_idx], result.normalizer)
    test_scores = result.head.score(x[splits.test_idx], result.normalizer)

    policy = fit_conformal_policy(
        calib_scores, ~correct[splits.calib_idx], alpha=args.alpha
    )

    # The conformal row must report the threshold that ships, measured on rows it
    # was not fitted on. Letting evaluate() refit the threshold on the test rows
    # would produce a number the bound holds on by construction, which is not a
    # result -- it is a restatement of the estimator.
    report = evaluate(
        test_scores,
        confidence[splits.test_idx],
        correct[splits.test_idx],
        targets=tuple(args.targets),
        conformal_policy=policy,
        wrong=~correct[splits.test_idx],
        n_options=None if n_options is None else n_options[splits.test_idx],
    )

    print(f"data source      : {source}")
    print(f"head parameters  : {result.head.n_parameters()}")
    print(f"best val AURC    : {result.best_val_aurc:.4f} (epoch {result.best_epoch})")
    print(f"policy threshold : {policy.threshold:.4f}")
    print(f"policy guarantee : {policy.guarantee}")
    print(f"policy coverage  : {policy.apply(test_scores).mean():.3f} on held-out test")
    print()
    print(format_report(report))

    # The honest competitor is not Laya's shipped bucketing -- it is Laya's own
    # repair mechanism applied with one temperature per option count instead of
    # one per bucket. Laya already rescales logits by temperature; the only
    # question is how fine the grid is. Fitted on the training split and measured
    # on held-out rows, exactly like the head, so neither gets an advantage.
    regime_temp = None
    temp_test = None
    if p_top1 is not None and n_options is not None:
        regime_temp = fit_regime_temperature(
            p_top1[splits.train_idx],
            correct[splits.train_idx],
            n_options[splits.train_idx],
            share_intercept=True,
        )
        temp_test = regime_temp.apply(p_top1[splits.test_idx], n_options[splits.test_idx])

    # The literal form of the claim: one *temperature* per option count, fitted on
    # the train split, applied to held-out published distributions. The exponent is
    # what you apply to Laya's own `probabilities`, so no raw logits are needed.
    option_temp = None
    option_temp_test = None
    shared_temp = None
    shared_temp_test = None
    if probs is not None and n_options is not None:
        applied = applied_temperatures(
            args.records, sorted(set(int(v) for v in n_options))
        )
        train_probs = [probs[int(i)] for i in splits.train_idx]
        test_probs = [probs[int(i)] for i in splits.test_idx]
        train_k = n_options[splits.train_idx]
        option_temp = fit_option_temperatures(
            train_probs, correct[splits.train_idx], train_k, applied=applied
        )
        option_temp_test = option_temp.apply(test_probs, n_options[splits.test_idx])
        # The control that decides whether a finer grid is justified at all. Fit
        # inside `choice:11+` only: pooling k=4 with the bucket would ask one
        # temperature to serve two buckets of the checkpoint's own grid, which is
        # a different question from "is this bucket too coarse?".
        above = sorted(k for k in option_temp.exponents if k >= 11)
        temp_min_k = above[0] if above else min(option_temp.exponents)
        shared_temp = fit_shared_temperature(
            train_probs, correct[splits.train_idx], train_k, applied=applied,
            min_options=temp_min_k,
        )
        shared_temp_test = shared_temp.apply(test_probs, n_options[splits.test_idx])
        # The zero-parameter control: set the offending bucket's temperature to 1.0
        # and change nothing else. No fit, no data, no tuning -- whatever this row
        # achieves is free, and it is the bar any proposed fix has to clear.
        test_selected_k = n_options[splits.test_idx]
        unity_test = np.asarray(p_top1[splits.test_idx], dtype=np.float64).copy()
        for i, row in enumerate(test_probs):
            k = int(test_selected_k[i])
            if k >= temp_min_k:
                unity_test[i] = top1_from_probs(row, 1.0, applied[k])

    # The head is not competing with confidence alone: a Laya deployment already
    # has two gating signals on the table. Measure against all of them, on the
    # same held-out rows, at the same selective risks.
    shipped: dict = {"head": test_scores, "confidence": confidence[splits.test_idx]}
    if p_top1 is not None:
        shipped["p_top1"] = p_top1[splits.test_idx]
    if temp_test is not None:
        shipped["p_top1+finer_temp"] = temp_test
    if shared_temp_test is not None:
        shipped["one temperature"] = shared_temp_test
    if option_temp_test is not None:
        shipped["per-k temperature"] = option_temp_test
    if act is not None:
        shipped["act_probability"] = act[splits.test_idx]
    comparison = compare_scorers(
        shipped, correct[splits.test_idx], targets=tuple(args.targets)
    )
    print()
    print(format_scorer_comparison(comparison, title="head vs the signals the model ships"))
    for name in shipped:
        if name == "head":
            continue
        gains = [comparison.coverage[name][i] for i in range(len(comparison.targets))]
        head = comparison.coverage["head"]
        print(f"  coverage gain over {name:>16} = "
              + " ".join(f"{head[i] - gains[i]:+.3f}" for i in range(len(head))))
    if temp_test is not None:
        rows_temp = bucket_calibration(
            temp_test, correct[splits.test_idx], n_options[splits.test_idx]
        )
        rows_p = bucket_calibration(
            p_top1[splits.test_idx], correct[splits.test_idx], n_options[splits.test_idx]
        )
        by_temp = {r.bucket: r for r in rows_temp}
        by_p = {r.bucket: r for r in rows_p}
        print()
        print("== is a finer temperature enough? (fit on train, measured on held-out) ==")
        print(
            f"{'bucket':>8} {'n':>6} {'acc':>7} {'ECE p_top1':>11} {'ECE +temp':>10} "
            f"{'bias p_top1':>12} {'bias +temp':>11}"
        )
        for label in BUCKET_LABELS:
            p = by_p.get(label)
            t = by_temp.get(label)
            if p is None or t is None:
                continue
            print(
                f"{label:>8} {p.n:>6} {p.accuracy:>7.3f} {p.ece:>11.4f} {t.ece:>10.4f} "
                f"{p.bias:>+12.4f} {t.bias:>+11.4f}"
            )
        sp = summarize(p_top1[splits.test_idx], correct[splits.test_idx])
        st = summarize(temp_test, correct[splits.test_idx])
        print(
            f"\nglobal ECE  p_top1={sp['ece']:.4f}  +finer temp={st['ece']:.4f}  "
            f"head={report.head_summary['ece']:.4f}"
        )
        print(
            f"global Brier p_top1={sp['brier']:.4f}  +finer temp={st['brier']:.4f}  "
            f"head={report.head_summary['brier']:.4f}"
        )
        print(
            f"slope spread across k = {regime_temp.slope_spread:.2f}x  "
            + "  ".join(
                f"k={k}:{v:.3f}" for k, v in sorted(regime_temp.slopes.items())
            )
        )

    if option_temp is not None:
        test_correct = correct[splits.test_idx]
        test_k = n_options[splits.test_idx]
        test_shipped = p_top1[splits.test_idx]

        # The bucket is the subject of the audit, and k=4 sits in a *different*
        # bucket it is not fair to judge against, so the headline comparison is
        # restricted to `choice:11+`.
        above_k = sorted(k for k in option_temp.exponents if k >= 11)
        if above_k:
            sel = np.isin(test_k, above_k)
            from laya_escalate import aurc, coverage_at_risk

            print()
            print(f"== the bucket itself: choice:11+ on held-out rows (n={int(sel.sum())}) ==")
            rows_hi = []
            for name, sc in (
                ("p_top1 as published", test_shipped),
                ("T=1.0 (no refit)", unity_test),
                ("one temperature", shared_temp_test),
                ("per-k temperature", option_temp_test),
                ("per-k logistic (7 params)", temp_test),
                ("escalation head (5543)", test_scores),
            ):
                if sc is None:
                    continue
                s = sc[sel]
                y = test_correct[sel]
                rows_hi.append(
                    (
                        name,
                        ece(s, y),
                        float(s.mean() - y.mean()),
                        aurc(s, y),
                        coverage_at_risk(s, y, 0.01),
                        coverage_at_risk(s, y, 0.02),
                        float(np.mean(s >= 1 - 5e-5)),
                    )
                )
            print(
                f"{'scorer':>26} {'ECE':>8} {'bias':>9} {'AURC':>8} "
                f"{'cov@1%':>8} {'cov@2%':>8} {'P==1':>7}"
            )
            for name, e, b, a, c1, c2, one in rows_hi:
                print(
                    f"{name:>26} {e:>8.4f} {b:>+9.4f} {a:>8.4f} "
                    f"{c1:>8.3f} {c2:>8.3f} {one * 100:>6.1f}%"
                )
            print("  (restricted to k >= 11; k=4 is a different bucket and is excluded)")

        print()
        print("== one temperature per option count, vs ONE for the whole bucket ==")
        print("   (both fit on train, both applied to the same held-out rows)")
        print(
            f"   restricted to k >= {temp_min_k}, the option counts inside choice:11+. k=4 is a"
        )
        print("   different bucket; the shared fit never saw it, so it is not scored here.")
        print(
            f"{'k':>4} {'n':>5} {'acc':>7} {'ship':>6} {'T=1.0':>6} {'T shared':>9} "
            f"{'T per-k':>9} | {'ECE ship':>9} {'ECE shrd':>9} {'ECE per-k':>10} "
            f"| {'bias ship':>10} {'bias shrd':>10} {'bias per-k':>11}"
        )
        # Per-option-count rows, not per-bucket: a bucket-level table is what hid
        # this in the first place.
        for k in above:
            m = test_k == k
            print(
                f"{k:>4} {int(m.sum()):>5} {test_correct[m].mean():>7.3f} "
                f"{option_temp.applied[k]:>6.3f} {1.0:>6.3f} "
                f"{shared_temp.temperatures[k]:>9.3f} "
                f"{option_temp.temperatures[k]:>9.3f} | "
                f"{ece(test_shipped[m], test_correct[m]):>9.4f} "
                f"{ece(shared_temp_test[m], test_correct[m]):>9.4f} "
                f"{ece(option_temp_test[m], test_correct[m]):>10.4f} "
                f"| {test_shipped[m].mean() - test_correct[m].mean():>+10.4f} "
                f"{shared_temp_test[m].mean() - test_correct[m].mean():>+10.4f} "
                f"{option_temp_test[m].mean() - test_correct[m].mean():>+11.4f}"
            )
        if above:
            sel = np.isin(test_k, above)
            st_ship = summarize(test_shipped[sel], test_correct[sel])
            st_unity = summarize(unity_test[sel], test_correct[sel])
            st_shrd = summarize(shared_temp_test[sel], test_correct[sel])
            st_opt = summarize(option_temp_test[sel], test_correct[sel])
            st_log = summarize(temp_test[sel], test_correct[sel])
            st_head = summarize(test_scores[sel], test_correct[sel])
            print()
            print(f"  pooled over choice:11+ (k >= 11), n={int(sel.sum())}:")
            print(f"  {'':>27} {'ECE':>8} {'Brier':>8} {'bias':>9}")
            for name, st, sc in (
                ("p_top1 as published", st_ship, test_shipped),
                ("T=1.0 (no refit)", st_unity, unity_test),
                ("one temperature (fitted)", st_shrd, shared_temp_test),
                ("per-k temperature (fitted)", st_opt, option_temp_test),
                ("per-k logistic (7 params)", st_log, temp_test),
                ("escalation head (5543)", st_head, test_scores),
            ):
                print(
                    f"  {name:>27} {st['ece']:>8.4f} {st['brier']:>8.4f} "
                    f"{sc[sel].mean() - test_correct[sel].mean():>+9.4f}"
                )
            span = [option_temp.temperatures[k] for k in above]
            one = shared_temp.temperatures[above[0]]
            print(
                f"  per-option-count temperatures span {max(span) / min(span):.2f}x "
                f"({min(span):.3f} .. {max(span):.3f}); the checkpoint applied a single "
                f"{option_temp.applied[above[0]]:.3f} to all of them."
            )
            print(
                f"  ONE temperature ({one:.3f}) for the whole bucket vs one per option "
                f"count differ by {abs(st_shrd['ece'] - st_opt['ece']):.4f} ECE, and the "
                f"per-count values span only {max(span) / min(span):.2f}x"
            )
            print(
                "  -> a finer *temperature* grid is not the fix. A temperature is a slope\n"
                "     through the origin on the logit scale, so it cannot express the offset\n"
                "     the data wants. The affine form (per-k slope + one shared offset) is\n"
                "     scored above as 'per-k logistic'."
            )

    if "act_probability" in shipped and float(np.nanstd(shipped["act_probability"])) < 1e-9:
        print(f"  note: act_probability is constant at "
              f"{float(np.nanmean(shipped['act_probability'])):.4f} on the held-out rows, "
              f"so it cannot gate anything at any threshold.")

    # The deployable artifact: the fitted correction as a file you can load
    # without re-running any of this. It ships *two* fitted forms, because the
    # negative one is half the finding. A bare per-option-count temperature is a
    # power transform on the published probabilities, which on the logit scale is
    # a slope through the origin; it cannot express the offset the data wants, and
    # the fitted values barely differ. An affine recalibration -- one slope per
    # option count plus a single shared offset -- does repair it.
    if regime_temp is not None and option_temp is not None:
        offsets = sorted(set(round(v, 12) for v in regime_temp.intercepts.values()))
        temps_path = args.out / "temperatures.json"
        temps_path.write_text(
            json.dumps(
                {
                    "form": (
                        "affine on the logit scale, applied to the published top-1 "
                        "probability: p_cal = sigmoid(slope[k] * logit(p_top1) + offset)"
                    ),
                    "slope_by_options": {
                        str(k): regime_temp.slopes[k]
                        for k in sorted(regime_temp.slopes)
                    },
                    "shared_offset": float(offsets[0]),
                    "offset_is_shared": len(offsets) == 1,
                    "slope_spread": regime_temp.slope_spread,
                    "n_parameters": len(regime_temp.slopes) + 1,
                    "negative_control": {
                        "what": (
                            "a bare per-option-count temperature: a power transform "
                            "on the published probabilities, with no offset"
                        ),
                        "temperatures_by_options": {
                            str(k): option_temp.temperatures[k]
                            for k in sorted(option_temp.temperatures)
                        },
                        "temperature_spread": option_temp.temperature_spread,
                        "applied_by_checkpoint": {
                            str(k): option_temp.applied[k]
                            for k in sorted(option_temp.applied)
                        },
                        "verdict": (
                            "not a fix. The fitted temperatures barely differ from each "
                            "other, and this form does not beat one temperature for the "
                            "whole bucket on ECE. Recorded because it is the obvious "
                            "guess -- Laya's own repair mechanism, made finer."
                        ),
                    },
                    "fit": {
                        "source": str(args.records),
                        "objective": (
                            "per-option-count logistic slope on logit(p_top1), "
                            "one shared intercept"
                        ),
                        "l2": regime_temp.l2,
                        "train_rows": int(splits.train_idx.size),
                    },
                    "caution": (
                        "Fitted to this checkpoint and the clinc_oos/plus English "
                        "intents. Refit before trusting these on other traffic."
                    ),
                },
                indent=2,
            )
        )
        print(f"saved temperatures: {temps_path}")

    bundle = EscalationBundle(
        head=result.head,
        normalizer=result.normalizer,
        threshold=policy.threshold,
        alpha=args.alpha,
        metadata={
            "source": source,
            "checkpoint": args.checkpoint,
            "grouped_split": groups is not None,
            "n_items": None if groups is None else int(np.unique(groups).size),
        },
    )
    args.out.mkdir(parents=True, exist_ok=True)
    bundle.save(args.out / "escalation_head.pt")
    payload = report.to_dict()
    payload["scorer_comparison"] = comparison.to_dict()
    (args.out / "report.json").write_text(json.dumps(payload, indent=2))
    size_kb = (args.out / "escalation_head.pt").stat().st_size / 1024
    print()
    print(f"saved bundle     : {args.out / 'escalation_head.pt'} ({size_kb:.1f} KB)")
    print(f"saved report     : {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    torch.set_num_threads(min(8, torch.get_num_threads()))
    raise SystemExit(main())
