"""The cardinality sweep: does Laya hide a within-bucket calibration defect?

The hypothesis
--------------
Laya applies one temperature per ``(question type, option-count bucket)``, where
the buckets are ``{2, 3-5, 6-10, 11+}``. Every question with 11 or more options
therefore shares a single temperature -- yet those questions range from 11 to
77+ options, and each option's share of the fixed ``head_max_len`` header budget
varies by 7x across that range. If accuracy and calibration both fall with
cardinality *inside* the bucket, no single temperature can repair it, and the
reported confidence is systematically wrong for some part of the bucket.

Laya's own loader says as much, on this checkpoint::

    RuntimeWarning: laya: this checkpoint ships temperatures outside [0.5, 5]
    which would distort confidence; clamping choice:11+=0.1006.
    Treat confidence from the affected buckets as uncalibrated.

Design
------
Take a labeled corpus with many short class names (Banking77, 77 intents). For
each item, build a choice question whose options are the true intent plus
``k - 1`` distractors sampled from the remaining intents. Sweep ``k``.

Two controls make the comparison fair:

* ``head_max_len`` is held **constant** across ``k``, so cardinality is not
  confounded with the header budget setting. Per-option starvation then varies
  naturally with ``k``, which is the documented failure mechanism.
* ``k = 4`` sits in the ``3-5`` bucket, whose temperature is *not* clamped, so
  it serves as a control for what a well-behaved bucket looks like.

Everything runs one question per call: a 20-question batch was observed to
exceed 8GB of VRAM and silently trigger Laya's sticky CPU fallback, which would
contaminate the latency numbers and slow the sweep by an order of magnitude.

Usage::

    USE_TF=0 python scripts/collect_sweep.py --items 200 --out artifacts/sweep.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_escalate import DecisionRecord, FeatureContext, featurize_many  # noqa: E402
from laya_escalate.laya_adapter import (  # noqa: E402
    answer_to_record,
    build_choice_question,
    cost_optimal_threshold,
    extract_act_probability,
    load_agent,
    probe_result,
)

INSTRUCTIONS = "What is the customer's intent in this message?"


def _char_ngrams(text: str, lo: int = 3, hi: int = 5) -> set:
    """Character n-grams of ``text`` with a space-padded boundary."""
    padded = f" {text.lower()} "
    return {
        padded[i:i + n]
        for n in range(lo, hi + 1)
        for i in range(len(padded) - n + 1)
    }


def char_ngram_vectors(
    labels: Sequence[str], lo: int = 3, hi: int = 5
) -> List[Dict[str, float]]:
    """TF-IDF weighted char-n-gram vectors, one sparse dict per label.

    Deliberately stdlib-only: the project does not depend on scikit-learn, and a
    char n-gram TF-IDF is short enough to own outright. Character n-grams rather
    than word tokens because intent names are snake_case compounds where the
    shared morphemes (``card_``, ``_amount``) are the signal.
    """
    grams = [_char_ngrams(label, lo, hi) for label in labels]
    df: Dict[str, int] = {}
    for g in grams:
        for term in g:
            df[term] = df.get(term, 0) + 1
    n = len(labels)
    # Smoothed idf, matching the usual convention so a term in one document
    # does not get infinite weight.
    idf = {t: 1.0 + math.log((1.0 + n) / (1.0 + c)) for t, c in df.items()}
    return [{t: idf[t] for t in g} for g in grams]


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity of two sparse weighted vectors."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb + 1e-12)


def similarity_ranking(
    gold_index: int,
    candidates: Sequence[int],
    vectors: Sequence[Dict[str, float]],
) -> List[int]:
    """``candidates`` ordered by descending name similarity to ``gold_index``.

    Ties break on index so the ranking -- and therefore the collected corpus --
    is fully deterministic.
    """
    gold = vectors[gold_index]
    scored = [
        (cosine_similarity(gold, vectors[i]), int(i))
        for i in candidates
        if int(i) != gold_index
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [i for _, i in scored]


def build_options(
    true_index: int,
    all_labels: Sequence[str],
    candidates: Sequence[int],
    k: int,
    rng: np.random.Generator,
    mode: str = "random",
) -> Tuple[List[str], List[int]]:
    """True label plus ``k - 1`` distractors, shuffled. Returns labels and indices.

    ``candidates`` excludes labels we never want as options (the out-of-scope
    class), while ``all_labels`` keeps the dataset's original indexing so the
    gold index still resolves correctly.

    ``mode`` decides *which* distractors:

    * ``random`` -- uniform over ``candidates`` (the shipped sweep's condition;
      callers pass a size-restricted pool, so this is uniform over that pool)
    * ``uniform`` -- uniform over the whole candidate pool
    * ``similar`` -- the most confusable ``k - 1``, with ``candidates`` already
      ordered most-similar-first by :func:`similarity_ranking`

    The shipped sweep used ``random``; its rng consumption is unchanged, so the
    published corpus stays reproducible from the documented command.
    """
    others = [i for i in candidates if i != true_index]
    if k - 1 > len(others):
        raise ValueError(f"k={k} needs {k - 1} distractors, pool has {len(others)}")
    if mode == "similar":
        chosen = list(range(k - 1))
    else:
        chosen = rng.choice(len(others), size=k - 1, replace=False)
    indices = [int(others[c]) for c in chosen] + [true_index]
    rng.shuffle(indices)
    return [all_labels[i] for i in indices], indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="clinc_oos",
                        help="parquet-hosted only: datasets>=4 dropped loading scripts")
    parser.add_argument("--subset", default="plus",
                        help="clinc_oos config: plus=151 intents, small/imbalance also exist")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="intent")
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--k", type=int, nargs="+", default=[4, 12, 20, 32, 64])
    parser.add_argument("--head-max-len", type=int, default=512,
                        help="held constant across k so cardinality is not confounded")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--checkpoint", default="english")
    parser.add_argument("--temperature-override", type=float, default=None,
                        help="force every temperature bucket to this value; the "
                             "checkpoint ships choice:11+=0.1006, which laya clamps to "
                             "0.5, so re-running at 1.0 isolates the temperature effect "
                             "from the cardinality effect")
    parser.add_argument("--describe", choices=("auto", "always", "never"), default="auto",
                        help="auto (default) describes options for k <= 10 and uses bare "
                             "label names above, which makes the small-k point a different "
                             "rendering as well as a different cardinality. 'never' removes "
                             "that confound and is what the k=4 counterfactual uses")
    parser.add_argument("--distractors", choices=("random", "uniform", "similar"),
                        default="random",
                        help="how distractors are chosen. 'random' (default) reproduces the "
                             "shipped sweep: uniform over the first max(k,64) in-scope "
                             "intents. 'uniform' draws from the whole in-scope pool, which "
                             "separates pool size from similarity. 'similar' takes the k-1 "
                             "intents whose names are most confusable with the gold one "
                             "(TF-IDF char-n-gram cosine) -- at fixed k this holds "
                             "cardinality constant and varies difficulty alone")
    parser.add_argument("--show-similar", type=int, default=0,
                        help="print the top distractors for N gold intents and exit; "
                             "audits that 'similar' really is selecting confusable names")
    parser.add_argument("--audit-options", action="store_true",
                        help="report achieved distractor similarity per k and exit, "
                             "loading no model: sizes the manipulation before paying "
                             "for a GPU pass")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/sweep.npz"))
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    info = ds.features[args.label_column]
    names = [str(n) for n in info.names]
    if not names:
        raise SystemExit(f"column {args.label_column!r} exposes no ClassLabel names")
    # clinc_oos appends an out-of-scope class. Keep the dataset's original
    # indexing (gold label indices must keep resolving) but never offer OOS as
    # an option and never score an OOS item -- otherwise the model could answer
    # "out of scope" for in-scope text and collect a free, meaningless point.
    oos_idx = {i for i, n in enumerate(names) if n.lower() in ("oos", "out_of_scope")}
    pool = [i for i in range(len(names)) if i not in oos_idx]
    print(f"dataset: {args.dataset}/{args.subset}  items={len(ds)}  "
          f"labels={len(names)} (excluding {len(oos_idx)} oos from the option pool)")

    # Similarity over intent names is the cheap proxy for confusability: it needs
    # no encoder and no extra corpus. It is a proxy -- the model sees the customer
    # message, not the option names alone -- so the collected corpus records the
    # achieved similarity per row, which lets the analysis check that the
    # manipulation actually moved anything.
    # Always built: the achieved similarity is recorded for *every* mode, so
    # 'random' / 'uniform' / 'similar' can be compared on the thing that actually
    # differs rather than on the name of the flag.
    vectors = char_ngram_vectors(names)

    if args.show_similar:
        probe_rng = np.random.default_rng(args.seed)
        probes = probe_rng.choice(pool, size=min(args.show_similar, len(pool)), replace=False)
        for probe_index in probes:
            probe_index = int(probe_index)
            ranked = similarity_ranking(probe_index, pool, vectors)[:5]
            scores = ", ".join(
                f"{names[i]}={cosine_similarity(vectors[probe_index], vectors[i]):.3f}"
                for i in ranked
            )
            print(f"  {names[probe_index]:<28} -> {scores}")
        return 0

    # Item selection lives above the model load so `--audit-options` can size the
    # manipulation without paying for a 1.7 GB checkpoint and a GPU pass.
    rng = np.random.default_rng(args.seed)
    # Draw candidates, then drop out-of-scope items so every scored decision has
    # a real intent as its gold label.
    in_scope = [
        int(i) for i in rng.permutation(len(ds))
        if int(ds[int(i)][args.label_column]) not in oos_idx
    ][: args.items]
    order = np.array(in_scope, dtype=int)
    print(f"in-scope items selected: {len(order)}")

    if args.audit_options:
        print()
        print(f"option audit (no model): distractors={args.distractors}")
        print(f"{'k':>5} {'sim/mean':>9} {'sim/p10':>8} {'sim/max':>8} {'sim/std':>8}")
        for k in args.k:
            means: List[float] = []
            maxes: List[float] = []
            for idx in order:
                gold = int(ds[int(idx)][args.label_column])
                candidates = (
                    similarity_ranking(gold, pool, vectors) if args.distractors == "similar"
                    else pool if args.distractors == "uniform"
                    else pool[: max(k, 64)]
                )
                _, option_indices = build_options(gold, names, candidates, k, rng,
                                                  mode=args.distractors)
                sims = [
                    cosine_similarity(vectors[gold], vectors[i])
                    for i in option_indices if i != gold
                ]
                means.append(float(np.mean(sims)))
                maxes.append(float(np.max(sims)))
            means_arr = np.asarray(means)
            print(f"{k:>5} {means_arr.mean():>9.3f} "
                  f"{np.percentile(means_arr, 10):>8.3f} "
                  f"{np.mean(maxes):>8.3f} {means_arr.std():>8.3f}")
        return 0

    # Warm up outside the timing loop, then pin the header budget.
    subfolder = None if args.checkpoint == "english" else args.checkpoint
    agent = load_agent(subfolder=subfolder)
    agent.cfg["head_max_len"] = args.head_max_len
    agent.cfg["max_len"] = args.max_len
    temperature_raw = dict(getattr(agent, "temperature_by_options_raw", {}) or {})
    if args.temperature_override is not None:
        t = float(args.temperature_override)
        agent.temperature_raw = [t, t, t]
        agent.temperature = [t, t, t]
        agent.temperature_by_options_raw = {k: t for k in agent.temperature_by_options}
        agent.temperature_by_options = {k: t for k in agent.temperature_by_options}
        print(f"temperature override: every bucket forced to {t:.4f}")
    print(f"head_max_len={agent.cfg['head_max_len']}  max_len={agent.cfg['max_len']}")
    print(f"activated checkpoints / device: {agent.device}  dtype: {agent.dtype}")

    ctx = FeatureContext(
        checkpoint=args.checkpoint,
        head_max_len=args.head_max_len,
        max_len=args.max_len,
    )

    records: List[DecisionRecord] = []
    meta: List[Dict[str, object]] = []
    latencies: List[float] = []

    # Warm-up call so the first timed decision is not paying compilation cost.
    agent.predict(
        str(ds[int(order[0])][args.text_column]),
        {"intent": build_choice_question(INSTRUCTIONS, [names[i] for i in pool[:2]],
                                         describe=False)},
    )

    total = args.items * len(args.k)
    done = 0
    for k_index, k in enumerate(args.k):
        describe = args.describe == "always" or (args.describe == "auto" and k <= 10)
        for position, idx in enumerate(order):
            row = ds[int(idx)]
            text = str(row[args.text_column])
            gold = int(row[args.label_column])
            if args.distractors == "similar":
                candidates = similarity_ranking(gold, pool, vectors)
            elif args.distractors == "uniform":
                candidates = pool
            else:
                candidates = pool[: max(k, 64)]
            options, option_indices = build_options(
                gold, names, candidates, k, rng, mode=args.distractors
            )
            gold_vec = vectors[gold]
            sims = [
                cosine_similarity(gold_vec, vectors[i])
                for i in option_indices if i != gold
            ]
            sim_mean, sim_max = float(np.mean(sims)), float(np.max(sims))

            question = {"intent": build_choice_question(INSTRUCTIONS, options,
                                                       describe=describe)}
            t0 = time.perf_counter()
            result = agent.predict(text, question)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            answer = result["answers"]["intent"]
            if args.probe and done == 0:
                print(probe_result(result))
                return 0

            predicted = str(answer["choice"])
            correct = predicted == names[gold]
            if predicted not in set(options):
                raise AssertionError(
                    f"model returned {predicted!r} outside the offered option set; "
                    "the option-key mapping is broken"
                )

            records.append(
                answer_to_record(
                    answer=answer,
                    qtype="choice",
                    correct=correct,
                    n_options=k,
                    context=ctx,
                    head_max_len=args.head_max_len,
                    max_len=args.max_len,
                    state_tokens=int(result.get("usage", {}).get("input_tokens", 0)),
                    act_probability=extract_act_probability(answer),
                    metadata={
                        "k": k,
                        "gold": gold,
                        "predicted": predicted,
                        "distractor_similarity_mean": sim_mean,
                        "distractor_similarity_max": sim_max,
                    },
                )
            )
            meta.append(
                {
                    "k": k,
                    "k_index": k_index,
                    "distractor_similarity_mean": sim_mean,
                    "distractor_similarity_max": sim_max,
                    # The same underlying item is asked at every k. Recorded so the
                    # split can be grouped by item: otherwise an item's five
                    # decisions straddle train and test, the head memorises it, and
                    # the exchangeability the conformal bound rests on is broken.
                    "item_index": position,
                    "gold": gold,
                    "predicted": predicted,
                    "correct": correct,
                    "confidence": float(answer.get("confidence", float("nan"))),
                    "act_probability": extract_act_probability(answer),
                }
            )
            done += 1
            if done % 100 == 0:
                acc_so_far = np.mean([r.correct for r in records])
                print(f"  {done}/{total}  running accuracy={acc_so_far:.4f}  "
                      f"median latency={np.median(latencies[-100:]):.0f} ms")

    x = featurize_many(records, ctx)
    correct = np.array([r.correct for r in records], dtype=bool)
    p_top1 = np.array([r.p_top1 for r in records], dtype=np.float64)
    confidence = np.array([float(m["confidence"]) for m in meta], dtype=np.float64)
    act = np.array(
        [float("nan") if m["act_probability"] is None else float(m["act_probability"])
         for m in meta],
        dtype=np.float64,
    )
    k_values = np.array([int(m["k"]) for m in meta], dtype=int)
    item_index = np.array([int(m["item_index"]) for m in meta], dtype=int)
    sim_mean = np.array([float(m["distractor_similarity_mean"]) for m in meta], dtype=np.float64)
    sim_max = np.array([float(m["distractor_similarity_max"]) for m in meta], dtype=np.float64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=x,
        correct=correct,
        p_top1=p_top1,
        confidence=confidence,
        act_probability=act,
        n_options=k_values,
        item_index=item_index,
        distractor_similarity_mean=sim_mean,
        distractor_similarity_max=sim_max,
        latency_ms=np.array(latencies, dtype=np.float64),
        probs=np.array([r.probs for r in records], dtype=object),
        qtype=np.array([r.qtype for r in records]),
    )

    # The sweep's whole point: does this bucket hold together?
    from laya_escalate.calibration import ece

    print()
    print(f"{'k':>5} {'bucket':>10} {'n':>5} {'acc':>7} {'ECE':>7} {'bias':>8} "
          f"{'sim/mean':>8} {'sim/max':>8} {'lat ms':>7}")
    for k in args.k:
        m = k_values == k
        if not m.any():
            continue
        bucket = "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        # sim_mean is the achieved distractor confusability. It replaces the old
        # (head_max_len - 16) / k column, which was a formula, not a measurement:
        # real option cost is ~7.4 tokens for every k here, so that column only
        # ever reported its own arithmetic.
        print(f"{k:>5} {'choice:' + bucket:>10} {m.sum():>5} {correct[m].mean():>7.3f} "
              f"{ece(confidence[m], correct[m]):>7.4f} "
              f"{confidence[m].mean() - correct[m].mean():>+8.4f} "
              f"{np.nanmean(sim_mean[m]):>8.3f} {np.nanmax(sim_max[m]) if np.isfinite(sim_max[m]).any() else float('nan'):>8.3f} "
              f"{np.median(np.array(latencies)[np.where(m)[0]]):>7.0f}")

    print()
    print("temperature actually applied per bucket:")
    for key, value in agent.temperature_by_options.items():
        print(f"  {key:>14}: {value:.4f}")
    print()
    print(f"saved: {args.out}")

    # Rank the option pool by size so the distractor draw is deterministic, and
    # report how many distinct intents the sweep actually exercised.
    print(f"distinct gold intents scored: {len({int(m['gold']) for m in meta})}")

    cfg_path = args.out.with_suffix(".meta.json")
    cfg_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "items": args.items,
                "k": args.k,
                "head_max_len": args.head_max_len,
                "max_len": args.max_len,
                "seed": args.seed,
                "distractors": args.distractors,
                "describe": args.describe,
                "temperature_by_options": agent.temperature_by_options,
                "temperature_by_options_raw": temperature_raw,
                "temperature_override": args.temperature_override,
                "cost_optimal_threshold": cost_optimal_threshold(
                    agent.cfg.get("act_costs", {}).get("escalate", 0.5),
                    agent.cfg.get("cost_wrong_act", 3.0),
                ),
            },
            indent=2,
        )
    )
    print(f"saved: {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
