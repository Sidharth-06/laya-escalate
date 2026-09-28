"""Measure the *actual* rendered option-token counts per ``k``.

Why this exists
---------------
The sweep reports ``tokens/option`` as ``(head_max_len - 16) / k`` -- a nominal
budget, computed from the truncation rule, not the tokens a question really
consumes. At ``k=64`` that prints ``7.8`` because ``(512-16)/64 = 7.75``, which
looks like starvation regardless of what the tokenizer does.

But Laya only truncates options when they actually overflow the header budget.
The collector asks ``k >= 11`` questions with one criterion per label rendered as
``"label: label"`` (``describe=False`` -> ``criteria = {label: label}``, and
``render_options`` renders ``"%s: %s" % (k, v)`` for any non-empty value). Those
are short. So the budget may never bind, in which case the collapse from 0.990 to
0.797 is difficulty discriminating among many similar short labels, not token
starvation -- and the mechanism paragraph changes completely.

This script replays the collector's exact option draw (same seed, same dataset,
same item order) and runs ``laya.common.build_sequence`` with the real tokenizer
to report, per ``k``:

* real per-option token counts (pre- and post-cap, and post-overflow-truncation)
* whether the overflow branch fires at all
* the resulting sequence length and how much room is left for the state text
* whether the impossible-marker condition (Laya raising) is reachable

CPU only; no model weights are loaded.

Usage::

    USE_TF=0 python scripts/measure_option_tokens.py --items 2000 --k 4 12 20 32 48 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# Must precede every transformers/datasets import: the HF calls hang on this host
# when the hub is unreachable, and the checkpoint and dataset are already cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INSTRUCTIONS = "What is the customer's intent in this message?"

# Every option in the sweep is one of ~150 label names rendered as "label: label",
# so the raw tokenizer calls are almost entirely redundant. Caching makes a
# full-scale replay cheap instead of quadratic.
_NTOK_CACHE: Dict[str, int] = {}


def ntok(tok, s: str) -> int:
    """Token count for ``s``, memoised on the string."""
    cached = _NTOK_CACHE.get(s)
    if cached is None:
        cached = len(tok(s, add_special_tokens=False)["input_ids"])
        _NTOK_CACHE[s] = cached
    return cached


def build_options(
    true_index: int,
    all_labels: Sequence[str],
    pool: Sequence[int],
    k: int,
    rng: np.random.Generator,
) -> List[str]:
    """Verbatim copy of the collector's draw so the token counts describe the same runs."""
    others = [i for i in pool if i != true_index]
    if k - 1 > len(others):
        raise ValueError(f"k={k} needs {k - 1} distractors, pool has {len(others)}")
    chosen = rng.choice(len(others), size=k - 1, replace=False)
    indices = [true_index] + [others[c] for c in chosen]
    rng.shuffle(indices)
    return [all_labels[i] for i in indices]


def measure(
    tok,
    state: str,
    options: Sequence[str],
    describe: bool,
    max_len: int,
    head_max_len: int,
) -> Dict[str, float]:
    """Mirror ``laya.common.build_sequence`` exactly and report every budget step.

    The lib's own arithmetic is duplicated on purpose: the point of this script is
    to report the intermediate quantities (where the budget binds, how much text it
    drops, whether the state survives) that ``build_sequence`` returns nowhere. The
    final ``seq_len`` is cross-checked against the real ``build_sequence`` below, so
    a divergence in this reimplementation cannot go unnoticed.
    """
    from laya.common import build_sequence, render_options

    crit = (
        {str(o): f"the text is about {o}" for o in options}
        if describe
        else {str(o): str(o) for o in options}
    )
    q = {"t": "choice", "ins": INSTRUCTIONS, "crit": crit}
    rendered = render_options(q)
    mask_id = tok.mask_token_id

    # Step 1: per-option counts as build_sequence computes them (first 48 tokens).
    raw_counts = [ntok(tok, " " + r.replace(tok.mask_token, " ")) for r in rendered]
    capped = [min(c, 48) for c in raw_counts]

    # Step 2: does the shared header budget bind? Each option also carries a mask token.
    opt_ids = [[mask_id] + [0] * c for c in capped]
    pre_budget = head_max_len - sum(len(o) for o in opt_ids)
    overflow = pre_budget < 16
    if overflow:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        kept = [min(1 + c, per) for c in capped]           # mask + text, then [:per]
        opt_budget = head_max_len - sum(kept)
    else:
        per = None
        kept = [1 + c for c in capped]
        opt_budget = pre_budget

    # Step 3: what is left for the instruction header, and how much of it survives.
    head_full = ntok(tok, "%s question: %s" % (q["t"], q["ins"]))
    head_used = min(head_full, max(8, opt_budget))

    # Step 4: how much room remains for the state, against what the state needs.
    body = 1 + head_used + 1 + sum(kept) + 1               # cls + head + sep + options + sep
    room = max(0, max_len - body - 1)
    st = tok(state.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]

    seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
    if len(seq) != min(body + min(len(st), room) + 1, max_len):
        raise AssertionError(
            f"reimplementation disagrees with build_sequence: {len(seq)} != "
            f"{min(body + min(len(st), room) + 1, max_len)}"
        )

    capped_total = sum(capped)
    kept_text_total = sum(k - 1 for k in kept)
    return {
        "n_options": len(rendered),
        "raw_tokens_per_option_mean": float(np.mean(raw_counts)),
        "raw_tokens_per_option_max": float(np.max(raw_counts)),
        "capped_tokens_per_option_mean": float(np.mean(capped)),
        "option_tokens_total": float(capped_total),
        "option_tokens_total_after_trunc": float(kept_text_total),
        "option_tokens_cut_fraction": (
            0.0 if capped_total == 0 else 1.0 - kept_text_total / capped_total
        ),
        "overflowed": bool(overflow),
        "per_option_cap_applied": per,
        "head_tokens_full": float(head_full),
        "head_tokens_used": float(head_used),
        "head_truncated": bool(head_used < head_full),
        "seq_len": len(seq),
        "state_room": float(room),
        "state_tokens": float(len(st)),
        "state_truncated": bool(len(st) > room),
        "markers": len(markers),
        "marker_shortfall": len(rendered) - len(markers),
        "nominal_tokens_per_option": (head_max_len - 16) / len(rendered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="clinc_oos")
    parser.add_argument("--subset", default="plus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="intent")
    parser.add_argument("--items", type=int, default=2000,
                        help="must match the collector for the option sets to line up")
    parser.add_argument("--k", type=int, nargs="+", default=[4, 12, 20, 32, 48, 64])
    parser.add_argument("--head-max-len", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--encoder", default=None,
                        help="local tokenizer dir; defaults to the cached Laya checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/option_tokens.json"))
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    if args.encoder is None:
        from huggingface_hub import snapshot_download

        root = snapshot_download("convaiinnovations/laya", local_files_only=True)
        args.encoder = str(Path(root) / "tokenizer")
    tok = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True)

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    info = ds.features[args.label_column]
    names = [str(n) for n in info.names]
    oos_idx = {i for i, n in enumerate(names) if n.lower() in ("oos", "out_of_scope")}
    pool = [i for i in range(len(names)) if i not in oos_idx]

    # Replay the collector's rng consumption exactly: the permutation first, then
    # one build_options draw per (k, item) in k-major order.
    rng = np.random.default_rng(args.seed)
    in_scope = [
        int(i) for i in rng.permutation(len(ds))
        if int(ds[int(i)][args.label_column]) not in oos_idx
    ][: args.items]
    print(f"replaying {len(in_scope)} items x k={args.k}")

    rows: List[Dict[str, float]] = []
    for k in args.k:
        for position, idx in enumerate(in_scope):
            row = ds[int(idx)]
            text = str(row[args.text_column])
            gold = int(row[args.label_column])
            options = build_options(gold, names, pool[: max(k, 64)], k, rng)
            describe = k <= 10  # collector's --describe auto
            stat = measure(tok, text, options, describe, args.max_len, args.head_max_len)
            stat["k"] = k
            stat["item_index"] = position
            rows.append(stat)

    # Aggregate per k.
    print()
    header = (f"{'k':>4} {'describe':>8} {'opt/raw':>8} {'opt/kept':>9} {'sum':>6} "
              f"{'overflow':>9} {'per':>5} {'cut%':>6} {'seq':>5} {'st/room':>8} {'short':>6}")
    print(header)
    print("-" * len(header))
    summary = {}
    for k in args.k:
        sel = [r for r in rows if r["k"] == k]
        overflowed = sum(1 for r in sel if r["overflowed"])
        per = next((r["per_option_cap_applied"] for r in sel if r["overflowed"]), None)
        short = max(r["marker_shortfall"] for r in sel)
        cut = float(np.mean([r["option_tokens_cut_fraction"] for r in sel]))
        print(f"{k:>4} {str(k <= 10):>8} "
              f"{np.mean([r['raw_tokens_per_option_mean'] for r in sel]):>8.2f} "
              f"{np.mean([r['option_tokens_total_after_trunc'] for r in sel]) / k:>9.2f} "
              f"{np.mean([r['option_tokens_total'] for r in sel]):>6.0f} "
              f"{f'{overflowed}/{len(sel)}':>9} "
              f"{(str(per) if per is not None else '-'):>5} "
              f"{cut * 100:>6.1f} "
              f"{np.mean([r['seq_len'] for r in sel]):>5.0f} "
              f"{np.mean([r['state_tokens'] for r in sel]):>3.0f}/"
              f"{np.mean([r['state_room'] for r in sel]):<4.0f} {short:>6}")
        summary[str(k)] = {
            "describe": k <= 10,
            "raw_tokens_per_option_mean": float(np.mean([r["raw_tokens_per_option_mean"] for r in sel])),
            "raw_tokens_per_option_max": float(np.max([r["raw_tokens_per_option_max"] for r in sel])),
            "capped_tokens_per_option_mean": float(np.mean([r["capped_tokens_per_option_mean"] for r in sel])),
            "kept_tokens_per_option_mean": float(np.mean([r["option_tokens_total_after_trunc"] for r in sel]) / k),
            "option_tokens_total_mean": float(np.mean([r["option_tokens_total"] for r in sel])),
            "option_tokens_total_after_trunc_mean": float(np.mean([r["option_tokens_total_after_trunc"] for r in sel])),
            "option_tokens_cut_fraction_mean": cut,
            "fraction_questions_overflowing": overflowed / len(sel),
            "fraction_head_truncated": float(np.mean([r["head_truncated"] for r in sel])),
            "fraction_state_truncated": float(np.mean([r["state_truncated"] for r in sel])),
            "per_option_cap_applied": per,
            "seq_len_mean": float(np.mean([r["seq_len"] for r in sel])),
            "state_room_mean": float(np.mean([r["state_room"] for r in sel])),
            "state_tokens_mean": float(np.mean([r["state_tokens"] for r in sel])),
            "state_tokens_max": float(np.max([r["state_tokens"] for r in sel])),
            "nominal_tokens_per_option": (args.head_max_len - 16) / k,
            "max_marker_shortfall": short,
            "option_tokens_total_max": float(np.max([r["option_tokens_total"] for r in sel])),
            "option_tokens_total_p95": float(np.percentile([r["option_tokens_total"] for r in sel], 95)),
            "raw_tokens_per_option_p99": float(np.percentile([r["raw_tokens_per_option_max"] for r in sel], 99)),
        }

    print()
    print("Truncation verdict per k:")
    for k in args.k:
        s = summary[str(k)]
        if s["fraction_questions_overflowing"] == 0:
            verdict = ("NO truncation -- every option reaches the model in full "
                       f"({s['option_tokens_total_mean']:.0f} option tok vs "
                       f"{args.head_max_len - 16} budget)")
        else:
            verdict = ("overflow branch fires on "
                       f"{s['fraction_questions_overflowing']*100:.0f}% of questions; "
                       f"per-option cap {s['per_option_cap_applied']}, "
                       f"{s['option_tokens_cut_fraction_mean']*100:.1f}% of option tokens dropped, "
                       f"header truncated on {s['fraction_head_truncated']*100:.0f}%, "
                       f"state truncated on {s['fraction_state_truncated']*100:.0f}%")
        print(f"  k={k:>3}: {verdict}")

    print()
    print("Describe=True counterfactual (the k<=10 rendering) at high k:")
    for k in [20, 32, 64]:
        r2 = np.random.default_rng(args.seed + 1)
        sel_gold = [int(ds[int(i)][args.label_column]) for i in in_scope[:50]]
        stats = [measure(tok, str(ds[int(in_scope[j])][args.text_column]),
                         build_options(g, names, pool[: max(k, 64)], k, r2),
                         True, args.max_len, args.head_max_len)
                 for j, g in enumerate(sel_gold)]
        frac = np.mean([s["overflowed"] for s in stats])
        print(f"  k={k:>3}: overflows on {frac*100:.0f}% of questions, "
              f"mean option tokens {np.mean([s['option_tokens_total'] for s in stats]):.0f} "
              f"(kept {np.mean([s['option_tokens_total_after_trunc'] for s in stats]):.0f}), "
              f"mean seq len {np.mean([s['seq_len'] for s in stats]):.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print()
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
