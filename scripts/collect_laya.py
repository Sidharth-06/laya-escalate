"""Collect real decision features by running a labeled corpus through Laya.

This is the step that produces the *published* weights. The synthetic simulator
validates the machinery; this script produces the evidence.

Example::

    python scripts/collect_laya.py \
        --dataset massive --split test \
        --text-column utterance --label-column intent \
        --checkpoint multilingual --limit 2000 \
        --out artifacts/laya_records.npz

Before trusting the output, probe the answer schema of your installed version::

    python scripts/collect_laya.py --dataset massive --limit 5 --probe

Option counts vary wildly across corpora (4 for AG News, 77 for Banking77, ~60
for MASSIVE), which is exactly the axis the escalation head needs to see.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_escalate import FeatureContext, featurize_many  # noqa: E402
from laya_escalate.laya_adapter import (  # noqa: E402
    answer_to_record,
    build_choice_question,
    load_agent,
    probe_result,
)


def resolve_text(row, column: str) -> str:
    value = row[column]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value)


def resolve_label(row, column: str) -> int:
    value = row[column]
    if isinstance(value, dict):
        value = value.get("label", value.get("int", 0))
    if isinstance(value, (list, tuple)):
        value = value[0]
    return int(value)


def label_names(features, column: str) -> List[str]:
    """Recover the string names of a ClassLabel column."""
    info = features[column]
    if hasattr(info, "names") and info.names:
        return [str(n) for n in info.names]
    return []


def count_tokens(agent, text: str) -> int:
    """Best-effort token count for the context-pressure feature."""
    for attr in ("tokenizer", "tok"):
        tok = getattr(agent, attr, None)
        if tok is not None and hasattr(tok, "encode"):
            try:
                return len(tok.encode(text))
            except Exception:  # pragma: no cover - tokenizer quirks
                return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--checkpoint", default="english",
                        choices=["english", "multilingual", "typed-decisions"])
    parser.add_argument("--instructions",
                        default="What is the topic or intent of this text?")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--probe", action="store_true",
                        help="print the answer schema for the first item and exit")
    parser.add_argument("--out", type=Path, default=Path("artifacts/laya_records.npz"))
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    names = label_names(ds.features, args.label_column)
    if not names:
        raise SystemExit(
            f"column {args.label_column!r} has no ClassLabel names; "
            "label names are required to build the option set"
        )

    subfolder = None if args.checkpoint == "english" else args.checkpoint
    agent = load_agent(subfolder=subfolder)
    ctx = FeatureContext(checkpoint=args.checkpoint)

    question = {"intent": build_choice_question(args.instructions, names)}

    records = []
    for i, row in enumerate(ds):
        text = resolve_text(row, args.text_column)
        gold = resolve_label(row, args.label_column)

        result = agent.predict(text, question)
        if args.probe and i == 0:
            print(probe_result(result))
            return 0

        answer = result["answers"]["intent"]

        predicted = None
        for key in ("choice", "value", "label"):
            if key in answer:
                predicted = answer[key]
                break
        if predicted is None:
            raise SystemExit(f"no choice field in answer keys {sorted(answer.keys())}")

        correct = str(predicted) == str(names[gold])
        records.append(
            answer_to_record(
                answer=answer,
                qtype="choice",
                correct=correct,
                n_options=len(names),
                context=ctx,
                state_tokens=count_tokens(agent, text),
                metadata={"gold": int(gold), "predicted": str(predicted)},
            )
        )
        if (i + 1) % 200 == 0:
            acc = np.mean([r.correct for r in records])
            print(f"  {i + 1}/{len(ds)}  running accuracy={acc:.4f}")

    x = featurize_many(records, ctx)
    correct = np.array([r.correct for r in records], dtype=bool)
    confidence = np.array([r.p_top1 for r in records], dtype=np.float64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=x,
        correct=correct,
        confidence=confidence,
        n_options=np.array([r.n_options for r in records]),
        qtype=np.array([r.qtype for r in records]),
    )

    print()
    print(f"dataset          : {args.dataset} ({args.split})")
    print(f"decisions        : {len(records)}")
    print(f"options          : {len(names)}")
    print(f"model accuracy   : {correct.mean():.4f}")
    print(f"mean p_top1      : {confidence.mean():.4f}")
    print(f"saved            : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
