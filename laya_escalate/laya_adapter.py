"""Adapter that turns real Laya answers into :class:`DecisionRecord` objects.

Everything Laya-specific lives here, behind a defensive interface, so the rest
of the package never imports ``laya`` and stays testable without it.

A note on robustness
--------------------
The published examples show answer *values* (``answers["department"]["choice"]``,
``answers["urgency"]["score"]``, ``answers["churn_risk"]["noul"]``) but not the
exact field name carrying the full per-option distribution. Rather than guess,
this module probes a list of plausible keys and, failing all of them, reconstructs
a degenerate distribution from the reported confidence. It also exposes
:func:`probe_result` so you can print the real schema from your installed version
and then tighten :data:`DISTRIBUTION_KEYS` to exactly one entry.

If you are collecting the corpus that produces the published weights, run
``probe_result`` first. Guessing wrong silently degrades the features, because
``p_top1`` and the entropy terms are the head's primary inputs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .features import DecisionRecord, FeatureContext

# Field names that carry the option distribution. Verified against laya 0.3.5:
# choices and score questions return ``"probabilities"`` as an ordered dict;
# noul questions return only a scalar ``"noul"``.
DISTRIBUTION_KEYS: Tuple[str, ...] = (
    "probabilities",
    "probs",
    "distribution",
    "scores",
    "option_probs",
    "logits",
)

CHOICE_KEYS: Tuple[str, ...] = ("choice", "value", "label", "answer")
NOUL_KEYS: Tuple[str, ...] = ("noul", "probability", "p", "prob")
SCORE_KEYS: Tuple[str, ...] = ("score", "value", "rating")
CONFIDENCE_KEYS: Tuple[str, ...] = ("confidence", "conf", "probability")

SELECTED_KEYS: Tuple[str, ...] = ("selected", "chosen", "index", "argmax", "option_index")


def require_laya():
    """Import the ``laya`` package or raise a helpful error."""
    try:
        import laya  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'laya' package is not installed. Install it with `pip install laya` "
            "(and run with USE_TF=0 if transformers deadlocks on TensorFlow import)."
        ) from exc
    return laya


def load_agent(repo: str = "convaiinnovations/laya", subfolder: Optional[str] = None):
    """Load a single Laya checkpoint.

    ``subfolder`` should be ``None`` for English, ``"multilingual"`` or
    ``"typed-decisions"`` for the bundled checkpoints.
    """
    laya = require_laya()
    if subfolder:
        return laya.load(repo, subfolder=subfolder)
    return laya.load(repo)


def _first_key(payload: Mapping[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _extract_distribution(answer: Mapping[str, Any], qtype: str) -> Optional[np.ndarray]:
    """Pull a probability vector out of one answer payload, if one is present."""
    for key in DISTRIBUTION_KEYS:
        if key in answer:
            raw = answer[key]
            if isinstance(raw, Mapping):
                # e.g. {"billing": 0.91, "technical": 0.04, ...}
                values = np.asarray(list(raw.values()), dtype=np.float64)
            else:
                values = np.asarray(raw, dtype=np.float64).reshape(-1)
            if values.size >= 2 and np.all(np.isfinite(values)):
                if values.size == 2 and qtype == "noul":
                    # A noul answer given as [1-p, p] should become the binary
                    # distribution over the two outcomes.
                    values = np.clip(values, 1e-12, None)
                    return values / values.sum()
                if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-2):
                    return _softmax(values)
                return values / values.sum()
    return None


def _degenerate_from_confidence(
    answer: Mapping[str, Any], qtype: str, n_options: int
) -> np.ndarray:
    """Reconstruct a distribution when only a scalar confidence is available.

    For ``noul`` this is exact: the answer *is* the probability. For ``choice``
    it spreads the residual mass evenly, which preserves ``p_top1`` and the
    margin -- the two features that matter most -- while being honest that the
    finer shape is unavailable.
    """
    if qtype == "noul":
        for key in NOUL_KEYS:
            if key in answer:
                p = float(answer[key])
                p = min(max(p, 1e-9), 1 - 1e-9)
                return np.array([1.0 - p, p], dtype=np.float64)
        conf = _first_key(answer, CONFIDENCE_KEYS)
        if conf is not None:
            p = float(conf)
            if float(answer.get("choice_index", 0)) == 0:
                p = 1.0 - p
            p = min(max(p, 1e-9), 1 - 1e-9)
            return np.array([1.0 - p, p], dtype=np.float64)

    conf = _first_key(answer, CONFIDENCE_KEYS)
    if conf is None:
        raise KeyError(
            f"could not find a distribution or a confidence in answer keys "
            f"{sorted(answer.keys())}; run probe_result() to inspect the schema"
        )
    conf = float(conf)
    conf = min(max(conf, 1e-9), 1 - 1e-9)
    if n_options < 2:
        raise ValueError("cannot build a distribution with fewer than 2 options")
    rest = (1.0 - conf) / (n_options - 1)
    probs = np.full(n_options, rest, dtype=np.float64)
    probs[0] = conf
    return probs / probs.sum()


def extract_act_probability(answer: Mapping[str, Any]) -> Optional[float]:
    """Pull the act/escalate head's output out of an answer payload.

    Laya returns ``{"action": {"act_probability": 1.0}}``. Older or other
    System One implementations may flatten it, so both shapes are accepted.
    """
    action = answer.get("action")
    if isinstance(action, Mapping):
        value = _first_key(action, ("act_probability", "act", "probability"))
        if value is not None:
            return float(value)
        # Some builds softmax over [act, escalate]; take the act channel.
        probs = _first_key(action, ("probabilities", "probs"))
        if probs is not None:
            try:
                return float(np.asarray(probs, dtype=np.float64).reshape(-1)[0])
            except Exception:  # pragma: no cover - defensive
                return None
    value = _first_key(answer, ("act_probability",))
    return float(value) if value is not None else None


def answer_to_record(
    answer: Mapping[str, Any],
    qtype: str,
    correct: bool,
    n_options: int,
    context: Optional[FeatureContext] = None,
    head_max_len: Optional[int] = None,
    max_len: Optional[int] = None,
    state_tokens: Optional[int] = None,
    act_probability: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DecisionRecord:
    """Convert one Laya answer payload into a :class:`DecisionRecord`."""
    probs = _extract_distribution(answer, qtype)
    if probs is None:
        probs = _degenerate_from_confidence(answer, qtype, n_options)

    if act_probability is None:
        act_probability = extract_act_probability(answer)

    ctx = (context or FeatureContext()).resolved()
    return DecisionRecord(
        probs=probs,
        qtype=qtype,
        correct=correct,
        n_options=n_options,
        head_max_len=head_max_len or ctx.head_max_len,
        max_len=max_len or ctx.max_len,
        state_tokens=state_tokens,
        act_probability=act_probability,
        metadata=metadata or {},
    )


def probe_result(result: Mapping[str, Any], max_questions: int = 2) -> str:
    """Render the answer schema of a live ``predict()`` result.

    Run this once against your installed Laya version and use the output to
    confirm (or correct) :data:`DISTRIBUTION_KEYS`.
    """
    answers = result.get("answers", result)
    lines: List[str] = []
    if not isinstance(answers, Mapping):
        return f"unexpected answers payload of type {type(answers)!r}: {answers!r}"
    for i, (name, payload) in enumerate(answers.items()):
        if i >= max_questions:
            break
        lines.append(f"question {name!r}:")
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                rendered = repr(value)
                if len(rendered) > 120:
                    rendered = rendered[:117] + "..."
                lines.append(f"  {key!r}: {type(value).__name__} = {rendered}")
        else:
            lines.append(f"  {type(payload).__name__} = {payload!r}")
    lines.append("")
    lines.append(f"routing keys: {sorted(result.keys())}")
    return "\n".join(lines)


def build_choice_question(
    instructions: str,
    labels: Sequence[str],
    describe: bool = True,
) -> Dict[str, Any]:
    """Build a Laya choice question whose options are dataset label names.

    Set ``describe=False`` for large option sets: each option's description
    consumes part of the shared ``head_max_len`` budget, and snake_case label
    names are already self-describing (``card_arrival``). With 77 options the
    budget is roughly 2-3 tokens per option, so a description per label pushes
    the question past the header budget and Laya raises on it.
    """
    criteria = (
        {str(label): f"the text is about {label}" for label in labels}
        if describe
        else {str(label): str(label) for label in labels}
    )
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": criteria,
    }


def build_noul_question(instructions: str) -> Dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def build_score_question(instructions: str, levels: Sequence[str]) -> Dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def predict_choice(
    agent: Any,
    state: str,
    question_name: str,
    instructions: str,
    labels: Sequence[str],
    describe: bool = True,
) -> Mapping[str, Any]:
    """Run a single choice question and return that question's answer payload."""
    questions = {
        question_name: build_choice_question(instructions, labels, describe=describe)
    }
    result = agent.predict(state, questions)
    return result["answers"][question_name]


def prediction_accuracy(agent: Any) -> Optional[float]:
    """Read the fraction of decisions this agent has been allowed to act on.

    Laya raises its own reported accuracy on the episodes it acted on; where it
    is unavailable this returns ``None`` rather than inventing a number.
    """
    for attr in ("accuracy", "act_accuracy", "observed_accuracy"):
        value = getattr(agent, attr, None)
        if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            return float(value)
    return None


def cost_optimal_threshold(
    escalate_cost: float, wrong_act_cost: float
) -> float:
    """Act-threshold implied by the model's own cost model.

    Acting on a decision costs ``wrong_act_cost`` if it is wrong; escalating
    costs ``escalate_cost`` unconditionally. Acting is therefore optimal when
    ``(1 - p_correct) * wrong_act_cost <= escalate_cost``, i.e. when
    ``p_correct >= 1 - escalate_cost / wrong_act_cost``.

    Laya ships ``escalate: 0.5`` and ``cost_wrong_act: 3.0`` in
    ``agent.cfg``, giving a threshold of 5/6. Using the model's own stated costs
    means the operating point is derived rather than picked to flatter a table.
    """
    if wrong_act_cost <= 0:
        raise ValueError("wrong_act_cost must be positive")
    return 1.0 - escalate_cost / wrong_act_cost
