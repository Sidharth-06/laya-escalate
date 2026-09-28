"""A simulator of a System One decision model's failure modes.

Why a simulator
---------------
The escalation head's value depends on a specific empirical claim: that a
decision model's top-line confidence is not a sufficient statistic for its
correctness, and that label-space geometry (how many options, how many tokens
each received) carries independent signal.

That claim is documented in the Laya model card -- overconfidence at 0.466 mean
ECE before refitting, refitting "one temperature per (question type, option
count)", 77-option questions collapsing to 0.425 accuracy because each label got
~3-4 tokens. But reproducing it end-to-end requires downloading ~800MB of
weights and running a labeled corpus through the model.

So this module builds a *synthetic* stand-in with the same qualitative
pathologies, which lets the head, the conformal guarantee and the full
evaluation harness be developed and tested without a GPU. It is explicitly a
simulator: numbers from it validate the *machinery*, never the claim about Laya
itself. The real numbers come from ``laya_escalate.laya_adapter``.

How the simulation is structured
--------------------------------
Two quantities are generated independently-but-correlated, which is the whole
point:

1. A belief distribution, peaked according to latent difficulty and the per
   option token budget. This is what the model "thinks".
2. Correctness, drawn from a Bernoulli whose rate depends on difficulty and
   cardinality.

Because correctness is not a deterministic function of the belief distribution,
confidence is an *imperfect* predictor of correctness -- exactly the situation
in which a learned head with extra features beats thresholding confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .features import DecisionRecord, FeatureContext, featurize_many

# Option counts that appear in the datasets the Laya card benchmarks against.
# 2 = noul, 4 = AG News, 6 = DAIR Emotion, 77 = Banking77, 60 = MASSIVE intents.
DEFAULT_OPTION_COUNTS = (2, 3, 4, 6, 8, 10, 16, 20, 32, 60, 77)


@dataclass
class SyntheticConfig:
    n_items: int = 4000
    option_counts: Sequence[int] = DEFAULT_OPTION_COUNTS
    checkpoint: str = "english"
    seed: int = 0
    # Difficulty prior: higher values mean the corpus is harder overall.
    difficulty_a: float = 2.0
    difficulty_b: float = 2.0
    # Scales the sharpness of the belief distribution.
    sharpness: float = 6.0
    # Correctness logit: intercept, sensitivity to the model's own confidence,
    # penalty for option starvation, and irreducible per-item noise.
    correctness_intercept: float = 1.2
    correctness_confidence_slope: float = 1.1
    correctness_starvation_slope: float = 2.5
    correctness_noise: float = 0.4


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate(cfg: Optional[SyntheticConfig] = None) -> Tuple[List[DecisionRecord], np.ndarray]:
    """Generate synthetic decisions.

    Returns
    -------
    (records, latent_difficulty)
        ``records`` are :class:`DecisionRecord` instances carrying ground-truth
        correctness; ``latent_difficulty`` is exposed for analysis only and must
        never be fed to the head (it is not observable at serving time).
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    ctx = FeatureContext(checkpoint=cfg.checkpoint).resolved()

    counts = np.asarray(cfg.option_counts, dtype=int)
    n_counts = len(counts)

    records: List[DecisionRecord] = []
    difficulties = np.zeros(cfg.n_items, dtype=np.float64)

    qtypes = np.array(["choice", "score", "noul"])
    # noul questions are always binary; the rest spread across the option counts.
    qtype_probs = np.array([0.78, 0.12, 0.10])

    for i in range(cfg.n_items):
        qtype = str(rng.choice(qtypes, p=qtype_probs))
        if qtype == "noul":
            n_options = 2
        else:
            # Skew toward smaller label spaces, as real corpora are.
            weights = 1.0 / (counts**0.6)
            weights = weights / weights.sum()
            n_options = int(rng.choice(counts, p=weights))

        difficulty = float(rng.beta(cfg.difficulty_a, cfg.difficulty_b))
        difficulties[i] = difficulty

        # Per-option token budget, exactly as the model card describes it.
        head_max_len = ctx.head_max_len or 192
        tokens_per_option = max((head_max_len - 16) / n_options, 0.0)
        # A saturating capability curve: past ~24 tokens per option the model is
        # no longer starved, below ~6 it degrades fast.
        capability = tokens_per_option / (tokens_per_option + 12.0)

        # --- 1. belief distribution -------------------------------------
        #
        # Note the *weak* coupling to ``capability``. This is deliberate and
        # follows the Laya card: the model "stays confident while being wrong"
        # on starved label spaces (Khmer: 0.000 accuracy at 0.952 confidence),
        # which is why the authors note that confidence gating cannot save you.
        # If confidence itself collapsed with cardinality, top-line confidence
        # would already encode the cardinality signal and the learned head would
        # have nothing to add -- which is exactly the failure mode an earlier
        # version of this simulator exhibited.
        sharp = cfg.sharpness * (1.0 - 0.55 * difficulty) * (0.85 + 0.15 * capability)
        sharp = max(sharp, 0.15)
        logits = rng.normal(0.0, 1.0, size=n_options) * sharp
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        pred = int(np.argmax(probs))

        # --- 2. correctness (correlated with, but not determined by, belief)
        #
        # Correctness tracks the model's own confidence, which is what gives the
        # risk-coverage curve its realistic shape (very confident decisions are
        # mostly right). The starvation penalty shifts that mapping as a
        # function of the *label space*: at identical confidence, a decision made
        # over 77 starved options is far less likely to be correct than one made
        # over 4 well-fed options. That shift is invisible to a confidence
        # threshold and is precisely the signal the head is meant to exploit.
        p_top1 = float(np.max(probs))
        own_confidence_logit = float(np.log(p_top1 / max(1.0 - p_top1, 1e-9)))
        p_correct_logit = (
            cfg.correctness_intercept
            + cfg.correctness_confidence_slope * own_confidence_logit
            - cfg.correctness_starvation_slope * (1.0 - capability)
            + rng.normal(0.0, cfg.correctness_noise)
        )
        p_correct = float(_sigmoid(np.array(p_correct_logit)))

        correct = bool(rng.random() < p_correct)
        if not correct:
            # Ground truth is some other option, so argmax is wrong.
            others = [j for j in range(n_options) if j != pred]
            label = int(rng.choice(others))
        else:
            label = pred

        records.append(
            DecisionRecord(
                probs=probs,
                qtype=qtype,
                correct=(label == pred),
                n_options=n_options,
                head_max_len=head_max_len,
                max_len=ctx.max_len,
                state_tokens=int(rng.integers(0, (ctx.max_len or 512))),
                metadata={
                    "latent_difficulty": difficulty,
                    "label": label,
                    # The generator's true correctness probability. This is an
                    # oracle used only to measure how much ranking headroom
                    # exists; it is never available at serving time and must not
                    # be fed to the head.
                    "p_correct_oracle": p_correct,
                },
            )
        )

    return records, difficulties


def simulate_features(
    cfg: Optional[SyntheticConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper returning ``(X, wrong, p_top1)``.

    ``wrong`` is the boolean ground truth needed to fit the conformal bound;
    ``p_top1`` is the raw confidence baseline the head must beat.
    """
    records, _ = simulate(cfg)
    ctx = FeatureContext(checkpoint=(cfg or SyntheticConfig()).checkpoint)
    x = featurize_many(records, ctx)
    wrong = np.array([not r.correct for r in records], dtype=bool)
    p_top1 = np.array([r.p_top1 for r in records], dtype=np.float64)
    return x, wrong, p_top1
