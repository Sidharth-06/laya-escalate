"""Training the escalation head.

Small data, small model, one split discipline that matters: the conformal bound
is only valid if the threshold is fitted on decisions the head never saw. So the
data is split three ways -- train the head, fit the threshold, then measure on
held-out traffic -- and the helper here enforces that discipline rather than
leaving it to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .conformal import aurc
from .head import EscalationHead, Normalizer


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    hidden: int = 64
    dropout: float = 0.1
    patience: int = 40
    seed: int = 0
    init_scale: float = 1.0
    """Scales the final layer init. A small value starts the head near a
    constant prior, which is a far better starting point than a random sigmoid
    when the positive rate is skewed."""


@dataclass
class TrainResult:
    head: EscalationHead
    normalizer: Normalizer
    history: List[Dict[str, float]] = field(default_factory=list)
    best_val_aurc: float = float("nan")
    best_epoch: int = -1


@dataclass
class Splits:
    train_idx: np.ndarray
    calib_idx: np.ndarray
    test_idx: np.ndarray


def split_dataset(
    n: int,
    calib_fraction: float = 0.25,
    test_fraction: float = 0.25,
    seed: int = 0,
    groups: Optional[Sequence[object]] = None,
) -> Splits:
    """Three-way split: train / conformal calibration / held-out test.

    Pass ``groups`` to split by group rather than by row. A cardinality sweep
    asks the *same* item at several option counts, so a row-level split puts one
    item's text on both sides of the boundary. That breaks the thing the bound
    rests on: conformal risk control assumes the calibration and test decisions
    are exchangeable, and a head that has already seen an item during training is
    not exchangeable with one that has not. Grouping by item keeps every item
    wholly inside a single split.
    """
    if calib_fraction + test_fraction >= 1.0:
        raise ValueError("calibration + test fractions must leave room for training")
    rng = np.random.default_rng(seed)

    if groups is None:
        perm = rng.permutation(n)
        n_calib = int(round(n * calib_fraction))
        n_test = int(round(n * test_fraction))
        return Splits(
            train_idx=perm[: n - n_calib - n_test],
            calib_idx=perm[n - n_calib - n_test : n - n_test],
            test_idx=perm[n - n_test :],
        )

    groups = np.asarray(groups)
    if groups.size != n:
        raise ValueError(f"groups has {groups.size} entries, expected one per row ({n})")
    unique = np.unique(groups)
    if unique.size < 3:
        raise ValueError(f"need at least 3 groups to split three ways, got {unique.size}")
    shuffled = unique[rng.permutation(unique.size)]
    n_calib = int(round(unique.size * calib_fraction))
    n_test = int(round(unique.size * test_fraction))
    if unique.size - n_calib - n_test < 1:
        raise ValueError("group split leaves no groups for training")

    train_groups = shuffled[: unique.size - n_calib - n_test]
    calib_groups = shuffled[unique.size - n_calib - n_test : unique.size - n_test]
    test_groups = shuffled[unique.size - n_test :]
    return Splits(
        train_idx=np.flatnonzero(np.isin(groups, train_groups)),
        calib_idx=np.flatnonzero(np.isin(groups, calib_groups)),
        test_idx=np.flatnonzero(np.isin(groups, test_groups)),
    )


def _split_train_val(
    n: int,
    val_fraction: float,
    seed: int,
    groups: Optional[Sequence[object]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train/validation row split, by group when groups are supplied."""
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(n)
        n_val = max(int(round(n * val_fraction)), 1)
        return perm[n_val:], perm[:n_val]
    groups = np.asarray(groups)
    unique = np.unique(groups)
    n_val = max(int(round(unique.size * val_fraction)), 1)
    perm = rng.permutation(unique.size)
    val_groups = unique[perm[:n_val]]
    is_val = np.isin(groups, val_groups)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def train_head(
    x: np.ndarray,
    correct: Sequence[bool],
    cfg: Optional[TrainConfig] = None,
    groups: Optional[Sequence[object]] = None,
) -> TrainResult:
    """Fit the escalation head to predict decision correctness.

    Model selection uses validation AURC rather than validation loss: AURC is
    the metric the artifact is judged on, so selecting on it keeps the reported
    number honest.

    ``groups`` is passed through to the internal train/validation split. It does
    not touch the conformal guarantee (that is fixed by
    :func:`split_dataset`), but early stopping selected on items the head has
    already memorised is optimistic, so pass the same item ids here too.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(correct, dtype=np.float32).reshape(-1)
    if x.shape[0] != y.size:
        raise ValueError("x and correct must agree on the number of rows")

    normalizer = Normalizer.fit(x)
    xn = normalizer.transform(x)

    train_idx, val_idx = _split_train_val(x.shape[0], 0.15, cfg.seed, groups)
    x_train = torch.from_numpy(xn[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(xn[val_idx])
    y_val_np = y[val_idx].astype(bool)

    head = EscalationHead(hidden=cfg.hidden, dropout=cfg.dropout)
    with torch.no_grad():
        head.net[-1].weight.mul_(cfg.init_scale)
        head.net[-1].bias.mul_(cfg.init_scale)

    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best_aurc = float("inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []
    stale = 0

    n_train = x_train.shape[0]
    for epoch in range(cfg.epochs):
        head.train()
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            logits = head(x_train[idx])
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1

        val_scores = head.score(x[val_idx], normalizer, batch_size=4096)
        val_aurc = aurc(val_scores, y_val_np)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": epoch_loss / max(n_batches, 1),
                "val_aurc": float(val_aurc),
            }
        )

        if np.isfinite(val_aurc) and val_aurc < best_aurc - 1e-6:
            best_aurc = float(val_aurc)
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    return TrainResult(
        head=head,
        normalizer=normalizer,
        history=history,
        best_val_aurc=best_aurc,
        best_epoch=best_epoch,
    )
