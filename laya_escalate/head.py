"""The escalation head: predicts whether a decision is safe to act on.

This is deliberately tiny. The signal that matters (label-space geometry,
confidence shape) is already summarised by the featurizer, so the head only has
to learn the interaction -- chiefly that high confidence is worth much less when
the label space is large. (On this corpus that degradation is difficulty
discriminating among many similar short labels, not option truncation; see
``scripts/measure_option_tokens.py``.)

Output is a single logit: ``P(this decision is correct)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_NAMES, N_FEATURES


@dataclass
class Normalizer:
    """Feature standardization fitted on the training split."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Normalizer":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float64),
                   std=np.asarray(payload["std"], dtype=np.float64))


class EscalationHead(nn.Module):
    """Small MLP scoring ``P(decision is correct)`` from decision features.

    Parameters
    ----------
    n_features:
        Must match :data:`laya_escalate.features.N_FEATURES`.
    hidden:
        Width of the two hidden layers. 64 keeps the whole head in the low tens
        of kilobytes, which is the entire point: the artifact is shippable as
        weights measured in KB, not hundreds of MB.
    dropout:
        Applied between hidden layers.
    """

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch,)``."""
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def score(self, x: np.ndarray, normalizer: Normalizer, batch_size: int = 8192) -> np.ndarray:
        """Return ``P(correct)`` for a feature matrix, in eval mode."""
        self.eval()
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        xn = normalizer.transform(x)
        out = []
        for start in range(0, xn.shape[0], batch_size):
            chunk = torch.from_numpy(xn[start : start + batch_size])
            out.append(torch.sigmoid(self.forward(chunk)).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class EscalationBundle:
    """Everything needed to serve a policy: model + normalizer + threshold."""

    head: EscalationHead
    normalizer: Normalizer
    threshold: float
    alpha: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.head.score(x, self.normalizer)

    def decide(self, x: np.ndarray) -> np.ndarray:
        """Boolean array: True where the decision may be acted on.

        Where False, the caller should escalate the decision to the fallback
        model rather than executing it.
        """
        return self.score(x) >= self.threshold

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.head.state_dict(),
                "n_features": self.head.n_features,
                "feature_names": list(FEATURE_NAMES),
                "hidden": int(self.head.net[1].out_features),
                "dropout": float(self.head.net[3].p),
                "normalizer": self.normalizer.to_dict(),
                "threshold": self.threshold,
                "alpha": self.alpha,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        hidden: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> "EscalationBundle":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        # Reconstruct the architecture from what was saved. The state dict is the
        # source of truth for ``hidden`` so bundles written before these keys
        # existed still load; an explicit argument overrides both.
        state = payload["state_dict"]
        resolved_hidden = int(
            hidden if hidden is not None else payload.get("hidden", state["net.1.weight"].shape[0])
        )
        resolved_dropout = float(
            dropout if dropout is not None else payload.get("dropout", 0.0)
        )
        head = EscalationHead(
            n_features=payload["n_features"], hidden=resolved_hidden, dropout=resolved_dropout
        )
        head.load_state_dict(payload["state_dict"])
        head.eval()
        return cls(
            head=head,
            normalizer=Normalizer.from_dict(payload["normalizer"]),
            threshold=float(payload["threshold"]),
            alpha=payload.get("alpha"),
            metadata=payload.get("metadata") or {},
        )
