"""Lightweight MLP: 30 -> 64 -> 32 -> 2 (ReLU, optional dropout), raw logits.

Trained with nn.CrossEntropyLoss on raw logits. That is mathematically the
same as the original repository's log_softmax + NLLLoss pairing, since
CrossEntropyLoss = NLLLoss(log_softmax(z)). Output index 1 = malignant.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int = 30, hidden_sizes=(64, 32), num_classes: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)          # raw logits, shape [batch, 2]


def build_model(cfg: dict) -> MLP:
    m = cfg["model"]
    return MLP(30, tuple(m["hidden_sizes"]), 2, float(m.get("dropout", 0.0)))
