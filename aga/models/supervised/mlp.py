
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseGADModel


class MLPModel(BaseGADModel):
    name = "mlp"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3, dropout: float = 0.2, **_):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers!r}")
        self.n_layers = n_layers
        self.dropout = dropout
        self.enc = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.enc.append(nn.Linear(d, hidden))
            d = hidden
        self.cls = nn.Linear(hidden, 2)

    def encode(self, g, x):
        h = x
        for i, layer in enumerate(self.enc):
            h = F.relu(layer(h))
            if i < self.n_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, g, x):
        h = self.encode(g, x)
        return self.cls(h)

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None, class_weight=None):
        logits = self.forward(g, x)
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        y = anomaly[node_mask].long()
        return F.cross_entropy(logits[node_mask], y, weight=class_weight)

    @torch.no_grad()
    def score(self, g, x):
        logits = self.forward(g, x)
        p = torch.softmax(logits, dim=1)[:, 1]
        return p / (p.max().clamp(min=1e-6))