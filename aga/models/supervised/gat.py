
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    from dgl.nn import GATConv
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel


class GATModel(BaseGADModel):
    name = "gat"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3, dropout: float = 0.2,
                 num_heads: int = 4, **_):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.enc = nn.ModuleList()
        d = in_dim
        for i in range(n_layers):
            heads = num_heads if i < n_layers - 1 else 1
            out_dim = hidden * (num_heads if i < n_layers - 1 else 1)
            self.enc.append(
                GATConv(d, hidden, num_heads=heads, activation=F.elu,
                        feat_drop=dropout, allow_zero_in_degree=True)
            )
            d = out_dim
        self.emb_dim = hidden
        self.cls = nn.Linear(self.emb_dim, 2)

    def encode(self, g, x):
        h = x
        for i, layer in enumerate(self.enc):
            h = layer(g, h)
            if h.dim() == 3:
                h = h.flatten(1)
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