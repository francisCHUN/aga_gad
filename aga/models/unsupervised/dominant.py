
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    from dgl.nn import GraphConv
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel


class DominantModel(BaseGADModel):
    name = "dominant"
    supervised = False

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3, dropout: float = 0.2,
                 attr_weight: float = 1.0, stru_weight: float = 1.0, **_):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.attr_weight = attr_weight
        self.stru_weight = stru_weight

        self.enc = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.enc.append(GraphConv(d, hidden, activation=F.relu, allow_zero_in_degree=True))
            d = hidden
        self.emb_dim = hidden
        self.attr_dec = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))
        self.emb_out = nn.Linear(hidden, hidden)

    def encode(self, g, x):
        h = x
        for i, layer in enumerate(self.enc):
            h = layer(g, h)
            if i < self.n_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, g, x):
        h = self.encode(g, x)
        attr_recon = self.attr_dec(h)
        z = self.emb_out(h)
        return h, attr_recon, z

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        _, attr_recon, z = self.forward(g, x)
        attr_err = (attr_recon - x).pow(2).mean(dim=1)

        src, dst = g.edges()
        pos = (z[src] * z[dst]).sum(dim=1)
        pos_target = torch.ones_like(pos)
        n_neg = min(int(src.numel()), 4096)
        if n_neg > 0:
            neg_src = torch.randint(0, g.num_nodes(), (n_neg,), device=z.device)
            neg_dst = torch.randint(0, g.num_nodes(), (n_neg,), device=z.device)
            neg = (z[neg_src] * z[neg_dst]).sum(dim=1)
            stru_err_nodes = torch.zeros(g.num_nodes(), device=z.device)
            edge_err = (pos - pos_target).pow(2)
            stru_err_nodes.scatter_add_(0, src, edge_err)
            stru_err_nodes.scatter_add_(0, dst, edge_err)
            cnt = torch.zeros(g.num_nodes(), device=z.device)
            cnt.scatter_add_(0, src, torch.ones_like(edge_err))
            cnt.scatter_add_(0, dst, torch.ones_like(edge_err))
            stru_err_nodes = stru_err_nodes / cnt.clamp(min=1.0)
            neg_term = (neg - 0).pow(2).mean()
        else:
            stru_err_nodes = torch.zeros(g.num_nodes(), device=z.device)
            neg_term = torch.tensor(0.0, device=z.device)

        node_mask = mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=x.device)
        attr_loss = attr_err[node_mask].mean()
        stru_loss = stru_err_nodes.mean() if stru_err_nodes.numel() else torch.tensor(0.0, device=x.device)
        return attr_loss + stru_loss + 0.1 * neg_term

    @torch.no_grad()
    def score(self, g, x, attr_weight=None, stru_weight=None):
        _, attr_recon, z = self.forward(g, x)
        attr_err = (attr_recon - x).pow(2).mean(dim=1)
        src, dst = g.edges()
        edge_err = ((z[src] * z[dst]).sum(dim=1) - 1.0).pow(2)
        stru_err_nodes = torch.zeros(g.num_nodes(), device=z.device)
        cnt = torch.zeros(g.num_nodes(), device=z.device)
        stru_err_nodes.scatter_add_(0, src, edge_err)
        stru_err_nodes.scatter_add_(0, dst, edge_err)
        cnt.scatter_add_(0, src, torch.ones_like(edge_err))
        cnt.scatter_add_(0, dst, torch.ones_like(edge_err))
        stru_err_nodes = stru_err_nodes / cnt.clamp(min=1.0)
        a = attr_err / (attr_err.max().clamp(min=1e-6))
        s = stru_err_nodes / (stru_err_nodes.max().clamp(min=1e-6))
        aw = self.attr_weight if attr_weight is None else attr_weight
        sw = self.stru_weight if stru_weight is None else stru_weight
        return aw * a + sw * s