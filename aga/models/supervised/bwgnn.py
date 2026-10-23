
from __future__ import annotations

import math

import numpy as np
import scipy
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.function as fn
    from torch.nn import init
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel


def calculate_theta2(d: int):
    thetas = []
    x = sympy.symbols("x")
    for i in range(d + 1):
        f = sympy.poly(
            (x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i))
        )
        coeff = f.all_coeffs()
        inv_coeff = []
        for j in range(d + 1):
            inv_coeff.append(float(coeff[d - j]))
        thetas.append(inv_coeff)
    return thetas


class PolyConv(nn.Module):

    def __init__(self, in_feats, out_feats, theta, activation=F.leaky_relu,
                 lin=False, bias=False):
        super().__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat - graph.ndata.pop("h") * D_invsqrt

        with graph.local_scope():
            D_invsqrt = (
                torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
                .unsqueeze(-1)
                .to(feat.device)
            )
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


class BWGNNModel(BaseGADModel):

    name = "bwgnn"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 64, d: int = 2,
                 num_classes: int = 2, graph=None, **_):
        super().__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList(
            [PolyConv(hidden, hidden, theta, lin=False) for theta in self.thetas]
        )
        self.linear = nn.Linear(in_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden * len(self.conv), hidden)
        self.linear4 = nn.Linear(hidden, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def _gather_graph(self, g):
        return g if g is not None else getattr(self, "g", None)

    def forward(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([in_feat.shape[0], 0], device=in_feat.device)
        for conv in self.conv:
            h0 = conv(g, h)
            h_final = torch.cat([h_final, h0], -1)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

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


class BWGNNHeteroModel(BaseGADModel):

    name = "bwgnn_hetero"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 64, d: int = 2,
                 num_classes: int = 2, graph=None, **_):
        super().__init__()
        self.thetas = calculate_theta2(d=d)
        self.h_feats = hidden
        self.conv = nn.ModuleList(
            [PolyConv(hidden, hidden, theta, lin=False) for theta in self.thetas]
        )
        self.linear = nn.Linear(in_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden * len(self.conv), hidden)
        self.linear4 = nn.Linear(hidden, num_classes)
        self.act = nn.LeakyReLU()
        self.g = graph

    def forward(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_all = []
        for relation in g.canonical_etypes:
            h_final = torch.zeros([in_feat.shape[0], 0], device=in_feat.device)
            for conv in self.conv:
                h0 = conv(g[relation], h)
                h_final = torch.cat([h_final, h0], -1)
            h_rel = self.linear3(h_final)
            h_all.append(h_rel)
        h_all = torch.stack(h_all).sum(0)
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        return h_all

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