
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    from dgl.nn import GraphConv
    import dgl.function as fn
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel


class AvgReadout(nn.Module):

    def forward(self, h):
        return torch.mean(h, dim=1)


class MaxReadout(nn.Module):
    def forward(self, h):
        return torch.max(h, dim=1)[0]


class MinReadout(nn.Module):
    def forward(self, h):
        return torch.min(h, dim=1)[0]


class WSReadout(nn.Module):

    def forward(self, h, h_last=None):
        if h_last is not None:
            w = torch.sigmoid((h * h_last).sum(dim=-1, keepdim=True))
        else:
            w = torch.softmax(h.sum(dim=-1, keepdim=True), dim=1)
        return (h * w).sum(dim=1)


class Discriminator(nn.Module):

    def __init__(self, n_h: int, negsamp_round: int = 1):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.negsamp_round = negsamp_round
        for m in self.modules():
            self._weights_init(m)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = [self.f_k(h_pl, c)]
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))
        return torch.cat(tuple(scs), dim=1)


def rwr_subgraphs(g: "dgl.DGLGraph", node_ids, subgraph_size: int, restart_prob: float = 0.8):
    src, dst = g.edges()
    src = src.tolist()
    dst = dst.tolist()
    adj = {}
    for s, d in zip(src, dst):
        adj.setdefault(s, []).append(d)

    out = []
    for u in node_ids:
        u = int(u)
        if u not in adj or len(adj[u]) == 0:
            out.append(np.array([u], dtype=np.int64))
            continue
        visited = set()
        cur = u
        steps = subgraph_size * 4
        rng = np.random.default_rng(abs(hash((u, id(out))) % (2 ** 32)))
        for _ in range(steps):
            visited.add(cur)
            if len(visited) >= subgraph_size:
                break
            if rng.random() < restart_prob:
                cur = u
            else:
                neighbors = adj.get(cur, [u])
                cur = int(rng.choice(neighbors))
        neigh = list(visited - {u})
        if len(neigh) >= subgraph_size - 1:
            neigh = neigh[: subgraph_size - 1]
        while len(neigh) < subgraph_size - 1 and neigh:
            neigh.append(rng.choice(neigh))
        nodes = neigh + [u]
        out.append(np.asarray(nodes[:subgraph_size], dtype=np.int64))
    return out


class CoLAModel(BaseGADModel):

    name = "cola"
    supervised = False

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3, dropout: float = 0.2,
                 readout: str = "avg", negsamp_round: int = 1, subgraph_size: int = 4,
                 batch_size: int = 256, **_):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.hidden = hidden
        self.read_mode = readout
        self.negsamp_round = negsamp_round
        self.subgraph_size = subgraph_size
        self.batch_size = batch_size

        self.gcn = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.gcn.append(GraphConv(d, hidden, activation=F.relu, allow_zero_in_degree=True))
            d = hidden

        if readout == "max":
            self.read = MaxReadout()
        elif readout == "min":
            self.read = MinReadout()
        elif readout == "avg":
            self.read = AvgReadout()
        elif readout == "weighted_sum":
            self.read = WSReadout()
        else:
            raise ValueError(f"unknown readout {readout!r}")

        self.disc = Discriminator(hidden, negsamp_round)

    def encode(self, g, x):
        h = x
        for i, layer in enumerate(self.gcn):
            h = layer(g, h)
            if i < self.n_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def _node_forward(self, h_all, sub_nodes):
        B = len(sub_nodes)
        H = h_all.shape[1]
        stack = np.full((B, self.subgraph_size), -1, dtype=np.int64)
        for i, s in enumerate(sub_nodes):
            n = min(len(s), self.subgraph_size)
            stack[i, :n] = s[:n]
            stack[i, n:] = s[-1]
        node_t = torch.from_numpy(stack).to(h_all.device)
        h_sub = h_all[node_t]

        if self.read_mode == "weighted_sum":
            h_mv = h_sub[:, -1, :]
            c = self.read(h_sub[:, :-1, :], h_sub[:, -2:-1, :])
        else:
            c = self.read(h_sub[:, :-1, :])
            h_mv = h_sub[:, -1, :]
        return c, h_mv

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        train_nodes = torch.where(node_mask)[0]
        if train_nodes.numel() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        h_all = self.encode(g, x)

        idx = train_nodes[torch.randperm(train_nodes.numel())[: self.batch_size]]
        sub_nodes = rwr_subgraphs(g, idx.tolist(), self.subgraph_size)
        c, h_pl = self._node_forward(h_all, sub_nodes)
        logits = self.disc(c, h_pl)

        lbl = torch.cat(
            [torch.ones(c.shape[0], 1), torch.zeros(c.shape[0], self.negsamp_round)], dim=1
        ).to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, lbl)

    @torch.no_grad()
    def score(self, g, x):
        h_all = self.encode(g, x)
        N = g.num_nodes()
        scores = torch.zeros(N, device=x.device)
        node_ids = list(range(N))
        bsz = max(self.batch_size, 256)
        for start in range(0, N, bsz):
            batch = node_ids[start : start + bsz]
            sub_nodes = rwr_subgraphs(g, batch, self.subgraph_size)
            c, h_pl = self._node_forward(h_all, sub_nodes)
            pos_logit = self.disc(c, h_pl)[:, 0]
            s = 1.0 - torch.sigmoid(pos_logit)
            for k, nid in enumerate(batch):
                scores[nid] = s[k]
        return scores / (scores.max().clamp(min=1e-6))