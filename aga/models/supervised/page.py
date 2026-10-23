
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel

log = logging.getLogger("aga.models.page")


class _PageGCN(nn.Module):

    def __init__(self, in_ft: int, out_ft: int):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        self.bias = nn.Parameter(torch.FloatTensor(out_ft))
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, adj):
        out = torch.mm(adj, self.fc(x))
        return self.act(out + self.bias)


def _random_structure_perturbation(
    adj_row: torch.Tensor, perturb_ratio: float, generator: torch.Generator
) -> torch.Tensor:
    adj_row = adj_row.clone()
    n = adj_row.shape[0]
    k = int(n * perturb_ratio)
    if k <= 0:
        return adj_row
    connected = (adj_row > 0).nonzero(as_tuple=True)[0]
    disconnected = (adj_row == 0).nonzero(as_tuple=True)[0]
    if len(connected) > 0 and k > 0:
        perm = torch.randperm(len(connected), generator=generator)[:k]
        adj_row[connected[perm]] = 0
    if len(disconnected) > 0 and k > 0:
        perm = torch.randperm(len(disconnected), generator=generator)[:k]
        adj_row[disconnected[perm]] = 1
    return adj_row


def _loss_anomaly_severity(emb_normal, emb_fake_abn, emb_adv_abn, margin=1.0):
    emb_normal = F.normalize(emb_normal, p=2, dim=1)
    emb_fake_abn = F.normalize(emb_fake_abn, p=2, dim=1)
    emb_adv_abn = F.normalize(emb_adv_abn, p=2, dim=1)
    d_pos = torch.norm(emb_fake_abn - emb_normal, dim=1)
    d_neg = torch.norm(emb_adv_abn - emb_normal, dim=1)
    loss_pos = d_pos.mean()
    loss_neg = torch.clamp(margin + d_pos - d_neg, min=0).mean()
    return loss_pos + loss_neg


class PAGEModel(BaseGADModel):

    name = "page"
    supervised = True

    def __init__(
        self,
        in_dim: int,
        hidden: int = 512,
        dropout: float = 0.5,
        noise_mean: float = 0.02,
        noise_var: float = 0.01,
        perturb_ratio: float = 0.02,
        margin: float = 1.0,
        w_rec: float = 1.0,
        w_bin: float = 1.0,
        w_page: float = 1.0,
        rng_seed: int = 42,
        **_,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.dropout = nn.Dropout(p=float(dropout))
        self.noise_mean = float(noise_mean)
        self.noise_var = float(noise_var)
        self.perturb_ratio = float(perturb_ratio)
        self.margin = float(margin)
        self.w_rec, self.w_bin, self.w_page = float(w_rec), float(w_bin), float(w_page)
        self.rng_seed = int(rng_seed)

        self.gcn1 = _PageGCN(in_dim, self.hidden)
        self.gcn2 = _PageGCN(self.hidden, self.hidden)
        h = self.hidden
        self.fc1 = nn.Linear(h, max(h // 2, 1), bias=False)
        self.fc2 = nn.Linear(max(h // 2, 1), max(h // 4, 1), bias=False)
        self.fc3 = nn.Linear(max(h // 4, 1), 1, bias=False)
        self.act = nn.ReLU()
        self.fc4 = nn.Linear(h, h, bias=False)

        self._adj_key = None
        self._adj: Optional[torch.Tensor] = None

    def _get_adj(self, g) -> torch.Tensor:
        key = (id(g), g.num_nodes(), g.num_edges())
        if self._adj_key != key or self._adj is None:
            n = g.num_nodes()
            src, dst = g.edges()
            src, dst = src.long(), dst.long()
            a = torch.zeros(n, n, device=src.device if src.is_cuda else "cpu")
            a[src, dst] = 1.0
            a[dst, src] = 1.0
            a.fill_diagonal_(1.0)
            deg = a.sum(dim=1)
            d_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
            d_mat = torch.diag(d_inv_sqrt)
            adj = d_mat @ a @ d_mat
            self._adj = adj
            self._adj_key = key
        return self._adj

    def _embed(self, x, adj):
        h1 = self.gcn1(x, adj)
        h1 = self.dropout(h1)
        emb = self.gcn2(h1, adj)
        return emb

    def _severity(self, emb):
        f = self.act(self.fc1(emb))
        f = self.act(self.fc2(f))
        f = self.dropout(f)
        return self.fc3(f).squeeze(-1)

    def _neigh_agg(self, emb):
        return self.act(self.fc4(emb))

    def forward(self, g, x, **_):
        adj = self._get_adj(g)
        emb = self._embed(x, adj)
        return self._severity(emb)

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None, class_weight=None):
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(
                g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        device = x.device
        adj = self._get_adj(g)

        if anomaly is not None:
            normal_idx = torch.where(node_mask & (anomaly.long() == 0))[0]
        else:
            normal_idx = torch.where(node_mask)[0]
        if normal_idx.numel() == 0:
            emb = self._embed(x, adj)
            return F.mse_loss(self.act(self.fc4(emb)), emb)

        gen = torch.Generator(device="cpu").manual_seed(self.rng_seed)
        self.rng_seed += 1

        emb = self._embed(x, adj)

        neigh_adj = adj[normal_idx, :]
        emb_con1 = self._neigh_agg(neigh_adj @ emb)

        noise = torch.randn(emb.size(), device=device)
        noise = noise * self.noise_var + self.noise_mean
        emb_noisy = emb + noise
        emb_con_fake = self._neigh_agg(neigh_adj @ emb_noisy)

        perturbed = neigh_adj.clone()
        n = adj.shape[0]
        k = int(n * self.perturb_ratio)
        if k > 0:
            conn = neigh_adj > 0
            disc = ~conn
            scores = torch.rand(neigh_adj.shape, generator=gen).to(neigh_adj.device)
            connected_scores = torch.where(conn, scores, torch.full_like(scores, float("inf")))
            row_conn = conn.sum(dim=1)
            k_drop = torch.minimum(torch.full_like(row_conn, k), row_conn)
            finite = torch.isfinite(connected_scores)
            sorted_idx = connected_scores.argsort(dim=1)
            rank = torch.empty_like(scores, dtype=torch.long)
            ar = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).expand_as(scores)
            rank.scatter_(1, sorted_idx, ar + 1)
            drop = finite & (rank <= k_drop.unsqueeze(1))
            perturbed[drop] = 0.0

            absent_scores = torch.where(disc, scores, torch.full_like(scores, float("inf")))
            finite_add = torch.isfinite(absent_scores)
            sorted_idx_a = absent_scores.argsort(dim=1)
            rank_a = torch.empty_like(rank)
            rank_a.scatter_(1, sorted_idx_a, ar + 1)
            row_disc = disc.sum(dim=1)
            k_add = torch.minimum(torch.full_like(row_disc, k), row_disc)
            add = finite_add & (rank_a <= k_add.unsqueeze(1))
            perturbed[add] = 1.0
        emb_con_adv = self._neigh_agg(perturbed @ emb_noisy)

        loss_page = _loss_anomaly_severity(
            emb_con1, emb_con_fake, emb_con_adv, margin=self.margin
        )

        s_clean = self._severity(emb_con1)
        s_fake = self._severity(emb_con_fake)
        s_adv = self._severity(emb_con_adv)
        logits = torch.cat([s_clean, s_fake, s_adv])
        labels = torch.cat([
            torch.zeros_like(s_clean),
            torch.ones_like(s_fake),
            torch.ones_like(s_adv),
        ])
        loss_bin = F.binary_cross_entropy_with_logits(
            logits, labels,
            pos_weight=torch.tensor(2.0, device=device),
        )

        loss_rec = F.mse_loss(emb_con_fake, emb_con1) + F.mse_loss(emb_con_adv, emb_con1)

        return self.w_rec * loss_rec + self.w_bin * loss_bin + self.w_page * loss_page

    @torch.no_grad()
    def score(self, g, x):
        s = self.forward(g, x)
        lo, hi = s.min(), s.max()
        return ((s - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)