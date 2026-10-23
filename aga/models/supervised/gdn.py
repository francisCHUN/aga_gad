
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel


class IntraAgg(nn.Module):

    def __init__(self, feat_dim: int, embed_dim: int, dropout: float = 0.0, **_):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.FloatTensor(2 * feat_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_ids, neigh_ids, x):
        device = x.device
        self_feats = x[node_ids]
        B = len(node_ids)
        agg = torch.zeros(B, self.feat_dim, device=device)
        lens = [len(neigh) for neigh in neigh_ids]
        if any(lens):
            rows = torch.repeat_interleave(
                torch.arange(B, device=device),
                torch.as_tensor(lens, device=device))
            flat = torch.as_tensor(
                [v for neigh in neigh_ids for v in neigh],
                dtype=torch.long, device=device)
            sums = torch.zeros(B, self.feat_dim, device=device) \
                .index_add_(0, rows, x[flat])
            agg = sums / torch.as_tensor(lens, dtype=torch.float,
                                         device=device).clamp(min=1).unsqueeze(1)
        cat = torch.cat((self_feats, agg), dim=1)
        out = F.relu(cat.mm(self.weight))
        return self.dropout(out)


class InterAgg(nn.Module):

    def __init__(self, feat_dim: int, embed_dim: int, n_relation: int = 3, dropout: float = 0.6, **_):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.n_relation = n_relation
        self.softmax = nn.Softmax(dim=-1)
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.intra_aggs = nn.ModuleList(
            [IntraAgg(feat_dim, embed_dim, dropout) for _ in range(n_relation)]
        )
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim * n_relation + feat_dim, embed_dim)
        )
        nn.init.xavier_uniform_(self.weight)
        self.pos_vector = None
        self.neg_vector = None
        self.sample_rates = [1.0, 0.5, 0.3][:n_relation] if n_relation <= 3 else [1.0] * n_relation
        self._adj_lists = None
        self._adj_key = None

    def _get_adj_lists(self, g):
        key = (id(g), g.num_nodes(), g.num_edges())
        if self._adj_key != key or self._adj_lists is None:
            succ: List[List[int]] = [[] for _ in range(g.num_nodes())]
            src, dst = g.edges()
            for s, d in zip(src.tolist(), dst.tolist()):
                succ[s].append(d)
            self._adj_lists = succ
            self._adj_key = key
        return self._adj_lists

    @staticmethod
    def _softmax_with_temperature(z, t=1.0):
        z = z / t
        z = z - z.max(dim=-1, keepdim=True).values
        ex = torch.exp(z)
        return ex / ex.sum(dim=-1, keepdim=True)

    def update_label_vector(self, x, pos_idx, neg_idx):
        x_pos = x[pos_idx]
        x_neg = x[neg_idx]
        if x_pos.shape[0] == 0 or x_neg.shape[0] == 0:
            return
        if self.pos_vector is None:
            self.pos_vector = x_pos.mean(dim=0, keepdim=True).detach()
            self.neg_vector = x_neg.mean(dim=0, keepdim=True).detach()
        else:
            cosine_pos = self.cos(self.pos_vector, x_pos)
            cosine_neg = self.cos(self.neg_vector, x_neg)
            w_pos = self._softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
            w_neg = self._softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
            self.pos_vector = torch.mm(w_pos, x_pos).detach()
            self.neg_vector = torch.mm(w_neg, x_neg).detach()

    def forward(self, g, node_ids, x):
        neigh_lists = self._sample_neighbors(g, node_ids)
        intra_feats = []
        for i, agg in enumerate(self.intra_aggs):
            r_neigh = neigh_lists[i]
            intra_feats.append(agg(g, node_ids, r_neigh, x))
        self_feats = x[node_ids]
        cat_feats = torch.cat([self_feats] + intra_feats, dim=1)
        combined = F.relu(cat_feats.mm(self.weight).t())
        return combined.t()

    def _sample_neighbors(self, g, node_ids, k=3):
        adj_lists = self._get_adj_lists(g)
        out = [[] for _ in range(self.n_relation)]
        for n in node_ids:
            succ = adj_lists[int(n)]
            for i in range(self.n_relation):
                rate = self.sample_rates[i]
                if len(succ) == 0:
                    out[i].append([])
                    continue
                keep = max(1, int(round(len(succ) * rate)))
                keep = min(keep, len(succ))
                sel = random.sample(succ, keep)
                out[i].append(sel)
        return out

    def label_aware_scores(self, x, node_ids):
        if self.pos_vector is None or self.neg_vector is None:
            return torch.zeros(len(node_ids), device=x.device)
        feats = x[node_ids]
        cos_pos = self.cos(self.pos_vector, feats)
        cos_neg = self.cos(self.neg_vector, feats)
        return (cos_pos - cos_neg).view(-1)


class GDNModel(BaseGADModel):

    name = "gdn"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.6,
                 n_relation: int = 3, fn_weight: float = 1.0, batch_size: int = 256,
                 neighbor_tax_cap: int = 8, layers: int = None, n_layer: int = None, **_):
        super().__init__()
        if n_layer is not None:
            n_layers = int(n_layer)
        if layers is not None:
            n_layers = int(layers)
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.fn_weight = fn_weight
        self.batch_size = batch_size
        self.neighbor_tax_cap = neighbor_tax_cap

        self.inter = InterAgg(in_dim, hidden, n_relation=n_relation, dropout=dropout)
        self.cls = nn.Linear(hidden, 2)

    def forward(self, g, x):
        return self.inter(g, list(range(g.num_nodes())), x)

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        device = x.device
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=device)
        )
        train_idx = torch.where(node_mask)[0]
        pos_idx = train_idx[anomaly[train_idx] == 1]
        neg_idx = train_idx[anomaly[train_idx] == 0]

        self.inter.update_label_vector(x.detach(), pos_idx, neg_idx)

        h = self.forward(g, x)
        logits = self.cls(h)

        if pos_idx.numel() > 0 and neg_idx.numel() > 0:
            n = min(pos_idx.numel(), neg_idx.numel())
            p = pos_idx[torch.randperm(pos_idx.numel())[:n]]
            nsel = neg_idx[torch.randperm(neg_idx.numel())[:n]]
            sel = torch.cat([p, nsel]).sort()[0]
            cls_loss = F.cross_entropy(logits[sel], anomaly[sel].long())
        else:
            cls_loss = F.cross_entropy(logits[train_idx], anomaly[train_idx].long())

        fn_loss = self._fn_loss(g, x, pos_idx)
        return cls_loss + self.fn_weight * fn_loss

    def _fn_loss(self, g, x, pos_idx):
        if pos_idx.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        dim = x.shape[1]
        n_dim = min(self.neighbor_tax_cap, dim)
        non_grad_idx = torch.randperm(dim)[:n_dim]
        pos_nodes = pos_idx[torch.randperm(pos_idx.numel())[: self.batch_size]]
        if pos_nodes.numel() == 0:
            return torch.tensor(0.0, device=x.device)

        x_pos = x[pos_nodes]
        n = pos_nodes.numel()
        N = g.num_nodes()
        adj_lists = self.inter._get_adj_lists(g)

        succ_lists = [adj_lists[int(nd)] for nd in pos_nodes.tolist()]
        lens = torch.as_tensor([len(s) for s in succ_lists], device=x.device)
        pos_proto = torch.zeros_like(x_pos)
        flat = [v for s in succ_lists for v in s]
        if flat:
            rows = torch.repeat_interleave(
                torch.arange(n, device=x.device), lens)
            flat_t = torch.as_tensor(flat, dtype=torch.long, device=x.device)
            sums = torch.zeros_like(x_pos).index_add_(0, rows, x[flat_t])
            pos_proto = sums / lens.clamp(min=1).unsqueeze(1)

        neg_list = torch.randint(0, N, (n,), device=x.device).tolist()
        for i, nd in enumerate(pos_nodes.tolist()):
            succ_set = adj_lists[nd]
            tries = 0
            while tries < 8 and (neg_list[i] == nd or neg_list[i] in succ_set):
                neg_list[i] = random.randint(0, N - 1)
                tries += 1
        neg_proto = x[torch.as_tensor(neg_list, dtype=torch.long, device=x.device)]

        sl = non_grad_idx
        logp = F.log_softmax(x_pos[:, sl], dim=-1)
        pos_t = F.softmax(pos_proto[:, sl], dim=-1)
        neg_t = F.softmax(neg_proto[:, sl], dim=-1)
        return self.inter.KLDiv(logp, pos_t) - 0.5 * self.inter.KLDiv(logp, neg_t)

    @torch.no_grad()
    def score(self, g, x):
        if self.inter.pos_vector is not None and self.inter.neg_vector is not None:
            s = self.inter.label_aware_scores(x, list(range(g.num_nodes())))
            s = s - s.min()
            return s / (s.max().clamp(min=1e-6))
        logits = self.forward(g, x)
        p = torch.softmax(self.cls(logits), dim=1)[:, 1]
        return p / (p.max().clamp(min=1e-6))