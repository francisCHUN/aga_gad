
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel


def hinge_loss(label: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.clamp(1.0 - label * score, min=0.0))


class RelationAware(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.d_liner = nn.Linear(input_dim, output_dim)
        self.f_liner = nn.Linear(3 * output_dim, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src - dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze(-1)
        return self.tanh(score)


class H2FDetectorLayer(nn.Module):

    def __init__(self, input_dim, output_dim, head, relation_aware, dropout, if_sum=False):
        super().__init__()
        self.hd = output_dim
        self.head = head
        self.if_sum = if_sum
        self.relation_aware = relation_aware
        self.w_liner = nn.Linear(input_dim, output_dim * head)
        self.atten = nn.Linear(2 * self.hd, 1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, g, h):
        n = g.num_nodes()
        src, dst = g.edges()
        sign = torch.sign(self.relation_aware(h[src], h[dst]))
        hp = self.w_liner(h)
        sf = sign.view(-1, 1, 1) * hp[src].view(-1, self.head, self.hd)
        z = torch.cat([sf, hp[dst].view(-1, self.head, self.hd)], dim=-1)
        alpha = self.leakyrelu(self.atten(z))

        E = src.numel()
        head_idx = torch.arange(self.head, device=h.device)
        seg = (dst.unsqueeze(1) * self.head + head_idx.view(1, -1)).reshape(-1)
        a = alpha.reshape(-1)
        nseg = n * self.head
        amax = torch.full((nseg,), float("-inf"), device=h.device)
        amax.scatter_reduce_(0, seg, a, reduce="amax", include_self=True)
        e = torch.exp(a - amax[seg])
        denom = torch.zeros(nseg, device=h.device)
        denom.index_add_(0, seg, e)
        anorm = (e / denom[seg].clamp(min=1e-16)).view(E, self.head, 1)

        out = torch.zeros(n, self.head, self.hd, device=h.device)
        out.index_add_(0, dst, anorm * sf)
        if not self.if_sum:
            return out.view(n, self.head * self.hd)
        return out.sum(dim=1)


class MultiRelationH2FDetectorLayer(nn.Module):

    def __init__(self, input_dim, output_dim, head, dropout, if_sum=False,
                 relations=("homo",), n_class=None):
        super().__init__()
        self.relations = list(relations)
        self.n_relation = len(self.relations)
        if not if_sum:
            self.liner = nn.Linear(self.n_relation * output_dim * head, output_dim * head)
        else:
            self.liner = nn.Linear(self.n_relation * output_dim, output_dim)
        self.relation_aware = RelationAware(input_dim, output_dim * head, dropout)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        for e in self.relations:
            self.minelayers[e] = H2FDetectorLayer(input_dim, output_dim, head,
                                                  self.relation_aware, dropout, if_sum)

    def forward(self, g, h):
        hs = [self.minelayers[e](g, h) for e in self.relations]
        h = torch.cat(hs, dim=1)
        h = self.dropout(h)
        h = self.liner(h)
        return h

    def edge_scores(self, g, h):
        with g.local_scope():
            g.ndata["feat"] = h
            src, dst = g.edges()
            return self.relation_aware(h[src], h[dst])


class H2FDetectorModel(BaseGADModel):

    name = "h2fdetector"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 32, n_layers: int = 2, dropout: float = 0.5,
                 head: int = 4, num_classes: int = 2, gamma1: float = 1.0, gamma2: float = 1.0,
                 n_layer: int = None, layers: int = None, **_):
        super().__init__()
        if n_layer is not None:
            n_layers = int(n_layer)
        if layers is not None:
            n_layers = int(layers)
        self.n_layer = n_layers
        self.intra_dim = hidden
        self.n_class = num_classes
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.head = head

        self.mine_layers = nn.ModuleList()
        if n_layers == 1:
            self.mine_layers.append(
                MultiRelationH2FDetectorLayer(in_dim, num_classes, head, dropout, if_sum=True)
            )
        else:
            self.mine_layers.append(
                MultiRelationH2FDetectorLayer(in_dim, hidden, head, dropout)
            )
            for _ in range(1, n_layers - 1):
                self.mine_layers.append(
                    MultiRelationH2FDetectorLayer(hidden * head, hidden, head, dropout)
                )
            self.mine_layers.append(
                MultiRelationH2FDetectorLayer(hidden * head, num_classes, head, dropout, if_sum=True)
            )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, g, x):
        feats = x.float()
        h = self.mine_layers[0](g, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers) - 1):
                h = self.mine_layers[i](g, h)
                h = self.relu(h)
                h = self.dropout(h)
            h = self.mine_layers[-1](g, h)
        return h

    def _edge_loss(self, g, h, anomaly):
        src, dst = g.edges()
        a = anomaly
        edge_label = torch.where((a[src] == 1) & (a[dst] == 1), 1, -1)
        pos_idx = torch.where(edge_label == 1)[0]
        neg_idx = torch.where(edge_label == -1)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return torch.tensor(0.0, device=h.device)
        n = min(pos_idx.numel(), neg_idx.numel(), 2048)
        pos_sample = pos_idx[torch.randperm(pos_idx.numel())[:n]]
        neg_sample = neg_idx[torch.randperm(neg_idx.numel())[:n]]
        idx = torch.cat([pos_sample, neg_sample])
        idx = idx.sort()[0]
        score = self.mine_layers[0].edge_scores(g, h)
        return hinge_loss(edge_label[idx].float(), score[idx])

    def _prototype_loss(self, h, train_mask, anomaly):
        train_h = h[train_mask]
        train_label = anomaly[train_mask]
        pos = train_label == 1
        neg = train_label == 0
        if pos.sum() == 0 or neg.sum() == 0:
            return torch.tensor(0.0, device=h.device)
        n = min(int(pos.sum()), int(neg.sum()))
        pos_idx = torch.where(pos)[0]
        neg_idx = torch.where(neg)[0]
        pos_sample = pos_idx[torch.randperm(pos_idx.numel())[:n]]
        neg_sample = neg_idx[torch.randperm(neg_idx.numel())[:n]]
        node_index = torch.cat([neg_sample, pos_sample])
        node_index = node_index.sort()[0]
        pos_proto = train_h[pos].mean(dim=0, keepdim=True)
        neg_proto = train_h[neg].mean(dim=0, keepdim=True)
        th = train_h[node_index]
        pos_exp = pos_proto.expand(th.shape)
        neg_exp = neg_proto.expand(th.shape)
        diff_pos = -F.pairwise_distance(th, pos_exp)
        diff_neg = -F.pairwise_distance(th, neg_exp)
        diff = torch.cat([diff_neg.view(-1, 1), diff_pos.view(-1, 1)], dim=1)
        return F.cross_entropy(diff, train_label[node_index])

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        h_raw = self.forward(g, x)
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        anomaly = anomaly.long()
        train_label = anomaly[node_mask]
        pos = train_label == 1
        neg = train_label == 0
        if pos.sum() > 0 and neg.sum() > 0:
            g_train_idx = torch.where(node_mask)[0]
            pos_idx = torch.where(pos)[0]
            neg_idx = torch.where(neg)[0]
            n = min(pos_idx.numel(), neg_idx.numel())
            p = g_train_idx[pos_idx[torch.randperm(pos_idx.numel())[:n]]]
            nn_ = g_train_idx[neg_idx[torch.randperm(neg_idx.numel())[:n]]]
            sel = torch.cat([p, nn_]).sort()[0]
            model_loss = F.cross_entropy(h_raw[sel], anomaly[sel])
        else:
            model_loss = F.cross_entropy(h_raw[node_mask], train_label)

        edge_loss = self._edge_loss(g, x, anomaly)
        proto_loss = self._prototype_loss(h_raw, node_mask, anomaly)
        return model_loss + self.gamma1 * edge_loss + self.gamma2 * proto_loss

    @torch.no_grad()
    def score(self, g, x):
        logits = self.forward(g, x)
        p = torch.softmax(logits, dim=1)[:, 1]
        return p / (p.max().clamp(min=1e-6))