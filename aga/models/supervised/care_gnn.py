
from __future__ import annotations

import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel


def rl_module(scores, scores_log, labels, thresholds, batch_num, step_size):
    relation_scores = []
    stop_flag = True
    pos_index = (labels == 1).nonzero().flatten().tolist()
    for score in scores:
        pos_scores = [score[i] for i in pos_index if i < len(score)]
        neigh_count = sum(1 if isinstance(s, float) else len(s) for s in pos_scores)
        pos_sum = sum(s if isinstance(s, float) else sum(s) for s in pos_scores)
        relation_scores.append(pos_sum / max(neigh_count, 1))
    if len(scores_log) % batch_num != 0 or len(scores_log) < 2 * batch_num:
        rewards = [0, 0, 0]
        new_thresholds = list(thresholds)
    else:
        previous = [sum(s) / batch_num for s in zip(*scores_log[-2 * batch_num : -batch_num])]
        current = [sum(s) / batch_num for s in zip(*scores_log[-batch_num :])]
        rewards = [1 if previous[i] - s >= 0 else -1 for i, s in enumerate(current)]
        new_thresholds = [
            (thresholds[i] + step_size) if r == 1 else (thresholds[i] - step_size)
            for i, r in enumerate(rewards)
        ]
        new_thresholds = [min(max(t, 0.001), 0.999) for t in new_thresholds]
    return relation_scores, rewards, new_thresholds, stop_flag


def filter_neighs_ada_threshold(center_scores, neigh_scores, neighs_list, sample_list):
    samp_neighs = []
    samp_scores = []
    for idx in range(len(center_scores)):
        center_score = center_scores[idx, 0]
        nsc = neigh_scores[idx][:, 0]
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]
        score_diff = torch.abs(center_score - nsc)
        sorted_scores, sorted_indices = torch.sort(score_diff, descending=False)
        selected_indices = sorted_indices.tolist()
        if len(nsc) > num_sample + 1 and len(selected_indices) >= num_sample:
            selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
            selected_scores = sorted_scores.tolist()[:num_sample]
        else:
            selected_neighs = list(neighs_indices)
            selected_scores = score_diff.tolist()
            if isinstance(selected_scores, float):
                selected_scores = [selected_scores]
        samp_neighs.append(set(selected_neighs))
        samp_scores.append(selected_scores if isinstance(selected_scores, list) else [selected_scores])
    return samp_neighs, samp_scores


def mean_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, n, device):
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)
    aggregated = torch.zeros(n, embed_dim, device=device)
    for r in range(num_relations):
        aggregated += neigh_h[r * n : (r + 1) * n, :]
    return F.relu((center_h + aggregated) / (num_relations + 1.0))


def weight_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, alpha, n, device):
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)
    w = F.softmax(alpha, dim=1)
    aggregated = torch.zeros(n, embed_dim, device=device)
    for r in range(num_relations):
        aggregated += neigh_h[r * n : (r + 1) * n, :] * w[:, r].unsqueeze(0)
    return F.relu(center_h + aggregated)


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, device):
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)
    combined = torch.cat((center_h.repeat(num_relations, 1), neigh_h), dim=1)
    e = att_layer(combined.mm(a))
    blocks = [e[r * n : (r + 1) * n, :] for r in range(num_relations)]
    attention = torch.cat(blocks, dim=1)
    ori_attention = F.softmax(attention, dim=1)
    attention = F.dropout(ori_attention, dropout, training=training)
    aggregated = torch.zeros(n, embed_dim, device=device)
    for r in range(num_relations):
        aggregated += attention[:, r].unsqueeze(1) * neigh_h[r * n : (r + 1) * n, :]
    return F.relu(center_h + aggregated)


def threshold_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, threshold, n, device):
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)
    aggregated = torch.zeros(n, embed_dim, device=device)
    for r in range(num_relations):
        aggregated += neigh_h[r * n : (r + 1) * n, :] * threshold[r]
    return F.relu(center_h + aggregated)


class IntraAgg(nn.Module):

    def forward(self, nodes, to_neighs_list, batch_scores, neigh_scores, sample_list, x):
        samp_neighs, samp_scores = filter_neighs_ada_threshold(
            batch_scores, neigh_scores, to_neighs_list, sample_list
        )
        device = x.device
        feats = []
        for i, neigh in enumerate(samp_neighs):
            if len(neigh) == 0:
                feats.append(torch.zeros(x.shape[1], device=device))
            else:
                feats.append(x[list(neigh)].mean(dim=0))
        to_feats = torch.stack(feats, dim=0)
        to_feats = F.relu(to_feats)
        return to_feats, samp_scores


class CAREGNNModel(BaseGADModel):

    name = "care_gnn"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.6,
                 inter: str = "GNN", step_size: float = 0.02, lambda_1: float = 1.0,
                 num_classes: int = 2, use_rl: bool = True, batch_size: int = 256,
                 layers: int = None, n_layer: int = None, **_):
        super().__init__()
        if n_layer is not None:
            n_layers = int(n_layer)
        if layers is not None:
            n_layers = int(layers)
        self.feat_dim = in_dim
        self.embed_dim = hidden
        self.inter = inter
        self.step_size = step_size
        self.lambda_1 = lambda_1
        self.dropout = dropout
        self.RL = use_rl
        self.batch_num = 1
        self.thresholds = [0.5, 0.5, 0.5]
        self.batch_size = batch_size
        self.relation_score_log: List = []
        self.thresholds_log = [list(self.thresholds)]

        self.n_relation = 3
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, hidden))
        nn.init.xavier_uniform_(self.weight)
        self.alpha = nn.Parameter(torch.FloatTensor(hidden, self.n_relation))
        nn.init.xavier_uniform_(self.alpha)
        self.a = nn.Parameter(torch.FloatTensor(2 * hidden, 1))
        nn.init.xavier_uniform_(self.a)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.label_clf = nn.Linear(in_dim, num_classes)
        self.cls = nn.Linear(hidden, num_classes)
        self.intra_agg = IntraAgg()
        self.sample_rates = [1.0, 0.6, 0.3]

    def _neighbors(self, g, node_id):
        return g.successors(int(node_id)).tolist()

    def _sample_relation_neighbors(self, g, nodes):
        out = [[] for _ in range(self.n_relation)]
        for n in nodes:
            succ = self._neighbors(g, n)
            for r in range(self.n_relation):
                if len(succ) == 0:
                    out[r].append([])
                    continue
                keep = max(1, int(round(len(succ) * self.sample_rates[r])))
                keep = min(keep, len(succ))
                out[r].append(random.sample(succ, keep))
        return out

    def _inter_aggregate(self, self_feats, neigh_feats, n, device):
        if self.inter == "Att":
            combined, _ = att_inter_agg(
                self.n_relation, self.leakyrelu, self_feats, neigh_feats,
                self.embed_dim, self.weight, self.a, n, self.dropout, self.training, device
            )
        elif self.inter == "Weight":
            combined = weight_inter_agg(
                self.n_relation, self_feats, neigh_feats, self.embed_dim, self.weight, self.alpha, n, device
            )
        elif self.inter == "Mean":
            combined = mean_inter_agg(
                self.n_relation, self_feats, neigh_feats, self.embed_dim, self.weight, n, device
            )
        else:
            combined = threshold_inter_agg(
                self.n_relation, self_feats, neigh_feats, self.embed_dim, self.weight, self.thresholds, n, device
            )
        return combined

    def _forward_batch(self, g, nodes, x, labels=None, train_flag=True):
        device = x.device
        node_idx = torch.tensor(nodes, dtype=torch.long, device=device)
        all_scores = self.label_clf(x)
        center_scores = all_scores[node_idx]
        rel_lists = self._sample_relation_neighbors(g, nodes)
        sample_lists = [
            [max(1, math.ceil(len(neighs) * self.thresholds[r])) for neighs in rel_lists[r]]
            for r in range(self.n_relation)
        ]
        rel_scores = [[all_scores[torch.tensor(nl, device=device)] if len(nl) > 0 else torch.zeros(0, 2, device=device)
                       for nl in rel_lists[r]] for r in range(self.n_relation)]
        intra_feats = []
        for r in range(self.n_relation):
            r_feats, _ = self.intra_agg(
                nodes, rel_lists[r], center_scores, rel_scores[r], sample_lists[r], x
            )
            intra_feats.append(r_feats)
        neigh_feats = torch.cat(intra_feats, dim=0)
        self_feats = x[node_idx]
        combined = self._inter_aggregate(self_feats, neigh_feats, len(nodes), device)
        gnn_scores = self.cls(combined)
        return gnn_scores, center_scores

    def forward(self, g, x):
        device = x.device
        N = g.num_nodes()
        out = torch.zeros(N, self.embed_dim, device=device)
        bsz = max(self.batch_size, 512)
        nodes_all = list(range(N))
        for start in range(0, N, bsz):
            chunk = nodes_all[start : start + bsz]
            rel_lists = self._sample_relation_neighbors(g, chunk)
            intra_feats = []
            for r in range(self.n_relation):
                feats = []
                for neigh in rel_lists[r]:
                    feats.append(x[neigh].mean(dim=0) if len(neigh) > 0 else torch.zeros(self.feat_dim, device=device))
                intra_feats.append(F.relu(torch.stack(feats, dim=0)))
            neigh_feats = torch.cat(intra_feats, dim=0)
            self_feats = x[chunk]
            combined = self._inter_aggregate(self_feats, neigh_feats, len(chunk), device)
            out[chunk] = combined
        return self.cls(out)

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        device = x.device
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=device)
        )
        train_idx = torch.where(node_mask)[0]
        pos = train_idx[anomaly[train_idx] == 1]
        neg = train_idx[anomaly[train_idx] == 0]
        if pos.numel() > 0 and neg.numel() > 0:
            n = min(pos.numel(), neg.numel())
            p = pos[torch.randperm(pos.numel())[:n]]
            nsel = neg[torch.randperm(neg.numel())[:n]]
            batch_nodes = torch.cat([p, nsel]).tolist()
        else:
            batch_nodes = train_idx.tolist()
        labels_t = anomaly[torch.tensor(batch_nodes, device=device)].long()

        gnn_scores, label_scores = self._forward_batch(
            g, batch_nodes, x, labels=labels_t, train_flag=True
        )
        gnn_loss = F.cross_entropy(gnn_scores, labels_t)
        label_loss = F.cross_entropy(label_scores, labels_t)
        return gnn_loss + self.lambda_1 * label_loss

    @torch.no_grad()
    def score(self, g, x):
        logits = self.forward(g, x)
        p = torch.softmax(logits, dim=1)[:, 1]
        return p / (p.max().clamp(min=1e-6))