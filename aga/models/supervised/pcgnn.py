
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from aga.models.base import BaseGADModel


def choose_step_neighs(center_scores, center_labels, neigh_scores, neighs_list,
                        minor_scores, minor_list, sample_list, sample_rate):
    samp_neighs = []
    samp_score_diff = []
    for idx, _ in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze()
        if score_diff_neigh.dim() == 0:
            score_diff_neigh = score_diff_neigh.view(1)
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist()

        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]
            selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
        else:
            selected_neighs = list(neighs_indices)
            selected_score_diff = score_diff_neigh.tolist()
            if isinstance(selected_score_diff, float):
                selected_score_diff = [selected_score_diff]

        if center_labels[idx] == 1:
            num_oversample = int(num_sample * sample_rate)
            if num_oversample > 0 and minor_scores is not None and len(minor_scores) > 0:
                center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
                score_diff_minor = torch.abs(center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
                if score_diff_minor.dim() == 0:
                    score_diff_minor = score_diff_minor.view(1)
                sorted_score_diff_minor, sorted_minor_indices = torch.sort(score_diff_minor, dim=0, descending=False)
                selected_minor_indices = sorted_minor_indices.tolist()
                selected_neighs.extend([minor_list[n] for n in selected_minor_indices[:num_oversample]])
                selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])

        samp_neighs.append(set(selected_neighs))
        samp_score_diff.append(selected_score_diff)

    return samp_neighs, samp_score_diff


def choose_step_test(center_scores, neigh_scores, neighs_list, sample_list):
    samp_neighs = []
    samp_scores = []
    for idx, _ in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        score_diff = torch.abs(center_score - neigh_score).squeeze()
        if score_diff.dim() == 0:
            score_diff = score_diff.view(1)
        sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
        selected_indices = sorted_indices.tolist()

        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
            selected_scores = sorted_scores.tolist()[:num_sample]
        else:
            selected_neighs = list(neighs_indices)
            selected_scores = score_diff.tolist()
            if isinstance(selected_scores, float):
                selected_scores = [selected_scores]

        samp_neighs.append(set(selected_neighs))
        samp_scores.append(selected_scores)

    return samp_neighs, samp_scores


class IntraAgg(nn.Module):

    def __init__(self, feat_dim: int, embed_dim: int, rho: float = 0.5):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.rho = rho
        self.weight = nn.Parameter(torch.FloatTensor(2 * feat_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, nodes, batch_labels, to_neighs_list, batch_scores,
                neigh_scores, pos_scores, sample_list, train_flag,
                x, train_pos_ids):
        device = x.device
        if train_flag:
            samp_neighs, _ = choose_step_neighs(
                batch_scores, batch_labels, neigh_scores, to_neighs_list,
                pos_scores, train_pos_ids, sample_list, self.rho,
            )
        else:
            samp_neighs, _ = choose_step_test(batch_scores, neigh_scores, to_neighs_list, sample_list)

        unique_nodes_list = list(set.union(*samp_neighs)) if samp_neighs else []
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(samp_neighs), max(len(unique_nodes_list), 1), device=device)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1.0
        num_neigh = mask.sum(1, keepdim=True).clamp(min=1.0)
        mask = mask.div(num_neigh)

        self_feats = x[torch.LongTensor(nodes).to(device)]
        embed_matrix = x[torch.LongTensor(unique_nodes_list).to(device)] if unique_nodes_list else \
            torch.zeros(1, self.feat_dim, device=device)
        agg_feats = mask.mm(embed_matrix)
        cat_feats = torch.cat((self_feats, agg_feats), dim=1)
        to_feats = F.relu(cat_feats.mm(self.weight))
        return to_feats


class InterAgg(nn.Module):

    def __init__(self, feat_dim: int, embed_dim: int, n_relation: int = 3,
                 rho: float = 0.5, thresholds=None, sample_rates=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.n_relation = n_relation
        self.thresholds = thresholds or [0.5] * n_relation
        self.sample_rates = sample_rates if sample_rates is not None else _default_sample_rates(n_relation)
        self.intra_aggs = nn.ModuleList(
            [IntraAgg(feat_dim, embed_dim, rho) for _ in range(n_relation)]
        )
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim * n_relation + feat_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.label_clf = nn.Linear(feat_dim, 2)
        self._adj_lists = None
        self._adj_key = None

    def _get_adj_lists(self, g):
        key = (id(g), g.num_nodes(), g.num_edges())
        if self._adj_key != key or self._adj_lists is None:
            succ = [[] for _ in range(g.num_nodes())]
            src, dst = g.edges()
            for s, d in zip(src.tolist(), dst.tolist()):
                succ[s].append(d)
            self._adj_lists = succ
            self._adj_key = key
        return self._adj_lists

    def _relation_neighbors(self, adj_lists, nodes, rate):
        out = []
        for n in nodes:
            nbrs = adj_lists[int(n)]
            if not nbrs:
                out.append([])
                continue
            keep = max(1, int(round(len(nbrs) * rate)))
            keep = min(keep, len(nbrs))
            out.append(random.sample(nbrs, keep))
        return out

    def forward(self, g, nodes, labels, x, train_flag, train_pos_ids):
        device = x.device
        adj_lists = self._get_adj_lists(g)

        to_neighs = [self._relation_neighbors(adj_lists, nodes, self.sample_rates[r]) for r in range(self.n_relation)]

        ref_nodes = set(nodes)
        for r in range(self.n_relation):
            for neighs in to_neighs[r]:
                ref_nodes.update(neighs)
        unique_nodes = list(ref_nodes)
        batch_scores = self.label_clf(x[torch.LongTensor(unique_nodes).to(device)])
        id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

        from operator import itemgetter
        center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

        scores_cpu = batch_scores.detach().cpu()
        center_scores_sel = scores_cpu[itemgetter(*nodes)(id_mapping), :]

        if train_flag and train_pos_ids is not None and len(train_pos_ids) > 0:
            pos_scores = self.label_clf(x[torch.LongTensor(list(train_pos_ids)).to(device)]).detach().cpu()
        else:
            pos_scores = None

        intra_feats = []
        for r in range(self.n_relation):
            r_list = to_neighs[r]
            r_scores = [
                scores_cpu[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) if len(to_neigh) > 0
                else torch.zeros(0, 2)
                for to_neigh in r_list
            ]
            r_sample_num = [max(1, math.ceil(len(neighs) * self.thresholds[r])) for neighs in r_list]
            r_feats = self.intra_aggs[r].forward(
                nodes, labels, r_list, center_scores_sel, r_scores, pos_scores,
                r_sample_num, train_flag, x, list(train_pos_ids) if train_pos_ids is not None else [],
            )
            intra_feats.append(r_feats)

        self_feats = x[torch.LongTensor(nodes).to(device)]
        cat_feats = torch.cat([self_feats] + intra_feats, dim=1)
        combined = F.relu(cat_feats.mm(self.weight).t())
        return combined.t(), center_scores


def _default_sample_rates(n_relation):
    return ([1.0, 0.5, 0.3][:n_relation] if n_relation <= 3 else [1.0] * n_relation)


def pick_step(idx_train, y_train, succ_counts, size):
    idx_train = list(idx_train)
    y = y_train
    if hasattr(y, "cpu"):
        y = y.cpu().numpy()
    degree = [succ_counts[n] for n in idx_train]
    S = float(y.sum())
    N = float(len(y))
    lf = [(S if int(y[i]) == 1 else N) for i in range(len(idx_train))]
    smp_prob = [degree[i] / max(lf[i], 1e-12) for i in range(len(idx_train))]
    return random.choices(idx_train, weights=smp_prob, k=size)


class PCGNNModel(BaseGADModel):

    name = "pcgnn"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 1,
                 dropout: float = 0.0, attr_weight=None, stru_weight=None,
                 rho: float = 0.5, alpha: float = 2.0, threshold: float = 0.5,
                 n_relation: int = 3, batch_size: int = 256, **_):
        super().__init__()
        self.feat_dim = in_dim
        self.embed_dim = hidden
        self.n_relation = n_relation
        self.rho = rho
        self.alpha = alpha
        self.thresholds = [threshold] * n_relation
        self.batch_size = batch_size

        self.inter1 = InterAgg(in_dim, hidden, n_relation=n_relation, rho=rho,
                               thresholds=self.thresholds)
        self.weight = nn.Parameter(torch.FloatTensor(2, hidden))
        nn.init.xavier_uniform_(self.weight)
        self.xent = nn.CrossEntropyLoss()

    def _forward_batch(self, g, nodes, labels, x, train_flag, train_pos_ids):
        embeds1, label_scores = self.inter1(g, nodes, labels, x, train_flag, train_pos_ids)
        scores = self.weight.mm(embeds1.t())
        return scores.t(), label_scores

    def _degree_counts(self, g):
        if not hasattr(self, "_succ_count_cache") or getattr(self, "_succ_n", -1) != g.num_nodes():
            self._succ_count_cache = g.out_degrees().tolist()
            self._succ_n = g.num_nodes()
        return self._succ_count_cache

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        device = x.device
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=device)
        )
        train_idx = torch.where(node_mask)[0]
        labels_np = anomaly.cpu().numpy()
        succ_counts = self._degree_counts(g)

        pos_ids = train_idx[anomaly[train_idx] == 1]
        neg_ids = train_idx[anomaly[train_idx] == 0]
        train_pos_ids = pos_ids.tolist() if pos_ids.numel() > 0 else []

        size = max(2 * int(pos_ids.numel()), 2)
        y_train = labels_np
        sampled = pick_step(
            train_idx.tolist(),
            labels_np[train_idx.tolist()],
            succ_counts,
            size,
        )
        sampled = torch.LongTensor(sampled).to(device)
        perm = torch.randperm(sampled.numel(), device=device)
        sampled = sampled[perm]

        total_loss = torch.zeros((), device=device)
        n_batches = 0
        for start in range(0, sampled.numel(), self.batch_size):
            batch_nodes = sampled[start:start + self.batch_size].tolist()
            batch_labels = anomaly[torch.LongTensor(batch_nodes).to(device)].long().squeeze()
            gnn_scores, label_scores = self._forward_batch(
                g, batch_nodes, batch_labels, x, train_flag=True,
                train_pos_ids=train_pos_ids,
            )
            gnn_loss = self.xent(gnn_scores, batch_labels)
            label_loss = self.xent(label_scores, batch_labels)
            total_loss = total_loss + (gnn_loss + self.alpha * label_loss)
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def score(self, g, x):
        device = x.device
        N = g.num_nodes()
        out_scores = torch.zeros(N, 2, device=device)
        bsz = max(self.batch_size, 1024)
        nodes_all = list(range(N))
        for start in range(0, N, bsz):
            chunk = nodes_all[start:start + bsz]
            dummy_labels = torch.zeros(len(chunk), dtype=torch.long, device=device)
            gnn_scores, _ = self._forward_batch(
                g, chunk, dummy_labels, x, train_flag=False, train_pos_ids=None,
            )
            out_scores[chunk] = gnn_scores
        return torch.sigmoid(out_scores)[:, 1]