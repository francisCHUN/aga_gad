
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel

log = logging.getLogger("aga.models.gaga")


def _build_token_matrix(hop_adj: sp.csr_matrix, member_colmask: np.ndarray,
                        feats: np.ndarray, restore_diag: bool = False,
                        drop_diag: bool = False, grp_norm: bool = False) -> np.ndarray:
    n = hop_adj.shape[0]
    adj = hop_adj.multiply(sp.diags(member_colmask.astype(hop_adj.dtype)))
    if drop_diag:
        adj = adj - sp.diags(adj.diagonal())
    if restore_diag:
        adj = adj + sp.diags(hop_adj.diagonal().astype(hop_adj.dtype))
    adj = adj.tocsr()
    adj.eliminate_zeros()
    counts = np.asarray(adj.sum(axis=1)).ravel()
    summed = adj @ feats
    mean = summed / np.maximum(counts, 1.0)[:, None]
    if grp_norm:
        mean = mean / np.sqrt(np.maximum(counts, 1.0))[:, None]
    mean[counts == 0] = 0.0
    return mean.astype(np.float32)


def build_gaga_sequence(
    g,
    x: torch.Tensor,
    anomaly: Optional[torch.Tensor],
    train_mask: Optional[torch.Tensor],
    n_hops: int = 2,
    norm_feat: bool = True,
    grp_norm: bool = False,
) -> torch.Tensor:
    n = g.num_nodes()
    device = x.device
    xn = x.detach().float().cpu().numpy()
    if norm_feat:
        rowsum = np.abs(xn).sum(axis=1, keepdims=True) + 0.01
        xn = (xn / rowsum).astype(np.float32)

    if anomaly is not None:
        lab = anomaly.detach().cpu().numpy().astype(np.int64)
    else:
        lab = np.zeros(n, dtype=np.int64)
    if train_mask is not None:
        tm = train_mask.detach().cpu().numpy().astype(bool)
    else:
        tm = np.zeros(n, dtype=bool)

    src, dst = g.edges()
    adj = sp.csr_matrix(
        (np.ones(len(src), dtype=np.float32),
         (src.detach().cpu().numpy(), dst.detach().cpu().numpy())),
        shape=(n, n),
    )
    a_sl = (adj + sp.eye(n, dtype=np.float32, format="csr")).tocsr()

    col_neg = (tm & (lab == 0)).astype(a_sl.dtype)
    col_pos = (tm & (lab == 1)).astype(a_sl.dtype)
    col_masked = (~tm).astype(a_sl.dtype)

    tokens = [xn]
    hop_adj = a_sl
    for _ in range(n_hops):
        neg = _build_token_matrix(hop_adj, col_neg, xn, drop_diag=True,
                                  grp_norm=grp_norm)
        pos = _build_token_matrix(hop_adj, col_pos, xn, drop_diag=True,
                                  grp_norm=grp_norm)
        masked = _build_token_matrix(hop_adj, col_masked, xn, restore_diag=True,
                                     grp_norm=grp_norm)
        tokens.extend([neg, pos, masked])
        hop_adj = ((hop_adj @ hop_adj) > 0).astype(np.float32).tocsr()

    seq = np.stack(tokens, axis=1)
    return torch.from_numpy(seq).to(device)


class _RoleEncoder(nn.Module):

    def __init__(self, feat_dim, emb_dim, n_relations, n_hops, n_groups, dropout):
        super().__init__()
        self.hop_embedding = nn.Embedding(n_hops + 1, emb_dim)
        self.relation_embedding = nn.Embedding(max(n_relations, 1), emb_dim)
        self.group_embedding = nn.Embedding(n_groups + 1, emb_dim)
        self.MLP = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_groups = n_groups
        self.base_seq_len = n_hops * n_groups + 1

    def forward(self, x):
        device = x.device
        hop_emb = self.hop_embedding(torch.arange(self.n_hops + 1, device=device))
        center_hop = hop_emb[0].unsqueeze(0)
        hop_list = [center_hop]
        for i in range(1, self.n_hops + 1):
            hop_list.append(hop_emb[i].unsqueeze(0).expand(self.n_groups, -1))
        hop_seq = torch.cat(hop_list, dim=0)

        rel_emb = self.relation_embedding(torch.arange(self.n_relations, device=device))
        rel_seq = rel_emb.repeat_interleave(self.base_seq_len, dim=0)

        grp_emb = self.group_embedding(torch.arange(self.n_groups + 1, device=device))
        center_grp = grp_emb[self.n_groups].unsqueeze(0)
        hop_grp = grp_emb[torch.arange(self.n_groups, device=device)]
        grp_seq = torch.cat([center_grp, hop_grp.repeat(self.n_hops, 1)], dim=0)

        enc_seq = (hop_seq + rel_seq + grp_seq).repeat(self.n_relations, 1)

        out = self.MLP(x)
        out = out + enc_seq.unsqueeze(1)
        return self.dropout(out)


class GAGAModel(BaseGADModel):

    name = "gaga"
    supervised = True

    def __init__(
        self,
        in_dim: int,
        hidden: int = 16,
        emb_dim: Optional[int] = None,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.0,
        n_hops: int = 2,
        grp_norm: bool = False,
        norm_feat: bool = True,
        n_relations: int = 1,
        **_,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim if emb_dim is not None else hidden)
        self.n_hops = int(n_hops)
        self.n_groups = 3
        self.n_relations = int(n_relations)
        self.grp_norm = bool(grp_norm)
        self.norm_feat = bool(norm_feat)
        self.base_seq_len = self.n_hops * self.n_groups + 1

        self.feat_encoder = _RoleEncoder(
            in_dim, self.emb_dim, self.n_relations, self.n_hops,
            self.n_groups, dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.projection = nn.Sequential(
            nn.Linear(self.emb_dim * self.n_relations, 2)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self._cache_key = None
        self._cache_seq: Optional[torch.Tensor] = None

    def _get_sequence(self, g, x, anomaly=None, train_mask=None):
        key = (id(g), id(x), g.num_nodes(), g.num_edges(), x.shape[1])
        if self._cache_key != key or self._cache_seq is None:
            self._cache_seq = build_gaga_sequence(
                g, x, anomaly, train_mask,
                n_hops=self.n_hops, norm_feat=self.norm_feat,
                grp_norm=self.grp_norm,
            )
            self._cache_key = key
            log.debug("GAGA: built group-aggregation sequence %s.",
                      tuple(self._cache_seq.shape))
        return self._cache_seq

    def forward(self, g, x):
        seq = self._get_sequence(g, x)
        return self._forward_seq(seq)

    def _forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        src = seq.transpose(0, 1).float()
        out = self.feat_encoder(src)
        out = self.transformer_encoder(out)
        block_len = self.base_seq_len
        indices = torch.arange(0, out.shape[0], block_len, device=out.device)
        mr_feats = out.index_select(0, indices)
        agg = mr_feats.transpose(0, 1).reshape(seq.shape[0], -1)
        return self.projection(agg)

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None, class_weight=None):
        seq = self._get_sequence(g, x, anomaly, train_mask)
        logits = self._forward_seq(seq)
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(
                g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        y = anomaly[node_mask].long()
        return F.cross_entropy(logits[node_mask], y, weight=class_weight)

    @torch.no_grad()
    def score(self, g, x):
        seq = self._get_sequence(g, x)
        p = torch.softmax(self._forward_seq(seq), dim=1)[:, 1]
        return p / (p.max().clamp(min=1e-6))