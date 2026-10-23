
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.function as fn
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel

log = logging.getLogger("aga.models.pmp")

_LAB_BENIGN = 0
_LAB_FRAUD = 1
_LAB_UNK = 2


class _LILinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, origin_infeat: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.trans_src = nn.Linear(origin_infeat, in_features)
        self.bias = nn.Parameter(torch.empty(out_features))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.trans_src.weight, a=5 ** 0.5)
        nn.init.zeros_(self.trans_src.bias)
        fan_in = self.in_features
        bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, h_self):
        t_src = self.trans_src(h_self)
        return F.linear(x * t_src, self.weight, self.bias)


class _LIMLP(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, origin_infeat):
        super().__init__()
        mods = []
        if num_layers == 1:
            mods.append(_LILinear(in_dim, out_dim, origin_infeat))
        else:
            mods.append(_LILinear(in_dim, hid_dim, origin_infeat))
            for _ in range(num_layers - 2):
                mods.append(nn.ReLU(inplace=True))
                mods.append(_LILinear(hid_dim, hid_dim, origin_infeat))
            mods.append(nn.ReLU(inplace=True))
            mods.append(_LILinear(hid_dim, out_dim, origin_infeat))
        self.mods = nn.ModuleList(mods)

    def forward(self, x, h_self):
        for m in self.mods:
            x = m(x, h_self) if isinstance(m, _LILinear) else m(x)
        return x


def _pmp_message(edges):
    return {"m": edges.src["h"], "lab": edges.src["label_unk"]}


def _pmp_reduce(nodes):
    mb = nodes.mailbox
    m, lab = mb["m"], mb["lab"]
    return {
        "neigh_fr": (m * (lab == _LAB_FRAUD).unsqueeze(-1)).sum(1),
        "neigh_be": (m * (lab == _LAB_BENIGN).unsqueeze(-1)).sum(1),
        "neigh_unk": (m * (lab == _LAB_UNK).unsqueeze(-1)).sum(1),
    }


class _PMPLayer(nn.Module):

    def __init__(self, in_feats, out_feats, hid_feats, num_trans, origin_infeat,
                 feat_drop=0.0):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.lin_before_mp = in_feats > out_feats
        bucket_dim = out_feats if self.lin_before_mp else in_feats
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_neigh_fraud = _LIMLP(bucket_dim, hid_feats, out_feats, num_trans,
                                     origin_infeat)
        self.fc_neigh_benign = _LIMLP(bucket_dim, hid_feats, out_feats, num_trans,
                                      origin_infeat)
        self.fc_balance = nn.Sequential(
            nn.Linear(in_feats, hid_feats), nn.ReLU(inplace=True),
            nn.Linear(hid_feats, 1),
        )
        self.balance_w = nn.Sigmoid()
        self.fc_self = nn.Linear(in_feats, out_feats)

    def forward(self, graph, feat, label_unk):
        with graph.local_scope():
            feat = self.feat_drop(feat)
            h_self = feat
            graph.srcdata["h"] = self.fc_neigh(feat) if self.lin_before_mp else feat
            graph.ndata["label_unk"] = label_unk
            graph.update_all(_pmp_message, _pmp_reduce)

            neigh_fr = graph.dstdata["neigh_fr"]
            neigh_be = graph.dstdata["neigh_be"]
            neigh_unk = graph.dstdata["neigh_unk"]

            neigh_fr = self.fc_neigh_fraud(neigh_fr, h_self)
            neigh_be = self.fc_neigh_benign(neigh_be, h_self)
            balance = self.balance_w(self.fc_balance(h_self))
            neigh_unk = (balance * self.fc_neigh_fraud(neigh_unk, h_self)
                         + (1.0 - balance) * self.fc_neigh_benign(neigh_unk, h_self))
            h_neigh = neigh_fr + neigh_be + neigh_unk
            rst = self.fc_self(h_self) + h_neigh
            return rst


class PMPModel(BaseGADModel):

    name = "pmp"
    supervised = True

    def __init__(
        self,
        in_dim: int,
        hidden: int = 48,
        n_layers: int = 1,
        num_trans: int = 1,
        dropout: float = 0.0,
        norm_feat: bool = True,
        synthetic_mode: str = "exclude",
        **_,
    ):
        super().__init__()
        if synthetic_mode not in ("exclude", "unknown", "fraud"):
            raise ValueError(
                f"synthetic_mode must be 'exclude', 'unknown' or 'fraud', "
                f"got {synthetic_mode!r}"
            )
        self.synthetic_mode = synthetic_mode
        self.norm_feat = bool(norm_feat)
        self.layers = nn.ModuleList()
        in_size = in_dim
        for i in range(n_layers):
            out_size = hidden if i < n_layers - 1 else 2
            self.layers.append(_PMPLayer(in_size, out_size, hidden, num_trans,
                                         origin_infeat=in_size))
            in_size = out_size
        self._lu_key = None
        self._label_unk: Optional[torch.Tensor] = None
        self._msg_key = None
        self._msg_graph = None

    def _get_label_unk(self, g, anomaly=None, train_mask=None):
        key = (id(g), id(anomaly), id(train_mask), g.num_nodes())
        if self._lu_key != key or self._label_unk is None:
            n = g.num_nodes()
            lab = torch.full((n,), _LAB_UNK, dtype=torch.long,
                             device=g.device)
            if anomaly is not None and train_mask is not None:
                tm = train_mask.bool()
                lab[tm] = anomaly[tm].long()
                if (self.synthetic_mode in ("unknown", "exclude")
                        and "is_synthetic" in g.ndata):
                    syn = g.ndata["is_synthetic"].bool()
                    lab[tm & syn] = _LAB_UNK
            self._label_unk = lab
            self._lu_key = key
        return self._label_unk

    def _get_message_graph(self, g):
        key = (id(g), g.num_nodes(), g.num_edges())
        if self._msg_key != key or self._msg_graph is None:
            g_msg = g
            if (self.synthetic_mode == "exclude"
                    and "is_synthetic" in g.ndata
                    and bool(g.ndata["is_synthetic"].any())):
                syn = g.ndata["is_synthetic"].bool()
                src, dst = g.edges()
                keep = ~(syn[src] | syn[dst])
                g_msg = dgl.graph(
                    (src[keep], dst[keep]), num_nodes=g.num_nodes()
                )
            self._msg_graph = g_msg
            self._msg_key = key
        return self._msg_graph

    def _forward(self, g, x, label_unk):
        h = x
        if self.norm_feat:
            h = h / (h.abs().sum(dim=1, keepdim=True) + 0.01)
        g_msg = self._get_message_graph(g)
        for i, layer in enumerate(self.layers):
            h = layer(g_msg, h, label_unk)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h

    def forward(self, g, x):
        label_unk = self._get_label_unk(g)
        return self._forward(g, x, label_unk)

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        label_unk = self._get_label_unk(g, anomaly, train_mask)
        logits = self._forward(g, x, label_unk)
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(
                g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        if (self.synthetic_mode == "exclude"
                and "is_synthetic" in g.ndata):
            node_mask = node_mask & ~g.ndata["is_synthetic"].bool()
        y = anomaly[node_mask].long()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        weight = None
        if n_pos > 0 and n_neg > 0:
            weight = torch.tensor([1.0, n_neg / n_pos],
                                  dtype=torch.float32, device=x.device)
        return F.cross_entropy(logits[node_mask], y, weight=weight)

    @torch.no_grad()
    def score(self, g, x):
        label_unk = self._get_label_unk(g)
        p = torch.softmax(self._forward(g, x, label_unk), dim=1)[:, 1]
        return p / (p.max().clamp(min=1e-6))