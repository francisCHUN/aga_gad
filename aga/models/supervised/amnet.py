
from __future__ import annotations

from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.function as fn
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel


class BernConv(nn.Module):

    def __init__(self, orders: int = 2):
        super().__init__()
        self.K = orders
        self.weight = nn.Parameter(torch.ones(orders + 1))

    def forward(self, graph, feat):
        def unnLaplacian1(feat, D_invsqrt, graph):
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat - graph.ndata.pop("h") * D_invsqrt

        def unnLaplacian2(feat, D_invsqrt, graph):
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat + graph.ndata.pop("h") * D_invsqrt

        with graph.local_scope():
            tmp = [feat]
            weight = nn.functional.relu(self.weight)
            D_invsqrt = (
                torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
                .unsqueeze(-1)
                .to(feat.device)
            )
            for _ in range(self.K):
                feat = unnLaplacian2(feat, D_invsqrt, graph)
                tmp.append(feat)

            out_feat = (comb(self.K, 0) / (2 ** self.K)) * weight[0] * tmp[self.K]
            for i in range(self.K):
                x = tmp[self.K - i - 1]
                for _ in range(i + 1):
                    x = unnLaplacian1(feat, D_invsqrt, graph)
                out_feat = out_feat + (comb(self.K, i + 1) / (2 ** self.K)) * weight[i + 1] * x
        return out_feat


class AMNetModel(BaseGADModel):

    name = "amnet"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 32, num_classes: int = 2, orders: int = 2,
                 num_layers: int = 3, mlp_layers: int = 1, dropout_rate: float = 0.0,
                 activation: str = "ReLU", n_layers: int = None, layers: int = None, **_):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.mlp_layers = nn.Sequential()
        if mlp_layers > 0:
            for _ in range(max(mlp_layers - 1, 0)):
                self.mlp_layers.append(nn.Linear(hidden, hidden))
            self.mlp_layers.append(nn.Linear(hidden, num_classes))
        self.act = getattr(nn, activation)()
        self.attn = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.num_layers = num_layers
        if n_layers is not None:
            self.num_layers = int(n_layers)
        if layers is not None:
            self.num_layers = int(layers)
        self.layers = nn.Sequential(*[BernConv(orders) for _ in range(self.num_layers)])
        self.linear_transform_in = nn.Sequential(self.linear1, self.act, self.linear2)
        self.W_f = nn.Sequential(self.linear2, self.attn)
        self.W_x = nn.Sequential(self.linear2, self.attn)
        self.linear_cls_out = nn.Sequential(self.mlp_layers)

    def forward(self, g, x):
        h = self.linear_transform_in(x)
        if self.dropout is not None:
            h = self.dropout(h)
        h_list = []
        for layer in self.layers:
            h_ = layer(g, h)
            h_list.append(h_)
        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(h).unsqueeze(-1)
        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score
        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.num_layers):
            res = res + h_filters[:, i, :] * score[:, i]
        y_hat = self.linear_cls_out(res)
        return y_hat

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