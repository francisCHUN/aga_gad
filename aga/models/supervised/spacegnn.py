
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.function as fn
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel

log = logging.getLogger("aga.models.spacegnn")


def _expmap(e, c):
    sqrt_c = abs(c) ** 0.5
    e_norm = torch.clamp_min(e.norm(dim=-1, p=2, keepdim=True), 1e-15)
    if float(c) < 0:
        hemb = torch.tanh(sqrt_c * e_norm) * e / (sqrt_c * e_norm)
    else:
        hemb = torch.tan(sqrt_c * e_norm) * e / (sqrt_c * e_norm)
    return hemb


def _logmap(h, c):
    sqrt_c = abs(c) ** 0.5
    h_norm = h.norm(dim=-1, p=2, keepdim=True).clamp_min(1e-15)
    if float(c) < 0:
        scale = 1.0 / sqrt_c * torch.arctanh(sqrt_c * h_norm) / h_norm
    else:
        scale = 1.0 / sqrt_c * torch.atan(sqrt_c * h_norm) / h_norm
    return scale * h


def _proj(x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-15)
    maxnorm = (1 - 1e-5) / (abs(c) ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


class _CustomLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


class _LinearLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, drop_rate, final=True):
        super().__init__()
        self.linear1 = _CustomLinear(in_dim, hid_dim)
        self.linear2 = _CustomLinear(hid_dim, out_dim)
        self.act = nn.ELU()
        self.drop = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(hid_dim)
        self.final = final

    def forward(self, h):
        h = self.linear1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.norm(h)
        h = self.linear2(h)
        if self.final:
            h = self.act(h)
            h = self.drop(h)
        return h


class _CurvLayer(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, drop_rate):
        super().__init__()
        self.edge_linear = _LinearLayer(in_dim * 2, hid_dim, hid_dim, drop_rate)
        self.out_linear = _CustomLinear(hid_dim, out_dim)
        self.edge_bn = nn.BatchNorm1d(hid_dim)

    def _edge_udf(self, c):
        edge_linear = self.edge_linear

        def weighted_message(edges):
            src = edges.src["h"]
            dst = edges.dst["h"]
            if float(c) == 0:
                coef = 1 - torch.sigmoid((src - dst).norm(dim=-1, p=2, keepdim=True))
            else:
                dist = (src - dst).norm(dim=-1, p=2, keepdim=True)
                multi = (src * dst).sum(dim=-1, keepdim=True)
                coef = 1 - torch.sigmoid(
                    2 * dist - 2 * c * ((dist ** 3) / 3 + multi * (dist ** 2))
                )
            msg = torch.cat([coef * src + src, dst], dim=-1)
            msg = edge_linear(msg)
            return {"msg": msg}

        return weighted_message

    def forward(self, graph, features, c):
        with graph.local_scope():
            if not (float(c) == 0):
                features = _expmap(features, c)
                features = _proj(features, c)
                features = _logmap(features, c)

            graph.ndata["h"] = features
            graph.apply_edges(self._edge_udf(c))
            graph.edata["msg"] = self.edge_bn(graph.edata["msg"])
            graph.update_all(fn.copy_e("msg", "msg"), fn.sum("msg", "out"))
            out = graph.ndata.pop("out")
            out = self.out_linear(out)
            if not (float(c) == 0):
                out = _expmap(out, c)
                out = _proj(out, c)
                out = _logmap(out, c)
                out = F.selu(out)
            out = out + features
        return out


class _CurvGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layer_num, drop_rate):
        super().__init__()
        self.in_linears = nn.ModuleList()
        self.gnns = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.out_linears = nn.ModuleList()
        self.layer_num = layer_num
        for _ in range(layer_num):
            self.in_linears.append(_LinearLayer(in_dim, hid_dim, hid_dim, drop_rate))
            self.gnns.append(_CurvLayer(hid_dim, hid_dim, hid_dim, drop_rate))
            self.bns.append(nn.BatchNorm1d(hid_dim))
            self.out_linears.append(_CustomLinear(hid_dim * 2, hid_dim))
        self.out_linear = _CustomLinear(hid_dim * layer_num, out_dim)
        self.act = F.selu

    def forward(self, graph, layer_feats: List[torch.Tensor], curvs) -> torch.Tensor:
        results = []
        for i in range(self.layer_num):
            inter_results = []
            h = self.in_linears[i](layer_feats[i])
            inter_results.append(h)
            h = self.gnns[i](graph, h, curvs[i])
            h = self.bns[i](h)
            h = self.act(h)
            inter_results.append(h)
            h = torch.cat(inter_results, dim=-1)
            h = self.out_linears[i](h)
            results.append(h)
        h = torch.cat(results, dim=-1)
        return self.out_linear(h).log_softmax(dim=-1)


class SpaceGNNModel(BaseGADModel):

    name = "spacegnn"
    supervised = True

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.05,
        alpha: float = 0.5,
        beta: float = 0.5,
        std_neg: float = 0.015,
        std_pos: float = 0.015,
        **_,
    ):
        super().__init__()
        self.layer_num = int(n_layers)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.curvgnnneg = _CurvGNN(in_dim, hidden, 2, self.layer_num, dropout)
        self.curvgnnpos = _CurvGNN(in_dim, hidden, 2, self.layer_num, dropout)
        self.eucgnn = _CurvGNN(in_dim, hidden, 2, self.layer_num, dropout)
        self.cneg = nn.Parameter(torch.empty(self.layer_num).normal_(-0.1, std_neg))
        self.cpos = nn.Parameter(torch.empty(self.layer_num).normal_(0.1, std_pos))
        self._zeros = [0] * self.layer_num
        self._cache_key = None
        self._cache_graph = None
        self._cache_feats: Optional[List[torch.Tensor]] = None

    def _prep(self, g, x) -> Tuple["dgl.DGLGraph", List[torch.Tensor]]:
        key = (id(g), id(x), g.num_nodes(), g.num_edges(), x.shape[1])
        if self._cache_key != key or self._cache_graph is None:
            gl = dgl.add_self_loop(g)
            feats = [x]
            h = x
            for _ in range(1, self.layer_num):
                with gl.local_scope():
                    gl.ndata["h"] = h
                    gl.update_all(fn.copy_u("h", "m"), fn.mean("m", "hh"))
                    h = gl.ndata.pop("hh")
                feats.append(h)
            self._cache_key = key
            self._cache_graph = gl
            self._cache_feats = feats
            log.debug("SpaceGNN: rebuilt self-loop graph and %d layer features.",
                      self.layer_num)
        return self._cache_graph, self._cache_feats

    def _three(self, g, x):
        gl, feats = self._prep(g, x)
        l1 = self.curvgnnneg(gl, feats, self.cneg)
        l2 = self.curvgnnpos(gl, feats, self.cpos)
        l3 = self.eucgnn(gl, feats, self._zeros)
        return l1, l2, l3

    def forward(self, g, x):
        l1, l2, l3 = self._three(g, x)
        a, b = self.alpha, self.beta
        return (1 - b) * ((1 - a) * l1 + a * l2) + b * l3

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None, class_weight=None):
        l1, l2, l3 = self._three(g, x)
        a, b = self.alpha, self.beta
        logprobs = (1 - b) * ((1 - a) * l1 + a * l2) + b * l3
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(
                g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        y = anomaly[node_mask].long()
        return F.nll_loss(logprobs[node_mask], y, weight=class_weight)

    @torch.no_grad()
    def score(self, g, x):
        l1, l2, l3 = self._three(g, x)
        a, b = self.alpha, self.beta
        p1 = l1.exp()[:, 1]
        p2 = l2.exp()[:, 1]
        p3 = l3.exp()[:, 1]
        p = (1 - b) * ((1 - a) * p1 + a * p2) + b * p3
        return p / (p.max().clamp(min=1e-6))