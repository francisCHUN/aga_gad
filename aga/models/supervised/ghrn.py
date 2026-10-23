
from __future__ import annotations

import logging
from typing import Optional

import scipy
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.function as fn
    from dgl.nn.pytorch.conv import EdgeWeightNorm
    from torch.nn import init
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel

log = logging.getLogger("aga.models.ghrn")


def _calculate_theta2(d: int):
    thetas = []
    x = sympy.symbols("x")
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for _ in range(d + 1):
            inv_coeff.append(float(coeff[d - _]))
        thetas.append(inv_coeff)
    return thetas


class _BernsteinPolyConv(nn.Module):

    def __init__(self, in_feats, out_feats, theta, activation=F.leaky_relu,
                 lin=False, bias=False):
        super().__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat - graph.ndata.pop("h") * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(
                graph.in_degrees().float().clamp(min=1), -0.5
            ).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h = h + self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


class GHRNModel(BaseGADModel):

    name = "ghrn"
    supervised = True

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        d: int = 2,
        prune_ratio: float = 0.015,
        prune_at_epoch: int = 50,
        adj_type: str = "sym",
        preserve_synthetic: bool = True,
        *,
        n_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        **_,
    ):
        super().__init__()
        if prune_ratio < 0 or prune_ratio >= 1:
            raise ValueError(f"prune_ratio must be in [0,1), got {prune_ratio}")
        if adj_type not in ("sym", "rw"):
            raise ValueError(f"adj_type must be 'sym' or 'rw', got {adj_type!r}")
        self.d = int(d)
        self.prune_ratio = float(prune_ratio)
        self.prune_at_epoch = int(prune_at_epoch)
        self.adj_type = adj_type
        self.preserve_synthetic = bool(preserve_synthetic)

        self.thetas = _calculate_theta2(d=self.d)
        self.conv = nn.ModuleList(
            [_BernsteinPolyConv(hidden, hidden, t, lin=False) for t in self.thetas]
        )
        self.linear1 = nn.Linear(in_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden * len(self.conv), hidden)
        self.linear4 = nn.Linear(hidden, 2)
        self.act = nn.ReLU()

        self._epoch = 0
        self._pruned: Optional["dgl.DGLGraph"] = None

    def _backbone(self, g, in_feat):
        h = self.act(self.linear1(in_feat))
        h = self.act(self.linear2(h))
        h_final = None
        for conv in self.conv:
            h0 = conv(g, h)
            h_final = h0 if h_final is None else torch.cat([h_final, h0], dim=-1)
        h = self.act(self.linear3(h_final))
        return self.linear4(h)

    def _current_graph(self, g):
        if self._pruned is not None and self._pruned.num_nodes() == g.num_nodes():
            return self._pruned
        return g

    def forward(self, g, x):
        return self._backbone(self._current_graph(g), x)

    def _maybe_prune(self, g, x):
        self._epoch += 1
        if (
            self._pruned is None
            and self.prune_ratio > 0
            and self._epoch > self.prune_at_epoch
        ):
            self._pruned = self._prune_heterophilous_edges(g, x)
            log.info(
                "GHRN phase 2: pruned the %s heterophilous edges by Laplacian "
                "high-frequency residual (|E| %d -> %d%s).",
                f"{self.prune_ratio:.3f}",
                g.num_edges(), self._pruned.num_edges(),
                "; synthetic-edge-preserving mode" if self.preserve_synthetic else "",
            )

    def _prune_heterophilous_edges(self, g, x) -> "dgl.DGLGraph":
        with torch.no_grad():
            was_training = self.training
            self.eval()
            pred_y = F.softmax(self._backbone(g, x), dim=1)
            if was_training:
                self.train()

            device = x.device
            g2 = dgl.add_self_loop(g)
            edge_weight = torch.ones(g2.num_edges(), device=device)
            norm = EdgeWeightNorm(norm="both" if self.adj_type == "sym" else "left")
            g2.edata["w"] = norm(g2, edge_weight)

            g2.ndata["h"] = pred_y
            g2.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "ay"))
            g2.ndata["ly"] = pred_y - g2.ndata["ay"]
            g2.apply_edges(
                lambda e: {"inner_black": (e.src["ly"] * e.dst["ly"]).sum(dim=1)}
            )
            black = g2.edata["inner_black"]

            if self.preserve_synthetic and "is_synthetic" in g2.ndata:
                syn = g2.ndata["is_synthetic"].bool()
                src, dst = g2.edges()
                candidate = ~syn[src] & ~syn[dst]
                cand_idx = torch.where(candidate)[0]
                if cand_idx.numel() > 0:
                    threshold = int(self.prune_ratio * cand_idx.numel())
                    order = black[cand_idx].sort()[1][:threshold]
                    edge_to_move = cand_idx[order].tolist()
                    if edge_to_move:
                        g2 = dgl.remove_edges(g2, edge_to_move)
            else:
                threshold = int(self.prune_ratio * g2.num_edges())
                edge_to_move = set(black.sort()[1][:threshold].tolist())
                if edge_to_move:
                    g2 = dgl.remove_edges(g2, list(edge_to_move))

            g2 = dgl.remove_self_loop(g2)
            for key in ("h", "ly", "ay"):
                g2.ndata.pop(key, None)
            for key in ("w", "inner_black"):
                g2.edata.pop(key, None)
        return g2

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        self._maybe_prune(g, x)
        logits = self.forward(g, x)
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(
                g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        y = anomaly[node_mask].long()
        weight = None
        n_pos = int((y == 1).sum().item())
        n_neg = int((y == 0).sum().item())
        if n_pos > 0 and n_neg > 0:
            weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32,
                                  device=x.device)
        return F.cross_entropy(logits[node_mask], y, weight=weight)

    @torch.no_grad()
    def score(self, g, x):
        logits = self.forward(g, x)
        p = torch.softmax(logits, dim=1)[:, 1]
        return p / (p.max().clamp(min=1e-6))