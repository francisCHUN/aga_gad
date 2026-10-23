
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    from dgl.nn import GraphConv
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel


class _MLP(nn.Module):

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(z)))


class DAGADModel(BaseGADModel):
    name = "dagad"
    supervised = True

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        dropout: float = 0.2,
        alpha: float = 1.5,
        beta: float = 0.5,
        q: float = 0.7,
        mlp_hidden: int = 32,
        **_,
    ):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        self.q = q

        self.enc_a = GraphConv(in_dim, hidden, activation=F.relu, allow_zero_in_degree=True)
        self.enc_b = GraphConv(in_dim, hidden, activation=F.relu, allow_zero_in_degree=True)

        self.ca = _MLP(2 * hidden, mlp_hidden, 2)
        self.cb = _MLP(2 * hidden, mlp_hidden, 2)

        self._last_omega = float("nan")
        self._last_l_org = float("nan")
        self._last_l_aug = float("nan")

    def encode(self, g, x):
        h_ca = self.enc_a(g, x)
        h_cb = self.enc_b(g, x)
        if self.training and self.dropout > 0:
            h_ca = F.dropout(h_ca, p=self.dropout, training=self.training)
            h_cb = F.dropout(h_cb, p=self.dropout, training=self.training)
        return h_ca, h_cb

    def forward(self, g, x):
        h_ca, h_cb = self.encode(g, x)
        h = torch.cat([h_ca, h_cb], dim=1)
        return self.cb(h)

    @staticmethod
    def _gather_correct(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        return probs.gather(1, y.view(-1, 1)).squeeze(1).clamp(min=1e-8)

    def _classwise_ce(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = self._gather_correct(logits, y)
        anom = y == 1
        norm = y == 0
        l_anom = -(p[anom].log()).mean() if anom.any() else torch.zeros((), device=y.device)
        l_norm = -(p[norm].log()).mean() if norm.any() else torch.zeros((), device=y.device)
        return l_anom + l_norm

    def _classwise_gce(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = self._gather_correct(logits, y)
        gce = (1.0 - p.pow(self.q)) / self.q
        anom = y == 1
        norm = y == 0
        l_anom = gce[anom].mean() if anom.any() else torch.zeros((), device=y.device)
        l_norm = gce[norm].mean() if norm.any() else torch.zeros((), device=y.device)
        return l_anom + l_norm

    def _instance_ce(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, y, reduction="mean")

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=x.device)
        )

        y_all = anomaly.long()
        h_ca, h_cb = self.encode(g, x)

        idx = torch.where(node_mask)[0]
        y = y_all[idx]
        h_ca_t = h_ca[idx]
        h_cb_t = h_cb[idx]

        m = idx.numel()
        perm = torch.randperm(m, device=x.device)
        h_cb_perm = h_cb_t[perm]
        y_tilde = y[perm]

        h_for_ca = torch.cat([h_ca_t, h_cb_t.detach()], dim=1)
        h_for_cb = torch.cat([h_ca_t.detach(), h_cb_t], dim=1)
        logits_ca = self.ca(h_for_ca)
        logits_cb = self.cb(h_for_cb)

        h_tilde_for_cb = torch.cat([h_ca_t.detach(), h_cb_perm], dim=1)
        logits_cb_aug = self.cb(h_tilde_for_cb)

        ce_ca = self._instance_ce(logits_ca, y)
        ce_cb = self._instance_ce(logits_cb, y)
        omega = ce_cb / (ce_ca + ce_cb + 1e-8)

        l_ca_org = self._classwise_ce(logits_ca, y)
        l_cb_org = self._classwise_gce(logits_cb, y)
        l_cb_aug = self._classwise_gce(logits_cb_aug, y_tilde)

        l_org = omega * l_ca_org + l_cb_org
        l_aug = l_cb_aug
        total = self.alpha * l_org + self.beta * l_aug

        self._last_omega = float(omega.detach().item())
        self._last_l_org = float(l_org.detach().item())
        self._last_l_aug = float(l_aug.detach().item())
        return total

    @torch.no_grad()
    def score(self, g, x):
        logits = self.forward(g, x)
        p = torch.softmax(logits, dim=1)[:, 1]
        return p / (p.max().clamp(min=1e-6))