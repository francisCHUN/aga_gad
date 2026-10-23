
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    from dgl.nn import GraphConv
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..base import BaseGADModel

NO_MIXUP = "no_mixup"
VANILLA_MIXUP = "vanilla_mixup"
ADAPTIVE = "adaptive"


class GADAMModel(BaseGADModel):
    name = "gadam"
    supervised = True

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
        mode: str = ADAPTIVE,
        margin: float = 10.0,
        eta: float = 1.6,
        gamma: float = 1.0,
        eps_border: float = 0.5,
        n_neighbors: int = 100,
        n_proposal_neighbors: int = 20,
        sampler_iters: int = 1,
        mixup_alpha: float = 1.0,
        proto_dim: int = 128,
        **_,
    ):
        super().__init__()
        assert mode in (NO_MIXUP, VANILLA_MIXUP, ADAPTIVE), f"unknown mode {mode!r}"
        self.hidden = hidden
        self.dropout = dropout
        self.mode = mode
        self.margin = float(margin)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.eps_border = float(eps_border)
        self.k = int(n_neighbors)
        self.h = int(n_proposal_neighbors)
        self.B = int(sampler_iters)
        self.mixup_alpha = float(mixup_alpha)
        self.proto_dim = int(proto_dim)
        self.lambda_temp = 2.0

        self.enc = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.enc.append(GraphConv(d, hidden, activation=F.relu, allow_zero_in_degree=True))
            d = hidden
        self.emb_proj = nn.Linear(hidden, hidden)
        self.score_head = nn.Linear(hidden, 1, bias=False)

        self.proto_attn = nn.Linear(hidden, self.proto_dim)
        self.attn_vec = nn.Linear(self.proto_dim, 1, bias=False)

        self._last_dev = float("nan")
        self._last_pro = float("nan")
        self._last_lambda = float("nan")

    def encode(self, g, x) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.enc):
            h = layer(g, h)
            if i < len(self.enc) - 1 and self.training and self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        z = F.relu(self.emb_proj(h))
        return z

    def score_of(self, z: torch.Tensor) -> torch.Tensor:
        return self.score_head(z).squeeze(-1)

    def forward(self, g, x) -> torch.Tensor:
        return self.score_of(self.encode(g, x))

    def _deviation(self, scores: torch.Tensor, pos_scores: torch.Tensor) -> torch.Tensor:
        s_r = pos_scores.detach().mean()
        sigma_r = pos_scores.detach().std().clamp(min=1e-6)
        return (scores - s_r) / sigma_r

    def _dev_loss(self, scores: torch.Tensor, y: torch.Tensor,
                  pos_scores: torch.Tensor,
                  mix_scores: Optional[torch.Tensor] = None,
                  mix_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        dev = self._deviation(scores, pos_scores)
        m = self.margin
        l1 = ((1 - y) * dev.abs() + y * torch.clamp(m - dev, min=0.0)).mean()
        l2 = scores.new_zeros(())
        if mix_scores is not None and mix_y is not None and mix_y.numel() > 0:
            dev_a = self._deviation(mix_scores, pos_scores)
            l2 = (mix_y * torch.clamp(m - dev_a, min=0.0)).mean()
        return l1 + l2

    def _prototype_loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.numel() == 0:
            return z.new_zeros(())
        r = torch.tanh(self.proto_attn(z))
        attn = F.softmax(self.attn_vec(r).squeeze(-1), dim=0)
        protos = []
        for cls in (0, 1):
            mask = (y == cls)
            if mask.sum() == 0:
                protos.append(z.new_zeros(z.shape[1]))
                continue
            w = F.softmax(self.attn_vec(r[mask]).squeeze(-1), dim=0)
            protos.append((z[mask] * w.unsqueeze(-1)).sum(dim=0))
        protos = torch.stack(protos, dim=0)
        dist = torch.cdist(z, protos)
        logp = F.log_softmax(-dist, dim=1)
        nll = -logp.gather(1, y.long().view(-1, 1)).squeeze(1)
        return nll.mean()

    @torch.no_grad()
    def _sample_negatives(self, z: torch.Tensor, labeled: torch.Tensor,
                          unlabeled: torch.Tensor) -> torch.Tensor:
        device = z.device
        nL = labeled.numel()
        nU = unlabeled.numel()
        if nL == 0 or nU == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        zL = z[labeled]
        d_to_label = torch.cdist(z[unlabeled], zL.mean(dim=0, keepdim=True)).squeeze(1)
        rho = d_to_label.clamp(min=1e-6).pow(self.gamma)

        init = torch.randint(0, nU, (nL,), device=device)
        cur = init.clone()
        for _ in range(max(1, self.B)):
            cand = torch.randint(0, nU, (nL,), device=device)
            half = torch.rand(nL, device=device) < 0.5
            if half.any() and nU > self.h:
                probe = torch.randint(0, nU, (nL, self.h), device=device)
                base = cur.unsqueeze(1).expand(-1, self.h)
                d_probe = ((z[unlabeled[probe]] - z[unlabeled[base]]) ** 2).sum(-1)
                nn = d_probe.argmin(dim=1)
                cand = torch.where(half, probe.gather(1, nn.unsqueeze(1)).squeeze(1), cand)
            acc = (rho[cand] / rho[cur].clamp(min=1e-12))
            r = torch.rand(nL, device=device)
            accept = r < torch.clamp(acc, max=1.0)
            cur = torch.where(accept, cand, cur)
            swap = torch.rand(nL, device=device) < self.eps_border
            if swap.any() and nU > self.k:
                probe = torch.randint(0, nU, (nL, self.k), device=device)
                base = cur.unsqueeze(1).expand(-1, self.k)
                d_probe = ((z[unlabeled[probe]] - z[unlabeled[base]]) ** 2).sum(-1)
                nn = d_probe.argmin(dim=1)
                border = probe.gather(1, nn.unsqueeze(1)).squeeze(1)
                cur = torch.where(swap, border, cur)
        return unlabeled[cur]

    def _mixup(self, z: torch.Tensor, labeled: torch.Tensor,
               sampled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zL = z[labeled]
        zS = z[sampled]
        if self.mode == ADAPTIVE:
            sim = F.cosine_similarity(zL, zS, dim=1)
            lam = (1.0 + torch.sigmoid(sim / self.lambda_temp)) / 2.0
            lam_z = lam.unsqueeze(1)
            z_mix = lam_z * zL + (1.0 - lam_z) * zS.detach()
            y_mix = lam
            self._last_lambda = float(lam.mean().detach().item())
            return z_mix, y_mix
        elif self.mode == VANILLA_MIXUP:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample(
                (zL.shape[0],)
            ).to(zL.device).clamp(min=0.0, max=1.0)
            lam_z = lam.unsqueeze(1)
            z_mix = lam_z * zL + (1.0 - lam_z) * zS.detach()
            y_mix = lam
            self._last_lambda = float(lam.mean().detach().item())
            return z_mix, y_mix
        return z.new_zeros((0, z.shape[1])), z.new_zeros((0,))

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None) -> torch.Tensor:
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        y_all = anomaly.long()

        z = self.encode(g, x)
        scores = self.score_of(z)

        labeled = torch.where(node_mask & (y_all == 1))[0]
        unlabeled = torch.where(node_mask & (y_all != 1))[0]

        if labeled.numel() == 0:
            return scores[node_mask].abs().mean()

        pos_scores = scores[labeled].detach()

        sampled = self._sample_negatives(z, labeled, unlabeled)
        sv = torch.cat([labeled, sampled], dim=0)
        sy = torch.cat([
            torch.ones(labeled.numel(), device=x.device),
            torch.zeros(sampled.numel(), device=x.device),
        ])

        if self.mode != NO_MIXUP:
            z_mix, y_mix = self._mixup(z, labeled, sampled)
            mix_scores = self.score_of(z_mix)
        else:
            mix_scores, y_mix = None, None

        l_dev = self._dev_loss(scores[sv], sy, pos_scores, mix_scores, y_mix)
        l_pro = self._prototype_loss(z[sv], sy)
        total = l_dev + self.eta * l_pro

        self._last_dev = float(l_dev.detach().item())
        self._last_pro = float(l_pro.detach().item())
        return total

    @torch.no_grad()
    def score(self, g, x) -> torch.Tensor:
        s = self.forward(g, x)
        return s / (s.abs().max().clamp(min=1e-6))