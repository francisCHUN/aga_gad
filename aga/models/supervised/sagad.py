
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.function as fn
    from dgl import KHopGraph
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from aga.models.base import BaseGADModel

EPS = 1e-12


def cheby(i, x):
    if i == 0:
        return 1.0
    elif i == 1:
        return x
    else:
        T0 = 1.0
        T1 = x
        for ii in range(2, i + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2


def presum_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length, device=h.device, dtype=h.dtype)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx - 1] + h[idx - 1]
    return temp


def preminus_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length, device=h.device, dtype=h.dtype)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx - 1] - h[idx - 1]
    return temp


def reverse_tensor(h):
    temp = torch.zeros_like(h)
    length = len(temp)
    for idx in range(length):
        temp[idx] = h[length - 1 - idx]
    return temp


def precompute_Tx(g, x, K, conv_norm="none"):

    def conv_normalize(x, conv_norm):
        if conv_norm != "none":
            dim = 0 if conv_norm == "batch" else 1
            x = (x - x.mean(dim, keepdim=True)) / (x.std(dim, unbiased=False, keepdim=True) + 1e-5)
        return x

    g = g.local_var()

    deg = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata["norm"] = norm

    def norm_edge_weight(edges):
        return {"lap_edge_weight": edges.src["norm"] * edges.dst["norm"]}

    g.apply_edges(norm_edge_weight)
    g.edata["lap_edge_weight"] = -g.edata["lap_edge_weight"]

    def cheb_propagate(g, x):
        g.ndata["h"] = x
        g.update_all(fn.u_mul_e("h", "lap_edge_weight", "m"), fn.sum("m", "h"))
        return conv_normalize(g.ndata["h"], conv_norm)

    Tx_list = []
    Tx_0 = x
    Tx_list.append(Tx_0)

    if K == 0:
        return Tx_list

    Tx_1 = cheb_propagate(g, x)
    Tx_list.append(Tx_1)

    for _ in range(2, K + 1):
        Tx_2 = cheb_propagate(g, Tx_1)
        Tx_2 = 2 * Tx_2 - Tx_0
        Tx_list.append(Tx_2)
        Tx_0, Tx_1 = Tx_1, Tx_2

    return Tx_list


class ChebnetIIPropDGL(nn.Module):

    def __init__(self, K):
        super().__init__()
        self.K = K
        self.register_buffer("initial_val_low", torch.tensor(2.0))
        self.temp_low = nn.Parameter(torch.Tensor(self.K))
        self.temp_high = nn.Parameter(torch.Tensor(self.K))
        self.register_buffer("initial_val_high", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.temp_low, 2.0 / self.K)
        nn.init.constant_(self.temp_high, 2.0 / self.K)

    def compute_coeffs(self, lowpass):
        if lowpass:
            TEMP = F.relu(self.temp_low)
            coe_tmp = preminus_tensor(TEMP, self.initial_val_low)
        else:
            TEMP = F.relu(self.temp_high)
            coe_tmp = presum_tensor(TEMP, self.initial_val_high)

        coe = coe_tmp.clone()
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] += coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)
        return coe

    def forward(self, Tx_list, lowpass=True):
        coe = self.compute_coeffs(lowpass)

        out = coe[0] / 2 * Tx_list[0]
        for i in range(1, self.K + 1):
            out = out + coe[i] * Tx_list[i]
        return out


class NodeDimGatedFusion(nn.Module):

    def __init__(self, in_dim, emb_dim, num_gate_layers):
        super(NodeDimGatedFusion, self).__init__()
        assert num_gate_layers <= 2
        if num_gate_layers == 1:
            self.gate = nn.Linear(in_dim, emb_dim)
        else:
            self.gate = nn.Sequential(
                nn.Linear(in_dim, 2 * in_dim),
                nn.ReLU(),
                nn.Linear(2 * in_dim, emb_dim),
            )

    def forward(self, Z_l, Z_h, X, return_coefs=False):
        C = torch.sigmoid(self.gate(X))
        Z = Z_l * C + Z_h * (1 - C)
        if return_coefs:
            return Z, C
        else:
            return Z


class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        hid_dims,
        dropout,
        activation,
        norm_type="none",
        last_activate=False,
    ):
        super(MLP, self).__init__()
        self.norm_type = norm_type
        self.last_activate = last_activate

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(hid_dims[i], hid_dims[i + 1]))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hid_dims[i + 1]))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hid_dims[i + 1]))

    def forward(self, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(h)
            if l != len(self.layers) - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
        if self.last_activate:
            h = self.activation(h)
        return h


def select_topk_star_normft(star_khop_graph, node_ids, central_node_id):
    h_xs, id_xs = star_khop_graph.ndata["feature_normed"][node_ids], node_ids
    h_x0, id_x0 = star_khop_graph.ndata["feature_normed"][central_node_id], central_node_id

    xs = list(zip(h_xs, id_xs))
    x0 = (h_x0, id_x0)

    up = 0
    down = torch.pow(x0[0], 2) + EPS
    best = up / down
    greedy = lambda xi: -(x0[0] - xi[0]) ** 2 / (xi[0] ** 2)
    xs_sorted = sorted(xs, key=greedy)
    nbs = []
    for i, xi in enumerate(xs_sorted):
        tmp_up = (x0[0] - xi[0]) ** 2
        tmp_down = xi[0] ** 2 + EPS
        if best < tmp_up / tmp_down:
            up += tmp_up
            down += tmp_down
            best = up / down
            nbs.append(xi[1])
        else:
            break
    return nbs


def get_star_topk_nbs(star_khop_graph_big, central_node_id, khop, select_topk):
    star_khop_graph_in = star_khop_graph_big.sample_neighbors([central_node_id], fanout=-1, edge_dir="in")
    star_khop_graph_out = star_khop_graph_big.sample_neighbors([central_node_id], fanout=-1, edge_dir="out")

    node_ids_in = star_khop_graph_in.edges()[0]
    node_ids_out = star_khop_graph_out.edges()[1]
    node_ids = torch.cat([node_ids_in, node_ids_out], dim=0)
    node_ids = torch.unique(node_ids)

    nbs = select_topk(star_khop_graph_big, node_ids, central_node_id)
    nbs.append(torch.tensor(central_node_id).long())
    weights = torch.ones(len(nbs), 1) * 0.5
    weights[:-1, 0] = weights[:-1, 0] / (len(nbs) + EPS)

    return nbs, weights


def mrqsample(graph, features, one_hop=True):
    g = graph.local_var()

    if one_hop:
        k_hop = 1
        get_sp_adj_list = get_star_topk_nbs
        select_topk_fn = select_topk_star_normft
    else:
        raise NotImplementedError("only one_hop MRQ is supported in this port")

    g.ndata["feature_normed"] = features
    g.ndata["feature_normed"] = g.ndata["feature_normed"] - g.ndata["feature_normed"].min(0, keepdim=True)[0]
    g.ndata["feature_normed"] = g.ndata["feature_normed"] / (g.ndata["feature_normed"].max(0, keepdim=True)[0] + EPS)
    g.ndata["feature_normed"] = torch.norm(g.ndata["feature_normed"], dim=1)

    sp_matrix_graph = dgl.graph(([], []))
    sp_matrix_graph.add_nodes(graph.num_nodes())

    transform = KHopGraph(k_hop)
    tmp_graph = transform(g)
    tmp_graph = tmp_graph.to_simple()
    tmp_graph = tmp_graph.remove_self_loop()

    for central_node_id in graph.nodes():
        adj_list, weight_list = get_sp_adj_list(tmp_graph, central_node_id.item(), k_hop, select_topk_fn)
        sp_matrix_graph.add_edges(adj_list, central_node_id.long())

    return sp_matrix_graph


class SAGADModel(BaseGADModel):

    name = "sagad"
    supervised = True

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3,
                 dropout: float = 0.0, attr_weight=None, stru_weight=None, **_):
        super().__init__()
        K: int = 5
        p = (0.8, 1.0)
        hid_dims = [32, 32]
        num_gate_layers = 1
        activation = "ReLU"
        mlp_norm = "none"
        conv_norm = "none"

        doubled_in_dim = in_dim * 2
        emb_dim = doubled_in_dim // 2

        self.K = K
        self.p = torch.tensor(list(p), dtype=torch.float32)
        self.conv_norm = conv_norm
        self.chebnet = ChebnetIIPropDGL(K)
        self.fusion = NodeDimGatedFusion(doubled_in_dim, emb_dim, num_gate_layers)
        self.mlp = MLP(emb_dim, hid_dims + [2], dropout, activation, mlp_norm)
        self.use_gate_reg = True

        self._cached = False
        self._cached_n = -1
        self._cached_feat_shape = None
        self._X_doubled = None
        self._Tx_list = None

    def _prepare(self, g, x):
        n = g.num_nodes()
        feat_shape = tuple(x.shape)
        stale = (
            not self._cached
            or self._cached_n != n
            or self._cached_feat_shape != feat_shape
            or (self._X_doubled is not None and self._X_doubled.device != x.device)
        )
        if stale:
            g_cpu = g.to("cpu") if g.device.type != "cpu" else g
            x_cpu = x.detach().to("cpu")
            g_mrq = mrqsample(g_cpu, x_cpu, one_hop=True)
            x_nei = dgl.ops.copy_u_mean(g_mrq, x_cpu)
            X_doubled = torch.cat([x, x_nei.to(x.device)], dim=1)
            Tx_list = precompute_Tx(g, x, self.K, conv_norm=self.conv_norm)
            Tx_list = [Tx.to(x.device) for Tx in Tx_list]
            self._X_doubled = X_doubled
            self._Tx_list = Tx_list
            self._cached = True
            self._cached_n = n
            self._cached_feat_shape = feat_shape
        return self._X_doubled, self._Tx_list

    def _forward(self, g, x, return_coefs=False):
        X_doubled, Tx_list = self._prepare(g, x)
        Z_l = self.chebnet(Tx_list, lowpass=True)
        Z_h = self.chebnet(Tx_list, lowpass=False)
        if return_coefs:
            Z, C = self.fusion(Z_l, Z_h, X_doubled, return_coefs=True)
            return self.mlp(Z), C
        Z = self.fusion(Z_l, Z_h, X_doubled)
        return self.mlp(Z)

    def forward(self, g, x):
        return self._forward(g, x, return_coefs=False)

    def loss(self, g, x, mask=None, anomaly=None, train_mask=None):
        node_mask = train_mask if train_mask is not None else (
            mask if mask is not None else torch.ones(g.num_nodes(), dtype=torch.bool, device=x.device)
        )
        y = anomaly[node_mask].long()
        device = x.device

        n_anom = int((y == 1).sum().item())
        n_norm = int((y == 0).sum().item())
        w = n_norm / max(n_anom, 1)
        weights_ce = torch.tensor([1.0, w], device=device)

        if self.use_gate_reg:
            logits, C = self._forward(g, x, return_coefs=True)
            logits = logits[node_mask]
            C = C[node_mask]
            loss_ce = F.cross_entropy(logits, y, weight=weights_ce)

            p = self.p.to(device)
            d_train = torch.where(y == 1, p[0], p[1]).float()
            weights_re = torch.ones_like(y, dtype=torch.float32)
            weights_re[y == 1] = float(w)
            coefs = C.mean(dim=1)
            loss_re = F.binary_cross_entropy(coefs, d_train, weight=weights_re)
            return loss_ce + loss_re

        logits = self._forward(g, x, return_coefs=False)
        logits = logits[node_mask]
        return F.cross_entropy(logits, y, weight=weights_ce)

    @torch.no_grad()
    def score(self, g, x):
        logits = self._forward(g, x, return_coefs=False)
        return torch.softmax(logits, dim=1)[:, 1]