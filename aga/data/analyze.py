
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError("networkx is required for community detection") from exc

import scipy.sparse as sp


def to_networkx(g: "dgl.DGLGraph") -> nx.Graph:
    src, dst = g.edges()
    src = src.tolist()
    dst = dst.tolist()
    G = nx.Graph()
    G.add_nodes_from(range(g.num_nodes()))
    G.add_edges_from(zip(src, dst))
    return G


def degree_stats(g: "dgl.DGLGraph") -> Tuple[float, float, np.ndarray]:
    deg = g.in_degrees().float().numpy()
    return float(deg.mean()), float(deg.std()), deg


def _to_bool_np(mask) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if hasattr(mask, "bool"):
        return mask.bool().numpy()
    return np.asarray(mask, dtype=bool)


def homophily(g: "dgl.DGLGraph", anomaly: torch.Tensor, mask=None) -> float:
    src, dst = g.edges()
    m = _to_bool_np(mask)
    if m is not None:
        keep = m[src.numpy()] & m[dst.numpy()]
        src, dst = src[torch.from_numpy(keep)], dst[torch.from_numpy(keep)]
    a = anomaly[src]
    b = anomaly[dst]
    if a.numel() == 0:
        return 0.0
    return float((a == b).float().mean().item())


def communities(g: "dgl.DGLGraph") -> Tuple[np.ndarray, int]:
    G = to_networkx(g)
    if hasattr(nx.algorithms.community, "louvain_communities"):
        comms = list(nx.algorithms.community.louvain_communities(G, seed=42))
    else:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    comm_id = np.zeros(g.num_nodes(), dtype=np.int64)
    for cid, nodes in enumerate(sorted(comms, key=len, reverse=True)):
        for n in nodes:
            comm_id[int(n)] = cid
    return comm_id, len(comms)


def train_mask(g: "dgl.DGLGraph") -> Optional[np.ndarray]:
    m = g.ndata.get("train_mask")
    if m is None:
        return None
    m = m.bool().numpy()
    if not m.any():
        return None
    return m


def _label_mask(g: "dgl.DGLGraph") -> np.ndarray:
    tm = train_mask(g)
    if tm is None:
        return np.ones(g.num_nodes(), dtype=bool)
    return tm


def anomaly_breakdown(
    anomaly: torch.Tensor, attr_ano: torch.Tensor, stru_ano: torch.Tensor,
    mask=None,
) -> Dict[str, int]:
    a = anomaly.numpy()
    at = attr_ano.numpy()
    st = stru_ano.numpy()
    m = _to_bool_np(mask)
    if m is not None:
        a, at, st = a[m], at[m], st[m]
    return {
        "attr": int(at.sum()),
        "stru": int(st.sum()),
        "attr_only": int(((at == 1) & (st == 0)).sum()),
        "stru_only": int(((at == 0) & (st == 1)).sum()),
        "hybrid": int(((at == 1) & (st == 1)).sum()),
        "any": int(a.sum()),
    }



_OBSERVED_FAMILIES = ("attribute", "structure", "hybrid")
_UNMEASURABLE_FAMILIES = ("context", "semantic_camouflage", "bridge")


def coverage_estimate(breakdown: Dict[str, int], families: List[str]) -> Dict[str, Dict]:
    total = max(breakdown.get("any", 1), 1)
    scores: Dict[str, float] = {
        "attribute": min(1.0, breakdown.get("attr_only", 0) / max(total, 1) * 4),
        "structure": min(1.0, breakdown.get("stru_only", 0) / max(total, 1) * 4),
        "hybrid": min(1.0, breakdown.get("hybrid", 0) / max(total, 1) * 4),
    }
    out: Dict[str, Dict] = {}
    for fam in families:
        if fam in _UNMEASURABLE_FAMILIES:
            out[fam] = {"score": None, "status": "unmeasured"}
        elif fam in scores:
            out[fam] = {"score": round(float(scores[fam]), 3), "status": "observed"}
    return out


def coverage_bar(score: Optional[float], width: int = 10) -> str:
    if score is None:
        return "·" * width
    score = max(0.0, min(1.0, float(score)))
    filled = int(round(score * width))
    return "█" * filled + "▒" * (width - filled)


def context_consistency(g: "dgl.DGLGraph") -> np.ndarray:
    feat = g.ndata["feat"].numpy().astype(np.float32)
    fnorm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
    fn = feat / fnorm
    src, dst = g.edges()
    n = g.num_nodes()
    a = sp.csr_matrix((np.ones(len(src), dtype=np.float32),
                      (src.numpy(), dst.numpy())), shape=(n, n))
    nb_sum = a @ fn
    deg = np.asarray(a.sum(axis=1)).ravel()
    nb_mean = nb_sum / np.maximum(deg, 1.0)[:, None]
    nb_mean /= np.linalg.norm(nb_mean, axis=1, keepdims=True) + 1e-12
    ctx = (fn * nb_mean).sum(axis=1)
    ctx[deg == 0] = 0.0
    return ctx.astype(np.float32)


def anomaly_geometry(g: "dgl.DGLGraph", mask=None) -> Dict[str, Optional[float]]:
    anomaly = g.ndata["anomaly"].numpy()
    at = g.ndata["attr_ano"].numpy()
    st = g.ndata["stru_ano"].numpy()
    feat = g.ndata["feat"].numpy()
    deg = np.asarray(g.in_degrees().float().numpy())
    m = _to_bool_np(mask)
    if m is not None:
        normal_sel = (anomaly == 0) & m
        attr_sel = (at == 1) & m
        stru_sel = (st == 1) & m
        anomaly_sel = (anomaly == 1) & m
    else:
        normal_sel = anomaly == 0
        attr_sel = at == 1
        stru_sel = st == 1
        anomaly_sel = anomaly == 1

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(arr.mean()) if len(arr) else None,
            "std": float(arr.std()) if len(arr) else None,
            "median": float(np.median(arr)) if len(arr) else None,
            "p90": float(np.percentile(arr, 90)) if len(arr) else None,
        }

    stru_deg = deg[stru_sel]
    stru_stats = _stats(stru_deg) if len(stru_deg) else {"mean": None, "std": None, "median": None, "p90": None}

    ctx = context_consistency(g)

    def _ctx(sel: np.ndarray) -> Optional[float]:
        m = sel & (deg > 0)
        return float(ctx[m].mean()) if m.any() else None

    norms = np.linalg.norm(feat, axis=1)

    out: Dict[str, Optional[float]] = {
        "stru_degree_mean": stru_stats["mean"],
        "stru_degree_std": stru_stats["std"],
        "stru_degree_median": stru_stats["median"],
        "stru_degree_p90": stru_stats["p90"],
        "normal_ctx_cos_mean": _ctx(normal_sel),
        "attr_ctx_cos_mean": _ctx(attr_sel),
        "normal_feat_norm_mean": float(norms[normal_sel].mean()) if normal_sel.any() else None,
        "anomaly_feat_norm_mean": float(norms[anomaly_sel].mean()) if anomaly_sel.any() else None,
        "attr_feat_norm_mean": float(norms[attr_sel].mean()) if attr_sel.any() else None,
    }
    return out


class GraphSnapshot:

    def __init__(self, g: "dgl.DGLGraph", comm_id: np.ndarray):
        self.g = g
        self.comm_id = comm_id

    @classmethod
    def from_graph(cls, g: "dgl.DGLGraph") -> "GraphSnapshot":
        comm_id, _ = communities(g)
        return cls(g, comm_id)