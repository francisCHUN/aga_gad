
from __future__ import annotations

import csv
import os
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

try:
    import dgl
    import dgl.backend as dgl_backend
except ImportError as exc:
    raise ImportError(
        "DGL is required. In the python312 env run with "
        "/opt/miniconda3/envs/python312/bin/python."
    ) from exc


def _as_row_vector(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).ravel()
    return arr


def load_dataset(
    mat_path: str,
    csv_path: str,
) -> Tuple["dgl.DGLGraph", Dict[int, str]]:

    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)

    mat = sio.loadmat(mat_path)

    adj = mat["Network"]
    if hasattr(adj, "tocoo"):
        adj = adj.tocoo()
    else:
        adj = sp.coo_matrix(np.asarray(adj))
    adj = (adj + adj.T)
    adj.data[:] = 1
    adj.eliminate_zeros()

    g = dgl.from_scipy(adj)

    feats = np.asarray(mat["Attributes"], dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    g.ndata["feat"] = torch.from_numpy(feats).float()

    anomaly = _as_row_vector(np.asarray(mat["Label"])).astype(np.int64)
    attr_ano = _as_row_vector(np.asarray(mat["attr_anomaly_label"])).astype(np.int64)
    stru_ano = _as_row_vector(np.asarray(mat["stru_anomaly_label"])).astype(np.int64)

    g.ndata["anomaly"] = torch.from_numpy(anomaly).long()
    g.ndata["attr_ano"] = torch.from_numpy(attr_ano).long()
    g.ndata["stru_ano"] = torch.from_numpy(stru_ano).long()
    g.ndata["is_synthetic"] = torch.zeros(g.num_nodes(), dtype=torch.bool)

    g.ndata["feat"] = g.ndata["feat"].float()
    return g, _load_text(csv_path)


def _load_text(csv_path: str) -> Dict[int, str]:
    text: Dict[int, str] = {}
    if not csv_path or not os.path.exists(csv_path):
        return text
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                nid = int(row[0])
            except ValueError:
                continue
            text[nid] = row[1] if len(row) > 1 else ""
    return text


def load_cora(data_dir: str = "data/") -> Tuple["dgl.DGLGraph", Dict[int, str]]:
    mat_path = os.path.join(data_dir, "Cora.mat")
    csv_path = os.path.join(data_dir, "Cora.csv")
    return load_dataset(mat_path, csv_path)


DATASETS = {
    "Cora": {"mat": "Cora.mat", "csv": "Cora.csv"},
    "Pubmed": {"mat": "Pubmed.mat", "csv": "Pubmed.csv"},
    "Arxiv": {"mat": "Arxiv.mat", "csv": "Arxiv.csv"},
    "History": {"mat": "History.mat", "csv": "History.csv"},
}


def load_by_name(name: str, data_dir: str = "data/") -> Tuple["dgl.DGLGraph", Dict[int, str]]:
    key = name.strip()
    spec = DATASETS.get(key) or DATASETS.get(key.capitalize())
    if spec is None:
        spec = {"mat": f"{key}.mat", "csv": f"{key}.csv"}
    mat_path = os.path.join(data_dir, spec["mat"])
    csv_path = os.path.join(data_dir, spec["csv"])
    return load_dataset(mat_path, csv_path)