
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc


def make_split(
    g: "dgl.DGLGraph",
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    n = g.num_nodes()
    anomaly = g.ndata["anomaly"].numpy()
    rng = np.random.default_rng(seed)

    n_val = int(round(val_ratio * n))
    n_test = int(round(test_ratio * n))
    n_train = n - n_val - n_test

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for cls in [0, 1]:
        idx = np.where(anomaly == cls)[0]
        rng.shuffle(idx)
        m = len(idx)
        v = int(round(val_ratio * m))
        t = int(round(test_ratio * m))
        val_idx = idx[:v]
        test_idx = idx[v : v + t]
        train_idx = idx[v + t :]
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        train_mask[train_idx] = True

    g.ndata["train_mask"] = torch.from_numpy(train_mask)
    g.ndata["val_mask"] = torch.from_numpy(val_mask)
    g.ndata["test_mask"] = torch.from_numpy(test_mask)

    return {
        "train_mask": g.ndata["train_mask"],
        "val_mask": g.ndata["val_mask"],
        "test_mask": g.ndata["test_mask"],
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def frozen_split_signature(masks: Dict[str, torch.Tensor]) -> Dict[str, int]:
    import hashlib

    def _digest(mask) -> int:
        if mask is None:
            return -1
        idx = sorted(int(i) for i in torch.where(mask)[0].tolist())
        return int.from_bytes(
            hashlib.md5(
                ",".join(map(str, idx)).encode("utf-8")
            ).digest()[:4], "big",
        )

    val = masks.get("val_mask")
    test = masks.get("test_mask")
    return {
        "n_val": int(val.sum().item()) if val is not None else -1,
        "n_test": int(test.sum().item()) if test is not None else -1,
        "val_sum": int(val.sum().item()) if val is not None else -1,
        "test_sum": int(test.sum().item()) if test is not None else -1,
        "val_digest": _digest(val),
        "test_digest": _digest(test),
    }