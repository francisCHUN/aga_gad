
from __future__ import annotations

from typing import List

import numpy as np
import torch

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from ..orchestration.schema import NewNodeBatch


def inject_batches(
    g: "dgl.DGLGraph",
    batches: List[NewNodeBatch],
) -> "dgl.DGLGraph":
    feat_dim = g.ndata["feat"].shape[1]
    base = g.num_nodes()

    all_new_feats: List[torch.Tensor] = []
    all_src: List[int] = []
    all_dst: List[int] = []
    new_train_mask: List[bool] = []
    new_anomaly: List[int] = []
    new_attr_ano: List[int] = []
    new_stru_ano: List[int] = []
    new_is_synth: List[bool] = []
    cum_offset = base

    for batch in batches:
        if batch.num_nodes == 0:
            continue
        feats = torch.tensor(batch.features, dtype=torch.float32)
        if feats.shape[1] != feat_dim:
            raise ValueError(
                f"batch {batch.pattern} feat dim {feats.shape[1]} != graph {feat_dim}"
            )
        all_new_feats.append(feats)

        for orig_id, local_idx in batch.new_edges:
            abs_new = cum_offset + int(local_idx)
            all_src.append(int(orig_id))
            all_dst.append(abs_new)
            all_src.append(abs_new)
            all_dst.append(int(orig_id))

        for li, lj in batch.new_new_edges:
            ai = cum_offset + int(li)
            aj = cum_offset + int(lj)
            all_src.append(ai)
            all_dst.append(aj)
            all_src.append(aj)
            all_dst.append(ai)

        ft = batch.family_type or "unknown"
        is_attr = ft in ("attribute", "both")
        is_stru = ft in ("structure", "both")
        is_both = ft == "both"
        for _ in range(batch.num_nodes):
            new_train_mask.append(True)
            new_anomaly.append(1)
            new_attr_ano.append(1 if (is_attr or is_both) else 0)
            new_stru_ano.append(1 if (is_stru or is_both) else 0)
            new_is_synth.append(True)

        cum_offset += batch.num_nodes

    if not all_new_feats:
        return g

    new_feats = torch.cat(all_new_feats, dim=0)
    n_new = new_feats.shape[0]

    base_tensors = {k: g.ndata[k].clone() for k in g.ndata.keys()}

    g.add_nodes(n_new)

    for key, vals, dtype in [
        ("feat", new_feats, torch.float32),
        ("is_synthetic", new_is_synth, torch.bool),
        ("anomaly", new_anomaly, torch.int64),
        ("attr_ano", new_attr_ano, torch.int64),
        ("stru_ano", new_stru_ano, torch.int64),
    ]:
        cur = base_tensors[key]
        g.ndata[key] = torch.cat([cur, torch.as_tensor(vals, dtype=dtype)], dim=0)

    for key in ("train_mask", "val_mask", "test_mask"):
        if key in base_tensors:
            pad_val = True if key == "train_mask" else False
            pad = torch.full((n_new,), pad_val, dtype=torch.bool)
            g.ndata[key] = torch.cat([base_tensors[key], pad], dim=0)
        else:
            pad_val = True if key == "train_mask" else False
            pad = torch.full((g.num_nodes(),), pad_val, dtype=torch.bool)
            g.ndata[key] = pad

    if all_src:
        src_t = torch.tensor(all_src, dtype=torch.int64)
        dst_t = torch.tensor(all_dst, dtype=torch.int64)
        g.add_edges(src_t, dst_t)

    return g


def synthetic_node_ids(g: "dgl.DGLGraph") -> torch.Tensor:
    return torch.where(g.ndata["is_synthetic"])[0]