
from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import torch

from .base import Agent
from ..data import loader, analyze
from ..data.analyze import GraphSnapshot
from ..orchestration.schema import (
    DatasetReport,
    DatasetStats,
    AnomalyCoverage,
    AnomalyGeometry,
)


FAMILIES = ["attribute", "structure", "context", "semantic_camouflage", "bridge", "hybrid"]


class DatasetInspector(Agent):
    name = "dataset_inspector"

    def __init__(self, data_dir: str = "data/", dataset: str = "Cora"):
        self.data_dir = data_dir
        self.dataset = dataset

    def run(
        self,
        snapshot: Optional[GraphSnapshot] = None,
        train_mask: Optional[torch.Tensor] = None,
    ) -> DatasetReport:
        if snapshot is None:
            g, _ = loader.load_by_name(self.dataset, self.data_dir)
            snapshot = GraphSnapshot.from_graph(g)
        g = snapshot.g

        deg_mu, deg_sd, _ = analyze.degree_stats(g)
        hom = analyze.homophily(g, g.ndata["anomaly"], mask=train_mask)
        comm_id, n_comm = analyze.communities(g) if not hasattr(snapshot, "comm_id") else (snapshot.comm_id, int(snapshot.comm_id.max()) + 1)
        snapshot.comm_id = comm_id
        breakdown = analyze.anomaly_breakdown(
            g.ndata["anomaly"], g.ndata["attr_ano"], g.ndata["stru_ano"],
            mask=train_mask,
        )
        cov = analyze.coverage_estimate(breakdown, FAMILIES)

        anomaly = g.ndata["anomaly"]
        if train_mask is not None:
            anomaly = anomaly[train_mask]
        anomaly = anomaly.numpy()
        cls_dist = {str(int(k)): int(v) for k, v in zip(*np.unique(anomaly, return_counts=True))}

        stats = DatasetStats(
            dataset=self.dataset,
            num_nodes=int(g.num_nodes()),
            num_edges=int(g.num_edges()),
            num_features=int(g.ndata["feat"].shape[1]),
            num_classes=int(len(np.unique(anomaly))),
            num_attr_anomalies=int(breakdown["attr"]),
            num_stru_anomalies=int(breakdown["stru"]),
            num_total_anomalies=int(breakdown["any"]),
            avg_degree=round(deg_mu, 4),
            degree_std=round(deg_sd, 4),
            homophily=round(hom, 4),
            num_communities=int(n_comm),
            class_distribution=cls_dist,
            stats_scope="train_only" if train_mask is not None else "full_graph",
        )

        geo_stats = analyze.anomaly_geometry(g, mask=train_mask)
        geometry = AnomalyGeometry(**{k: (round(v, 4) if v is not None else None)
                                      for k, v in geo_stats.items()})

        coverage: List[AnomalyCoverage] = []
        for f in FAMILIES:
            entry = cov.get(f, {"score": None, "status": "unmeasured"})
            score = entry["score"]
            status = entry["status"]
            if status == "unmeasured":
                note = "unmeasured (labels cannot express this family; review flow decides)"
            else:
                note = "present" if (score or 0.0) > 0 else "absent"
            coverage.append(
                AnomalyCoverage(family=f, score=score, status=status, note=note)
            )
        bars: Dict[str, str] = {f: analyze.coverage_bar(cov.get(f, {}).get("score")) for f in FAMILIES}

        unmeasured_fams = [f for f in FAMILIES if cov.get(f, {}).get("status") == "unmeasured"]
        missing_fams = [
            f for f in FAMILIES
            if cov.get(f, {}).get("status") == "observed" and (cov.get(f, {}).get("score") or 0.0) == 0.0
        ]
        present_fams = [
            f for f in FAMILIES
            if cov.get(f, {}).get("status") == "observed" and (cov.get(f, {}).get("score") or 0.0) > 0.0
        ]
        geo_line = ""
        if geometry.stru_degree_mean is not None:
            geo_line = (
                f" Real structural anomalies are hubs (degree mean "
                f"{geometry.stru_degree_mean}, median {geometry.stru_degree_median}) "
                f"vs normal nodes' avg degree {stats.avg_degree} -- structural "
                f"synthesis must match this degree profile, not low-degree bridges."
            )
        if geometry.attr_ctx_cos_mean is not None and geometry.normal_ctx_cos_mean is not None:
            geo_line += (
                f" Real attribute anomalies deviate from their neighbourhoods "
                f"only mildly (ctx cosine {geometry.attr_ctx_cos_mean} vs normal "
                f"{geometry.normal_ctx_cos_mean})."
            )
        scope_line = (
            f" Label-dependent statistics cover the TRAINING split only."
            if train_mask is not None else
            " WARNING: label-dependent statistics cover the FULL graph (no train mask)."
        )
        summary = (
            f"Dataset has {stats.num_nodes} nodes / {stats.num_edges} edges, "
            f"{stats.num_total_anomalies} labeled anomalies in scope "
            f"({stats.num_attr_anomalies} attr, {stats.num_stru_anomalies} stru, "
            f"{breakdown['hybrid']} hybrid), "
            f"avg degree {stats.avg_degree}, homophily {stats.homophily}, "
            f"{stats.num_communities} communities. "
            f"Covered families: {present_fams}. Missing: {missing_fams}. "
            f"Unmeasured families (no label evidence; NOT evidence of absence): "
            f"{unmeasured_fams}."
            f"{geo_line}{scope_line}"
        )

        report = DatasetReport(stats=stats, coverage=coverage, coverage_bars=bars,
                               anomaly_geometry=geometry, summary=summary)
        return report