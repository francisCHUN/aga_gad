
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import (
    AnomalyGenerator,
    GenerationContext,
    feature_scale,
    mixup_real_features,
)
from ..orchestration.schema import NewNodeBatch, PatternPlan
from ..data.analyze import GraphSnapshot

_DEGREE_CAP = 50


class HubAttack(AnomalyGenerator):
    name = "hub_attack"

    def generate(self, snap, plan, node_text, rng, ctx: Optional[GenerationContext] = None) -> NewNodeBatch:
        count = int(plan.count)
        feat_dim = snap.g.ndata["feat"].shape[1]
        if count <= 0:
            return self._make_batch(self.name, np.zeros((0, feat_dim)), [], [], self.name)

        degrees = self._hub_degrees(snap, count, rng)
        eligible = self._eligible_pool(snap, rng)

        scale = feature_scale(snap)
        donors = self._sample_real_feats(snap, None, 2 * count, rng)
        primary, partner = donors[:count], donors[count:]
        feats_list: List[np.ndarray] = [
            mixup_real_features(a, b, float(rng.uniform(0.6, 0.9)), scale, rng,
                                rel_noise=0.1)
            for a, b in zip(primary, partner)
        ]

        new_feats: List[np.ndarray] = []
        new_edges: List[List[int]] = []
        labels: List[int] = []
        for i in range(count):
            k = int(degrees[i])
            n = min(k, len(eligible))
            if n <= 0:
                break
            neigh = rng.choice(eligible, size=n,
                               replace=k > len(eligible))
            new_feats.append(feats_list[i])
            for nb in neigh.tolist():
                new_edges.append([int(nb), i])
            labels.append(0)

        feats = np.stack(new_feats) if new_feats else np.zeros((0, feat_dim), dtype=np.float32)
        return self._make_batch(self.name, feats, new_edges, labels, self.name)

    @staticmethod
    def _hub_degrees(snap: GraphSnapshot, n: int, rng) -> np.ndarray:
        deg = snap.g.in_degrees().float().numpy()
        stru_mask = snap.g.ndata["stru_ano"].numpy() == 1
        real = np.round(deg[stru_mask]).astype(np.int64)
        if real.size >= 5:
            sample = rng.choice(real, size=n, replace=True)
        else:
            deg_std = float(deg.std()) if deg.size else 1.0
            factors = rng.uniform(3.0, 5.0, size=n)
            sample = np.maximum(3, (factors * max(deg_std, 1.0))).astype(np.int64)
        return np.clip(sample, 2, _DEGREE_CAP).astype(np.int64)

    @staticmethod
    def _eligible_pool(snap: GraphSnapshot, rng) -> np.ndarray:
        eligible = AnomalyGenerator._synth_eligible_mask(snap)
        pool = np.where(eligible)[0]
        if len(pool) == 0:
            pool = np.arange(snap.g.num_nodes())
        rng.shuffle(pool)
        return pool