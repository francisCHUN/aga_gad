
from __future__ import annotations

from typing import List

import numpy as np

from .base import AnomalyGenerator, feature_scale, mixup_real_features
from ..orchestration.schema import NewNodeBatch, PatternPlan
from ..data.analyze import GraphSnapshot


class BridgeAttack(AnomalyGenerator):
    name = "bridge_attack"

    def generate(self, snap, plan, node_text, rng, ctx=None) -> NewNodeBatch:
        count = int(plan.count)
        feat_dim = snap.g.ndata["feat"].shape[1]
        if count <= 0:
            return self._make_batch(self.name, np.zeros((0, feat_dim)), [], [], self.name)

        comm = snap.comm_id
        communities = np.unique(comm)
        communities_sorted = [int(c) for c in communities if np.sum(comm == c) >= 5]
        if len(communities_sorted) < 2:
            communities_sorted = [int(communities[0]), int(communities[-1])]
        cA, cB = int(communities_sorted[0]), int(communities_sorted[1])
        scale = feature_scale(snap)

        eligible = AnomalyGenerator._synth_eligible_mask(snap)
        poolA = np.where((comm == cA) & eligible)[0]
        poolB = np.where((comm == cB) & eligible)[0]
        if len(poolA) == 0:
            poolA = np.where(comm == cA)[0]
        if len(poolB) == 0:
            poolB = np.where(comm == cB)[0]
        rng.shuffle(poolA)
        rng.shuffle(poolB)

        new_feats: List[np.ndarray] = []
        new_edges: List[List[int]] = []
        labels: List[int] = []

        donorA_feats = self._sample_real_feats(snap, cA, count, rng)
        donorB_feats = self._sample_real_feats(snap, cB, count, rng)

        for i in range(count):
            lam = float(rng.uniform(0.4, 0.6))
            new_feats.append(mixup_real_features(donorA_feats[i], donorB_feats[i], lam, scale, rng))
            a = int(poolA[i % len(poolA)])
            b = int(poolB[i % len(poolB)])
            new_edges.append([a, i])
            new_edges.append([b, i])
            labels.append(0)

        feats = np.stack(new_feats)
        return self._make_batch(self.name, feats, new_edges, labels, self.name)