
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import (
    AnomalyGenerator,
    GenerationContext,
    feature_scale,
    generate_text_features,
    mixup_real_features,
)
from ..orchestration.schema import NewNodeBatch, PatternPlan
from ..data.analyze import GraphSnapshot


class SemanticCamouflage(AnomalyGenerator):
    name = "semantic_camouflage"

    def generate(self, snap, plan, node_text, rng, ctx: Optional[GenerationContext] = None) -> NewNodeBatch:
        count = int(plan.count)
        feat_dim = snap.g.ndata["feat"].shape[1]
        if count <= 0:
            return self._make_batch(self.name, np.zeros((0, feat_dim)), [], [], self.name)

        comm = snap.comm_id
        communities = np.unique(comm)
        scale = feature_scale(snap)

        donor = plan.transfer_attribute_from
        if donor is None and len(communities) >= 2:
            donor = int(communities[1])
        if donor is None:
            donor = int(communities[0])

        target_comm = plan.target_community if plan.target_community is not None else int(communities[0])

        new_feats = None
        if ctx is not None and ctx.text_gen and node_text:
            exemplars = ctx.community_texts.get(donor, []) or self._texts_for_community(
                node_text, comm, donor
            )
            topic = f"the research theme of community #{donor} (mimic it while hiding among community #{target_comm})"
            new_feats = generate_text_features(count, topic, exemplars, ctx)
        if new_feats is None:
            donor_feats = self._sample_real_feats(snap, donor, count, rng)
            target_feats = self._sample_real_feats(snap, target_comm, count, rng)
            feats_list: List[np.ndarray] = [
                mixup_real_features(xd, xt, float(rng.uniform(0.7, 0.95)), scale, rng)
                for xd, xt in zip(donor_feats, target_feats)
            ]
            new_feats = np.stack(feats_list)

        new_edges: List[List[int]] = []
        labels: List[int] = []
        normal = self._sample_normal_nodes(snap, max(count * 4, count), rng, community=target_comm)
        anchor_iter = iter(normal.tolist())

        for i in range(count):
            for _ in range(int(rng.integers(2, 4))):
                try:
                    n = next(anchor_iter)
                except StopIteration:
                    n = int(self._sample_normal_nodes(snap, 1, rng, community=target_comm)[0])
                new_edges.append([n, i])
            labels.append(0)

        if new_feats.ndim == 1:
            new_feats = new_feats.reshape(1, -1)
        return self._make_batch(self.name, new_feats, new_edges, labels, self.name)

    @staticmethod
    def _texts_for_community(node_text: dict, comm: np.ndarray, c: int) -> List[str]:
        idxs = np.where(comm == c)[0]
        out = []
        for i in idxs:
            t = node_text.get(int(i))
            if t:
                out.append(t)
            if len(out) >= 3:
                break
        return out