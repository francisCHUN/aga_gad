
from __future__ import annotations

import logging
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

log = logging.getLogger("aga.patterns.context_attack")


class ContextAttack(AnomalyGenerator):
    name = "context_attack"

    def generate(self, snap, plan, node_text, rng, ctx: Optional[GenerationContext] = None) -> NewNodeBatch:
        count = int(plan.count)
        feat_dim = snap.g.ndata["feat"].shape[1]
        if count <= 0:
            return self._make_batch(self.name, np.zeros((0, feat_dim)), [], [], self.name)

        comm = snap.comm_id
        communities = np.unique(comm)
        scale = feature_scale(snap)

        per_anchor = int(plan.extra.get("per_anchor", 0)) or self._default_per_anchor(count)
        per_anchor = max(1, min(per_anchor, 4))
        n_anchors = max(1, int(np.ceil(count / per_anchor)))

        target_comm_hint = plan.target_community
        anchors, anchor_comms = self._select_anchors(
            snap, n_anchors, target_comm_hint, rng
        )

        new_feats: List[np.ndarray] = []
        new_edges: List[List[int]] = []
        labels: List[int] = []
        produced = 0

        for ai, anchor in enumerate(anchors):
            if produced >= count:
                break
            this_target_comm = anchor_comms[ai]

            other_comms = [c for c in communities if c != this_target_comm]
            donor_comm = int(rng.choice(other_comms)) if len(other_comms) else int(this_target_comm)

            k = min(per_anchor, count - produced)

            feats_list: Optional[List[np.ndarray]] = None
            if ctx is not None and ctx.text_gen and node_text:
                exemplars = ctx.community_texts.get(donor_comm, []) or self._texts_for_community(
                    node_text, comm, donor_comm
                )
                topic = self._topic_label(comm, donor_comm) or "a topic unlike the node's neighbors"
                feats_k = generate_text_features(k, topic, exemplars, ctx)
                if feats_k is not None:
                    feats_list = [np.asarray(f, dtype=np.float32) for f in np.atleast_2d(feats_k)]
            if feats_list is None:
                donors = self._sample_real_feats(snap, donor_comm, 2 * k, rng)
                primary, partner = donors[:k], donors[k:]
                feats_list = [
                    mixup_real_features(a, b, float(rng.uniform(0.6, 0.9)), scale, rng,
                                        rel_noise=0.1)
                    for a, b in zip(primary, partner)
                ]

            for j in range(k):
                local_idx = produced + j
                new_feats.append(feats_list[j])
                new_edges.append([int(anchor), local_idx])
                if rng.random() < 0.5:
                    extra = self._one_neighbor_in_community(snap, this_target_comm, exclude=anchor, rng=rng)
                    if extra is not None:
                        new_edges.append([int(extra), local_idx])
                labels.append(0)

            produced += k

        feats = np.stack(new_feats) if new_feats else np.zeros((0, feat_dim), dtype=np.float32)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        return self._make_batch(self.name, feats, new_edges, labels, self.name)

    @staticmethod
    def _default_per_anchor(count: int) -> int:
        return 2 if count <= 200 else 3

    @staticmethod
    def _select_anchors(
        snap: GraphSnapshot,
        n_anchors: int,
        target_comm_hint: Optional[int],
        rng,
    ):
        comm = snap.comm_id
        eligible = AnomalyGenerator._synth_eligible_mask(snap)
        comm_list = np.unique(comm)
        pools: List[np.ndarray] = []
        ordered = list(comm_list)
        rng.shuffle(ordered)
        order_iter = iter(ordered)

        def _next_comm():
            nonlocal order_iter
            try:
                return next(order_iter)
            except StopIteration:
                rng.shuffle(ordered)
                order_iter = iter(ordered)
                try:
                    return next(order_iter)
                except StopIteration:
                    return None

        anchors: List[int] = []
        anchor_comms: List[int] = []

        if target_comm_hint is not None:
            comm_pool = np.where((comm == target_comm_hint) & eligible)[0]
            rng.shuffle(comm_pool)
            for a in comm_pool[:n_anchors]:
                anchors.append(int(a))
                anchor_comms.append(int(target_comm_hint))

        if len(anchors) < n_anchors:
            if target_comm_hint is not None:
                log.warning(
                    "context_attack: target community %s has no eligible "
                    "(train, non-anomalous) nodes -- falling back to "
                    "round-robin anchor spread across communities.",
                    target_comm_hint,
                )
            seen = set(anchors)
            guard = 0
            while len(anchors) < n_anchors and guard < n_anchors * 4:
                guard += 1
                c = _next_comm()
                if c is None:
                    break
                pool = np.where((comm == c) & eligible)[0]
                rng.shuffle(pool)
                picked = False
                for a in pool:
                    if int(a) in seen:
                        continue
                    seen.add(int(a))
                    anchors.append(int(a))
                    anchor_comms.append(int(c))
                    picked = True
                    break
                if not picked:
                    pools.append(pool)

        return np.array(anchors, dtype=np.int64), anchor_comms

    @staticmethod
    def _one_neighbor_in_community(
        snap: GraphSnapshot,
        community: int,
        exclude: int,
        rng,
    ) -> Optional[int]:
        comm = snap.comm_id
        eligible = AnomalyGenerator._synth_eligible_mask(snap)
        pool = np.where((comm == community) & eligible)[0]
        pool = np.setdiff1d(pool, [int(exclude)])
        if len(pool) == 0:
            return None
        return int(rng.choice(pool))

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

    @staticmethod
    def _topic_label(comm: np.ndarray, c: int) -> str:
        size = int(np.sum(comm == c))
        return f"the research theme of community #{c} (≈{size} papers)"