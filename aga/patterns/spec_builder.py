
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .base import (
    AnomalyGenerator,
    GenerationContext,
    feature_scale,
    generate_text_features,
    mixup_real_features,
)
from ..orchestration.schema import AnomalySpec, NewNodeBatch, PatternPlan
from ..data.analyze import GraphSnapshot

log = logging.getLogger("aga.specbuilder")


class SpecBuilder(AnomalyGenerator):

    name = "spec_attack"

    def generate(
        self,
        snap: GraphSnapshot,
        plan: PatternPlan,
        node_text: dict,
        rng,
        ctx: Optional[GenerationContext] = None,
    ) -> NewNodeBatch:
        spec: Optional[AnomalySpec] = plan.spec
        if spec is None:
            raise ValueError(
                "SpecBuilder requires plan.spec; fixed-core names should route to "
                "their own builder."
            )
        count = int(plan.count)
        feat_dim = snap.g.ndata["feat"].shape[1]
        if count <= 0:
            return self._make_batch(spec.family_name, np.zeros((0, feat_dim)), [], [],
                                    spec.family_name, family_type=spec.family_type)

        feats = self._resolve_features(spec, snap, count, rng, ctx, node_text)
        new_edges, new_new_edges = self._resolve_edges(spec, snap, count, rng)
        labels = self._resolve_labels(spec, snap, count)
        new_edges, new_new_edges = self._enforce_degree_cap(
            new_edges, new_new_edges, count, spec.degree_cap, rng
        )
        return self._make_batch(
            spec.family_name, feats, new_edges, labels, spec.family_name,
            family_type=spec.family_type, new_new_edges=new_new_edges,
        )

    def _resolve_features(
        self,
        spec: AnomalySpec,
        snap: GraphSnapshot,
        count: int,
        rng,
        ctx: Optional[GenerationContext],
        node_text: dict,
    ) -> np.ndarray:
        fs = spec.feature_source
        communities = [int(c) for c in np.unique(snap.comm_id)]
        scale = feature_scale(snap)
        rel_noise = float(fs.noise_scale)

        def borrow(community: Optional[int], n: int) -> np.ndarray:
            noise_floor = max(rel_noise, 0.05)
            donors = self._sample_real_feats(snap, community, 2 * n, rng)
            primary, partner = donors[:n], donors[n:]
            return np.stack(
                [
                    mixup_real_features(a, b, 0.75, scale, rng, noise_floor)
                    for a, b in zip(primary, partner)
                ]
            )

        kind = fs.kind

        if kind == "own_community_centroid":
            return borrow(fs.target_community, count)

        if kind == "donor_centroid":
            donor = fs.donor_community
            if donor is None:
                cand = [c for c in communities if c != fs.target_community] or communities
                donor = int(rng.choice(cand))
            return borrow(donor, count)

        if kind == "blend_centroids":
            a = fs.donor_community if fs.donor_community is not None else fs.target_community
            b = fs.second_community if fs.second_community is not None else fs.target_community
            alpha = float(fs.alpha)
            feats_a = self._sample_real_feats(snap, a, count, rng)
            feats_b = self._sample_real_feats(snap, b, count, rng)
            return np.stack(
                [
                    mixup_real_features(x, y, alpha, scale, rng, rel_noise)
                    for x, y in zip(feats_a, feats_b)
                ]
            )

        if kind == "mean_plus_noise":
            return borrow(None, count)

        if kind == "permutation":
            target_c = fs.target_community
            pool = self._sample_normal_nodes(snap, max(count, 1), rng, community=target_c)
            if len(pool) == 0:
                return borrow(None, count)
            feat = snap.g.ndata["feat"].numpy()
            feats = []
            for i in range(count):
                src = feat[int(pool[i % len(pool)])].astype(np.float32)
                feats.append(src[rng.permutation(src.shape[0])].astype(np.float32))
            return np.stack(feats)

        if kind == "text_gen":
            if ctx is not None and ctx.text_gen and node_text:
                donor = fs.donor_community if fs.donor_community is not None else fs.target_community
                exemplars = ctx.community_texts.get(int(donor), []) if donor is not None else []
                topic = f"the research theme of community #{donor}" if donor is not None else "a graph anomaly topic"
                feats_arr = generate_text_features(count, topic, exemplars, ctx)
                if feats_arr is not None:
                    return np.atleast_2d(np.asarray(feats_arr, dtype=np.float32))
            log.warning("spec feature_source=text_gen but text_gen ctx unavailable; "
                        "falling back to real donor-row features.")
            donor = fs.donor_community
            if donor is None:
                cand = [c for c in communities if c != fs.target_community] or communities
                donor = int(rng.choice(cand))
            return borrow(donor, count)

        return borrow(None, count)

    def _resolve_edges(
        self,
        spec: AnomalySpec,
        snap: GraphSnapshot,
        count: int,
        rng,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        et = spec.edge_topology
        kind = et.kind

        if kind == "isolated":
            return [], []

        if kind == "bridge_two_communities":
            cA = et.target_community
            cB = et.second_community
            communities_sorted = self._two_distant_communities(snap)
            if cA is None:
                cA = communities_sorted[0]
            if cB is None:
                cB = communities_sorted[1] if communities_sorted[1] != cA else communities_sorted[0]
            poolA = self._pool(snap, et, cA, rng)
            poolB = self._pool(snap, et, cB, rng)
            new_edges: List[List[int]] = []
            for i in range(count):
                new_edges.append([int(poolA[i % len(poolA)]), i])
                new_edges.append([int(poolB[i % len(poolB)]), i])
            return new_edges, []

        if kind == "star":
            anchor = self._pick_anchors(snap, et, 1, rng)[0]
            return [[int(anchor), i] for i in range(count)], []

        if kind == "single_edge":
            anchor = self._pick_anchors(snap, et, 1, rng)[0]
            return [[int(anchor), 0]] if count > 0 else [], []

        if kind == "clique":
            new_edges: List[List[int]] = []
            if et.target_community is not None:
                anchors = self._pick_anchors(snap, et, count, rng)
                for i in range(count):
                    new_edges.append([int(anchors[i]), i])
            clique_n = count if et.clique_size is None else min(int(et.clique_size), count)
            new_new: List[List[int]] = []
            start = 0
            while start < count:
                end = min(start + clique_n, count)
                for a in range(start, end):
                    for b in range(a + 1, end):
                        new_new.append([a, b])
                start = end
            return new_edges, new_new

        return self._attach_to_anchors(spec, snap, count, rng), []

    def _attach_to_anchors(self, spec, snap, count, rng) -> List[List[int]]:
        et = spec.edge_topology
        per_node = max(1, int(et.edges_per_node))
        anchors = self._pick_anchors(snap, et, count * per_node + 4, rng)
        comm = snap.comm_id
        new_edges: List[List[int]] = []
        ai = 0
        for i in range(count):
            for _ in range(per_node):
                new_edges.append([int(anchors[ai % len(anchors)]), i])
                ai += 1
            if rng.random() < float(et.extra_neighbor_prob) and et.target_community is not None:
                extra = self._one_neighbor_in_community(snap, int(et.target_community), exclude=int(anchors[ai % len(anchors)]), rng=rng)
                if extra is not None:
                    new_edges.append([extra, i])
        return new_edges

    def _resolve_labels(self, spec: AnomalySpec, snap: GraphSnapshot, count: int) -> List[int]:
        explicit = spec.feature_source.label
        if explicit is not None:
            return [int(explicit)] * count
        target_c = spec.feature_source.target_community or spec.edge_topology.target_community
        if "label" in snap.g.ndata and target_c is not None:
            comm = snap.comm_id
            labels = snap.g.ndata["label"].numpy()
            mask = comm == int(target_c)
            if mask.sum() > 0:
                vals, counts = np.unique(labels[mask], return_counts=True)
                maj = int(vals[np.argmax(counts)])
                return [maj] * count
        return [0] * count

    @staticmethod
    def _two_distant_communities(snap: GraphSnapshot) -> List[int]:
        comm = snap.comm_id
        communities = np.unique(comm)
        communities_sorted = [int(c) for c in communities if np.sum(comm == c) >= 5]
        if len(communities_sorted) < 2:
            communities_sorted = [int(communities[0]), int(communities[-1])]
        return [communities_sorted[0], communities_sorted[1]]

    def _pool(self, snap, et, community, rng) -> np.ndarray:
        comm = snap.comm_id
        eligible = AnomalyGenerator._synth_eligible_mask(snap)
        candidates = np.where((comm == int(community)) & eligible)[0]
        if len(candidates) == 0:
            candidates = np.where(comm == int(community))[0]
        if len(candidates) == 0:
            candidates = np.where(eligible)[0]
        rng.shuffle(candidates)
        return candidates

    def _pick_anchors(self, snap, et, n: int, rng) -> np.ndarray:
        sel = et
        target_c = sel.target_community
        if target_c is None:
            comm_list = list(np.unique(snap.comm_id))
            rng.shuffle(comm_list)
            eligible = AnomalyGenerator._synth_eligible_mask(snap)
            anchors: List[int] = []
            guard = 0
            it = iter(comm_list)
            while len(anchors) < n and guard < n * 6:
                guard += 1
                try:
                    c = int(next(it))
                except StopIteration:
                    rng.shuffle(comm_list)
                    it = iter(comm_list)
                    continue
                pool = np.where((snap.comm_id == c) & eligible)[0]
                if len(pool) == 0:
                    continue
                anchors.append(int(rng.choice(pool)))
            if not anchors:
                anchors = [int(x) for x in np.where(eligible)[0][:max(n, 1)]]
            return np.array(anchors[:n] if len(anchors) >= n else anchors, dtype=np.int64)
        pool = self._pool(snap, et, target_c, rng)
        if len(pool) == 0:
            pool = np.where(AnomalyGenerator._synth_eligible_mask(snap))[0]
        if et.anchor_spread:
            return pool[:n] if len(pool) >= n else pool
        return np.array([int(pool[0])] * n, dtype=np.int64) if len(pool) else np.array([], dtype=np.int64)

    @staticmethod
    def _one_neighbor_in_community(snap, community: int, exclude: int, rng) -> Optional[int]:
        comm = snap.comm_id
        eligible = AnomalyGenerator._synth_eligible_mask(snap)
        pool = np.where((comm == community) & eligible)[0]
        pool = np.setdiff1d(pool, [int(exclude)])
        if len(pool) == 0:
            return None
        return int(rng.choice(pool))

    @staticmethod
    def _enforce_degree_cap(
        new_edges: List[List[int]],
        new_new_edges: List[List[int]],
        count: int,
        degree_cap: int,
        rng,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        if degree_cap < 0:
            degree_cap = 0
        if degree_cap >= 20:
            return new_edges, new_new_edges
        per_node: List[List[List[int]]] = [[] for _ in range(count)]
        per_node_intra: List[List[List[int]]] = [[] for _ in range(count)]
        for e in new_edges:
            li = int(e[1])
            if 0 <= li < count:
                per_node[li].append(e)
        for e in new_new_edges:
            for li in (int(e[0]), int(e[1])):
                if 0 <= li < count:
                    per_node_intra[li].append(e)
        out_orig: List[List[int]] = []
        kept_intra: List[List[int]] = []
        used_pairs = set()
        for li in range(count):
            orig_edges = per_node[li]
            if len(orig_edges) > degree_cap:
                rng.shuffle(orig_edges)
                orig_edges = orig_edges[:degree_cap]
            out_orig.extend(orig_edges)
            remaining = degree_cap - len(orig_edges)
            intra = per_node_intra[li]
            remaining_intra = [e for e in intra if (e[0], e[1]) not in used_pairs]
            if remaining <= 0:
                continue
            if len(remaining_intra) > remaining:
                rng.shuffle(remaining_intra)
                remaining_intra = remaining_intra[:remaining]
            for e in remaining_intra:
                key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                if key in used_pairs:
                    continue
                used_pairs.add(key)
                kept_intra.append([int(e[0]), int(e[1])])
        return out_orig, kept_intra