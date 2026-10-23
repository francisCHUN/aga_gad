
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import analyze
from .analyze import GraphSnapshot

log = logging.getLogger("aga.review.sampling")


@dataclass
class ReviewSamplingConfig:

    targeted: int = 10
    stratified: int = 10
    boundary: int = 5
    seed: int = 42
    max_rounds: int = 2


@dataclass
class SampleRecord:

    sample_id: int
    gap_id: str
    group: str
    reason: str = ""
    stratum: str = ""
    round_no: int = 1


class SamplingContext:

    def __init__(self, snapshot: GraphSnapshot, train_mask: torch.Tensor):
        g = snapshot.g
        self.snapshot = snapshot
        self.g = g
        self.n = int(g.num_nodes())
        mask = train_mask.bool().numpy() if hasattr(train_mask, "bool") \
            else np.asarray(train_mask, dtype=bool)
        self.train_mask = mask
        self.train_idx: np.ndarray = np.where(mask)[0]

        self.comm_id: np.ndarray = (
            snapshot.comm_id if getattr(snapshot, "comm_id", None) is not None
            else analyze.communities(g)[0]
        )
        self.degree: np.ndarray = np.asarray(g.in_degrees().numpy())
        ctx = analyze.context_consistency(g)
        self.ctx_cos: np.ndarray = np.asarray(ctx)
        self.cross_ratio: np.ndarray = self._cross_community_ratio()

        self.label_anomaly = g.ndata["anomaly"].numpy()
        self.label_attr = g.ndata["attr_ano"].numpy()
        self.label_stru = g.ndata["stru_ano"].numpy()
        self._feat_norm = np.linalg.norm(g.ndata["feat"].numpy(), axis=1)

    def _cross_community_ratio(self) -> np.ndarray:
        g = self.g
        n = self.n
        src, dst = g.edges()
        s = src.numpy()
        d = dst.numpy()
        own = np.zeros(n, dtype=np.int64)
        other = np.zeros(n, dtype=np.int64)
        for a, b in zip(s.tolist(), d.tolist()):
            if a == b:
                continue
            same = self.comm_id[a] == self.comm_id[b]
            own[a] += 1 if same else 0
            other[a] += 0 if same else 1
            own[b] += 1 if same else 0
            other[b] += 0 if same else 1
        total = own + other
        ratio = np.zeros(n, dtype=np.float32)
        nz = total > 0
        ratio[nz] = other[nz] / total[nz]
        return ratio

    def neighbours(self, node_id: int) -> np.ndarray:
        src, dst = self.g.in_edges(node_id)
        s = set(src.tolist())
        src2, dst2 = self.g.out_edges(node_id)
        s.update(dst2.tolist())
        s.discard(int(node_id))
        return np.fromiter(sorted(s), dtype=np.int64)

    def train_percentile(self, values: np.ndarray, value: float) -> float:
        train_vals = values[self.train_idx]
        if len(train_vals) == 0:
            return 50.0
        return float(np.mean(train_vals <= value) * 100.0)

    def reference_profile(self) -> Dict:
        anomaly = self.label_anomaly.astype(bool)
        attr = self.label_attr.astype(bool)
        stru = self.label_stru.astype(bool)
        train = self.train_mask

        def _prof(sel: np.ndarray, name: str) -> Optional[Dict]:
            m = sel & train
            if not m.any():
                return None
            deg = self.degree[m]
            ctx = self.ctx_cos[m][self.degree[m] > 0]
            xr = self.cross_ratio[m][self.degree[m] > 0]
            return {
                "n": int(m.sum()),
                "degree_mean": round(float(deg.mean()), 4) if len(deg) else None,
                "ctx_cos_std": round(float(ctx.std()), 4) if len(ctx) else None,
                "ctx_cos_mean": round(float(ctx.mean()), 4) if len(ctx) else None,
                "cross_ratio_mean": round(float(xr.mean()), 4) if len(xr) else None,
                "degree_p10": round(float(np.percentile(deg, 10)), 4) if len(deg) else None,
                "degree_p90": round(float(np.percentile(deg, 90)), 4) if len(deg) else None,
                "note": f"training-labeled {name} nodes (original labels)",
            }

        return {
            "train_nodes": int(train.sum()),
            "labeled_anomaly": _prof(anomaly, "anomaly"),
            "labeled_attr": _prof(attr, "attribute-anomaly"),
            "labeled_stru": _prof(stru, "structure-anomaly"),
            "normal": _prof(~anomaly, "normal"),
            "metric_semantics": (
                "ctx_cos: cosine own-feature vs mean-neighbour-feature (low = "
                "clashes with neighbourhood); cross_community_ratio: fraction "
                "of edges leaving the node's community (high = bridging); "
                "degree_percentile etc. are positions within the TRAIN "
                "distribution. Signals, never verdicts."
            ),
        }

    def _nbr_feat_cos(self, node_id: int) -> Optional[float]:
        nid = int(node_id)
        if self.degree[nid] == 0:
            return None
        return float(self.ctx_cos[nid])

    def evidence_hub(self, node_id: int) -> Dict:
        nid = int(node_id)
        nbr = self.neighbours(nid)
        own = int(sum(1 for x in nbr if self.comm_id[x] == self.comm_id[nid]))
        cross = int(len(nbr) - own)
        ref = self.reference_profile()["labeled_stru"]
        degp = round(self.train_percentile(self.degree, self.degree[nid]), 2)
        ref_deg_mean = ref["degree_mean"] if ref else None
        ref_deg_p90 = ref["degree_p90"] if ref else None
        return {
            "degree": int(self.degree[nid]),
            "degree_train_percentile": degp,
            "labeled_stru_degree_mean": ref_deg_mean,
            "labeled_stru_degree_p90": ref_deg_p90,
            "degree_in_labeled_band": bool(
                ref is not None and ref_deg_p90 is not None
                and self.degree[nid] >= (ref_deg_mean or 0.0)
            ),
            "in_community_edges": own,
            "cross_community_edges": cross,
            "cross_community_ratio": round(float(self.cross_ratio[nid]), 4),
            "n_communities_touched": int(
                len({int(self.comm_id[x]) for x in nbr}) if len(nbr) else 0),
        }

    def evidence_context(self, node_id: int) -> Dict:
        nid = int(node_id)
        cos = self._nbr_feat_cos(nid)
        ref = self.reference_profile()
        attr = ref["labeled_attr"]
        normal = ref["normal"]
        pct = (round(self.train_percentile(self.ctx_cos, cos), 2)
               if cos is not None else None)
        return {
            "nbr_feat_cos": (round(cos, 4) if cos is not None else None),
            "nbr_feat_cos_train_percentile": pct,
            "labeled_attr_ctx_cos_mean": attr["ctx_cos_mean"] if attr else None,
            "normal_ctx_cos_mean": normal["ctx_cos_mean"] if normal else None,
            "deviation_direction": (
                "below_normal_mean" if (cos is not None and normal
                                        and normal["ctx_cos_mean"] is not None
                                        and cos < normal["ctx_cos_mean"]) else
                ("above_normal_mean" if cos is not None else "isolated_node")),
            "n_neighbours": int(self.degree[nid]),
        }

    def evidence_bridge(self, node_id: int) -> Dict:
        nid = int(node_id)
        nbr = self.neighbours(nid)
        by_comm: Dict[int, int] = {}
        for x in nbr:
            c = int(self.comm_id[x])
            by_comm[c] = by_comm.get(c, 0) + 1
        own = self.comm_id[nid]
        pool = self.train_idx[(self.comm_id[self.train_idx] == own)
                              & (self.degree[self.train_idx] > 0)]
        pool_xr = self.cross_ratio[pool]
        control = (round(float(np.median(pool_xr)), 4)
                   if len(pool_xr) else None)
        return {
            "degree": int(self.degree[nid]),
            "n_communities_connected": sorted(by_comm.keys()),
            "edges_per_community": {str(k): v for k, v in sorted(by_comm.items())},
            "cross_community_ratio": round(float(self.cross_ratio[nid]), 4),
            "cross_ratio_train_percentile": round(
                self.train_percentile(self.cross_ratio, self.cross_ratio[nid]), 2),
            "own_community_control_median_cross": control,
            "own_community": int(own),
        }

    def evidence_camouflage(self, node_id: int) -> Dict:
        nid = int(node_id)
        sim = self._nbr_feat_cos(nid)
        ref = self.reference_profile()
        normal = ref["normal"]
        attr = ref["labeled_attr"]
        norm_pct = round(self.train_percentile(
            self._feat_norm, self._feat_norm[nid]), 2)
        return {
            "surface_similarity_nbr_cos": (round(sim, 4)
                                           if sim is not None else None),
            "normal_ctx_cos_mean": normal["ctx_cos_mean"] if normal else None,
            "surface_matches_normal": bool(
                sim is not None and normal and normal["ctx_cos_mean"] is not None
                and abs(sim - normal["ctx_cos_mean"]) <= 0.1),
            "attr_anomaly_ctx_cos_mean": attr["ctx_cos_mean"] if attr else None,
            "feat_norm_train_percentile": norm_pct,
            "degree_train_percentile": round(
                self.train_percentile(self.degree, self.degree[nid]), 2),
        }

    def evidence_for_family(self, family: str, node_id: int) -> Dict:
        f = (family or "").lower()
        if "hub" in f:
            return self.evidence_hub(node_id)
        if "context" in f:
            return self.evidence_context(node_id)
        if "bridge" in f:
            return self.evidence_bridge(node_id)
        if "camouflage" in f:
            return self.evidence_camouflage(node_id)
        if "hybrid" in f:
            return {**self.evidence_context(node_id),
                    **self.evidence_hub(node_id)}
        return self.evidence_context(node_id)


def _probe_low_ctx(ctx: SamplingContext, k: int) -> List[int]:
    pool = ctx.train_idx
    cand = pool[(ctx.degree[pool] > 0) & (ctx.ctx_cos[pool] > -1)]
    order = np.argsort(ctx.ctx_cos[cand], kind="stable")
    return [int(i) for i in cand[order[:k]]]


def _probe_top_degree(ctx: SamplingContext, k: int) -> List[int]:
    pool = ctx.train_idx
    order = np.argsort(-ctx.degree[pool], kind="stable")
    return [int(i) for i in pool[order[:k]]]


def _probe_xcomm(ctx: SamplingContext, k: int) -> List[int]:
    pool = ctx.train_idx
    cand = pool[(ctx.degree[pool] > 0) & (ctx.cross_ratio[pool] > 0)]
    order = np.argsort(-ctx.cross_ratio[cand], kind="stable")
    return [int(i) for i in cand[order[:k]]]


_PROBES = {
    "context": (_probe_low_ctx, _probe_xcomm),
    "semantic_camouflage": (_probe_low_ctx, _probe_xcomm),
    "bridge": (_probe_xcomm, _probe_top_degree),
    "hub": (_probe_top_degree, _probe_xcomm),
    "hybrid": (_probe_low_ctx, _probe_top_degree),
}


def _family_probes(name: str):
    low = name.lower()
    for key, probes in _PROBES.items():
        if key in low:
            return probes
    return (_probe_low_ctx, _probe_top_degree)


_REPO_REASONS = {
    "low_ctx": "lowest ctx_consistency (feature vs neighbourhood)",
    "top_degree": "top degree among train nodes",
    "xcomm": "top cross-community ratio among train nodes",
    "stratified": "stratified random draw (coverage-estimable)",
    "boundary": "near-boundary / normal control",
}


def _targeted(gap, ctx: SamplingContext, k: int) -> List[SampleRecord]:
    probes = _family_probes(gap.name)
    per = max(1, int(np.ceil(k / len(probes))))
    out: List[SampleRecord] = []
    for p in probes:
        ids = p(ctx, per)[:per]
        rname = _REPO_REASONS.get(
            {id(_probe_low_ctx): "low_ctx", id(_probe_top_degree): "top_degree",
             id(_probe_xcomm): "xcomm"}.get(id(p), ""), "probe")
        for nid in ids:
            out.append(SampleRecord(
                sample_id=int(nid), gap_id=gap.gap_id, group="targeted",
                reason=rname, stratum=rname))
    while len(out) > k:
        out.pop()
    return out


def _stratified(gap, ctx: SamplingContext, k: int, rng: np.random.Generator) -> List[SampleRecord]:
    pool = ctx.train_idx
    deg = ctx.degree
    comm = ctx.comm_id
    pool_deg = deg[pool]
    edges = np.quantile(pool_deg, [0.0, 1 / 3, 2 / 3, 1.0]) if len(pool_deg) \
        else np.array([0, 1, 2, 3])
    bands = ("low", "mid", "high")
    cells: List[tuple] = []
    train_set = set(pool.tolist())
    for c in np.unique(comm[pool]):
        cm = pool[comm[pool] == c]
        d = deg[cm]
        for b in range(3):
            if b < 2:
                sel = cm[(d >= edges[b]) & (d < edges[b + 1])]
            else:
                sel = cm[d >= edges[b]]
            if len(sel):
                cells.append((f"comm={int(c)};deg_band={bands[b]}", sel))
    if not cells:
        cells = [("all", pool)]
    out: List[SampleRecord] = []
    order = list(range(len(cells)))
    exhausted = set()
    while len(out) < k and len(exhausted) < len(cells):
        for ci in list(order):
            if ci in exhausted:
                continue
            label, sel = cells[ci]
            if len(sel) == 0:
                exhausted.add(ci)
                continue
            pick = int(sel[rng.integers(0, len(sel))])
            out.append(SampleRecord(
                sample_id=pick, gap_id=gap.gap_id, group="stratified",
                reason=_REPO_REASONS["stratified"], stratum=label))
            cells[ci] = (label, sel[sel != pick])
            if len(cells[ci][1]) == 0:
                exhausted.add(ci)
            if len(out) >= k:
                break
    return out


def _boundary(gap, ctx: SamplingContext, k: int, rng: np.random.Generator) -> List[SampleRecord]:
    out: List[SampleRecord] = []
    normal = ctx.train_idx[ctx.label_anomaly[ctx.train_idx] == 0]
    if len(normal) == 0:
        return out
    kb = k // 2
    if kb > 0:
        cand = normal[ctx.degree[normal] > 0]
        if len(cand):
            order = np.argsort(ctx.ctx_cos[cand], kind="stable")
            mid = order[len(order) // 4: len(order) * 3 // 4]
            for nid in cand[mid[:kb]].tolist():
                out.append(SampleRecord(
                    sample_id=int(nid), gap_id=gap.gap_id, group="boundary",
                    reason="normal-labelled node in the ctx mid-quartiles",
                    stratum="normal;ctx_mid"))
    kr = k - len(out)
    if kr > 0:
        picks = rng.choice(normal, size=min(kr, len(normal)), replace=False)
        for nid in np.sort(picks).tolist():
            out.append(SampleRecord(
                sample_id=int(nid), gap_id=gap.gap_id, group="boundary",
                reason="seeded random normal node (baseline control)",
                stratum="normal;random"))
    return out


def sample_for_gap(
    gap,
    ctx: SamplingContext,
    cfg: ReviewSamplingConfig,
    round_no: int = 1,
    exclude: Optional[set] = None,
) -> List[SampleRecord]:
    rng = np.random.default_rng(int(cfg.seed) + 9973 * (round_no - 1) + 17)
    exclude = exclude or set()

    def _filt(records: List[SampleRecord]) -> List[SampleRecord]:
        return [r for r in records if r.sample_id not in exclude]

    targeted = _filt(_targeted(gap, ctx, cfg.targeted))
    strat = _filt(_stratified(gap, ctx, cfg.stratified, rng))
    boundary = _filt(_boundary(gap, ctx, cfg.boundary, rng))

    for name, want, got in (("targeted", cfg.targeted, len(targeted)),
                            ("stratified", cfg.stratified, len(strat)),
                            ("boundary", cfg.boundary, len(boundary))):
        if got < want:
            log.warning("gap %s round %d: %s drew %d/%d after exclusions "
                        "(pool exhausted; shortfall recorded, never padded)",
                        gap.gap_id, round_no, name, got, want)
    return targeted + strat + boundary


def sample_all_gaps(
    gaps: Sequence,
    ctx: SamplingContext,
    cfg: ReviewSamplingConfig,
    round_no: int = 1,
    previous: Optional[Dict[str, List[SampleRecord]]] = None,
) -> Dict[str, List[SampleRecord]]:
    out: Dict[str, List[SampleRecord]] = {}
    for g in gaps:
        gid = g.gap_id or "gap-000"
        prev = {r.sample_id for r in (previous or {}).get(gid, [])}
        out[gid] = sample_for_gap(g, ctx, cfg, round_no=round_no,
                                  exclude=(prev or None))
    return out