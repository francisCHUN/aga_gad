
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..orchestration.schema import NewNodeBatch, PatternPlan
from ..data.analyze import GraphSnapshot

log = logging.getLogger("aga.patterns.base")


_FAMILY_TYPE_MAP: Dict[str, str] = {
    "context_attack": "attribute",
    "semantic_camouflage": "attribute",
    "bridge_attack": "structure",
    "hybrid_attack": "both",
    "hub_attack": "structure",
}


@dataclass
class GenerationContext:

    text_gen: bool = False
    embedder: Optional[object] = None
    community_texts: Dict[int, List[str]] = field(default_factory=dict)


class AnomalyGenerator(ABC):

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        snap: GraphSnapshot,
        plan: PatternPlan,
        node_text: dict,
        rng,
        ctx: Optional[GenerationContext] = None,
    ) -> NewNodeBatch:
        raise NotImplementedError

    @staticmethod
    def _feature_centroids_by_community(snap: GraphSnapshot) -> Dict[int, np.ndarray]:
        feat = snap.g.ndata["feat"].numpy()
        comm = snap.comm_id
        out: Dict[int, np.ndarray] = {}
        for c in np.unique(comm):
            mask = comm == c
            if mask.sum() == 0:
                continue
            out[int(c)] = feat[mask].mean(axis=0)
        return out

    @staticmethod
    def _synth_eligible_mask(snap: GraphSnapshot) -> np.ndarray:
        anomaly = snap.g.ndata["anomaly"].numpy()
        normal = anomaly == 0
        if "train_mask" in snap.g.ndata:
            eligible = normal & snap.g.ndata["train_mask"].bool().numpy()
            if eligible.any():
                return eligible
        return normal

    @staticmethod
    def _sample_normal_nodes(
        snap: GraphSnapshot,
        n: int,
        rng,
        community: int = None,
    ) -> np.ndarray:
        eligible = AnomalyGenerator._synth_eligible_mask(snap)
        candidates = np.where(eligible)[0]
        if community is not None:
            comm = snap.comm_id
            in_comm = candidates[comm[candidates] == community]
            if len(in_comm) > 0:
                candidates = in_comm
            else:
                pass
        if len(candidates) == 0:
            candidates = np.where(snap.g.ndata["anomaly"].numpy() == 0)[0]
        rng.shuffle(candidates)
        return candidates[:n]

    @staticmethod
    def _sample_real_feats(
        snap: GraphSnapshot,
        community: Optional[int],
        n: int,
        rng,
    ) -> np.ndarray:
        comm = snap.comm_id
        anomaly = snap.g.ndata["anomaly"].numpy()
        eligible = AnomalyGenerator._synth_eligible_mask(snap)
        if community is not None:
            pool = np.where((comm == community) & eligible)[0]
            if len(pool) == 0:
                pool = np.where(comm == community)[0]
        else:
            pool = np.where(eligible)[0]
        if len(pool) == 0:
            pool = np.arange(comm.shape[0])
        if len(pool) == 0:
            raise ValueError("empty graph: no real feature donors available")
        rng.shuffle(pool)
        idx = pool[np.arange(n) % len(pool)]
        return snap.g.ndata["feat"].numpy()[idx].astype(np.float32)

    @staticmethod
    def _make_batch(
        pattern: str,
        features: np.ndarray,
        new_edges: List[List[int]],
        labels: List[int],
        anomaly_family: str,
        family_type: Optional[str] = None,
        new_new_edges: Optional[List[List[int]]] = None,
    ) -> NewNodeBatch:
        feats_list = features.tolist()
        if len(feats_list) > 0 and not isinstance(feats_list[0], list) and features.ndim == 1:
            feats_list = [[float(x)] for x in feats_list]
        if family_type is None:
            family_type = _FAMILY_TYPE_MAP.get(anomaly_family, "unknown")
        return NewNodeBatch(
            pattern=pattern,
            num_nodes=int(features.shape[0]),
            features=feats_list,
            new_edges=new_edges,
            labels=labels,
            anomaly_family=anomaly_family,
            family_type=family_type,
            new_new_edges=new_new_edges or [],
        )



DEFAULT_REL_NOISE = 0.05


def feature_scale(snap: GraphSnapshot) -> np.ndarray:
    feat = snap.g.ndata["feat"].numpy()
    std = feat.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return std.astype(np.float32)


def calibrate_norm(y: np.ndarray, target_norm: float) -> np.ndarray:
    cur = float(np.linalg.norm(y))
    if cur > 1e-9 and target_norm > 1e-9:
        y = y * (target_norm / cur)
    return y


def borrow_real_feature(
    x: np.ndarray,
    scale: np.ndarray,
    rng,
    rel_noise: float = DEFAULT_REL_NOISE,
) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    if rel_noise > 0:
        noise = rng.normal(0.0, 1.0, y.shape).astype(np.float32) * (scale * rel_noise)
        y = y + noise
    return calibrate_norm(y, float(np.linalg.norm(x))).astype(np.float32)


def mixup_real_features(
    xa: np.ndarray,
    xb: np.ndarray,
    lam: float,
    scale: np.ndarray,
    rng,
    rel_noise: float = DEFAULT_REL_NOISE,
) -> np.ndarray:
    xa = np.asarray(xa, dtype=np.float32)
    xb = np.asarray(xb, dtype=np.float32)
    y = lam * xa + (1.0 - lam) * xb
    if rel_noise > 0:
        noise = rng.normal(0.0, 1.0, y.shape).astype(np.float32) * (scale * rel_noise)
        y = y + noise
    target = lam * float(np.linalg.norm(xa)) + (1.0 - lam) * float(np.linalg.norm(xb))
    return calibrate_norm(y.astype(np.float32), target).astype(np.float32)


_TEXT_SYSTEM = (
    "You generate realistic synthetic paper/abstract text for graph anomaly "
    "research. You are given a few real example texts and a target topic. Write "
    "{n} NEW distinct short paper title+abstract snippets (each 1-3 sentences) "
    "that read as belonging to the TOPIC but are NOT copies of the examples. "
    "Return them as a plain numbered list, one per line, starting with '1.'. "
    "No commentary, no code."
)


def _build_text_prompt(
    n: int,
    target_topic: str,
    exemplar_texts: List[str],
) -> str:
    examples = "\n".join(f"- {t[:400]}" for t in exemplar_texts[:3]) or "- (no examples)"
    return (
        f"Generate {n} synthetic texts.\n"
        f"Seed topic: {target_topic}\n"
        f"Real example texts to imitate the *style* and length of:\n{examples}\n\n"
        f"Now write {n} new texts as a numbered list."
    )


def generate_text_features(
    count: int,
    target_topic: str,
    exemplar_texts: List[str],
    ctx: GenerationContext,
):
    from ..llm import registry as llm
    system = _TEXT_SYSTEM.format(n=count)
    user = _build_text_prompt(count, target_topic, exemplar_texts)
    try:
        texts = llm.generate_text(system, user, n=count)
    except Exception as exc:
        log.warning(
            "text-gen LLM call failed (%s); degrading this batch of %d nodes "
            "to deterministic real-node-mixup features (count preserved).",
            exc, count,
        )
        return None
    while len(texts) < count:
        texts = texts + [texts[-1] if texts else f"{target_topic}. (synthetic)"]
    texts = texts[:count]
    embedder = ctx.embedder
    feats = embedder.encode(texts)
    return np.asarray(feats, dtype=np.float32)