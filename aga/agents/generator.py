
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .base import Agent
from ..data.analyze import GraphSnapshot
from ..patterns import registry as patreg
from ..patterns.base import GenerationContext
from ..orchestration.schema import GenerationStrategy, NewNodeBatch

log = logging.getLogger("aga.generator")


class Generator(Agent):
    name = "generator"

    def __init__(self, seed: int = 42, text_gen: bool = False, embedder: Optional[object] = None):
        self.seed = seed
        self.text_gen = text_gen
        self.embedder = embedder

    def run(
        self,
        strategy: GenerationStrategy,
        snapshot: GraphSnapshot,
        node_text: dict,
        ctx: Optional[GenerationContext] = None,
    ) -> List[NewNodeBatch]:
        batches: List[NewNodeBatch] = []
        if ctx is None:
            ctx = GenerationContext()
        for i, plan in enumerate(strategy.per_pattern):
            builder = patreg.get(plan.pattern)
            if builder is None:
                if plan.spec is not None:
                    sb = patreg.get("spec_attack")
                    if sb is not None:
                        builder = sb
                        plan.spec.family_name = plan.pattern
                        log.info("Using SpecBuilder for proposed pattern %r.", plan.pattern)
                    else:
                        log.warning(
                            "Skipping proposed pattern %r (spec_attack builder not registered).",
                            plan.pattern,
                        )
                        continue
                else:
                    log.warning(
                        "Skipping proposed pattern %r (no builder and no spec). "
                        "Provide a spec or implement + aga.patterns.registry.register to enable.",
                        plan.pattern,
                    )
                    continue
            rng = np.random.default_rng(self.seed + i)
            batch = builder.generate(snapshot, plan, node_text, rng, ctx=ctx)
            batches.append(batch)
            tag = " (text-gen)" if (ctx.text_gen and plan.pattern in ("context_attack", "semantic_camouflage")) else ""
            log.info("Generated %d '%s' anomaly nodes%s", batch.num_nodes, plan.pattern, tag)
        total = sum(b.num_nodes for b in batches)
        log.info("Generator produced %d synthetic anomalies across %d batches.", total, len(batches))
        return batches


def build_generation_context(
    snapshot: GraphSnapshot,
    node_text: dict,
    text_gen: bool,
    embedder: Optional[object] = None,
    adapt_samples: int = 400,
) -> GenerationContext:
    comm = snapshot.comm_id
    community_texts: Dict[int, List[str]] = {}
    if text_gen and node_text:
        for c in np.unique(comm):
            idxs = np.where(comm == c)[0]
            txts = []
            for idx in idxs:
                t = node_text.get(int(idx))
                if t:
                    txts.append(t)
                if len(txts) >= 5:
                    break
            if txts:
                community_texts[int(c)] = txts

    if text_gen and embedder is not None:
        feat = snapshot.g.ndata["feat"].numpy()
        keys = [k for k in node_text.keys() if node_text.get(k)]
        rng = np.random.default_rng(0)
        rng.shuffle(keys)
        keys = keys[:adapt_samples]
        real_texts = [node_text[k] for k in keys]
        real_feats = feat[np.asarray(keys)]
        if real_texts and len(real_feats) > 0:
            embedder.fit_adapter(real_texts, real_feats)
            if not getattr(embedder, "available", False):
                log.warning(
                    "text_gen enabled but encoder unavailable; using deterministic "
                    "pseudo-embeddings (set a real backend + model to get true LLM text)."
                )

    return GenerationContext(
        text_gen=text_gen,
        embedder=embedder,
        community_texts=community_texts,
    )