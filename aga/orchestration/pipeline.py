
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..data import loader, analyze
from ..data.analyze import GraphSnapshot
from ..data.review_sampling import ReviewSamplingConfig, SamplingContext
from ..data import review_evidence
from ..models.split import make_split, frozen_split_signature
from ..agents.inspector import DatasetInspector
from ..agents.gap_analyzer import GapAnalyzer
from ..agents.gap_reviewer import (
    GapReviewer,
    feedback_digest,
    select_approved_gaps,
    validate_strategy_against_review,
    make_empty_strategy,
)
from ..agents.strategy_planner import StrategyPlanner
from ..agents.generator import Generator, build_generation_context
from ..augmentation.inject import inject_batches, synthetic_node_ids
from ..llm import registry as llm
from ..patterns.text_embedder import get_embedder
from ..orchestration.schema import (
    DatasetReport,
    MissingPatterns,
    GenerationStrategy,
    GapReview,
    ReviewState,
)
from ..orchestration import cache as cache_mod

log = logging.getLogger("aga.pipeline")


@dataclass
class PipelineResult:
    report: DatasetReport
    missing: MissingPatterns
    strategy: Optional[GenerationStrategy] = None
    baseline_graph: object = None
    augmented_graph: object = None
    num_synthetic: int = 0
    split: Dict = field(default_factory=dict)
    text_gen: bool = False
    from_cache: bool = False
    review_state: Optional[ReviewState] = None
    reviewed_gaps: List[GapReview] = field(default_factory=list)
    review_paused: bool = False
    review_dir: Optional[str] = None
    refine_rounds: int = 0


_NON_SAMPLING_REVIEW_KEYS = {"llm_precheck", "coverage_criterion", "mode", "max_rounds"}


def _sampling_config(review_config: Optional[Dict]) -> ReviewSamplingConfig:
    cfg = dict(review_config or {})
    known = {f: cfg.pop(f) for f in list(cfg) if f in ReviewSamplingConfig.__dataclass_fields__}
    for k in _NON_SAMPLING_REVIEW_KEYS:
        cfg.pop(k, None)
    if cfg:
        log.warning("Unknown review-config keys ignored: %s", sorted(cfg))
    return ReviewSamplingConfig(**known)


def _generation_budget_hint(split_cfg: Optional[Dict], report: DatasetReport) -> int:
    return int(round((split_cfg or {}).get("budget_ratio", 0.1) * report.stats.num_nodes))


def _generate_and_inject(
    strategy: GenerationStrategy,
    snapshot: GraphSnapshot,
    g,
    node_text: Dict[int, str],
    split: Dict,
    generation_seed: int,
    text_gen: bool,
    embed_model: str,
    allow_download: bool,
    baseline_graph,
    apply_augmentation: bool,
):
    embedder = None
    if text_gen:
        feat_dim = g.ndata["feat"].shape[1]
        embedder = get_embedder(model_name=embed_model, target_dim=feat_dim,
                                allow_download=allow_download)
    ctx = build_generation_context(snapshot, node_text, text_gen, embedder)
    gen = Generator(seed=generation_seed, text_gen=text_gen, embedder=embedder)
    batches = gen.run(strategy, snapshot, node_text, ctx=ctx)

    if apply_augmentation:
        augmented_graph = inject_batches(g, batches)
        syn = synthetic_node_ids(augmented_graph)
        num_synthetic = int(syn.numel())
        log.info("Augmented graph: %d nodes (%d synthetic, train-only)%s.",
                 augmented_graph.num_nodes(), num_synthetic,
                 " [text-gen]" if text_gen else "")
    else:
        augmented_graph = baseline_graph
        num_synthetic = 0
    return augmented_graph, num_synthetic


def _review_cache_block(reviewed: List[GapReview], review_cfg: Optional[Dict]) -> Dict:
    return {
        "sampling": review_cfg or {},
        "feedback_digest": feedback_digest(reviewed) if reviewed else "",
    }


def run_pipeline(
    data_dir: str = "data/",
    dataset: str = "Cora",
    backend: str = "dummy",
    llm_cfg: Optional[Dict] = None,
    split_cfg: Optional[Dict] = None,
    generation_seed: int = 42,
    apply_augmentation: bool = True,
    text_gen: bool = False,
    embed_model: str = "sentence-transformers/msmarco-MiniLM-L-12-v3",
    allow_download: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
    generation_scope: Optional[str] = None,
    review_only: bool = False,
    gap_loop: bool = False,
    review_dir: Optional[str] = None,
    review_config: Optional[Dict] = None,
) -> PipelineResult:
    key = None
    if cache_dir and apply_augmentation and not review_only:
        if gap_loop:
            rc = review_config or {}
            key = cache_mod.build_cache_key(
                dataset=dataset,
                data_dir=data_dir,
                backend=backend,
                llm_cfg=llm_cfg,
                split_cfg=split_cfg,
                generation_seed=generation_seed,
                text_gen=text_gen,
                embed_model=embed_model,
                allow_download=allow_download,
                generation_scope=generation_scope,
                review=cache_mod.gap_loop_cache_component(
                    rc, _sampling_config(rc), backend, llm_cfg),
            )
            if not force_refresh:
                cached = cache_mod.load_augmented(
                    cache_dir, key, data_dir=data_dir, dataset=dataset
                )
                if cached is not None:
                    log.info(
                        "Cache HIT for %s gap-loop (entry %s): refined gaps "
                        "+ strategy + augmented graph reloaded; loop and "
                        "planner LLM calls skipped.",
                        dataset, cache_mod.entry_dir(cache_dir, key).name,
                    )
                    return PipelineResult(
                        report=cached["report"],
                        missing=cached["missing"],
                        strategy=cached["strategy"],
                        baseline_graph=cached["baseline_graph"],
                        augmented_graph=cached["augmented_graph"],
                        num_synthetic=cached["num_synthetic"],
                        split=cached["split"],
                        text_gen=cached["text_gen"],
                        from_cache=True,
                        reviewed_gaps=cached.get("reviewed_gaps", []),
                        review_dir=review_dir or "review_run",
                        refine_rounds=-1,
                    )
        else:
            key = cache_mod.build_cache_key(
                dataset=dataset,
                data_dir=data_dir,
                backend=backend,
                llm_cfg=llm_cfg,
                split_cfg=split_cfg,
                generation_seed=generation_seed,
                text_gen=text_gen,
                embed_model=embed_model,
                allow_download=allow_download,
                generation_scope=generation_scope,
                review={},
            )
            cached = None
            if not force_refresh:
                cached = cache_mod.load_augmented(
                    cache_dir, key, data_dir=data_dir, dataset=dataset
                )
            if cached is not None:
                log.info(
                    "Cache HIT for %s (entry %s): augmented graph reloaded; "
                    "LLM planning (Gap Analyzer + Strategy Planner) skipped.",
                    dataset, cache_mod.entry_dir(cache_dir, key).name,
                )
                return PipelineResult(
                    report=cached["report"],
                    missing=cached["missing"],
                    strategy=cached["strategy"],
                    baseline_graph=cached["baseline_graph"],
                    augmented_graph=cached["augmented_graph"],
                    num_synthetic=cached["num_synthetic"],
                    split=cached["split"],
                    text_gen=cached["text_gen"],
                    from_cache=True,
                    reviewed_gaps=cached.get("reviewed_gaps", []),
                )

    llm.set_backend(backend, **(llm_cfg or {}))

    g, node_text = loader.load_by_name(dataset, data_dir)
    split = make_split(g, seed=(split_cfg or {}).get("seed", 42),
                       val_ratio=(split_cfg or {}).get("val_ratio", 0.10),
                       test_ratio=(split_cfg or {}).get("test_ratio", 0.20))

    baseline_graph = copy.deepcopy(g)

    snapshot = GraphSnapshot.from_graph(g)
    inspector = DatasetInspector(data_dir, dataset=dataset)
    report = inspector.run(snapshot, train_mask=split["train_mask"])

    gap = GapAnalyzer().run(report)

    if gap_loop:
        from .gap_loop import gap_refinement_loop

        rdir = review_dir or "review_run"
        rc = review_config or {}
        loop_result = gap_refinement_loop(
            report=report,
            gaps=gap,
            snapshot=snapshot,
            node_text=node_text,
            split=split,
            review_dir=rdir,
            sampling_config=_sampling_config(rc),
            coverage_criterion=rc.get("coverage_criterion"),
            max_rounds=int(rc.get("max_rounds", 3)),
            data_fingerprint=cache_mod.data_fingerprint(data_dir, dataset),
            pause_after_loop=bool(rc.get("final_review", False)),
        )
        gap, loop_verdicts, loop_state, loop_trace = loop_result

        if loop_state is not None:
            from .gap_loop import GapLoopState as _GLS
            return PipelineResult(
                report=report,
                missing=gap,
                strategy=None,
                baseline_graph=baseline_graph,
                augmented_graph=baseline_graph,
                num_synthetic=0,
                split=split,
                text_gen=text_gen,
                reviewed_gaps=loop_verdicts,
                review_dir=rdir,
                review_state=loop_state,
                review_paused=True,
                refine_rounds=len(loop_trace.rounds),
            )
        budget_hint = _generation_budget_hint(split_cfg, report)
        strategy = StrategyPlanner(total_budget_hint=budget_hint).run(
            report, gap, review_refined=True)
        augmented_graph, num_synthetic = _generate_and_inject(
            strategy, snapshot, g, node_text, split, generation_seed,
            text_gen, embed_model, allow_download, baseline_graph,
            apply_augmentation,
        )
        loop_result = PipelineResult(
            report=report,
            missing=gap,
            strategy=strategy,
            baseline_graph=baseline_graph,
            augmented_graph=augmented_graph,
            num_synthetic=num_synthetic,
            split=split,
            text_gen=text_gen,
            reviewed_gaps=loop_verdicts,
            review_dir=rdir,
            refine_rounds=len(loop_trace.rounds),
        )
        if key is not None and num_synthetic > 0:
            try:
                entry = cache_mod.save_augmented(cache_dir, key, loop_result)
                log.info("Cached gap-loop result at %s (key digest %s).",
                         entry, cache_mod.key_digest(key))
            except Exception:
                log.warning("Failed to cache gap-loop result; continuing.",
                            exc_info=True)
        return loop_result

    if review_only:
        rdir = review_dir or "review_run"
        reviewer = GapReviewer(_sampling_config(review_config))
        sampl_ctx = SamplingContext(snapshot, split["train_mask"])
        state = reviewer.prepare(
            report, gap, sampl_ctx, node_text, rdir,
            data_fingerprint=cache_mod.data_fingerprint(data_dir, dataset),
            split=split,
            llm_precheck=bool((review_config or {}).get("llm_precheck", False)),
            coverage_criterion=(review_config or {}).get("coverage_criterion"),
        )
        return PipelineResult(
            report=report,
            missing=gap,
            strategy=None,
            baseline_graph=baseline_graph,
            augmented_graph=baseline_graph,
            num_synthetic=0,
            split=split,
            text_gen=text_gen,
            review_state=state,
            review_paused=True,
            review_dir=rdir,
        )

    budget_hint = _generation_budget_hint(split_cfg, report)
    strategy = StrategyPlanner(total_budget_hint=budget_hint).run(report, gap)

    augmented_graph, num_synthetic = _generate_and_inject(
        strategy, snapshot, g, node_text, split, generation_seed,
        text_gen, embed_model, allow_download, baseline_graph,
        apply_augmentation,
    )

    result = PipelineResult(
        report=report,
        missing=gap,
        strategy=strategy,
        baseline_graph=baseline_graph,
        augmented_graph=augmented_graph,
        num_synthetic=num_synthetic,
        split=split,
        text_gen=text_gen,
    )

    if key is not None and apply_augmentation and num_synthetic > 0:
        try:
            entry = cache_mod.save_augmented(cache_dir, key, result)
            log.info("Cached augmented graph at %s (key digest %s).",
                     entry, cache_mod.key_digest(key))
        except Exception:
            log.warning("Failed to cache augmented graph; continuing.",
                        exc_info=True)

    return result


class ResumeError(RuntimeError):


def resume_review_pipeline(
    review_dir: str,
    data_dir: str = "data/",
    dataset: str = "Cora",
    backend: str = "dummy",
    llm_cfg: Optional[Dict] = None,
    split_cfg: Optional[Dict] = None,
    generation_seed: int = 42,
    apply_augmentation: bool = True,
    text_gen: bool = False,
    embed_model: str = "sentence-transformers/msmarco-MiniLM-L-12-v3",
    allow_download: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
    generation_scope: Optional[str] = None,
    accept_insufficient: bool = False,
) -> PipelineResult:
    reviewer = GapReviewer()
    state = GapReviewer.load_state(review_dir)

    g, node_text = loader.load_by_name(dataset, data_dir)
    sc = split_cfg or {}
    split = make_split(g, seed=sc.get("seed", 42),
                       val_ratio=sc.get("val_ratio", 0.10),
                       test_ratio=sc.get("test_ratio", 0.20))

    fingerprint = cache_mod.data_fingerprint(data_dir, dataset)
    if fingerprint != state.data_fingerprint:
        raise ResumeError(
            f"dataset files changed since the review was exported "
            f"(fingerprint {state.data_fingerprint} -> {fingerprint}); "
            f"the old evidence/feedback cannot be reused -- re-run --review-only."
        )
    split_sig = frozen_split_signature(split)
    if split_sig != state.split_signature:
        raise ResumeError(
            f"split mismatch (paused {state.split_signature} vs now {split_sig}); "
            f"use the same split seed/ratios as the paused run."
        )

    snapshot = GraphSnapshot.from_graph(g)
    baseline_graph = copy.deepcopy(g)

    inspector = DatasetInspector(data_dir, dataset=dataset)
    report = inspector.run(snapshot, train_mask=split["train_mask"])

    sample_fb = review_evidence.load_sample_feedback(
        os.path.join(review_dir, "sample_feedback.csv")
    )
    gap_fb = review_evidence.load_gap_feedback(
        os.path.join(review_dir, "gap_feedback.json")
    )

    sampl_ctx = SamplingContext(snapshot, split["train_mask"])
    cfg = _sampling_config(state.review_config)
    reviewer = GapReviewer(cfg)
    reviewed, new_state = reviewer.finalize(
        state, sample_fb, gap_fb,
        ctx=sampl_ctx, node_text=node_text, review_dir=review_dir,
        accept_insufficient=accept_insufficient,
    )

    if new_state is not None:
        log.info("Review requested a resample round; pipeline re-paused at %s.",
                 review_dir)
        return PipelineResult(
            report=report,
            missing=MissingPatterns(existing_patterns=[], missing=list(new_state.gaps)),
            strategy=None,
            baseline_graph=baseline_graph,
            augmented_graph=baseline_graph,
            num_synthetic=0,
            split=split,
            text_gen=text_gen,
            review_state=new_state,
            reviewed_gaps=reviewed,
            review_paused=True,
            review_dir=review_dir,
        )

    key = None
    if cache_dir and apply_augmentation:
        key = cache_mod.build_cache_key(
            dataset=dataset,
            data_dir=data_dir,
            backend=backend,
            llm_cfg=llm_cfg,
            split_cfg=split_cfg,
            generation_seed=generation_seed,
            text_gen=text_gen,
            embed_model=embed_model,
            allow_download=allow_download,
            generation_scope=generation_scope,
            review=_review_cache_block(reviewed, state.review_config),
        )
        if not force_refresh:
            cached = cache_mod.load_augmented(
                cache_dir, key, data_dir=data_dir, dataset=dataset
            )
            if cached is not None:
                log.info("Cache HIT for resumed review of %s; generation skipped.",
                         dataset)
                return PipelineResult(
                    report=cached["report"],
                    missing=cached["missing"],
                    strategy=cached["strategy"],
                    baseline_graph=cached["baseline_graph"],
                    augmented_graph=cached["augmented_graph"],
                    num_synthetic=cached["num_synthetic"],
                    split=cached["split"],
                    text_gen=cached["text_gen"],
                    from_cache=True,
                    reviewed_gaps=cached.get("reviewed_gaps", []),
                    review_dir=review_dir,
                )

    gaps_all = MissingPatterns(existing_patterns=[], missing=list(state.gaps))
    approved = select_approved_gaps(reviewed, gaps_all)

    llm.set_backend(backend, **(llm_cfg or {}))
    if approved.missing:
        budget_hint = _generation_budget_hint(split_cfg, report)
        strategy = StrategyPlanner(total_budget_hint=budget_hint).run(
            report, approved, review_approved=True,
        )
        strategy = validate_strategy_against_review(strategy, reviewed)
    else:
        log.info("No gap approved by review; using zero-budget strategy.")
        strategy = make_empty_strategy()

    augmented_graph, num_synthetic = _generate_and_inject(
        strategy, snapshot, g, node_text, split, generation_seed,
        text_gen, embed_model, allow_download, baseline_graph,
        apply_augmentation,
    )

    result = PipelineResult(
        report=report,
        missing=gaps_all,
        strategy=strategy,
        baseline_graph=baseline_graph,
        augmented_graph=augmented_graph,
        num_synthetic=num_synthetic,
        split=split,
        text_gen=text_gen,
        reviewed_gaps=reviewed,
        review_dir=review_dir,
    )

    if key is not None and apply_augmentation and num_synthetic > 0:
        try:
            entry = cache_mod.save_augmented(cache_dir, key, result)
            log.info("Cached augmented graph at %s (key digest %s).",
                     entry, cache_mod.key_digest(key))
        except Exception:
            log.warning("Failed to cache augmented graph; continuing.",
                        exc_info=True)

    return result


def load_gap_loop_state(review_dir: str) -> "GapLoopState":
    from ..orchestration.schema import GapLoopState

    path = os.path.join(review_dir, "loop", "loop_state.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no loop/loop_state.json under {review_dir}: not a paused "
            f"gap-loop final-review run")
    with open(path, encoding="utf-8") as fh:
        return GapLoopState.model_validate_json(fh.read())


def apply_gap_loop_curation(
    state: "GapLoopState",
    gap_feedback: List["GapFeedback"],
) -> MissingPatterns:
    from ..agents.gap_reviewer import gap_definition_digest

    fb = {f.gap_id: f for f in gap_feedback}
    kept: List[MissingPattern] = []
    for g in state.final_gaps:
        f = fb.get(g.gap_id)
        if f is None:
            kept.append(g)
            continue
        if f.review_status == "rejected":
            continue
        if f.review_status == "revised_confirmed" and f.revised_claim:
            kept.append(g.model_copy(deep=True))
            kept[-1].claim = f.revised_claim
            if f.reviewed_scope:
                kept[-1].scope = f.reviewed_scope
            continue
        kept.append(g)
    note = (
        f"[final review applied] kept {len(kept)}, "
        f"rejected {sum(1 for f in fb.values() if f.review_status == 'rejected')}, "
        f"revised {sum(1 for f in fb.values() if f.review_status == 'revised_confirmed')}"
    )
    mp = MissingPatterns(existing_patterns=[], missing=kept,
                         global_note=note)
    log.info("gap-loop final review applied: %s", note)
    return mp


def resume_gap_loop_pipeline(
    review_dir: str,
    data_dir: str = "data/",
    dataset: str = "Cora",
    backend: str = "dummy",
    llm_cfg: Optional[Dict] = None,
    split_cfg: Optional[Dict] = None,
    generation_seed: int = 42,
    apply_augmentation: bool = True,
    text_gen: bool = False,
    embed_model: str = "sentence-transformers/msmarco-MiniLM-L-12-v3",
    allow_download: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
    generation_scope: Optional[str] = None,
) -> PipelineResult:
    state = load_gap_loop_state(review_dir)

    g, node_text = loader.load_by_name(dataset, data_dir)
    sc = split_cfg or {}
    split = make_split(g, seed=sc.get("seed", 42),
                       val_ratio=sc.get("val_ratio", 0.10),
                       test_ratio=sc.get("test_ratio", 0.20))

    fingerprint = cache_mod.data_fingerprint(data_dir, dataset)
    if state.data_fingerprint and fingerprint != state.data_fingerprint:
        raise ResumeError(
            f"dataset files changed since the loop paused "
            f"({state.data_fingerprint} -> {fingerprint}); re-run the loop.")
    split_sig = frozen_split_signature(split)
    if split_sig != state.split_signature:
        raise ResumeError(
            f"split mismatch (paused {state.split_signature} vs now "
            f"{split_sig}); use the same split seed/ratios as the loop run.")

    snapshot = GraphSnapshot.from_graph(g)
    baseline_graph = copy.deepcopy(g)
    inspector = DatasetInspector(data_dir, dataset=dataset)
    report = inspector.run(snapshot, train_mask=split["train_mask"])

    from ..agents.gap_reviewer import gap_definition_digest
    cur_digest = gap_definition_digest(state.final_gaps)
    if state.final_gap_digest and cur_digest != state.final_gap_digest:
        raise ResumeError(
            "final gap set drift: loop_state.json was edited; re-run the loop")

    gap_fb = review_evidence.load_gap_feedback(
        os.path.join(review_dir, "loop", "gap_feedback.json"))
    for f in gap_fb:
        if f.source == "llm" and f.review_status not in ("pending",):
            f.review_status = "pending"
    curated = apply_gap_loop_curation(state, gap_fb)

    key = None
    if cache_dir and apply_augmentation:
        key = cache_mod.build_cache_key(
            dataset=dataset,
            data_dir=data_dir,
            backend=backend,
            llm_cfg=llm_cfg,
            split_cfg=split_cfg,
            generation_seed=generation_seed,
            text_gen=text_gen,
            embed_model=embed_model,
            allow_download=allow_download,
            generation_scope=generation_scope,
            review=cache_mod.gap_loop_cache_component(
                dict(state.review_config or {}),
                ReviewSamplingConfig(**(state.review_config or {}).get(
                    "sampling", {})),
                backend, llm_cfg),
        )
        cur_blob = sorted(
            ({"gap_id": f.gap_id, "status": f.review_status,
              "revised_claim": f.revised_claim,
              "reviewed_scope": f.reviewed_scope}
             for f in (gap_fb or [])),
            key=lambda d: d["gap_id"])
        key["review"]["curation_digest"] = cache_mod.key_digest(
            {"curation": cur_blob})
        if not force_refresh:
            cached = cache_mod.load_augmented(
                cache_dir, key, data_dir=data_dir, dataset=dataset)
            if cached is not None:
                log.info("Cache HIT for gap-loop final review of %s.", dataset)
                return PipelineResult(
                    report=cached["report"],
                    missing=cached["missing"],
                    strategy=cached["strategy"],
                    baseline_graph=cached["baseline_graph"],
                    augmented_graph=cached["augmented_graph"],
                    num_synthetic=cached["num_synthetic"],
                    split=cached["split"],
                    text_gen=cached["text_gen"],
                    from_cache=True,
                    reviewed_gaps=cached.get("reviewed_gaps", []),
                    review_dir=review_dir,
                    refine_rounds=len(state.trace.rounds),
                )

    llm.set_backend(backend, **(llm_cfg or {}))
    budget_hint = _generation_budget_hint(split_cfg, report)
    strategy = StrategyPlanner(total_budget_hint=budget_hint).run(
        report, curated, review_refined=True)
    augmented_graph, num_synthetic = _generate_and_inject(
        strategy, snapshot, g, node_text, split, generation_seed,
        text_gen, embed_model, allow_download, baseline_graph,
        apply_augmentation,
    )
    result = PipelineResult(
        report=report,
        missing=curated,
        strategy=strategy,
        baseline_graph=baseline_graph,
        augmented_graph=augmented_graph,
        num_synthetic=num_synthetic,
        split=split,
        text_gen=text_gen,
        reviewed_gaps=[],
        review_dir=review_dir,
        refine_rounds=len(state.trace.rounds),
    )
    if key is not None and num_synthetic > 0:
        try:
            entry = cache_mod.save_augmented(cache_dir, key, result)
            log.info("Cached gap-loop final-review result at %s (digest %s).",
                     entry, cache_mod.key_digest(key))
        except Exception:
            log.warning("Failed to cache final-review result; continuing.",
                        exc_info=True)
    return result