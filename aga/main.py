from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import yaml

from aga.orchestration.pipeline import (
    run_pipeline,
    resume_review_pipeline,
    resume_gap_loop_pipeline,
)
from aga.models.train import train_compare, TrainConfig

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _print_report(result):
    r = result.report
    print("=" * 72)
    print("AGENT 1 — Dataset Inspector (deterministic)")
    print("=" * 72)
    s = r.stats
    print(f"  dataset      : {s.dataset}")
    print(f"  nodes/edges  : {s.num_nodes} / {s.num_edges}")
    print(f"  feat dim     : {s.num_features}")
    print(f"  anomalies    : {s.num_total_anomalies} "
          f"(attr {s.num_attr_anomalies}, stru {s.num_stru_anomalies})")
    print(f"  avg degree   : {s.avg_degree} (std {s.degree_std})")
    print(f"  homophily    : {s.homophily}")
    print(f"  communities  : {s.num_communities}")
    print("  existing anomaly coverage:")
    for fam, bar in r.coverage_bars.items():
        cov = next((c for c in r.coverage if c.family == fam), None)
        if cov is not None and cov.status == "unmeasured":
            print(f"    {fam:22s} {bar}  (unmeasured)")
        else:
            score = cov.score if cov is not None else 0.0
            print(f"    {fam:22s} {bar}  ({score:.2f})")
    print(f"  summary      : {r.summary}")
    print()


def _print_missing(result):
    m = result.missing
    print("=" * 72)
    print("AGENT 2 — Gap Analyzer (LLM reasoning)")
    print("=" * 72)
    print(f"  existing patterns : {m.existing_patterns}")
    print("  missing patterns  :")
    for x in m.missing:
        tag = " (PROPOSED, no builder yet)" if x.proposed else ""
        print(f"    - {x.name:22s} severity={x.severity:.2f}{tag}")
        print(f"        {x.rationale}")
    print()


def _print_strategy(result):
    if result.strategy is None:
        print("=" * 72)
        print("AGENT 3 — Strategy Planner: SKIPPED (review paused; no generation)")
        print("=" * 72)
        print()
        return
    st = result.strategy
    print("=" * 72)
    print("AGENT 3 — Strategy Planner (LLM reasoning -> typed plan)")
    print("=" * 72)
    print(f"  total new anomalies : {st.total_new_anomalies}")
    for p in st.per_pattern:
        print(f"    - {p.pattern:22s} count={p.count} "
              f"target_community={p.target_community} transfer_from={p.transfer_attribute_from} "
              f"keep_degree={p.keep_degree_distribution}")
    print(f"  global constraints  : {st.global_constraints.model_dump(exclude_none=True)}")
    print(f"  rationale           : {st.rationale}")
    print()


def _print_review(result):
    if not result.reviewed_gaps and not result.review_state:
        return
    print("=" * 72)
    print("GAP REVIEWER — evidence-grounded gap verification")
    print("=" * 72)
    if getattr(result, "refine_rounds", 0) and not result.review_paused:
        print(f"  gap refinement loop : {result.refine_rounds} round(s) "
              f"(unconstrained handoff to the planner)")
        print(f"  audit bundle        : {result.review_dir}/loop/ "
              f"(loop_trace.json + final review package)")
    if result.review_paused:
        is_loop_final = getattr(result, "refine_rounds", 0) and \
            result.review_state is not None and \
            result.review_state.__class__.__name__ == "GapLoopState"
        print(f"  status       : PAUSED awaiting human feedback in '{result.review_dir}'")
        if is_loop_final:
            print("  files        : loop/review_report.html / loop/gap_feedback.json")
            print(f"  refined gaps : {len(result.review_state.final_gaps)} "
                  f"gap(s) from {result.refine_rounds} loop round(s)")
            print("  Curate the refined gap set (confirmed=keep, "
                  "rejected=remove, revised_confirmed=edit), then resume "
                  "with --resume-gap-loop.")
        else:
            print("  files        : review_report.html / sample_feedback.csv / gap_feedback.json")
            if result.review_state is not None:
                n_gaps = len(result.review_state.gaps)
                n_cards = len(result.review_state.samples)
                print(f"  gaps/samples : {n_gaps} gap(s), {n_cards} sample card(s), "
                      f"sampling round {result.review_state.sampling_round}")
            print("  Fill in the two feedback files, then resume with --resume-review.")
        print()
        return
    for r in result.reviewed_gaps:
        print(f"  [{r.gap_id}] {r.pattern_name:22s} -> {r.review_status}")
        if r.review_status in ("confirmed", "revised_confirmed"):
            eff_claim = r.revised_claim or r.original_claim
            eff_scope = r.reviewed_scope or r.original_scope
            print(f"      effective claim : {eff_claim}")
            print(f"      effective scope : {eff_scope}")
        print(f"      evidence        : {r.evidence_summary}")
        if r.limitations:
            for lim in r.limitations[:3]:
                print(f"      limitation      : {lim}")
        if r.human_comment:
            print(f"      human comment   : {r.human_comment}")
    n_approved = sum(1 for r in result.reviewed_gaps
                     if r.review_status in ("confirmed", "revised_confirmed"))
    print(f"  approved gaps: {n_approved}/{len(result.reviewed_gaps)} "
          f"(only approved gaps enter generation)")
    print()


def _print_generation(result):
    print("=" * 72)
    print("AGENT 4 — Generator (deterministic DGL builders) + Injection")
    print("=" * 72)
    print(f"  original nodes  : {result.baseline_graph.num_nodes()}")
    print(f"  augmented nodes : {result.augmented_graph.num_nodes()}")
    print(f"  synthetic nodes : {result.num_synthetic}  (train_mask=True only)")
    if result.text_gen:
        print("  feature source  : LLM-written text -> sentence-transformer embedding")
    else:
        print("  feature source  : community-centroid blend (no LLM text)")
    print()


def main(argv=None):
    p = argparse.ArgumentParser(description="AGA: Agent-Guided Adaptive Anomaly Generation for GAD.")
    p.add_argument("--config", default=None, help="path to a YAML config (defaults to configs/default.yaml)")
    p.add_argument("--dataset", default=None,
                   help="dataset name (Cora|Pubmed|Arxiv|History); falls back to config dataset.name")
    p.add_argument("--data-dir", default=None,
                   help="data directory; falls back to config dataset.data_dir")
    p.add_argument("--backend", default=None,
                   help="dummy|openai|sohoyo|anthropic|local "
                        "(sohoyo: https://openai.sohoyo.io gateway, model gpt-6-astra; "
                        "needs the SOHOYO_API_KEY env var)")
    p.add_argument("--report-only", action="store_true",
                   help="run agents 1-3 and print the plan; do not mutate the graph")
    p.add_argument("--review-only", action="store_true",
                   help="run agents 1-2, export the gap-review package "
                        "(review_report.html / sample_feedback.csv / gap_feedback.json) "
                        "to --review-dir and PAUSE: no strategy, no generation. "
                        "Resume later with --resume-review.")
    p.add_argument("--llm-precheck", action="store_true",
                   help="with --review-only: run the LLM pre-reviewer after sampling "
                        "and pre-fill its suggestions into the two feedback templates "
                        "(marked source=llm). A human still confirms/overrides every "
                        "suggestion in the workbench. Also settable via config "
                        "review.llm_precheck.")
    p.add_argument("--coverage-criterion", default=None,
                   help="the agreed basis for CONFIRMING a gap (task coverage "
                        "requirement / community coverage / review guideline), "
                        "forwarded to the LLM gap pre-review and recorded with the "
                        "review. Without it, weak evidence honestly leans "
                        "insufficient_evidence. Also settable via config "
                        "review.coverage_criterion.")
    p.add_argument("--trust-llm-review", action="store_true",
                   help="(reserved for the ablation study) accept the LLM pre-review "
                        "suggestions directly on resume, skipping human confirmation. "
                        "Currently a no-op placeholder: the prefill+human-confirm flow "
                        "is the default.")
    p.add_argument("--review-dir", default="review_run",
                   help="directory for the review package (export + feedback + state); "
                        "default: review_run")
    p.add_argument("--resume-review", default=None, metavar="REVIEW_DIR",
                   help="resume a paused review run: load the human feedback from the "
                        "review directory, finalize the review, and continue with ONLY "
                        "the approved gaps (zero generation if none pass)")
    p.add_argument("--accept-insufficient", action="store_true",
                   help="resume exit for evidence-dead gaps: insufficient_evidence gaps "
                        "are APPROVED for generation despite unresolved evidence, each "
                        "flagged accepted_despite_insufficient in the verdicts and "
                        "cache key (never silently); rejected/pending still never pass")
    p.add_argument("--review-gap-loop", dest="review_gap_loop",
                   action="store_true",
                   help="gap-loop WITH a human final review: the automatic "
                        "refinement loop runs to convergence/budget, then "
                        "PAUSES -- loop/gap_feedback.json + workbench are "
                        "exported for curating the refined gap set (drop a "
                        "gap / revise its claim/scope). Continue with "
                        "--resume-gap-loop. Implies --gap-loop.")
    p.add_argument("--resume-gap-loop", default=None, metavar="REVIEW_DIR",
                   help="continue a paused --review-gap-loop run: apply the "
                        "human curation from loop/gap_feedback.json to the "
                        "refined gap set and plan + generate (planner still "
                        "unconstrained). Mutually exclusive with --review-only"
                        " / --resume-review / --gap-loop.")
    p.add_argument("--gap-loop", dest="gap_loop", action="store_true",
                   help="fully-automatic gap refinement loop: after the Gap Analyzer, "
                        "run LLM-reviewer <-> gap-refiner rounds (fresh sampled "
                        "evidence each round, verdict-driven refine, stop on stable "
                        "gap definitions or the round budget) and hand the refined "
                        "gap set to the Strategy Planner UNCONSTRAINED (no "
                        "approved-gap filter, no strategy pruning). Audit bundle: "
                        "--review-dir/loop/. No pause; CI-safe; mutually exclusive "
                        "with --review-only / --resume-review.")
    p.add_argument("--train-compare", action="store_true",
                   help="train a GAD model baseline vs augmented and print AUC/F1 on the frozen test set")
    p.add_argument("--embedding-check", action="store_true",
                   help="比较原始/生成节点的原始特征(送入模型前的 Embedding)余弦相似度: "
                        "对每个合成异常节点取其与最近原始正常/最近原始异常节点的相似度, "
                        "输出 直方图+箱线图 统计图")
    p.add_argument("--model", default=None,
                   help="GAD model: dominant | cola | gcn | gat (default: dominant, overridable in config training.model)")
    p.add_argument("--text-gen", action="store_true",
                   help="use the LLM to WRITE each synthetic node's text, then embed via sentence-transformers (context_attack & semantic_camouflage). Falls back to pseudo-embeddings if no encoder/model is available.")
    p.add_argument("--embed-model", default="/Users/wuchunqi/.cache/modelscope/hub/models/sentence-transformers/msmarco-MiniLM-L-12-v3",
                   help="sentence-transformers model id for encoding generated node text")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--runs", type=int, default=5,
                   help="number of repeated train_compare runs (each with a different seed) "
                   "to average over; reports mean +/- std per metric (default: 5)")
    p.add_argument("--cache-dir", default=None,
                   help="augmented-graph cache directory (enables the cache). Hits skip the "
                        "two LLM planning calls. Default: config cache.dir when cache.enabled; "
                        "pass an empty string to force-disable.")
    p.add_argument("--force-refresh", action="store_true",
                   help="ignore any cached augmented graph, regenerate it and overwrite the cache entry")
    p.add_argument("--cache-scope", default=None,
                   help="generation-scope tag for the cache key (default: config cache.generation_scope "
                        "or the fixed 'gen-v1'). Stable across code edits; bump it (or use --force-refresh) "
                        "when a generator change must regenerate the datasets. Special value 'code' restores "
                        "the strict md5-over-source fingerprint.")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(name)s %(levelname)s %(message)s")

    cfg = {}
    cfg_path = args.config
    if cfg_path is None and os.path.exists(os.path.join("configs", "default.yaml")):
        cfg_path = os.path.join("configs", "default.yaml")
    if cfg_path:
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}

    dataset_cfg = cfg.get("dataset", {}) or {}
    dataset = args.dataset or dataset_cfg.get("name", "Cora")
    data_dir = args.data_dir or dataset_cfg.get("data_dir", "data/")

    backend = args.backend or cfg.get("llm", {}).get("backend", "dummy")
    tg_cfg = cfg.get("text_gen", {})
    text_gen = args.text_gen or bool(tg_cfg.get("enabled", False))
    embed_model = args.embed_model if args.text_gen else tg_cfg.get(
        "model", "/Users/wuchunqi/.cache/modelscope/hub/models/sentence-transformers/msmarco-MiniLM-L-12-v3"
    )

    cache_cfg = cfg.get("cache", {}) or {}
    if args.cache_dir is not None:
        cache_dir = args.cache_dir or None
    elif cache_cfg.get("enabled", False):
        cache_dir = cache_cfg.get("dir", "cache/")
    else:
        cache_dir = None
    force_refresh = args.force_refresh or bool(cache_cfg.get("force_refresh", False))
    generation_scope = args.cache_scope or cache_cfg.get("generation_scope") or None

    if args.resume_gap_loop:
        result = resume_gap_loop_pipeline(
            review_dir=args.resume_gap_loop,
            data_dir=data_dir,
            dataset=dataset,
            backend=backend,
            split_cfg={
                "seed": cfg.get("split", {}).get("seed", 42),
                "val_ratio": cfg.get("split", {}).get("val_ratio", 0.0),
                "test_ratio": cfg.get("split", {}).get("test_ratio", 0.60),
                "budget_ratio": cfg.get("generation", {}).get("default_anomaly_ratio", 0.05),
            },
            generation_seed=cfg.get("generation", {}).get("seed", 42),
            apply_augmentation=not args.report_only,
            text_gen=text_gen,
            embed_model=embed_model,
            allow_download=bool(tg_cfg.get("allow_download", True)),
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            generation_scope=generation_scope,
        )
    elif args.resume_review:
        result = resume_review_pipeline(
            review_dir=args.resume_review,
            data_dir=data_dir,
            dataset=dataset,
            backend=backend,
            split_cfg={
                "seed": cfg.get("split", {}).get("seed", 42),
                "val_ratio": cfg.get("split", {}).get("val_ratio", 0.0),
                "test_ratio": cfg.get("split", {}).get("test_ratio", 0.60),
                "budget_ratio": cfg.get("generation", {}).get("default_anomaly_ratio", 0.05),
            },
            generation_seed=cfg.get("generation", {}).get("seed", 42),
            apply_augmentation=not args.report_only,
            text_gen=text_gen,
            embed_model=embed_model,
            allow_download=bool(tg_cfg.get("allow_download", True)),
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            generation_scope=generation_scope,
            accept_insufficient=(args.accept_insufficient
                                 or bool((cfg.get("review", {}) or {}).get("accept_insufficient", False))),
        )
    else:
        review_cfg = dict(cfg.get("review", {}) or {})
        if args.llm_precheck:
            review_cfg["llm_precheck"] = True
        if args.coverage_criterion:
            review_cfg["coverage_criterion"] = args.coverage_criterion
        gap_loop = args.gap_loop or args.review_gap_loop \
            or (review_cfg.get("mode") == "loop")
        if gap_loop and (args.review_only or args.resume_review
                         or args.resume_gap_loop):
            p.error("--gap-loop/--review-gap-loop cannot be combined with "
                    "--review-only / --resume-review / --resume-gap-loop")
        if args.review_gap_loop:
            review_cfg["final_review"] = True
        elif review_cfg.get("final_review") and gap_loop:
            pass
        elif review_cfg.get("final_review"):
            review_cfg.pop("final_review")
        if gap_loop:
            review_cfg.setdefault("targeted", 5)
            review_cfg.setdefault("stratified", 3)
            review_cfg.setdefault("boundary", 2)
        result = run_pipeline(
            data_dir=data_dir,
            dataset=dataset,
            backend=backend,
            split_cfg={
                "seed": cfg.get("split", {}).get("seed", 42),
                "val_ratio": cfg.get("split", {}).get("val_ratio", 0.0),
                "test_ratio": cfg.get("split", {}).get("test_ratio", 0.60),
                "budget_ratio": cfg.get("generation", {}).get("default_anomaly_ratio", 0.05),
            },
            generation_seed=cfg.get("generation", {}).get("seed", 42),
            apply_augmentation=not args.report_only,
            text_gen=text_gen,
            embed_model=embed_model,
            allow_download=bool(tg_cfg.get("allow_download", True)),
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            generation_scope=generation_scope,
            review_only=args.review_only,
            gap_loop=gap_loop,
            review_dir=args.review_dir,
            review_config=review_cfg,
        )

    if cache_dir:
        status = "HIT (LLM planning skipped)" if result.from_cache else "MISS (fresh generation, entry written)"
        print(f"augmented-graph cache : {status} under {cache_dir}")

    _print_report(result)
    _print_missing(result)
    _print_review(result)
    _print_strategy(result)
    _print_generation(result)

    if args.review_only:
        print(f"(--review-only: review package exported to '{args.review_dir}'; "
              f"no strategy, no generation. Fill in sample_feedback.csv and "
              f"gap_feedback.json, then resume with --resume-review {args.review_dir}.)")
        return 0

    if result.review_paused:
        print(f"(review re-paused for a resample round in '{result.review_dir}'; "
              f"review the refreshed package and resume again.)")
        return 0

    if args.report_only:
        print("(--report-only: no graph mutation performed.)")
        return 0

    if args.embedding_check:
        from aga.models.train import embedding_similarity_check

        emb_tcfg = cfg.get("training", {})
        emb_seed = cfg.get("split", {}).get("seed", 42)
        emb_plot_path = emb_tcfg.get("emb_plot_path", "embedding_similarity.png")
        emb_stats = embedding_similarity_check(
            result.baseline_graph,
            result.augmented_graph,
            out_path=emb_plot_path,
            seed=emb_seed,
        )
        print("=" * 72)
        print("EMBEDDING SIMILARITY CHECK — 原始特征 (cosine, 送入模型前)")
        print(f"  plot saved to   : {emb_plot_path}")
        print(f"  n_synthetic      : {emb_stats['n_synthetic']}")
        print(f"  n_orig_anomaly   : {emb_stats['n_orig_anomaly']}")
        for key, label in (
            ("to_anomaly", "→ 最近原始异常"),
            ("to_synthetic", "→ 最近合成异常"),
        ):
            s = emb_stats[key]
            print(f"  {label:14s} mean={s['mean']:.4f} median={s['median']:.4f} "
                  f"std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f}")
        print("=" * 72)

    if args.train_compare:
        import numpy as np

        tcfg = cfg.get("training", {})
        model_name = args.model or tcfg.get("model", "dominant")
        model_kwargs = cfg.get("models", {}).get(model_name, {}) or {}
        _per = tcfg.get("per_model", {}).get(model_name, {}) or {}
        _shared_keys = ("epochs", "lr", "hidden", "layers", "dropout",
                        "class_weight", "attr_weight", "stru_weight", "device")
        for k, v in _per.items():
            if k in _shared_keys:
                tcfg = {**tcfg, k: v}
            else:
                model_kwargs = {**model_kwargs, k: v}
        base_seed = cfg.get("split", {}).get("seed", 42)
        runs = max(1, int(args.runs))

        agg = {
            split: {m: [] for m in ("auc", "ap", "f1")}
            for split in ("baseline", "augmented")
        }
        num_eval = None
        plot_path = tcfg.get("plot_path", "tsne_compare.png")
        for i in range(runs):
            print(f"[run {i + 1}/{runs}] seed={base_seed + i} "
                  f"(baseline + augmented) ...", flush=True)
            comp = train_compare(
                result.baseline_graph,
                result.augmented_graph,
                TrainConfig(
                    model=model_name,
                    epochs=tcfg.get("epochs", 100),
                    lr=tcfg.get("lr", 1e-3),
                    hidden=tcfg.get("hidden", 64),
                    layers=tcfg.get("layers", 3),
                    dropout=tcfg.get("dropout", 0.2),
                    device=tcfg.get("device", "cpu"),
                    attr_weight=tcfg.get("attr_weight", 1.0),
                    stru_weight=tcfg.get("stru_weight", 1.0),
                    class_weight=tcfg.get("class_weight", "inverse"),
                    seed=base_seed + i,
                    model_kwargs=dict(model_kwargs),
                ),
                eval_mask_name="test_mask",
                visualize=False,
                plot_path=plot_path,
            )
            num_eval = comp["baseline"]["num_eval"]
            for split, res in comp.items():
                for m in ("auc", "ap", "f1"):
                    agg[split][m].append(res[m])

        print("=" * 72)
        print(f"TRAINING COMPARISON — {model_name.upper()} on frozen test set")
        print(f"  {runs} run(s), seeds {base_seed}..{base_seed + runs - 1}, n_eval={num_eval}")
        print("=" * 72)
        for split in ("baseline", "augmented"):
            vals = agg[split]
            if runs == 1:
                line = (f"  {split:10s} "
                        f"AUC={vals['auc'][0]:.4f}  "
                        f"AP={vals['ap'][0]:.4f}  "
                        f"F1={vals['f1'][0]:.4f}")
            else:
                line = (f"  {split:10s} "
                        f"AUC={np.mean(vals['auc']):.4f}+-{np.std(vals['auc'], ddof=1):.4f}  "
                        f"AP={np.mean(vals['ap']):.4f}+-{np.std(vals['ap'], ddof=1):.4f}  "
                        f"F1={np.mean(vals['f1']):.4f}+-{np.std(vals['f1'], ddof=1):.4f}")
            print(line)
        print("  (val/test node sets verified identical across both runs.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())