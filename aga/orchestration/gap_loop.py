
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from ..agents.gap_refiner import GapRefiner, REFINE_DEGRADED_MARKER
from ..agents.gap_reviewer import (
    GapReviewer,
    gap_definition_digest,
)
from ..agents.llm_reviewer import LLMReviewer
from ..data import review_evidence
from ..data.review_sampling import (
    ReviewSamplingConfig,
    SampleRecord,
    SamplingContext,
    sample_all_gaps,
)
from ..models.split import frozen_split_signature
from ..orchestration.schema import (
    DatasetReport,
    GapFeedback,
    GapLoopRound,
    GapLoopState,
    GapLoopTrace,
    MissingPatterns,
    ReviewState,
)

log = logging.getLogger("aga.pipeline.gap_loop")


def gap_refinement_loop(
    *,
    report: DatasetReport,
    gaps: MissingPatterns,
    snapshot,
    node_text: Dict[int, str],
    split: Optional[Dict],
    review_dir: str,
    sampling_config: ReviewSamplingConfig,
    coverage_criterion: Optional[str] = None,
    max_rounds: int = 3,
    data_fingerprint: str = "",
    pause_after_loop: bool = False,
) -> Tuple[MissingPatterns, List, "Optional[GapLoopState]", GapLoopTrace]:
    ctx = SamplingContext(snapshot, split["train_mask"])
    split_sig = frozen_split_signature(split) if split else {}
    reviewer = GapReviewer(sampling_config)
    refiner = GapRefiner()
    cfg = sampling_config

    current_gaps: MissingPatterns = gaps.model_copy(deep=True)
    current_verdicts: List = []
    prev_records: Dict[str, List[SampleRecord]] = {}
    prev_after_digest: Optional[str] = None

    rounds: List[GapLoopRound] = []
    converged = False
    stopped_reason = "budget"
    last_cards: List = []
    last_pre = None
    k = 0

    for k in range(1, int(max_rounds) + 1):
        digest_before = gap_definition_digest(current_gaps.missing)

        records_k = sample_all_gaps(
            current_gaps.missing, ctx, cfg, round_no=k, previous=prev_records)
        cards = review_evidence.build_sample_cards(
            current_gaps.missing, records_k, ctx, node_text)
        for gid, recs in records_k.items():
            prev_records.setdefault(gid, []).extend(recs)
        last_cards = cards
        sample_ids = sorted({c.sample_id for c in cards})

        pre = LLMReviewer().review(
            current_gaps.missing, cards,
            checklists={
                (g.gap_id or "gap-000"): review_evidence._checklist_for(g.name)
                for g in current_gaps.missing
            },
            reference_profile=ctx.reference_profile(),
            ctx=ctx,
            coverage_criterion=coverage_criterion,
        )
        last_pre = pre
        round_degraded = bool(pre.degraded)

        state_for_round = ReviewState(
            data_fingerprint=data_fingerprint,
            split_signature=split_sig,
            gap_definition_digest=digest_before,
            review_config={"loop": True, "round": k},
            sampling_seed=cfg.seed,
            sampling_round=k,
            gaps=list(current_gaps.missing),
            samples=cards,
        )
        verdicts, new_state = reviewer.finalize(
            state_for_round, pre.sample_feedback, pre.gap_feedback,
            ctx=None, node_text=None, review_dir=None,
        )
        current_verdicts = verdicts

        refined = refiner.refine(
            report, current_gaps, verdicts, round_no=k)
        refined_degraded = REFINE_DEGRADED_MARKER in (refined.global_note or "")
        round_degraded = round_degraded or refined_degraded

        digest_after = gap_definition_digest(refined.missing)
        converged_k = bool(
            prev_after_digest is not None and digest_after == prev_after_digest)

        ids_before = {g.gap_id for g in current_gaps.missing}
        ids_after = {g.gap_id for g in refined.missing}
        rounds.append(GapLoopRound(
            round_no=k,
            digest_before=digest_before,
            digest_after=digest_after,
            verdicts=list(verdicts),
            kept_gap_ids=sorted(ids_before & ids_after),
            dropped_gap_ids=sorted(ids_before - ids_after),
            added_gap_ids=sorted(ids_after - ids_before),
            converged=converged_k,
            degraded=round_degraded,
            sample_ids=sample_ids,
        ))
        log.info("gap-loop round %d/%d: %d -> %d gaps (kept %d, dropped %d, "
                 "added %d); digest %s -> %s%s",
                 k, max_rounds, len(ids_before), len(ids_after),
                 len(ids_before & ids_after), len(ids_before - ids_after),
                 len(ids_after - ids_before),
                 digest_before, digest_after,
                 " [converged]" if converged_k else "")

        current_gaps = refined
        prev_after_digest = digest_after

        if converged_k:
            converged = True
            stopped_reason = "converged"
            break
        if refined_degraded and k >= 2:
            stopped_reason = "degraded_stable"
            break

    final_digest = gap_definition_digest(current_gaps.missing)
    trace = GapLoopTrace(
        max_rounds=int(max_rounds),
        rounds=rounds,
        converged=converged,
        stopped_reason=stopped_reason,
        final_digest=final_digest,
        final_gaps=list(current_gaps.missing),
    )

    kept = sorted({g.gap_id for g in current_gaps.missing})
    summary_note = (
        f"gap-loop: {len(rounds)} round(s), converged={converged} "
        f"({stopped_reason}); final gaps {kept}; full audit in "
        f"{os.path.join(review_dir, 'loop', 'loop_trace.json')}"
    )
    final_gaps = current_gaps.model_copy(deep=True)
    final_gaps.global_note = (summary_note + " | " +
                              (current_gaps.global_note or "")).strip(" |")

    _export_loop_audit(
        review_dir, trace, final_gaps, last_cards, last_pre, ctx,
        node_text=node_text, verdicts=current_verdicts)

    loop_state: Optional[GapLoopState] = None
    if pause_after_loop:
        loop_state = _export_final_review(
            review_dir, trace, final_gaps, data_fingerprint, split,
            sampling_config=cfg)
        log.info("gap-loop: final review PAUSED at %s (curate "
                 "loop/gap_feedback.json, then --resume-gap-loop).",
                 os.path.join(review_dir, "loop"))

    return final_gaps, current_verdicts, loop_state, trace


def _export_loop_audit(
    review_dir: str,
    trace: GapLoopTrace,
    final_gaps: MissingPatterns,
    last_cards: List,
    last_pre,
    ctx: SamplingContext,
    *,
    node_text: Dict[int, str],
    verdicts: List,
) -> None:
    try:
        loop_dir = os.path.join(review_dir, "loop")
        os.makedirs(loop_dir, exist_ok=True)
        with open(os.path.join(loop_dir, "loop_trace.json"), "w",
                  encoding="utf-8") as fh:
            fh.write(trace.model_dump_json(indent=2))

        evidence_summary = review_evidence.occurrence_summary(last_cards)
        context_notes = {
            "loop": {
                "rounds": len(trace.rounds),
                "converged": trace.converged,
                "stopped_reason": trace.stopped_reason,
                "degraded_rounds": [r.round_no for r in trace.rounds
                                    if r.degraded],
                "final_digest": trace.final_digest,
                "trace_file": "loop_trace.json",
            },
            "unmeasured_note": (
                "unmeasured families have NO label evidence; absence of "
                "labels is not evidence of absence"
            ),
            "labeled_anomaly_reference": ctx.reference_profile(),
            "final_verdicts": [
                {
                    "gap_id": v.gap_id,
                    "review_status": v.review_status,
                    "evidence_summary": v.evidence_summary,
                    "counterexample_sample_ids": v.counterexample_sample_ids,
                }
                for v in verdicts
            ],
        }
        prefill_sample = last_pre.sample_feedback if last_pre else []
        prefill_gap = last_pre.gap_feedback if last_pre else []
        review_evidence.export_review_package(
            loop_dir,
            final_gaps.missing,
            last_cards,
            evidence_summary,
            context_notes=context_notes,
            draft_key=trace.final_digest,
            prefill_sample=prefill_sample,
            prefill_gap=prefill_gap,
        )
        log.info("gap-loop audit exported to %s (loop_trace.json + final "
                 "review package)", loop_dir)
    except Exception:
        log.warning("gap-loop audit export failed; continuing.",
                    exc_info=True)


def _export_final_review(
    review_dir: str,
    trace: GapLoopTrace,
    final_gaps: MissingPatterns,
    data_fingerprint: str,
    split: Optional[Dict],
    sampling_config: Optional[ReviewSamplingConfig] = None,
) -> GapLoopState:
    loop_dir = os.path.join(review_dir, "loop")
    os.makedirs(loop_dir, exist_ok=True)

    verdict_status = {}
    for rd in trace.rounds:
        for v in rd.verdicts:
            verdict_status[v.gap_id] = v.review_status
    prefill_gap = [
        GapFeedback(
            gap_id=g.gap_id,
            review_status=(verdict_status.get(g.gap_id, "pending")
                           if verdict_status.get(g.gap_id, "pending") in
                           ("confirmed", "rejected") else "pending"),
            human_comment=("suggested from the loop verdict: "
                           + verdict_status.get(g.gap_id, "pending")),
            source="llm",
        )
        for g in final_gaps.missing
    ]

    context_notes = {
        "gap_loop_final_review": True,
        "loop": {
            "rounds": len(trace.rounds),
            "converged": trace.converged,
            "stopped_reason": trace.stopped_reason,
            "final_digest": trace.final_digest,
        },
        "instructions": (
            "CURATE the refined gap set: confirmed = keep this gap; "
            "rejected = REMOVE it from generation; revised_confirmed = keep "
            "with your rewritten claim/scope. Unmarked gaps are KEPT as-is "
            "(loop output). No strategy-level constraints apply -- your "
            "curation IS the gap set the planner will plan freely over."
        ),
    }
    review_evidence.export_review_package(
        loop_dir,
        final_gaps.missing,
        [],
        "final review: curate the refined gap set below",
        context_notes=context_notes,
        draft_key=trace.final_digest,
        prefill_gap=prefill_gap,
    )

    state = GapLoopState(
        data_fingerprint=data_fingerprint,
        split_signature=frozen_split_signature(split) if split else {},
        final_gap_digest=trace.final_digest,
        review_config={
            "loop": True,
            "max_rounds": trace.max_rounds,
            "sampling": (
                {"targeted": sampling_config.targeted,
                 "stratified": sampling_config.stratified,
                 "boundary": sampling_config.boundary,
                 "seed": sampling_config.seed}
                if sampling_config is not None else {}
            ),
        },
        trace=trace,
        final_gaps=list(final_gaps.missing),
    )
    path = os.path.join(loop_dir, "loop_state.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(state.model_dump_json(indent=2))
    return state