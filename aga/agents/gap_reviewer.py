
from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from .base import Agent
from ..data.review_sampling import (
    ReviewSamplingConfig,
    SamplingContext,
    SampleRecord,
    sample_all_gaps,
)
from ..data import review_evidence
from ..orchestration.schema import (
    DatasetReport,
    GapFeedback,
    GapReview,
    GenerationStrategy,
    GlobalConstraints,
    MissingPattern,
    MissingPatterns,
    ReviewState,
    SampleCard,
    SampleFeedback,
)
from ..models.split import frozen_split_signature

log = logging.getLogger("aga.review")

APPROVED_STATUSES = ("confirmed", "revised_confirmed")


def gap_definition_digest(gaps: Sequence[MissingPattern]) -> str:
    payload = [
        {
            "gap_id": g.gap_id,
            "name": g.name,
            "claim": g.claim,
            "scope": g.scope,
            "pattern_definition": g.pattern_definition,
            "evidence_needed": list(g.evidence_needed),
        }
        for g in gaps
    ]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]


class GapReviewer(Agent):
    name = "gap_reviewer"

    def __init__(self, config: Optional[ReviewSamplingConfig] = None):
        self.config = config or ReviewSamplingConfig()

    def run(self, *args, **kwargs):
        return self.prepare(*args, **kwargs)

    def prepare(
        self,
        report: DatasetReport,
        gaps: MissingPatterns,
        ctx: SamplingContext,
        node_text: Dict[int, str],
        review_dir: str,
        data_fingerprint: str = "",
        split: Optional[Dict] = None,
        llm_precheck: bool = False,
        coverage_criterion: Optional[str] = None,
    ) -> ReviewState:
        if not gaps.missing:
            log.info("Gap Reviewer: no gaps to review; empty review state.")
        cfg = self.config
        records = sample_all_gaps(gaps.missing, ctx, cfg, round_no=1)
        cards = review_evidence.build_sample_cards(gaps.missing, records, ctx, node_text)
        evidence_summary = review_evidence.occurrence_summary(cards)

        pre_sample: List[SampleFeedback] = []
        pre_gap: List[GapFeedback] = []
        pre_summary = ""
        reference_profile = ctx.reference_profile()
        if llm_precheck and gaps.missing and cards:
            from .llm_reviewer import LLMReviewer

            pre = LLMReviewer().review(
                gaps.missing, cards,
                checklists={
                    g.gap_id or "gap-000": review_evidence._checklist_for(g.name)
                    for g in gaps.missing
                },
                reference_profile=reference_profile,
                ctx=ctx,
                coverage_criterion=coverage_criterion,
            )
            pre_sample = pre.sample_feedback
            pre_gap = pre.gap_feedback
            pre_summary = pre.summary()
            log.info("Gap Reviewer: LLM pre-review done (%s); suggestions "
                     "prefilled into the feedback templates.", pre_summary)
            if pre.uniform_not_matches:
                log.warning(
                    "Gap Reviewer: LLM pre-review judged EVERY sampled card "
                    "not_matches. Either the pattern truly has no instances "
                    "in this sample, or the model defaulted to the "
                    "conservative answer -- spot-check the targeted cards by "
                    "hand before trusting the suggestions."
                )

        split_sig = frozen_split_signature(split) if split else {}
        state = ReviewState(
            data_fingerprint=data_fingerprint,
            split_signature=split_sig,
            gap_definition_digest=gap_definition_digest(gaps.missing),
            review_config={
                "targeted": cfg.targeted,
                "stratified": cfg.stratified,
                "boundary": cfg.boundary,
                "seed": cfg.seed,
                "max_rounds": cfg.max_rounds,
                "llm_precheck": bool(llm_precheck),
            },
            sampling_seed=cfg.seed,
            sampling_round=1,
            gaps=list(gaps.missing),
            samples=cards,
        )
        ctx_notes: Dict = {
            "dataset": report.stats.dataset,
            "num_train_nodes": int(len(ctx.train_idx)),
            "stats_scope": report.stats.stats_scope,
            "coverage": {
                c.family: {"score": c.score, "status": c.status}
                for c in report.coverage
            },
            "unmeasured_note": (
                "unmeasured families have NO label evidence; absence of "
                "labels is not evidence of absence"
            ),
            "labeled_anomaly_reference": reference_profile,
        }
        if pre_summary:
            ctx_notes["llm_precheck"] = pre_summary
        review_evidence.export_review_package(
            review_dir,
            gaps.missing,
            cards,
            evidence_summary,
            context_notes=ctx_notes,
            draft_key=state.gap_definition_digest,
            prefill_sample=pre_sample,
            prefill_gap=pre_gap,
        )
        self._save_state(review_dir, state)
        log.info(
            "Gap Reviewer: review package for %d gap(s), %d sample card(s) "
            "exported to %s; pipeline PAUSED before generation.",
            len(gaps.missing), len(cards), review_dir,
        )
        return state

    def finalize(
        self,
        review: ReviewState,
        sample_feedback: Sequence[SampleFeedback],
        gap_feedback: Sequence[GapFeedback],
        ctx: Optional[SamplingContext] = None,
        node_text: Optional[Dict[int, str]] = None,
        review_dir: Optional[str] = None,
        accept_insufficient: bool = False,
    ) -> Tuple[List[GapReview], Optional[ReviewState]]:
        verdicts = {(fb.gap_id, fb.sample_id): fb.verdict for fb in sample_feedback}
        fb_by_gap = {fb.gap_id: fb for fb in gap_feedback}
        gap_by_id = {g.gap_id: g for g in review.gaps}

        resample_requested: List[str] = []
        reviewed: List[GapReview] = []

        for gap in review.gaps:
            gid = gap.gap_id
            fb = fb_by_gap.get(gid)
            status = fb.review_status if fb is not None else "pending"

            supporting: List[int] = []
            counterexamples: List[int] = []
            undecidable: List[int] = []
            for card in review.samples:
                if card.gap_id != gid:
                    continue
                v = verdicts.get((gid, card.sample_id), "undecidable")
                if v == "matches":
                    counterexamples.append(card.sample_id)
                elif v == "not_matches":
                    supporting.append(card.sample_id)
                else:
                    undecidable.append(card.sample_id)

            limitations: List[str] = []
            evidence_parts = []
            if counterexamples:
                evidence_parts.append(
                    f"{len(counterexamples)} sampled node(s) MATCH the pattern "
                    f"(existing instances: refutes 'completely missing'; "
                    f"sufficiency still needs scope + coverage criteria)"
                )
            if supporting:
                evidence_parts.append(
                    f"{len(supporting)} sampled node(s) do not match "
                    f"(consistent with the claim, weak support)"
                )
            if undecidable:
                evidence_parts.append(
                    f"{len(undecidable)} sampled node(s) undecidable/blank "
                    f"(blank feedback never auto-passes)"
                )
            if not review.samples:
                evidence_parts.append("no samples were drawn for this gap")
                limitations.append("no sample evidence available")

            if status == "insufficient_evidence" and fb is not None and fb.request_resample:
                if ctx is not None and review.sampling_round < self.config.max_rounds:
                    resample_requested.append(gid)
                    limitations.append(
                        f"resample round {review.sampling_round + 1} requested; "
                        f"gap stays insufficient_evidence until re-reviewed"
                    )
                else:
                    limitations.append(
                        "resample requested but no sampling context or budget "
                        "exhausted; gap remains unknown (insufficient_evidence)"
                    )
            elif status == "pending":
                limitations.append("no human gap feedback recorded; kept pending")

            if status in APPROVED_STATUSES:
                criterion = fb.coverage_criterion if fb else None
                if not criterion:
                    limitations.append(
                        "confirmed without a documented coverage criterion "
                        "(task requirement / community coverage / guideline); "
                        "frequency alone never justifies a confirm"
                    )

            accepted = bool(accept_insufficient
                            and status == "insufficient_evidence"
                            and not fb.request_resample)
            if accepted:
                limitations.append(
                    "ACCEPTED DESPITE INSUFFICIENT EVIDENCE (operator "
                    "--accept-insufficient): continuing generation without "
                    "settling the evidence; revisit when a coverage criterion "
                    "is agreed"
                )

            if not counterexamples:
                limitations.append(
                    "pattern not observed in this round's sample -- at most "
                    "'not observed in this round', NOT 'absent from the dataset'"
                )

            reviewed.append(GapReview(
                gap_id=gid,
                pattern_name=gap.name,
                review_status=status,
                supporting_sample_ids=supporting,
                counterexample_sample_ids=counterexamples,
                undecidable_sample_ids=undecidable,
                revised_claim=fb.revised_claim if fb else None,
                reviewed_scope=fb.reviewed_scope if fb else None,
                evidence_summary="; ".join(evidence_parts),
                limitations=limitations,
                human_comment=fb.human_comment if fb else "",
                coverage_criterion=fb.coverage_criterion if fb else None,
                original_claim=gap.claim,
                original_scope=gap.scope,
                pattern_definition=gap.pattern_definition,
                accepted_despite_insufficient=accepted,
            ))

        new_state: Optional[ReviewState] = None
        if resample_requested and ctx is not None:
            new_state = self._resample(review, resample_requested, ctx, node_text or {},
                                       review_dir)
        return reviewed, new_state

    def _resample(
        self,
        review: ReviewState,
        gap_ids: Sequence[str],
        ctx: SamplingContext,
        node_text: Dict[int, str],
        review_dir: Optional[str],
    ) -> ReviewState:
        next_round = review.sampling_round + 1
        log.info("Gap Reviewer: resample round %d for gap(s): %s",
                 next_round, list(gap_ids))
        prev_records: Dict[str, List[SampleRecord]] = {}
        for card in review.samples:
            prev_records.setdefault(card.gap_id, []).append(SampleRecord(
                sample_id=card.sample_id, gap_id=card.gap_id, group=card.group,
                reason=card.reason, stratum=card.stratum,
                round_no=review.sampling_round,
            ))
        target_gaps = [g for g in review.gaps if g.gap_id in set(gap_ids)]
        new_records = sample_all_gaps(
            target_gaps, ctx, self.config, round_no=next_round, previous=prev_records
        )
        all_gaps = review.gaps
        cards = review_evidence.build_sample_cards(all_gaps, new_records, ctx, node_text)
        evidence_summary = review_evidence.occurrence_summary(cards)
        state = review.model_copy(update={
            "sampling_round": next_round,
            "samples": cards,
        })
        if review_dir:
            review_evidence.export_review_package(
                review_dir, all_gaps, cards, evidence_summary,
                context_notes={"resample_round": next_round, "gaps": list(gap_ids)},
                draft_key=state.gap_definition_digest,
            )
            self._save_state(review_dir, state)
        return state

    def _save_state(self, review_dir: str, state: ReviewState) -> str:
        import os

        path = os.path.join(review_dir, "review_state.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(state.model_dump_json(indent=2))
        return path

    @staticmethod
    def load_state(review_dir: str) -> ReviewState:
        import os

        path = os.path.join(review_dir, "review_state.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no review_state.json under {review_dir}")
        with open(path, encoding="utf-8") as fh:
            return ReviewState.model_validate_json(fh.read())


def select_approved_gaps(
    reviewed: Sequence[GapReview],
    gaps: MissingPatterns,
) -> MissingPatterns:
    approved_ids = {
        r.gap_id for r in reviewed
        if r.review_status in APPROVED_STATUSES
        or r.accepted_despite_insufficient
    }
    kept: List[MissingPattern] = []
    rev_by_id = {r.gap_id: r for r in reviewed}
    for m in gaps.missing:
        if m.gap_id not in approved_ids:
            continue
        r = rev_by_id[m.gap_id]
        revised = m.model_copy(update={
            "claim": r.revised_claim or m.claim,
            "scope": r.reviewed_scope or m.scope,
        })
        kept.append(revised)
    log.info("Gap Reviewer: %d/%d gap(s) approved for strategy planning.",
             len(kept), len(gaps.missing))
    return MissingPatterns(
        existing_patterns=gaps.existing_patterns,
        missing=kept,
        global_note=gaps.global_note,
    )


def make_empty_strategy(rationale: str = "") -> GenerationStrategy:
    return GenerationStrategy(
        total_new_anomalies=0,
        per_pattern=[],
        global_constraints=GlobalConstraints(),
        rationale=rationale or (
            "No gap was approved by the evidence-grounded review; "
            "generation budget is zero by design."
        ),
    )


def _pattern_family(name: str) -> str:
    n = name.lower()
    if "camouflage" in n:
        return "semantic_camouflage"
    if "bridge" in n:
        return "bridge"
    if "hybrid" in n:
        return "hybrid"
    if "context" in n:
        return "context"
    if "hub" in n:
        return "hub"
    return n


def validate_strategy_against_review(
    strategy: GenerationStrategy,
    reviewed: Sequence[GapReview],
) -> GenerationStrategy:
    approved: Dict[str, GapReview] = {
        _pattern_family(r.pattern_name): r
        for r in reviewed
        if r.review_status in APPROVED_STATUSES
        or r.accepted_despite_insufficient
    }
    kept_rows = []
    for row in strategy.per_pattern:
        fam = _pattern_family(row.pattern)
        rev = approved.get(fam)
        if rev is None:
            log.warning(
                "Strategy row %s (count=%d) dropped: pattern family not "
                "approved by review (approved: %s).",
                row.pattern, row.count, sorted(approved),
            )
            continue
        scope = rev.reviewed_scope or rev.original_scope or "train_all"
        if scope.startswith("comm="):
            try:
                allowed = int(scope.split("=", 1)[1])
            except ValueError:
                allowed = None
            if allowed is not None and row.target_community is not None \
                    and int(row.target_community) != allowed:
                log.warning(
                    "Strategy row %s dropped: target_community=%s outside "
                    "reviewed scope %s.",
                    row.pattern, row.target_community, scope,
                )
                continue
        kept_rows.append(row)

    total = sum(r.count for r in kept_rows)
    if total != strategy.total_new_anomalies:
        log.info("Strategy budget adjusted %d -> %d after review-scope pruning.",
                 strategy.total_new_anomalies, total)
    return strategy.model_copy(update={
        "per_pattern": kept_rows,
        "total_new_anomalies": total,
    })


def feedback_digest(reviewed: Sequence[GapReview]) -> str:
    payload = [
        {
            "gap_id": r.gap_id,
            "review_status": r.review_status,
            "accepted_despite_insufficient": bool(r.accepted_despite_insufficient),
            "revised_claim": r.revised_claim,
            "reviewed_scope": r.reviewed_scope,
            "supporting": r.supporting_sample_ids,
            "counterexamples": r.counterexample_sample_ids,
        }
        for r in sorted(reviewed, key=lambda x: x.gap_id)
    ]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]