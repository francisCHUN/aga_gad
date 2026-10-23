
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from .base import Agent
from ..llm import registry as llm
from ..llm.prompts import (
    SYSTEM_LLM_GAP_REVIEWER,
    SYSTEM_LLM_SAMPLE_REVIEWER,
    build_gap_review_user_prompt,
    build_sample_review_user_prompt,
)
from ..orchestration.schema import (
    ConditionAssessment,
    GapFeedback,
    GapSuggestions,
    MissingPattern,
    SampleCard,
    SampleFeedback,
    SampleJudgement,
    SampleJudgements,
)

log = logging.getLogger("aga.review.llm")

_TEXT_LIMIT_STRUCTURAL = 600
_TEXT_LIMIT_SEMANTIC = 4000
_NEIGHBOUR_TEXT_LIMIT = 300
_MAX_NEIGHBOURS = 3

_TEXT_FAMILIES = ("context", "semantic_camouflage")
_BRIDGE_FAMILIES = ("bridge", "hybrid")


def family_key(pattern_name: str) -> str:
    n = (pattern_name or "").lower()
    if "camouflage" in n:
        return "semantic_camouflage"
    if "bridge" in n:
        return "bridge"
    if "hybrid" in n:
        return "hybrid"
    if "context" in n or "attribute" in n:
        return "context"
    if "hub" in n or "structure" in n or "stru" in n:
        return "hub"
    return "default"


def _pct_word(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    if p < 1:
        return "lowest 1%"
    if p < 5:
        return "lowest 5%"
    if p < 10:
        return "low (below 10%)"
    if p >= 99:
        return "highest 1%"
    if p >= 95:
        return "highest 5%"
    if p >= 90:
        return "high (above 90%)"
    return f"{p:.0f}th pct"


def card_evidence(
    card: SampleCard,
    ctx=None,
    family: str = "default",
) -> Dict:
    limit = _TEXT_LIMIT_SEMANTIC if family in _TEXT_FAMILIES else _TEXT_LIMIT_STRUCTURAL
    raw_len = len(card.text)
    text_card_truncated = bool(card.text_truncated)
    shown_truncated = raw_len > limit
    text = card.text[:limit] + ("...[TRUNCATED]" if shown_truncated else "")

    nbrs = []
    for nid, ntext in zip(card.neighbour_ids[:_MAX_NEIGHBOURS],
                          card.neighbour_texts[:_MAX_NEIGHBOURS]):
        ntext = ntext or ""
        nbr_trunc = len(ntext) > _NEIGHBOUR_TEXT_LIMIT
        shown = ntext[:_NEIGHBOUR_TEXT_LIMIT] + \
            ("...[TRUNCATED]" if nbr_trunc else "")
        entry = {
            "node": nid,
            "text": shown,
            "text_truncated": bool(nbr_trunc),
            "original_text_chars": len(ntext),
        }
        if ctx is not None and 0 <= nid < len(ctx.comm_id):
            entry["community"] = int(ctx.comm_id[nid])
            entry["same_community_as_node"] = bool(
                int(ctx.comm_id[nid]) == card.community)
        nbrs.append(entry)

    family_facts: Dict = {}
    if ctx is not None and family != "default":
        try:
            family_facts = ctx.evidence_for_family(family, card.sample_id)
        except Exception:
            family_facts = {"error": "family facts unavailable"}

    missing: List[str] = []
    if card.degree == 0:
        missing.append("no_neighbours: every neighbourhood condition reads unknown")
    if nbrs and not any((n.get("text") or "").strip() for n in nbrs):
        missing.append("no_neighbour_texts: semantic comparison with the "
                       "neighbourhood reads unknown")
    if card.ctx_cos is None:
        missing.append("nbr_feat_cos unavailable (isolated node)")

    return {
        "sample_id": card.sample_id,
        "group": card.group,
        "sampling_reason": card.reason,
        "community": card.community,
        "metrics_note": (
            "ctx_cos / cross_community_ratio are computed over ALL neighbours "
            "and edges of this node (not just the sampled ones shown below). "
            "family_facts are PROGRAM-COMPUTED deterministic evidence: cite "
            "them by field name; do not recompute or dispute them."
        ),
        "metrics": {
            "degree": card.degree,
            "degree_percentile": card.degree_percentile,
            "degree_percentile_word": _pct_word(card.degree_percentile),
            "ctx_cos": card.ctx_cos,
            "ctx_cos_percentile": card.ctx_cos_percentile,
            "ctx_cos_percentile_word": _pct_word(card.ctx_cos_percentile),
            "cross_community_ratio": card.cross_community_ratio,
            "cross_ratio_percentile": card.cross_ratio_percentile,
            "cross_ratio_percentile_word": _pct_word(card.cross_ratio_percentile),
        },
        "family_facts": family_facts,
        "node_text": text,
        "text_truncated": bool(shown_truncated or text_card_truncated),
        "text_truncated_full_length": raw_len if shown_truncated else None,
        "neighbours": nbrs,
        "neighbour_selection": card.neighbour_selection,
        "evidence_missing": missing,
        "limitations": list(card.limitations),
    }


def aggregate_verdict(
    assessments: Sequence[ConditionAssessment],
    n_conditions: int,
) -> Tuple[str, List[str]]:
    status: Dict[int, str] = {}
    for a in assessments:
        if 0 <= a.condition_index < n_conditions:
            status[a.condition_index] = a.status
    lines = []
    for i in range(n_conditions):
        st = status.get(i, "unknown")
        lines.append(f"c{i}={st}")
    if n_conditions == 0:
        return "undecidable", ["no necessary conditions defined"]
    if any(status.get(i, "unknown") == "contradicted" for i in range(n_conditions)):
        return "not_matches", lines
    if any(status.get(i, "unknown") == "unknown" for i in range(n_conditions)):
        return "undecidable", lines
    return "matches", lines


def _gap_payload(gap: MissingPattern, checklist: Sequence[str]) -> Dict:
    return {
        "gap_id": gap.gap_id,
        "pattern": gap.name,
        "claim": gap.claim,
        "scope": gap.scope,
        "pattern_definition": gap.pattern_definition,
        "evidence_needed": list(gap.evidence_needed),
        "necessary_conditions": list(checklist),
    }


class LLMReviewer(Agent):

    name = "llm_reviewer"

    def __init__(self):
        pass

    def run(self, *args, **kwargs):
        return self.review(*args, **kwargs)

    def review(
        self,
        gaps: Sequence[MissingPattern],
        cards: Sequence[SampleCard],
        checklists: Optional[Dict[str, Sequence[str]]] = None,
        reference_profile: Optional[Dict] = None,
        ctx=None,
        coverage_criterion: Optional[str] = None,
    ) -> "LLMPreReview":
        checklists = checklists or {}
        by_gap: Dict[str, List[SampleCard]] = {}
        for c in cards:
            by_gap.setdefault(c.gap_id, []).append(c)

        sample_fb: List[SampleFeedback] = []
        for gap in gaps:
            gid = gap.gap_id or "gap-000"
            sample_fb.extend(
                self.review_samples(
                    gap, by_gap.get(gid, []),
                    checklists.get(gid, []),
                    reference_profile=reference_profile,
                    ctx=ctx,
                )
            )
        gap_fb = self.review_gaps(
            gaps, sample_fb, cards,
            reference_profile=reference_profile,
            coverage_criterion=coverage_criterion,
        )
        return LLMPreReview(sample_feedback=sample_fb, gap_feedback=gap_fb)

    def review_samples(
        self,
        gap: MissingPattern,
        cards: Sequence[SampleCard],
        checklist: Sequence[str],
        reference_profile: Optional[Dict] = None,
        ctx=None,
    ) -> List[SampleFeedback]:
        gid = gap.gap_id or "gap-000"
        if not cards:
            return []
        import json as _json

        fam = family_key(gap.name)
        gap_json = _json.dumps(_gap_payload(gap, checklist), ensure_ascii=False)
        cards_json = _json.dumps(
            [card_evidence(c, ctx=ctx, family=fam) for c in cards],
            ensure_ascii=False,
        )
        ref_json = _json.dumps(reference_profile, ensure_ascii=False) \
            if reference_profile else ""
        user = build_sample_review_user_prompt(
            gap_json,
            _json.dumps(list(checklist), ensure_ascii=False),
            cards_json,
            reference_json=ref_json,
        )
        by_id: Dict[int, "SampleJudgement"] = {}
        try:
            result = llm.complete(SYSTEM_LLM_SAMPLE_REVIEWER, user, SampleJudgements)
            known_ids = {c.sample_id for c in cards}
            for j in result.judgements:
                if j.sample_id not in known_ids:
                    log.warning("%s: LLM returned unknown sample_id %s; dropped",
                                gid, j.sample_id)
                else:
                    by_id[j.sample_id] = j
        except Exception as exc:
            log.warning("LLM sample pre-review FAILED for %s (%d cards); "
                        "marking all undecidable: %s", gid, len(cards), exc)
            by_id = {}

        out: List[SampleFeedback] = []
        for c in cards:
            j = by_id.get(c.sample_id)
            if j is None:
                out.append(SampleFeedback(
                    gap_id=gid, sample_id=c.sample_id,
                    verdict="undecidable",
                    comment="[pre-review unavailable] no LLM judgement for this card",
                    source="llm",
                ))
            else:
                verdict, cond_lines = aggregate_verdict(
                    j.assessments, len(checklist))
                comment = (j.reason or "").strip()
                if cond_lines:
                    comment = (comment + " | " if comment else "") + \
                        " ".join(cond_lines)
                ev = card_evidence(c, ctx=ctx, family=fam)
                missing = ev.get("evidence_missing") or []
                if missing and verdict == "not_matches":
                    verdict = "undecidable"
                    comment = (comment + " | " if comment else "") + \
                        "[program: " + "; ".join(missing) + "]"
                out.append(SampleFeedback(
                    gap_id=gid, sample_id=c.sample_id,
                    verdict=verdict,
                    comment=comment[:500],
                    source="llm",
                ))
        return out

    def review_gaps(
        self,
        gaps: Sequence[MissingPattern],
        sample_feedback: Sequence[SampleFeedback],
        cards: Optional[Sequence[SampleCard]] = None,
        reference_profile: Optional[Dict] = None,
        coverage_criterion: Optional[str] = None,
    ) -> List[GapFeedback]:
        import json as _json

        if not gaps:
            return []
        outcomes = self._sample_outcomes(
            gaps, sample_feedback, cards, coverage_criterion)
        gaps_json = _json.dumps(
            [{"gap_id": g.gap_id, "pattern": g.name, "claim": g.claim,
              "scope": g.scope, "pattern_definition": g.pattern_definition}
             for g in gaps],
            ensure_ascii=False,
        )
        ref_json = _json.dumps(reference_profile, ensure_ascii=False) \
            if reference_profile else ""
        user = build_gap_review_user_prompt(
            gaps_json, _json.dumps(outcomes, ensure_ascii=False),
            reference_json=ref_json,
        )
        try:
            result = llm.complete(SYSTEM_LLM_GAP_REVIEWER, user, GapSuggestions)
            by_id = {s.gap_id: s for s in result.suggestions}
        except Exception as exc:
            log.warning("LLM gap pre-review FAILED (%d gaps); keeping all "
                        "pending: %s", len(gaps), exc)
            by_id = {}

        out: List[GapFeedback] = []
        for g in gaps:
            gid = g.gap_id or "gap-000"
            s = by_id.get(gid)
            if s is None:
                out.append(GapFeedback(
                    gap_id=gid, review_status="pending",
                    human_comment="[pre-review unavailable] LLM gap suggestion failed",
                    source="llm",
                ))
            else:
                comment = (s.reason or "").strip()
                if s.evidence_summary:
                    comment = (comment + " | " if comment else "") + s.evidence_summary
                out.append(GapFeedback(
                    gap_id=gid,
                    review_status=s.suggested_status,
                    human_comment=comment[:1000],
                    source="llm",
                ))
        return out

    @staticmethod
    def _sample_outcomes(
        gaps: Sequence[MissingPattern],
        sample_feedback: Sequence[SampleFeedback],
        cards: Optional[Sequence[SampleCard]] = None,
        coverage_criterion: Optional[str] = None,
    ) -> List[Dict]:
        card_by_key = {(c.gap_id, c.sample_id): c for c in (cards or [])}
        fb_by_gap: Dict[str, List[SampleFeedback]] = {}
        for f in sample_feedback:
            fb_by_gap.setdefault(f.gap_id, []).append(f)

        out: List[Dict] = []
        for g in gaps:
            gid = g.gap_id or "gap-000"
            fbs = fb_by_gap.get(gid, [])
            by_group: Dict[str, Dict[str, int]] = {}
            for f in fbs:
                grp = card_by_key.get((gid, f.sample_id)).group \
                    if (gid, f.sample_id) in card_by_key else "unknown"
                d = by_group.setdefault(grp, {"matches": 0, "not_matches": 0,
                                              "undecidable": 0})
                d[f.verdict] = d.get(f.verdict, 0) + 1
            comm_dist: Dict[int, int] = {}
            for f in fbs:
                c = card_by_key.get((gid, f.sample_id))
                if c is not None:
                    comm_dist[c.community] = comm_dist.get(c.community, 0) + 1
            judgements = [
                {
                    "sample_id": f.sample_id,
                    "group": card_by_key.get((gid, f.sample_id)).group
                             if (gid, f.sample_id) in card_by_key else "unknown",
                    "community": card_by_key.get((gid, f.sample_id)).community
                                 if (gid, f.sample_id) in card_by_key else None,
                    "verdict": f.verdict,
                    "reason": f.comment,
                }
                for f in fbs
            ]
            out.append({
                "gap_id": gid,
                "pattern": g.name,
                "claim": g.claim,
                "scope": g.scope,
                "pattern_definition": g.pattern_definition,
                "coverage_criterion": coverage_criterion or
                    "(none agreed -- do not invent one; without it, weak "
                    "evidence should lean insufficient_evidence)",
                "verdict_counts": {
                    "matches": sum(1 for f in fbs if f.verdict == "matches"),
                    "not_matches": sum(1 for f in fbs if f.verdict == "not_matches"),
                    "undecidable": sum(1 for f in fbs if f.verdict == "undecidable"),
                },
                "by_group": by_group,
                "per_card_judgements": judgements,
                "sample_community_distribution": comm_dist,
                "note": "matches = existing instances (counterexamples to the "
                        "missing claim); targeted counts are retrieval results, "
                        "never population ratios; if no matches, at most 'not "
                        "observed in this round', never 'absent'.",
            })
        return out


class LLMPreReview:

    def __init__(
        self,
        sample_feedback: List[SampleFeedback],
        gap_feedback: List[GapFeedback],
    ):
        self.sample_feedback = sample_feedback
        self.gap_feedback = gap_feedback

    @property
    def degraded(self) -> bool:
        return (
            any("[pre-review unavailable]" in f.comment for f in self.sample_feedback)
            or any("[pre-review unavailable]" in g.human_comment for g in self.gap_feedback)
        )

    @property
    def uniform_not_matches(self) -> bool:
        return (len(self.sample_feedback) > 0
                and all(f.verdict == "not_matches"
                        for f in self.sample_feedback))

    def summary(self) -> str:
        n_cards = len(self.sample_feedback)
        n_undecidable = sum(1 for f in self.sample_feedback
                            if f.verdict == "undecidable")
        n_gaps = len(self.gap_feedback)
        n_pending = sum(1 for g in self.gap_feedback
                        if g.review_status == "pending")
        parts = [f"LLM pre-review: {n_cards} cards "
                 f"({n_undecidable} undecidable), {n_gaps} gaps "
                 f"({n_pending} pending)"]
        if self.uniform_not_matches:
            parts.append("[UNIFORM not_matches: every card judged not_matches "
                         "-- possible miscalibration or the pattern truly has "
                         "no instances; spot-check cards by hand]")
        if self.degraded:
            parts.append("[DEGRADED: some LLM calls failed]")
        return " ".join(parts)