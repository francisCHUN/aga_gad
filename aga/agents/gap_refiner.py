
from __future__ import annotations

import logging
import re
from typing import Sequence

from .base import Agent
from .gap_analyzer import fill_verification_fields
from ..llm import registry as llm
from ..llm.prompts import SYSTEM_GAP_REFINER, build_gap_refine_user_prompt
from ..orchestration.schema import (
    DatasetReport,
    GapReview,
    MissingPatterns,
)

log = logging.getLogger("aga.review.refiner")

REFINE_DEGRADED_MARKER = "[refine unavailable]"


def _next_gap_number(existing_ids: Sequence[str]) -> int:
    top = 0
    for gid in existing_ids:
        m = re.match(r"^gap-(\d+)$", (gid or "").strip())
        if m:
            top = max(top, int(m.group(1)))
    return top + 1


class GapRefiner(Agent):
    name = "gap_refiner"

    def run(self, *args, **kwargs):
        return self.refine(*args, **kwargs)

    def refine(
        self,
        report: DatasetReport,
        previous_gaps: MissingPatterns,
        feedback: Sequence[GapReview],
        round_no: int = 1,
    ) -> MissingPatterns:
        report_json = report.model_dump_json(indent=2)
        gaps_json = previous_gaps.model_dump_json(indent=2)
        feedback_json = "[" + ", ".join(
            v.model_dump_json() for v in feedback) + "]"
        user = build_gap_refine_user_prompt(
            report_json, gaps_json, feedback_json, round_no)
        try:
            result = llm.complete(SYSTEM_GAP_REFINER, user, MissingPatterns)
        except Exception as exc:
            log.warning("GapRefiner round %d failed (%s); returning previous "
                        "gap set unchanged.", round_no, exc)
            prev = previous_gaps.model_copy(deep=True)
            prev.global_note = (prev.global_note or "") + " " + REFINE_DEGRADED_MARKER
            return prev
        return self._normalize(result, previous_gaps, feedback)

    def _normalize(
        self,
        result: MissingPatterns,
        previous_gaps: MissingPatterns,
        feedback: Sequence[GapReview],
    ) -> MissingPatterns:
        prev_by_id = {g.gap_id: g for g in previous_gaps.missing}
        prev_by_name = {g.name: g for g in previous_gaps.missing}
        used = {g.gap_id for g in previous_gaps.missing}
        nxt = _next_gap_number([g.gap_id for g in previous_gaps.missing])

        hard_drop = {
            v.gap_id for v in feedback
            if v.review_status == "rejected" and v.counterexample_sample_ids
        }

        out = []
        for m in result.missing:
            gid = (m.gap_id or "").strip()
            if gid and gid in prev_by_id:
                pass
            elif gid and gid in used and gid not in prev_by_id:
                m.gap_id = ""
            if not (m.gap_id or "").strip():
                prev = prev_by_name.get(m.name)
                if prev is not None and prev.gap_id not in hard_drop:
                    m.gap_id = prev.gap_id
                else:
                    m.gap_id = f"gap-{nxt:03d}"
                    nxt += 1
            used.add(m.gap_id)
            out.append(m)

        kept_ids = [m.gap_id for m in out]
        revived = [gid for gid in kept_ids if gid in hard_drop]
        if revived:
            log.warning("GapRefiner tried to revive rejected gap(s) %s "
                        "carrying reviewer counterexamples; removed.",
                        revived)
            out = [m for m in out if m.gap_id not in hard_drop]

        result.missing = out
        result.existing_patterns = (result.existing_patterns
                                    or previous_gaps.existing_patterns)
        fill_verification_fields(result)
        return result