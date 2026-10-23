
from __future__ import annotations

from .base import Agent
from ..llm import registry as llm
from ..llm.prompts import SYSTEM_GAP_ANALYZER, build_gap_user_prompt
from ..orchestration.schema import DatasetReport, MissingPatterns

FIXED_CORE = {"context_attack", "semantic_camouflage", "bridge_attack", "hybrid_attack", "hub_attack"}
CENSUS = {"attribute", "structure", "hybrid"}

_FAMILY_EVIDENCE: dict = {
    "context_attack": [
        "targeted candidates with low node-vs-neighbour semantic similarity",
        "stratified random sample to estimate occurrence",
        "neighbour texts to judge context conflict by hand",
    ],
    "semantic_camouflage": [
        "surface-similar candidates with potential relational-context conflicts",
        "neighbour texts to judge whether a real semantic clash exists",
        "stratified random sample to estimate occurrence",
    ],
    "bridge_attack": [
        "community partition + cross-community connection ratio per node",
        "degree statistics of cross-community connectors",
        "neighbour texts to judge whether the bridging lacks semantic support",
    ],
    "hybrid_attack": [
        "attribute-deviation candidates AND structural-deviation candidates",
        "evidence for both conditions separately (they are verified apart)",
    ],
    "hub_attack": [
        "degree distribution vs the real structural-anomaly degree profile",
        "high-degree candidates with their neighbour texts",
    ],
}
_DEFAULT_EVIDENCE = [
    "targeted candidates most likely to already exhibit the pattern",
    "stratified random sample to estimate occurrence",
    "normal/boundary controls to separate pattern from ordinary variation",
]


def _family_key(name: str) -> str:
    if name in FIXED_CORE:
        return name
    alias = {
        "context": "context_attack",
        "bridge": "bridge_attack",
        "semantic_camouflage": "semantic_camouflage",
        "hybrid": "hybrid_attack",
        "structure": "hub_attack",
        "attribute": "context_attack",
    }
    return alias.get(name, "")


def _default_claim(m) -> str:
    return (
        f"Training data may under-cover the {m.name} anomaly pattern "
        f"(scope: full training set). This is a hypothesis to verify, not a fact."
    )


def _default_definition(m) -> str:
    return (
        f"A node matches the {m.name} pattern if it satisfies the pattern's "
        f"fixed-core definition; see rationale/proposed_spec for the specifics "
        f"the LLM gave. If ambiguous, refine this definition BEFORE sampling."
    )


def fill_verification_fields(gaps: MissingPatterns) -> MissingPatterns:
    for i, m in enumerate(gaps.missing, start=1):
        if m.name not in FIXED_CORE and m.name not in CENSUS:
            m.proposed = True
            if not m.proposed_spec:
                m.proposed_spec = "LLM-proposed pattern; implement builder + register to enable."
        if not m.gap_id:
            m.gap_id = f"gap-{i:03d}"
        if not m.initial_rationale:
            m.initial_rationale = m.rationale
        if not m.claim:
            m.claim = _default_claim(m)
        if not m.pattern_definition:
            m.pattern_definition = _default_definition(m)
        if not m.scope:
            m.scope = "train_all"
        if not m.evidence_needed:
            m.evidence_needed = list(
                _FAMILY_EVIDENCE.get(_family_key(m.name), _DEFAULT_EVIDENCE))
    return gaps


class GapAnalyzer(Agent):
    name = "gap_analyzer"

    def run(self, report: DatasetReport) -> MissingPatterns:
        report_json = report.model_dump_json(indent=2)
        user = build_gap_user_prompt(report_json)
        result = llm.complete(SYSTEM_GAP_ANALYZER, user, MissingPatterns)
        return fill_verification_fields(result)