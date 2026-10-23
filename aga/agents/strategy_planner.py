
from __future__ import annotations

from .base import Agent
from ..llm import registry as llm
from ..llm.prompts import SYSTEM_STRATEGY_PLANNER, build_strategy_user_prompt
from ..orchestration.schema import (
    DatasetReport,
    MissingPatterns,
    GenerationStrategy,
)


class StrategyPlanner(Agent):
    name = "strategy_planner"

    def __init__(self, total_budget_hint: int | None = None):
        self.total_budget_hint = total_budget_hint

    def run(
        self,
        report: DatasetReport,
        missing: MissingPatterns,
        review_approved: bool = False,
        review_refined: bool = False,
    ) -> GenerationStrategy:
        report_json = report.model_dump_json(indent=2)
        missing_json = missing.model_dump_json(indent=2)
        user = build_strategy_user_prompt(report_json, missing_json,
                                          review_approved=review_approved,
                                          review_refined=review_refined)
        if self.total_budget_hint is not None:
            user += (
                f"\n\nBUDGET HINT: keep total_new_anomalies around "
                f"{self.total_budget_hint} (~5% of the {report.stats.num_nodes} original nodes)."
            )
        return llm.complete(SYSTEM_STRATEGY_PLANNER, user, GenerationStrategy)