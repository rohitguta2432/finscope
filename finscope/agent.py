"""FinScope agent: a transparent analysis loop.

No agent framework — just a readable pipeline so reviewers can see exactly
how the agent works. The loop:
  parse → analyze_allocation → detect_overlap → find_tax_leakage
  → expense_leakage → score_health → generate_report

All numbers come from the Python analysers. The LLM only phrases the
educational narrative.

COMPLIANCE: the agent never outputs recommendation language.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .llm import MockLLM, get_llm
from .tools import (
    DISCLAIMER,
    analyze_allocation,
    detect_overlap,
    detect_regular_plans,
    expense_leakage,
    find_tax_leakage,
    generate_report,
    parse_holdings,
    score_health,
)


@dataclass
class AnalysisResult:
    """Full output of one FinScope run."""
    holdings: list[dict[str, Any]]
    allocation: dict[str, Any]
    overlap: dict[str, Any]
    tax: dict[str, Any]
    expense: dict[str, Any]
    regular_plans: dict[str, Any]
    health: dict[str, Any]
    narrative: str
    report_path: str
    report_text: str

    @property
    def scores(self) -> dict[str, int]:
        return self.health["scores"]

    def all_flags(self) -> list[str]:
        return (
            self.allocation.get("flags", [])
            + self.overlap.get("flags", [])
            + self.tax.get("flags", [])
            + self.expense.get("flags", [])
            + self.regular_plans.get("flags", [])
            + self.health.get("concentration_flags", [])
            + self.health.get("emergency_flags", [])
        )


class Agent:
    def __init__(
        self,
        llm: Any | None = None,
        out_dir: str = "out",
        age: int = 35,
        risk: str = "moderate",
        monthly_expenses: float | None = None,
        today: date | None = None,
        concentration_threshold: float = 20.0,
    ) -> None:
        self.llm = llm or get_llm()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.age = age
        self.risk = risk
        self.monthly_expenses = monthly_expenses
        self.today = today or date.today()
        self.concentration_threshold = concentration_threshold

    def run(self, csv_path: str) -> AnalysisResult:
        """Run the full analysis pipeline on a holdings CSV."""
        # Step 1: parse
        holdings = parse_holdings(csv_path)

        # Step 2: analyzers — all numbers come from Python, not the LLM
        allocation = analyze_allocation(holdings, age=self.age, risk=self.risk)
        overlap = detect_overlap(holdings)
        tax = find_tax_leakage(holdings, today=self.today)
        expense = expense_leakage(holdings)
        regular_plans = detect_regular_plans(holdings)
        health = score_health(
            holdings,
            allocation,
            overlap,
            tax,
            expense,
            monthly_expenses=self.monthly_expenses,
            concentration_threshold=self.concentration_threshold,
        )

        # Step 3: LLM phrases the narrative only
        summary = {
            "scores": health["scores"],
            "total_portfolio": allocation.get("total_portfolio", 0),
            "drift_pp": allocation.get("drift_pp", 0),
            "flagged_overlap_pairs": sum(1 for p in overlap.get("pairs", []) if p["flagged"]),
            "annual_drag_inr": expense.get("total_annual_drag_inr", 0),
            "commission_drag_inr": regular_plans.get("total_commission_drag_inr", 0),
            "total_flags": len(
                allocation.get("flags", [])
                + overlap.get("flags", [])
                + tax.get("flags", [])
                + expense.get("flags", [])
                + regular_plans.get("flags", [])
                + health.get("concentration_flags", [])
                + health.get("emergency_flags", [])
            ),
        }
        narrative = self.llm.generate_narrative(summary)

        # Step 4: report
        report_path = str(self.out_dir / "report.md")
        report_text = generate_report(
            holdings, allocation, overlap, tax, expense, health, narrative, report_path,
            regular_plans_result=regular_plans,
        )

        return AnalysisResult(
            holdings=holdings,
            allocation=allocation,
            overlap=overlap,
            tax=tax,
            expense=expense,
            regular_plans=regular_plans,
            health=health,
            narrative=narrative,
            report_path=report_path,
            report_text=report_text,
        )


def run_analysis(
    csv_path: str,
    age: int = 35,
    risk: str = "moderate",
    monthly_expenses: float | None = None,
    today: date | None = None,
    llm: Any | None = None,
    out_dir: str = "out",
) -> AnalysisResult:
    """One-shot helper used by the eval harness and CLI."""
    agent = Agent(
        llm=llm,
        out_dir=out_dir,
        age=age,
        risk=risk,
        monthly_expenses=monthly_expenses,
        today=today,
    )
    return agent.run(csv_path)
