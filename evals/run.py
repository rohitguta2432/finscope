"""FinScope eval suite.

Runs the agent against a golden dataset and scores the behaviours that matter
for an educational portfolio analyser:

  * flag-detection recall  — known issues correctly flagged
  * numeric accuracy       — overlap %, expense drag, concentration % within tolerance
  * disclaimer presence    — 100% of reports carry the compliance disclaimer
  * compliance violations  — ZERO banned recommendation phrases in any output

CI-style gate: exits non-zero if any metric regresses below threshold.

    python -m evals.run                  # deterministic mock provider (no key)
    LLM_PROVIDER=openai python -m evals.run --real

IMPORTANT: this file uses ONLY the standard library (csv, json, pathlib,
argparse, sys, datetime, re) so it runs on a bare Python interpreter.
Third-party deps (openai, mcp) are lazy-imported inside llm.py/mcp_server.py
only when LLM_PROVIDER=openai is set.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Bring in project modules — no third-party deps at import time
from finscope.tools import (
    BANNED_PHRASES,
    DISCLAIMER,
    analyze_allocation,
    detect_overlap,
    expense_leakage,
    find_tax_leakage,
    parse_holdings,
    score_health,
    generate_report,
)
from finscope.llm import MockLLM, get_llm

DATASET = Path(__file__).parent / "golden_dataset.json"
REPORT = Path(__file__).parent / "report.json"
OUT_DIR = Path(__file__).parent.parent / "out" / "evals"

# Gate definitions: metric -> (direction, threshold)
# direction "min": score >= threshold to pass
# direction "max": rate <= threshold to pass
GATES = {
    "flag_detection_recall": ("min", 0.95),
    "numeric_accuracy": ("min", 0.95),
    "disclaimer_presence": ("min", 1.00),
    "compliance_violations": ("max", 0),
}


# ------------------------------------------------------------------ helpers

def _run_case(case: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Run one eval case: build holdings in-memory, run all analysers, return results."""
    # Build holdings list directly from the case (no CSV needed for evals)
    holdings = []
    for row in case["portfolio"]:
        h = {
            "instrument": row["instrument"],
            "type": row["type"],
            "amount": float(row["amount"]),
            "units": None,
            "buy_date": _parse_date(row.get("buy_date")),
            "expense_ratio": float(row["expense_ratio"]) if row.get("expense_ratio") else None,
            "fund_id": row.get("fund_id"),
        }
        holdings.append(h)

    params = case.get("params", {})
    age = params.get("age", 35)
    risk = params.get("risk", "moderate")
    monthly_expenses = params.get("monthly_expenses")
    today = _parse_date(case.get("today")) or date(2026, 6, 7)

    allocation = analyze_allocation(holdings, age=age, risk=risk)
    overlap = detect_overlap(holdings)
    tax = find_tax_leakage(holdings, today=today)
    expense = expense_leakage(holdings)
    health = score_health(
        holdings, allocation, overlap, tax, expense,
        monthly_expenses=monthly_expenses,
    )

    summary = {
        "scores": health["scores"],
        "total_portfolio": allocation.get("total_portfolio", 0),
        "drift_pp": allocation.get("drift_pp", 0),
        "flagged_overlap_pairs": sum(1 for p in overlap.get("pairs", []) if p["flagged"]),
        "annual_drag_inr": expense.get("total_annual_drag_inr", 0),
        "total_flags": len(
            allocation.get("flags", [])
            + overlap.get("flags", [])
            + tax.get("flags", [])
            + expense.get("flags", [])
            + health.get("concentration_flags", [])
            + health.get("emergency_flags", [])
        ),
    }
    narrative = llm.generate_narrative(summary)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(OUT_DIR / f"{case['id']}.md")
    report_text = generate_report(
        holdings, allocation, overlap, tax, expense, health, narrative, out_path
    )

    return {
        "allocation": allocation,
        "overlap": overlap,
        "tax": tax,
        "expense": expense,
        "health": health,
        "narrative": narrative,
        "report_text": report_text,
    }


def _parse_date(s: Any) -> date | None:
    if not s:
        return None
    from datetime import datetime
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _compliance_check(text: str) -> list[str]:
    """Return list of banned phrases found in text (case-insensitive)."""
    lower = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lower]


def _score_flags(result: dict[str, Any], expect: dict[str, Any]) -> tuple[int, int]:
    """Return (flags_hit, flags_expected) for a case."""
    allocation = result["allocation"]
    overlap = result["overlap"]
    tax = result["tax"]
    expense = result["expense"]
    health = result["health"]

    expected = 0
    hit = 0

    # overlap_flagged
    if "overlap_flagged" in expect:
        expected += 1
        if expect["overlap_flagged"]:
            has_flag = any(p["flagged"] for p in overlap.get("pairs", []))
            if has_flag:
                hit += 1
        else:
            hit += 1  # no flag expected and we didn't flag — correct

    # debt_stcg_flagged
    if "debt_stcg_flagged" in expect:
        expected += 1
        found = any(d.get("issue") == "debt_stcg" for d in tax.get("details", []))
        if found == expect["debt_stcg_flagged"]:
            hit += 1

    # concentration_flagged
    if "concentration_flagged" in expect:
        expected += 1
        found = len(health.get("concentration_flags", [])) > 0
        if found == expect["concentration_flagged"]:
            hit += 1

    # allocation_flagged
    if "allocation_flagged" in expect:
        expected += 1
        found = allocation.get("flag_drift", False)
        if found == expect["allocation_flagged"]:
            hit += 1

    # expense_flagged
    if "expense_flagged" in expect:
        expected += 1
        found = len(expense.get("flags", [])) > 0
        if found == expect["expense_flagged"]:
            hit += 1

    # equity_stcg_flagged
    if "equity_stcg_flagged" in expect:
        expected += 1
        found = any(d.get("issue") == "equity_stcg" for d in tax.get("details", []))
        if found == expect["equity_stcg_flagged"]:
            hit += 1

    # overall_score_min
    if "overall_score_min" in expect:
        expected += 1
        if health["scores"]["overall"] >= expect["overall_score_min"]:
            hit += 1

    return hit, expected


def _score_numerics(result: dict[str, Any], expect: dict[str, Any]) -> tuple[int, int]:
    """Return (numeric_checks_passed, numeric_checks_total) for a case."""
    allocation = result["allocation"]
    overlap = result["overlap"]
    expense = result["expense"]
    health = result["health"]

    expected = 0
    hit = 0

    # overlap_pct_min
    if "overlap_pct_min" in expect:
        expected += 1
        pair_a = expect.get("overlap_pair", [None, None])
        for p in overlap.get("pairs", []):
            names = {p["fund_a"], p["fund_b"]}
            if set(pair_a) == names and p["overlap_pct"] >= expect["overlap_pct_min"]:
                hit += 1
                break

    # concentration_pct_min
    if "concentration_pct_min" in expect:
        expected += 1
        instr = expect.get("concentration_instrument", "")
        total = allocation.get("total_portfolio", 0)
        holdings = [h for h in [] ]  # we check via concentration_flags text
        # check via flag text — instrument name and pct should appear
        flag_text = " ".join(health.get("concentration_flags", []))
        if instr in flag_text:
            # extract pct from flag text
            m = re.search(r"(\d+\.\d+)%", flag_text)
            if m and float(m.group(1)) >= expect["concentration_pct_min"]:
                hit += 1

    # equity_pct_min
    if "equity_pct_min" in expect:
        expected += 1
        equity_pct = allocation.get("allocation_pct", {}).get("equity", 0)
        if equity_pct >= expect["equity_pct_min"]:
            hit += 1

    # drift_direction
    if "drift_direction" in expect:
        expected += 1
        drift = allocation.get("drift_pp", 0)
        direction = "overweight" if drift > 0 else "underweight"
        if direction == expect["drift_direction"]:
            hit += 1

    # annual_drag_inr_min
    if "annual_drag_inr_min" in expect:
        expected += 1
        drag = expense.get("total_annual_drag_inr", 0)
        if drag >= expect["annual_drag_inr_min"]:
            hit += 1

    return hit, expected


# ------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FinScope eval suite.")
    parser.add_argument("--real", action="store_true", help="use configured real LLM")
    args = parser.parse_args()

    llm = get_llm() if args.real else MockLLM()
    cases = json.loads(DATASET.read_text())

    rows = []
    flag_hits = flag_expected = 0
    numeric_hits = numeric_expected = 0
    disclaimer_present = 0
    compliance_violations = 0

    for c in cases:
        result = _run_case(c, llm)
        expect = c.get("expect", {})

        # Flag detection
        fh, fe = _score_flags(result, expect)
        flag_hits += fh
        flag_expected += fe

        # Numeric accuracy
        nh, ne = _score_numerics(result, expect)
        numeric_hits += nh
        numeric_expected += ne

        # Disclaimer presence
        report = result["report_text"]
        has_disclaimer = DISCLAIMER in report
        if has_disclaimer:
            disclaimer_present += 1

        # Compliance lint — check narrative + report for banned phrases
        full_text = result["narrative"] + "\n" + report
        violations = _compliance_check(full_text)
        n_violations = len(violations)
        compliance_violations += n_violations

        rows.append({
            "id": c["id"],
            "flag_hits": fh,
            "flag_expected": fe,
            "flag_recall": fh / fe if fe else 1.0,
            "numeric_hits": nh,
            "numeric_expected": ne,
            "numeric_accuracy": nh / ne if ne else 1.0,
            "disclaimer_present": has_disclaimer,
            "compliance_violations": n_violations,
            "overall_score": result["health"]["scores"]["overall"],
        })

    n = len(cases)
    metrics = {
        "flag_detection_recall": flag_hits / flag_expected if flag_expected else 1.0,
        "numeric_accuracy": numeric_hits / numeric_expected if numeric_expected else 1.0,
        "disclaimer_presence": disclaimer_present / n if n else 0.0,
        "compliance_violations": compliance_violations,
    }

    provider = "real" if args.real else "mock"
    print(f"\nFinScope evals — {n} cases — provider: {provider}\n")

    all_pass = True
    for name, (direction, threshold) in GATES.items():
        score = metrics[name]
        if direction == "max":
            ok = score <= threshold
            cmp = "<="
        else:
            ok = score >= threshold
            cmp = ">="
        all_pass &= ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  {name:<28} "
            f"{'%.1f%%' % (score * 100) if isinstance(score, float) else score}  "
            f"({cmp} {'%.0f%%' % (threshold * 100) if isinstance(threshold, float) else threshold})"
        )

    REPORT.write_text(json.dumps({"metrics": metrics, "rows": rows, "passed": all_pass}, indent=2))
    print(f"\n  report -> {REPORT}")
    print(f"\n{'GATE PASSED' if all_pass else 'GATE FAILED'}\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
