"""FinScope analysis tools.

All number-crunching lives here. The LLM only phrases the narrative.
Each function returns a plain dict so the agent loop, evals, and MCP
server can all consume the same output without adaptation.

COMPLIANCE: these tools FLAG issues and EXPLAIN them. They never
recommend buying, selling, or switching a specific instrument.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

_DATA = Path(__file__).parent / "data"

with (_DATA / "fund_holdings.json").open() as _f:
    FUND_HOLDINGS: dict[str, dict[str, Any]] = json.load(_f)

with (_DATA / "category_medians.json").open() as _f:
    CATEGORY_MEDIANS: dict[str, float] = json.load(_f)

# Compliance disclaimer — must appear in every generated report.
DISCLAIMER = "Educational diagnostic · Not investment advice · Not a SEBI-registered RIA"

# Phrases that are BANNED from any output — hard compliance gate.
BANNED_PHRASES = [
    "you should buy",
    "you should sell",
    "you should switch",
    "i recommend",
    "we recommend",
    "i suggest buying",
    "i suggest selling",
    "i suggest switching",
    "buy this",
    "sell this",
    "switch to",
    "invest in",
    "purchase",
    "recommend buying",
    "recommend selling",
]


def parse_holdings(csv_path: str) -> list[dict[str, Any]]:
    """Parse a holdings CSV and normalise each row.

    Expected columns (extras ignored):
      instrument, type, amount, units (opt), buy_date (opt),
      expense_ratio (opt), fund_id (opt)

    Returns a list of dicts with numeric fields coerced.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            instrument = raw.get("instrument", "").strip()
            if not instrument:
                continue
            row: dict[str, Any] = {
                "instrument": instrument,
                "type": raw.get("type", "").strip(),
                "amount": float(raw.get("amount") or 0),
                "units": float(raw.get("units") or 0) if raw.get("units") else None,
                "buy_date": _parse_date(raw.get("buy_date", "")),
                "expense_ratio": float(raw.get("expense_ratio") or 0) if raw.get("expense_ratio") else None,
                "fund_id": raw.get("fund_id", "").strip() or None,
            }
            rows.append(row)
    return rows


def _parse_date(s: str) -> date | None:
    s = s.strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def analyze_allocation(
    holdings: list[dict[str, Any]],
    age: int,
    risk: str = "moderate",
) -> dict[str, Any]:
    """Compare actual asset allocation vs a rule-of-thumb band.

    Rule: equity target ≈ (110 - age)% for moderate risk.
    Flags drift beyond ±10 pp as EDUCATIONAL information.
    Returns actual percentages, target band, drift flag, and a flag message.
    """
    total = sum(h["amount"] for h in holdings)
    if total == 0:
        return {"error": "portfolio total is zero", "flags": []}

    # Bucket amounts by broad asset class
    buckets: dict[str, float] = {
        "equity": 0.0,
        "debt": 0.0,
        "gold": 0.0,
        "cash": 0.0,
        "other": 0.0,
    }
    for h in holdings:
        t = h["type"]
        if t in ("equity_mf", "stock"):
            buckets["equity"] += h["amount"]
        elif t in ("debt_mf", "fd"):
            buckets["debt"] += h["amount"]
        elif t == "gold":
            buckets["gold"] += h["amount"]
        elif t == "cash":
            buckets["cash"] += h["amount"]
        else:
            buckets["other"] += h["amount"]

    pcts = {k: round(v / total * 100, 1) for k, v in buckets.items()}

    # Rule-of-thumb equity target
    risk_adj = {"conservative": -10, "moderate": 0, "aggressive": 10}.get(risk, 0)
    target_equity = max(0, min(100, 110 - age + risk_adj))
    band_low = max(0, target_equity - 10)
    band_high = min(100, target_equity + 10)
    actual_equity = pcts["equity"]
    drift = round(actual_equity - target_equity, 1)
    flag_drift = not (band_low <= actual_equity <= band_high)

    flags = []
    if flag_drift:
        direction = "overweight" if drift > 0 else "underweight"
        flags.append(
            f"Allocation flag: equity is {actual_equity}% of portfolio, "
            f"vs rule-of-thumb target {band_low}%–{band_high}% for age {age} ({risk} risk). "
            f"You are {abs(drift)} pp {direction} equity. "
            f"Question to ask a SEBI RIA: 'Does my equity allocation match my actual risk capacity and goals?'"
        )

    return {
        "total_portfolio": total,
        "allocation_pct": pcts,
        "target_equity_pct": target_equity,
        "band": [band_low, band_high],
        "actual_equity_pct": actual_equity,
        "drift_pp": drift,
        "flag_drift": flag_drift,
        "age": age,
        "risk": risk,
        "flags": flags,
    }


def detect_overlap(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect underlying-stock overlap between mutual funds.

    Uses fund_holdings.json (fund_id -> top_holdings list).
    Reports pairwise Jaccard overlap %. Flags pairs above 40%.
    """
    # Collect MF holdings that have a fund_id in our reference data
    mf_rows = [h for h in holdings if h["type"] in ("equity_mf", "debt_mf") and h["fund_id"]]

    pairs = []
    for i, a in enumerate(mf_rows):
        for b in mf_rows[i + 1:]:
            a_data = FUND_HOLDINGS.get(a["fund_id"], {})
            b_data = FUND_HOLDINGS.get(b["fund_id"], {})
            a_stocks = set(a_data.get("top_holdings", []))
            b_stocks = set(b_data.get("top_holdings", []))
            if not a_stocks or not b_stocks:
                continue
            union = a_stocks | b_stocks
            intersection = a_stocks & b_stocks
            jaccard = round(len(intersection) / len(union) * 100, 1) if union else 0.0
            pairs.append({
                "fund_a": a["instrument"],
                "fund_b": b["instrument"],
                "overlap_pct": jaccard,
                "shared_stocks": sorted(intersection),
                "flagged": jaccard >= 40.0,
            })

    flags = []
    for p in pairs:
        if p["flagged"]:
            flags.append(
                f"Overlap flag: {p['fund_a']} and {p['fund_b']} share "
                f"{p['overlap_pct']}% of their top holdings "
                f"({', '.join(p['shared_stocks'][:5])}{'…' if len(p['shared_stocks']) > 5 else ''}). "
                f"Holding both may not add the diversification you expect. "
                f"Question to ask a SEBI RIA: 'Are these two funds genuinely diversifying my portfolio?'"
            )

    return {"pairs": pairs, "flags": flags}


def find_tax_leakage(
    holdings: list[dict[str, Any]],
    today: date | None = None,
) -> dict[str, Any]:
    """Flag potential tax-inefficiency issues as educational information.

    Checks:
    - Debt MF held < 3 years → STCG taxed at slab rate (post-Apr-2023 rules)
    - Equity MF / stock held < 1 year → STCG at 15%
    - Equity gains that may fall under ₹1L LTCG exemption (harvesting headroom)

    All figures are educational estimates, not tax advice.
    """
    today = today or date.today()
    flags = []
    details = []

    total_equity_gains_estimate = 0.0

    for h in holdings:
        if h["buy_date"] is None:
            continue
        held_days = (today - h["buy_date"]).days
        held_years = held_days / 365.25

        if h["type"] == "debt_mf" and held_years < 3:
            flags.append(
                f"Tax flag (STCG debt): {h['instrument']} held {held_days} days "
                f"(< 3 years). Gains taxed at your income-slab rate (STCG). "
                f"Question to ask a SEBI RIA: 'Does the post-tax return still meet my goal?'"
            )
            details.append({"instrument": h["instrument"], "issue": "debt_stcg", "held_days": held_days})

        elif h["type"] in ("equity_mf", "stock") and 0 < held_years < 1:
            flags.append(
                f"Tax flag (STCG equity): {h['instrument']} held {held_days} days "
                f"(< 1 year). Short-term capital gains taxed at 15%. "
                f"Question to ask a SEBI RIA: 'Is the short-term churn intentional?'"
            )
            details.append({"instrument": h["instrument"], "issue": "equity_stcg", "held_days": held_days})

        elif h["type"] in ("equity_mf", "stock") and held_years >= 1:
            # Rough estimate: assume 12% CAGR for illustration only
            cost = h["amount"]
            est_current = cost * ((1.12) ** held_years)
            gain_est = max(0.0, est_current - cost)
            total_equity_gains_estimate += gain_est

    # LTCG harvesting headroom
    ltcg_exemption = 100000  # ₹1L per year
    harvesting_headroom = max(0.0, ltcg_exemption - total_equity_gains_estimate)
    ltcg_flag = total_equity_gains_estimate < ltcg_exemption
    if ltcg_flag and total_equity_gains_estimate >= 0:
        flags.append(
            f"LTCG harvesting: estimated unrealised long-term equity gains ≈ "
            f"₹{int(total_equity_gains_estimate):,} vs ₹1,00,000 annual exemption. "
            f"Approximately ₹{int(harvesting_headroom):,} of exemption may be unused this year. "
            f"Question to ask a SEBI RIA: 'Should I harvest gains before year-end to use the LTCG exemption?'"
        )

    return {
        "flags": flags,
        "details": details,
        "ltcg_gains_estimate": round(total_equity_gains_estimate, 2),
        "ltcg_headroom": round(harvesting_headroom, 2),
    }


def expense_leakage(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag funds with above-median expense ratios and compute annual ₹ drag.

    Uses category_medians.json for category-specific medians.
    """
    flags = []
    details = []
    total_annual_drag = 0.0

    for h in holdings:
        if h["type"] not in ("equity_mf", "debt_mf") or h["expense_ratio"] is None:
            continue
        fund_id = h["fund_id"]
        category = "unknown"
        if fund_id and fund_id in FUND_HOLDINGS:
            category = FUND_HOLDINGS[fund_id].get("category", "unknown")
        median_er = CATEGORY_MEDIANS.get(category, CATEGORY_MEDIANS["unknown"])
        er = h["expense_ratio"]
        excess = round(er - median_er, 3)
        annual_drag = round(h["amount"] * excess / 100, 2) if excess > 0 else 0.0
        total_annual_drag += annual_drag
        above_median = er > median_er
        details.append({
            "instrument": h["instrument"],
            "expense_ratio": er,
            "category": category,
            "category_median_er": median_er,
            "excess_er": excess,
            "annual_drag_inr": annual_drag,
            "above_median": above_median,
        })
        if above_median:
            flags.append(
                f"Expense flag: {h['instrument']} has expense ratio {er:.2f}% "
                f"vs category median {median_er:.2f}%. "
                f"Annual drag on ₹{int(h['amount']):,} corpus ≈ ₹{int(annual_drag):,}/yr. "
                f"Question to ask a SEBI RIA: 'Is there a lower-cost option in the same category?'"
            )

    return {
        "funds": details,
        "total_annual_drag_inr": round(total_annual_drag, 2),
        "flags": flags,
    }


def score_health(
    holdings: list[dict[str, Any]],
    allocation_result: dict[str, Any],
    overlap_result: dict[str, Any],
    tax_result: dict[str, Any],
    expense_result: dict[str, Any],
    monthly_expenses: float | None = None,
    concentration_threshold: float = 20.0,
) -> dict[str, Any]:
    """Compute per-dimension health scores (0-100) and an overall score.

    Dimensions:
    - allocation: drift penalty
    - overlap: flag penalty
    - tax: flag penalty
    - expense: drag penalty
    - concentration: single holding > threshold% of portfolio
    - emergency_fund: liquid assets vs 6-month expenses (if provided)
    """
    total = allocation_result.get("total_portfolio", 0)
    flags_all = []

    # --- allocation score ---
    drift = abs(allocation_result.get("drift_pp", 0))
    allocation_score = max(0, 100 - int(drift * 3))

    # --- overlap score ---
    flagged_pairs = sum(1 for p in overlap_result.get("pairs", []) if p["flagged"])
    overlap_score = max(0, 100 - flagged_pairs * 25)

    # --- tax score ---
    tax_flags = len(tax_result.get("flags", []))
    tax_score = max(0, 100 - tax_flags * 20)

    # --- expense score ---
    annual_drag = expense_result.get("total_annual_drag_inr", 0)
    drag_pct = (annual_drag / total * 100) if total else 0
    expense_score = max(0, 100 - int(drag_pct * 40))

    # --- concentration score ---
    concentration_flags = []
    concentration_score = 100
    if total > 0:
        for h in holdings:
            pct = h["amount"] / total * 100
            if pct >= concentration_threshold:
                concentration_flags.append(
                    f"Concentration flag: {h['instrument']} is {pct:.1f}% of your portfolio "
                    f"(threshold: {concentration_threshold}%). "
                    f"Question to ask a SEBI RIA: 'Am I taking excess single-instrument risk?'"
                )
                concentration_score = max(0, concentration_score - int(pct))
    flags_all.extend(concentration_flags)

    # --- emergency fund score ---
    emergency_score = 100
    emergency_flags = []
    if monthly_expenses and monthly_expenses > 0:
        liquid = sum(
            h["amount"] for h in holdings if h["type"] in ("cash", "fd")
        )
        months_covered = liquid / monthly_expenses
        if months_covered < 3:
            emergency_flags.append(
                f"Emergency-fund flag: liquid assets (cash + FD) cover "
                f"{months_covered:.1f} months of expenses vs recommended ≥3–6 months. "
                f"Question to ask a SEBI RIA: 'How much liquidity buffer do I need given my situation?'"
            )
            emergency_score = max(0, int(months_covered / 6 * 100))
        flags_all.extend(emergency_flags)

    overall = int(
        allocation_score * 0.20
        + overlap_score * 0.15
        + tax_score * 0.15
        + expense_score * 0.15
        + concentration_score * 0.20
        + emergency_score * 0.15
    )

    return {
        "scores": {
            "allocation": allocation_score,
            "overlap": overlap_score,
            "tax": tax_score,
            "expense": expense_score,
            "concentration": concentration_score,
            "emergency_fund": emergency_score,
            "overall": overall,
        },
        "concentration_flags": concentration_flags,
        "emergency_flags": emergency_flags,
        "flags_all": flags_all,
    }


def generate_report(
    holdings: list[dict[str, Any]],
    allocation_result: dict[str, Any],
    overlap_result: dict[str, Any],
    tax_result: dict[str, Any],
    expense_result: dict[str, Any],
    health_result: dict[str, Any],
    narrative: str,
    out_path: str,
) -> str:
    """Write a markdown report to out_path and return it as a string.

    The report carries the compliance disclaimer on every page.
    It FLAGS issues and EXPLAINS them. It NEVER recommends buying/selling.
    """
    scores = health_result["scores"]

    def score_bar(s: int) -> str:
        filled = s // 10
        return "█" * filled + "░" * (10 - filled) + f" {s}/100"

    # Collect all flags
    all_flags = (
        allocation_result.get("flags", [])
        + overlap_result.get("flags", [])
        + tax_result.get("flags", [])
        + expense_result.get("flags", [])
        + health_result.get("concentration_flags", [])
        + health_result.get("emergency_flags", [])
    )

    total = allocation_result.get("total_portfolio", 0)

    lines = [
        f"# FinScope Portfolio X-Ray Report",
        f"",
        f"> **{DISCLAIMER}**",
        f"",
        f"---",
        f"",
        f"## Health Scores",
        f"",
        f"| Dimension | Score |",
        f"|---|---|",
        f"| Allocation | {score_bar(scores['allocation'])} |",
        f"| Overlap | {score_bar(scores['overlap'])} |",
        f"| Tax Efficiency | {score_bar(scores['tax'])} |",
        f"| Expense Ratio | {score_bar(scores['expense'])} |",
        f"| Concentration | {score_bar(scores['concentration'])} |",
        f"| Emergency Fund | {score_bar(scores['emergency_fund'])} |",
        f"| **Overall** | **{score_bar(scores['overall'])}** |",
        f"",
        f"---",
        f"",
        f"## Portfolio Summary",
        f"",
        f"- Total corpus: **₹{int(total):,}**",
        f"- Holdings: {len(holdings)} instruments",
        f"",
        f"### Allocation breakdown",
        f"",
    ]

    alloc = allocation_result.get("allocation_pct", {})
    for k, v in alloc.items():
        lines.append(f"- {k.capitalize()}: {v}%")

    lines += [
        f"",
        f"---",
        f"",
        f"## Issues Flagged",
        f"",
        f"*This section lists educational observations about your portfolio.*",
        f"*No action is recommended. Discuss each flag with a SEBI-registered RIA.*",
        f"",
    ]

    if all_flags:
        for i, flag in enumerate(all_flags, 1):
            lines.append(f"### Flag {i}")
            lines.append(f"")
            lines.append(flag)
            lines.append(f"")
    else:
        lines.append("No significant issues flagged in this analysis.")
        lines.append("")

    # Expense drag summary
    drag = expense_result.get("total_annual_drag_inr", 0)
    if drag > 0:
        lines += [
            f"### Expense Leakage Summary",
            f"",
            f"Total estimated annual drag from above-median expense ratios: **₹{int(drag):,}/yr**",
            f"",
        ]

    # Tax leakage summary
    ltcg_headroom = tax_result.get("ltcg_headroom", 0)
    if ltcg_headroom > 0:
        lines += [
            f"### LTCG Harvesting Headroom",
            f"",
            f"Estimated unused LTCG exemption this year: **₹{int(ltcg_headroom):,}**",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"## Educational Narrative",
        f"",
        narrative,
        f"",
        f"---",
        f"",
        f"## Questions to Ask a SEBI-Registered RIA",
        f"",
        f"The flags above each end with a specific question. Additional general questions:",
        f"",
        f"1. Does my overall allocation match my actual risk tolerance (not just age)?",
        f"2. Are all my funds genuinely diversified, or do they overlap significantly?",
        f"3. Am I holding debt funds in the most tax-efficient manner for my income slab?",
        f"4. Are there lower-cost alternatives in the same category as my current funds?",
        f"5. Is my emergency fund adequate for my specific income and job situation?",
        f"",
        f"---",
        f"",
        f"*{DISCLAIMER}*",
        f"*Generated by FinScope — an educational portfolio X-ray tool.*",
        f"",
    ]

    report_text = "\n".join(lines)
    Path(out_path).write_text(report_text)
    return report_text
