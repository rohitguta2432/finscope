"""LLM providers for FinScope.

Two interchangeable implementations behind one interface:

* ``OpenAILLM`` — real narrative generation via any OpenAI-compatible API.
* ``MockLLM``   — deterministic, key-free stand-in. Returns a plain-English
  educational narrative built entirely from the analyser outputs. No numbers
  are invented; the mock only phrases what the Python tools already computed.

The mock is what makes the eval gate reproducible in CI.
"""
from __future__ import annotations

import os
from typing import Any


class OpenAILLM:
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        from openai import OpenAI  # lazy import — not needed for mock/evals

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_narrative(self, summary: dict[str, Any]) -> str:
        """Ask the real LLM to phrase the educational narrative."""
        prompt = _build_narrative_prompt(summary)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""


class MockLLM:
    """Deterministic narrative builder — no keys, no network."""

    def generate_narrative(self, summary: dict[str, Any]) -> str:
        return _mock_narrative(summary)


def _mock_narrative(summary: dict[str, Any]) -> str:
    """Build a plain-English educational narrative from tool outputs."""
    scores = summary.get("scores", {})
    overall = scores.get("overall", 0)
    total = summary.get("total_portfolio", 0)
    n_flags = summary.get("total_flags", 0)
    overlap_pairs = summary.get("flagged_overlap_pairs", 0)
    drag = summary.get("annual_drag_inr", 0)
    drift = summary.get("drift_pp", 0)

    health_label = (
        "strong" if overall >= 80
        else "reasonable" if overall >= 60
        else "worth reviewing with a professional"
    )

    parts = [
        f"Your portfolio of ₹{int(total):,} scores {overall}/100 overall — a {health_label} result "
        f"based on the dimensions analysed.",
    ]

    if abs(drift) > 0:
        direction = "above" if drift > 0 else "below"
        parts.append(
            f"The allocation analysis found your equity exposure is {abs(drift):.1f} percentage points "
            f"{direction} the rule-of-thumb band for your age and risk profile. "
            f"Whether this gap is appropriate depends on your specific goals and situation — "
            f"a SEBI RIA can give you a personalised view."
        )

    if overlap_pairs > 0:
        parts.append(
            f"{overlap_pairs} fund pair(s) show significant overlap in their top holdings. "
            f"This means your effective diversification may be lower than the number of funds suggests. "
            f"It is worth asking whether each fund adds genuinely different exposure."
        )

    if drag > 0:
        parts.append(
            f"Above-median expense ratios are costing an estimated ₹{int(drag):,} per year in additional fees. "
            f"Over long periods, expense ratios compound — small differences in cost can have a meaningful "
            f"impact on final corpus. Ask your RIA whether lower-cost options in the same categories exist."
        )

    if n_flags > 0:
        parts.append(
            f"In total, {n_flags} issue(s) were flagged across allocation, overlap, tax, expense, "
            f"and concentration dimensions. Each flag in the report comes with a specific question "
            f"you can bring to a SEBI-registered RIA."
        )

    parts.append(
        "This X-ray is a starting point for a conversation, not a conclusion. "
        "A licensed RIA will consider your complete financial picture — income, liabilities, "
        "tax situation, and goals — before giving personalised guidance."
    )

    return "\n\n".join(parts)


_SYSTEM_PROMPT = """\
You are FinScope, an educational portfolio analysis tool for Indian investors.
You produce plain-English explanations of portfolio analysis results.

STRICT COMPLIANCE RULES — you MUST follow these without exception:
1. NEVER use the phrases: "you should buy", "you should sell", "you should switch",
   "I recommend", "we recommend", "I suggest buying/selling/switching", "invest in",
   "purchase", "recommend buying/selling", "buy this", "sell this", "switch to".
2. You may only FLAG issues, EXPLAIN what they mean, and suggest QUESTIONS to ask
   a SEBI-registered RIA.
3. Every response must treat the user as an educated adult seeking to understand
   their portfolio, not someone seeking investment advice.
4. Always note that the analysis is educational and not personalised advice.
"""


def _build_narrative_prompt(summary: dict[str, Any]) -> str:
    return (
        f"Write a plain-English educational narrative for a portfolio X-ray with these findings:\n"
        f"Overall score: {summary.get('scores', {}).get('overall', 0)}/100\n"
        f"Total corpus: ₹{int(summary.get('total_portfolio', 0)):,}\n"
        f"Allocation drift: {summary.get('drift_pp', 0):.1f} pp\n"
        f"Flagged overlap pairs: {summary.get('flagged_overlap_pairs', 0)}\n"
        f"Annual expense drag: ₹{int(summary.get('annual_drag_inr', 0)):,}\n"
        f"Total flags raised: {summary.get('total_flags', 0)}\n\n"
        f"Explain what these numbers mean in plain language. Do NOT recommend any action. "
        f"Only flag, explain, and suggest questions for a SEBI RIA."
    )


def get_llm() -> Any:
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "openai":
        return OpenAILLM(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    return MockLLM()
