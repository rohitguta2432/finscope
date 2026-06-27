"""FinScope as an MCP server.

Exposes three core analysers over the Model Context Protocol so any MCP
client (Claude Desktop, an IDE, another agent) can call them directly.
The tools share no mutable state — each call is stateless and reproducible.

Run over stdio:

    python -m finscope.mcp_server

Claude Desktop config (claude_desktop_config.json):

    {
      "mcpServers": {
        "finscope": {
          "command": "/abs/path/.venv/bin/python",
          "args": ["-m", "finscope.mcp_server"],
          "cwd": "/abs/path/to/finscope"
        }
      }
    }
"""
from __future__ import annotations

from datetime import date
from typing import Any


def main() -> None:
    from mcp.server.fastmcp import FastMCP  # lazy import — not needed for evals

    from .tools import detect_overlap, detect_regular_plans, find_tax_leakage, parse_holdings, score_health
    from .tools import analyze_allocation, expense_leakage

    mcp = FastMCP("finscope")

    @mcp.tool()
    def mcp_detect_overlap(csv_path: str) -> dict[str, Any]:
        """Detect underlying-stock overlap between mutual funds in a holdings CSV.

        Returns pairwise Jaccard overlap % and flags pairs above 40%.
        Educational output only — not investment advice.
        """
        holdings = parse_holdings(csv_path)
        return detect_overlap(holdings)

    @mcp.tool()
    def mcp_find_tax_leakage(csv_path: str) -> dict[str, Any]:
        """Flag potential tax-inefficiency in a holdings CSV.

        Checks: debt MF held <3yr (STCG at slab), equity held <1yr (STCG 15%),
        unused LTCG exemption headroom. Educational flags only — not tax advice.
        """
        holdings = parse_holdings(csv_path)
        return find_tax_leakage(holdings, today=date.today())

    @mcp.tool()
    def mcp_detect_regular_plans(csv_path: str) -> dict[str, Any]:
        """Detect Regular plan mutual fund holdings in a holdings CSV.

        Regular plans embed a distributor commission (~0.5–1.5% above Direct plans).
        Returns per-fund plan type, estimated annual commission drag in ₹, and flags.
        Detection uses an explicit plan_type column (values: 'regular'|'direct') or
        'Regular'/'Direct' keywords in the fund name as a fallback.
        Educational output only — not investment advice.
        """
        holdings = parse_holdings(csv_path)
        return detect_regular_plans(holdings)

    @mcp.tool()
    def mcp_score_health(
        csv_path: str,
        age: int = 35,
        risk: str = "moderate",
        monthly_expenses: float = 0.0,
    ) -> dict[str, Any]:
        """Compute per-dimension health scores (0-100) for a holdings CSV.

        Dimensions: allocation, overlap, tax, expense, concentration,
        emergency_fund, overall. Educational only — not investment advice.
        """
        holdings = parse_holdings(csv_path)
        allocation = analyze_allocation(holdings, age=age, risk=risk)
        overlap = detect_overlap(holdings)
        tax = find_tax_leakage(holdings, today=date.today())
        expense = expense_leakage(holdings)
        me = monthly_expenses if monthly_expenses > 0 else None
        return score_health(holdings, allocation, overlap, tax, expense, monthly_expenses=me)

    mcp.run()


if __name__ == "__main__":
    main()
