# FinScope

**An educational portfolio X-ray agent for Indian investors.**
Ingests your holdings and flags fund overlap, allocation drift, tax inefficiency,
expense-ratio leakage, and concentration gaps — in plain English, with questions
to take to a SEBI-registered RIA. Ships with a **reproducible eval suite and a CI gate**.

> Why this market: India has ~200 million mutual-fund investors but only ~967
> SEBI-registered investment advisers. Personalised paid advice is legally gated
> to SEBI RIAs — but a free educational "portfolio X-ray" sits squarely outside
> that definition and serves as the highest-intent top-of-funnel for the
> ₹15L+ earner segment that myfinancial.in targets.

---

## Compliance — read this first

**FinScope is EDUCATIONAL ONLY.**

In India, personalised investment advice for consideration is legally restricted
to SEBI-registered Investment Advisers (Regulation 3, SEBI (IA) Regulations 2013).
FinScope is not a SEBI-registered RIA and does not provide personalised investment
advice. Every report it produces carries the mandatory disclaimer:

> **Educational diagnostic · Not investment advice · Not a SEBI-registered RIA**

The agent **never** uses phrases like "you should buy / sell / switch", "I recommend",
or "invest in". It only **flags issues**, **explains what they mean**, and suggests
**questions to ask a licensed RIA**. This constraint is enforced by a hard compliance
lint in the eval gate — zero violations is a non-negotiable gate.

---

## What the agent does

A transparent **analysis pipeline** (no agent framework — the logic is readable):

1. **Parse** the holdings CSV (`data/sample_portfolio.csv`).
2. **Analyze allocation** — actual vs rule-of-thumb equity band for age + risk.
3. **Detect overlap** — pairwise Jaccard overlap between mutual fund top holdings.
4. **Flag tax leakage** — debt MF held <3yr (STCG), equity <1yr (STCG), unused LTCG headroom.
5. **Flag expense leakage** — funds above category-median expense ratio; ₹ annual drag.
6. **Detect Regular plans** — flags funds in Regular distributor plans vs Direct plans; estimates annual commission drag (~0.5–0.9% of corpus). Detection uses an optional `plan_type` CSV column or 'Regular'/'Direct' keywords in the fund name.
7. **Score health** — per-dimension 0-100 + overall; concentration >20%; emergency-fund gap.
8. **Generate report** — plain-English markdown to `out/report.md` with disclaimer + RIA questions.

All numbers come from deterministic Python analysers. The LLM **only phrases** the narrative.

---

## Eval results — the headline

Run on 8 golden-dataset cases with a CI-style gate (`python -m evals.run`):

| Metric | Score | Gate |
|---|---|---|
| Flag detection recall | **100%** | ≥ 95% |
| Numeric accuracy | **100%** | ≥ 95% |
| Disclaimer presence | **100%** | = 100% |
| Compliance violations | **0** | = 0 |

*Cases cover: high-overlap fund pair, debt STCG flag, concentrated single stock,
allocation drift at age 55, above-median expense ratio, equity STCG, Regular plan
commission flag, and a clean portfolio baseline. Suite exits non-zero on any regression.*

---

## Architecture

```
holdings CSV
     │
     ▼
 parse_holdings()
     │
     ├──► analyze_allocation()    ← age, risk params
     ├──► detect_overlap()        ← fund_holdings.json
     ├──► find_tax_leakage()      ← date math, LTCG rules
     ├──► expense_leakage()       ← category_medians.json
     ├──► detect_regular_plans()  ← plan_type field / fund name keywords
     └──► score_health()          ← per-dimension 0-100
               │
               ▼
          LLM.generate_narrative()
          mock (deterministic) │ OpenAI-compatible
               │
               ▼
          generate_report() ──► out/report.md
          [disclaimer on every page, no recommendations]

Evals: golden_dataset.json ──► analysers ──► scorer ──► PASS/FAIL gate ──► report.json

MCP:   detect_overlap  }
       find_tax_leakage } ──► mcp_server.py (FastMCP, stdio)
       score_health     }    → Claude Desktop, IDE, other agents
```

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Run the eval gate (deterministic, no API key)
python -m evals.run

# 2) Run the agent on the sample portfolio
python -c "
from finscope.agent import run_analysis
r = run_analysis('finscope/data/sample_portfolio.csv', age=35, risk='moderate')
print(f'Overall score: {r.scores[\"overall\"]}/100')
print(f'Report: {r.report_path}')
print(f'Flags: {len(r.all_flags())}')
"
```

---

## Expose the tools over MCP

Three core analysers are published as an **MCP server**, so any
[Model Context Protocol](https://modelcontextprotocol.io) client can call them:

```bash
python -m finscope.mcp_server      # serves over stdio
```

Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "finscope": {
      "command": "/abs/path/.venv/bin/python",
      "args": ["-m", "finscope.mcp_server"],
      "cwd": "/abs/path/to/finscope"
    }
  }
}
```

---

## Using a real LLM

The demo and evals default to a deterministic **mock** provider so they run with no
key. To drive a real model for the narrative, copy `.env.example` to `.env` and set:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # or OpenRouter / local endpoint
OPENAI_MODEL=gpt-4o-mini
```

Then score with a real LLM:

```bash
LLM_PROVIDER=openai python -m evals.run --real
```

---

## What this demonstrates

- A production-style **agentic analysis pipeline**: deterministic analysers + LLM narrative.
- **Compliance engineering** — a hard banned-phrase lint baked into the eval gate.
- **Evals engineering** — golden dataset, regression gate, numeric-accuracy checks, tracked
  failure modes (the most under-supplied skill in 2026 AI hiring).
- Clean provider abstraction (swap mock ↔ real LLM) and a deterministic CI story.
- **Interoperability** — core tools served over MCP, so they plug into Claude Desktop,
  IDEs, or other agents.
- Domain expertise: Indian tax rules (STCG/LTCG), SEBI regulatory context, fund-overlap
  analysis — all encoded in plain, auditable Python.

---

## Repo layout

```
finscope/
  finscope/
    agent.py        # the analysis pipeline loop
    llm.py          # OpenAI-compatible client + deterministic mock
    tools.py        # parse, analyze, detect_overlap, tax, expense, score, report
    mcp_server.py   # three tools served over MCP (stdio)
    data/
      sample_portfolio.csv
      fund_holdings.json
      category_medians.json
  evals/
    golden_dataset.json
    run.py          # scorer + CI gate
  out/              # generated reports (gitignored)
  .env.example
  ADR.md
  requirements.txt
```

See [ADR.md](ADR.md) for the key design decisions.

---

### 🤝 Work with me

I'm an **AI Consultant · Forward Deployed Engineer** — I embed with teams and ship AI to production: agents, MCP integrations, and LLM features, with evals proving they work.

**→ [rohitraj.tech/en/hire](https://rohitraj.tech/en/hire)**
