# Architecture Decision Records

## ADR-001 — A framework-light analysis pipeline

**Context.** Agent frameworks (LangChain, CrewAI, AutoGen) add abstraction layers that
hide the control flow. For a single-purpose portfolio analyser — and for a portfolio
piece meant to *show* how an agent works — that indirection is a cost, not a benefit.

**Decision.** Implement the agent as a small, explicit analysis pipeline
([`agent.py`](finscope/agent.py)) that calls each analyser function in sequence over
the standard Python call interface. No framework, no chains, no magic.

**Consequences.**
- A reviewer can read the entire decision path in one short file.
- The pipeline runs anywhere Python runs — no framework version pinning, no framework
  upgrade surprises.
- We own the ordering, error handling, and output shape — no surprises from a framework
  update changing behaviour.
- Trade-off: a multi-agent orchestration scenario (e.g. one agent per user question)
  would need more code, but it's out of scope for a portfolio X-ray tool.

---

## ADR-002 — Deterministic numbers; LLM only for narrative

**Context.** A financial diagnostic tool must be trustworthy and auditable. LLMs are
non-deterministic and can hallucinate numbers. But plain-Python maths is deterministic,
free, and fast. Meanwhile, converting a set of flags into plain-English prose *is*
where a language model adds genuine value.

**Decision.** Split responsibility sharply ([`tools.py`](finscope/tools.py) vs
[`llm.py`](finscope/llm.py)):

- **All numbers, flags, and scores** come from Python analysers that use only stdlib
  maths and JSON seed data. The LLM cannot influence these values.
- **The LLM only phrases the narrative** — the explanation paragraph the user reads
  alongside the structured flag table.
- A deterministic `MockLLM` generates the narrative in the mock path using the same
  computed numbers, making the eval gate reproducible with zero keys/network.

**Consequences.**
- `python -m evals.run` produces the **same flag results every run** → a trustworthy
  regression gate and a diffable `report.json`.
- Demo and evals run with **zero setup** (no key, no network, no cost).
- A real model is one env var away (`LLM_PROVIDER=openai`), and the eval harness
  re-runs with the real narrative for quality scoring.
- Trade-off: the mock validates the *analysers and pipeline*, not the model's prose
  quality. Real-provider runs cover that dimension.

---

## ADR-003 — Educational-only / no-recommendation design and compliance lint

**Context.** In India, personalised investment advice for consideration is legally
restricted to SEBI-registered Investment Advisers (Regulation 3, SEBI (IA)
Regulations 2013). A product that outputs phrases like "you should buy X" or
"I recommend switching to Y" could constitute unlicensed investment advice and
expose the operator to regulatory risk.

The target market — salaried Indians and NRIs earning ₹15L+ — is sophisticated
enough to benefit from structured educational diagnostics, and will engage more
deeply if the product is clearly positioned as a second-opinion tool rather than a
robo-advisor.

**Decision.** Make the no-recommendation constraint structural, not aspirational:

1. **Every report** carries the disclaimer on the first content block and the last line:
   `"Educational diagnostic · Not investment advice · Not a SEBI-registered RIA"`.
2. **Every flag** in the output ends with "Question to ask a SEBI RIA: …" rather than
   a suggested action.
3. **A hard banned-phrase list** (`BANNED_PHRASES` in `tools.py`) is checked by the
   eval harness on every generated output. Zero violations is a non-negotiable gate
   (`"compliance_violations": ("max", 0)`).
4. **The system prompt** for the real-LLM path lists the banned phrases explicitly and
   instructs the model it may only flag, explain, and suggest questions.

**Consequences.**
- The product is clearly positioned as educational → legal risk is minimal and the
  value proposition (free "second opinion") is differentiated from SEBI-gated advice.
- The compliance lint is machine-checkable and CI-enforced → a future contributor
  cannot accidentally ship a recommendation phrase without the gate catching it.
- Users are directed toward a SEBI RIA for personalised guidance → FinScope works
  *with* the regulated ecosystem rather than against it, which is a stronger long-term
  position.
- Trade-off: the output is necessarily less actionable than a personalised advisory
  report. This is a feature, not a bug, for the legal context.
