# Customer Success AI Workflow — Simulation

End-to-end simulation of an AI system handling daily account monitoring,
prioritization, inbound issue routing, customer check-in support, output
quality review, and targeted intervention design for a ~750-account B2B
SaaS customer success portfolio.

## Setup

```bash
cd cs_workflow
python3 -m venv venv && source venv/bin/activate   # optional
pip install --break-system-packages -r requirements.txt  # none required beyond stdlib
```

No external packages are required (uses only `csv`, `json`, `urllib`, `argparse`
from the standard library).

To use **real Claude API calls** (instead of deterministic mock responses),
set your key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run (single entry point)

```bash
python3 run_workflow.py --all --runs 5
```

- `--mock` forces mock mode even if `ANTHROPIC_API_KEY` is set (useful for
  fast, free demo runs).
- `--runs N` controls how many full end-to-end demo runs are executed
  (default 5, satisfying the "at least 5 representative runs" requirement).

Each run executes all six stages against the full provided dataset in
`/data` and writes:

- `outputs/run_<n>.json` — full structured output of that run (scored
  accounts, priority list, routed issues, check-in preps, quality review
  results, intervention plan, and per-stage token/cost report)
- `outputs/priority_list.json`, `outputs/intervention_plan.json` — latest
  versions of those artifacts
- `outputs/summary.json` — per-run cost, average cost/run, and a naive
  annualized cost projection (avg_cost_per_run × 250 business days)

## Workflow stages

| Stage | Model tier | What it does |
|---|---|---|
| `account_monitoring` | Haiku | Batched daily health/risk scoring across all accounts (10/batch) |
| `prioritization` | Sonnet | Ranks top-risk accounts from scored output, with rationale |
| `inbound_issues` | Haiku | Classifies + routes each support ticket, drafts a short reply |
| `checkins` | Sonnet | Builds agenda + continuity notes for scheduled check-ins using prior call notes |
| `quality_review` | Sonnet | Checks junior-staff drafts against `quality_standards.csv`, flags missing standards |
| `intervention` | Sonnet | Designs a corrective action + success metric for the declining account segment |

## Data

All six CSVs from the provided synthetic dataset live in `/data` and are
read directly — no transformation needed:
`accounts.csv`, `usage_events.csv`, `support_tickets.csv`,
`scheduled_checkins.csv`, `call_notes.csv`, `junior_outputs.csv`,
`quality_standards.csv`.

## Token & cost measurement

Every model call (real or mocked) is logged with input/output token counts
per stage in `token_log` (in-memory) and surfaced per run in
`run_<n>.json -> token_report`. Pricing constants in `PRICING` at the top
of `run_workflow.py` should be replaced with the exact per-model rates from
the Token Math Sheet template before finalizing cost numbers.

## Notes / limitations

- This is a simulation: mock-mode responses are deterministic stand-ins for
  real model output, used so the workflow is runnable and reproducible
  without API spend. Switch off `--mock` with a real key to get measured
  real-world token counts for the Token Math Sheet.
- Quality-review and intervention logic is intentionally simple
  (rule-assisted + LLM call) to stay within the time budget — flagged in
  the session log as an area to harden further (e.g., structured scoring
  rubric, confidence thresholds for auto-escalation).
