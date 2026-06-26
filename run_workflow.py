#!/usr/bin/env python3
"""
Customer Success AI Workflow — end-to-end simulation
Run: python run_workflow.py --all [--mock]

Stages:
  1. account_monitoring   - daily batched health scoring
  2. prioritization       - rollup of top-risk/opportunity accounts
  3. inbound_issues       - classify + route + draft response
  4. checkins             - prep + continuity for scheduled check-ins
  5. quality_review       - review junior outputs vs quality_standards
  6. intervention         - design + log a targeted intervention for a declining segment

Each stage calls `call_model()` which either calls the real Anthropic API
(if ANTHROPIC_API_KEY set and --mock not passed) or returns a deterministic
mock response. Every call logs input/output token counts (real, from the
API usage field, or estimated via a simple chars/4 heuristic in mock mode)
to token_log.json, bucketed by stage.
"""
import argparse, csv, json, os, time, math
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6"

# ---------- token logging ----------
token_log = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})

def log_tokens(stage, in_tok, out_tok):
    token_log[stage]["calls"] += 1
    token_log[stage]["input_tokens"] += in_tok
    token_log[stage]["output_tokens"] += out_tok

def estimate_tokens(text):
    return max(1, math.ceil(len(text) / 4))

# ---------- model call wrapper ----------
def call_model(stage, model, system, user_prompt, mock=True, mock_output="OK"):
    if mock or not os.environ.get("ANTHROPIC_API_KEY"):
        in_tok = estimate_tokens(system + user_prompt)
        out_tok = estimate_tokens(mock_output)
        log_tokens(stage, in_tok, out_tok)
        return mock_output, in_tok, out_tok
    else:
        import urllib.request
        body = json.dumps({
            "model": model,
            "max_tokens": 600,
            "system": system,
            "messages": [{"role": "user", "content": user_prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in data.get("content", []))
        usage = data.get("usage", {})
        in_tok = usage.get("input_tokens", estimate_tokens(user_prompt))
        out_tok = usage.get("output_tokens", estimate_tokens(text))
        log_tokens(stage, in_tok, out_tok)
        return text, in_tok, out_tok

# ---------- data loading ----------
def load_csv(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ---------- Stage 1: Account Monitoring ----------
def stage_account_monitoring(accounts, usage_events, mock):
    usage_by_acct = defaultdict(list)
    for u in usage_events:
        usage_by_acct[u["account_id"]].append(u)

    results = []
    batch_size = 10
    for i in range(0, len(accounts), batch_size):
        batch = accounts[i:i + batch_size]
        prompt_rows = []
        for a in batch:
            u = usage_by_acct.get(a["account_id"], [])
            trend = u[-1]["usage_trend"] if u else "unknown"
            prompt_rows.append(
                f"{a['account_id']} health={a['current_health_score']} "
                f"prev={a['previous_health_score']} usage_trend={trend} "
                f"tickets30d={a['support_ticket_count_30d']} nps={a['nps_score']}"
            )
        system = "You are a CS health-scoring assistant. Score risk 0-100 per account and flag the top risk drivers."
        user = "Score these accounts for churn risk and expansion opportunity:\n" + "\n".join(prompt_rows)
        mock_out = json.dumps([
            {"account_id": a["account_id"], "risk_score": 100 - int(a["current_health_score"]),
             "flag": "declining" if int(a["current_health_score"]) < int(a["previous_health_score"]) else "stable"}
            for a in batch
        ])
        text, _, _ = call_model("account_monitoring", MODEL_HAIKU, system, user, mock, mock_out)
        try:
            results.extend(json.loads(text))
        except Exception:
            results.extend(json.loads(mock_out))
    return results

# ---------- Stage 2: Prioritization ----------
def stage_prioritization(scored_accounts, mock):
    top = sorted(scored_accounts, key=lambda r: r["risk_score"], reverse=True)[:5]
    system = "You are a CS portfolio prioritization assistant. Rank accounts needing attention today and explain why."
    user = f"Top scored accounts: {json.dumps(top)}"
    mock_out = json.dumps({"priority_list": [t["account_id"] for t in top],
                            "rationale": "Highest risk scores / declining flags first."})
    text, _, _ = call_model("prioritization", MODEL_SONNET, system, user, mock, mock_out)
    (OUT / "priority_list.json").write_text(text if not mock else mock_out)
    return top

# ---------- Stage 3: Inbound Issues ----------
def stage_inbound_issues(tickets, mock):
    routed = []
    for t in tickets:
        system = "You are a CS issue triage assistant. Classify severity->route as immediate_resolution, scheduled_follow_up, or escalation, and draft a short reply."
        user = f"Ticket: {t['issue_summary']} | severity={t['severity']} | sentiment={t['customer_sentiment']}"
        route = "escalation" if t["severity"] == "High" else ("immediate_resolution" if t["customer_sentiment"] == "frustrated" else "scheduled_follow_up")
        mock_out = json.dumps({"ticket_id": t["ticket_id"], "route": route,
                                "draft_reply": "Thanks for flagging this — we're on it and will update you shortly."})
        text, _, _ = call_model("inbound_issues", MODEL_HAIKU, system, user, mock, mock_out)
        try:
            routed.append(json.loads(text))
        except Exception:
            routed.append(json.loads(mock_out))
    return routed

# ---------- Stage 4: Check-ins ----------
def stage_checkins(checkins, call_notes, mock):
    notes_by_acct = defaultdict(list)
    for n in call_notes:
        notes_by_acct[n["account_id"]].append(n)
    preps = []
    for c in checkins:
        history = notes_by_acct.get(c["account_id"], [])
        system = "You are a CS check-in prep assistant. Summarize prior context and propose an agenda."
        user = f"Checkin {c['checkin_type']} priority={c['priority']} topics={c['topics_to_cover']} history={json.dumps(history)}"
        mock_out = json.dumps({"checkin_id": c["checkin_id"],
                                "agenda": c["topics_to_cover"],
                                "continuity_note": history[-1]["follow_up_items"] if history else "No prior notes."})
        text, _, _ = call_model("checkins", MODEL_SONNET, system, user, mock, mock_out)
        try:
            preps.append(json.loads(text))
        except Exception:
            preps.append(json.loads(mock_out))
    return preps

# ---------- Stage 5: Quality Review ----------
def stage_quality_review(outputs, standards, mock):
    std_ids = {s["standard_id"] for s in standards}
    reviews = []
    for o in outputs:
        covered = set(o["quality_standard_ids"].split(";"))
        missing = std_ids - covered
        system = "You are a CS output quality reviewer. Check draft against required quality standards and flag gaps."
        user = f"Draft: {o['draft_text']} | Required standards: {[s['standard_name'] for s in standards]}"
        mock_out = json.dumps({"output_id": o["output_id"],
                                "pass": len(missing) == 0,
                                "missing_standards": sorted(missing)})
        text, _, _ = call_model("quality_review", MODEL_SONNET, system, user, mock, mock_out)
        try:
            reviews.append(json.loads(text))
        except Exception:
            reviews.append(json.loads(mock_out))
    return reviews

# ---------- Stage 6: Intervention Design ----------
def stage_intervention(scored_accounts, mock):
    declining = [a for a in scored_accounts if a["flag"] == "declining"]
    system = "You are a CS intervention design assistant. Given a declining segment, propose a corrective action plan and a success metric."
    user = f"Declining accounts: {json.dumps(declining)}"
    mock_out = json.dumps({
        "segment_size": len(declining),
        "intervention": "Targeted exec check-in + adoption workshop for accounts with health drop >5pts.",
        "success_metric": "Health score recovery >=5pts within 30 days for 60% of segment."
    })
    text, _, _ = call_model("intervention", MODEL_SONNET, system, user, mock, mock_out)
    (OUT / "intervention_plan.json").write_text(text if not mock else mock_out)
    return text if not mock else mock_out

# ---------- Run + cost reporting ----------
PRICING = {  # $ per million tokens — matches Token Math Sheet Pricing Reference tab
    MODEL_HAIKU: {"in": 1.00, "out": 5.00},
    MODEL_SONNET: {"in": 3.00, "out": 15.00},
}
# NOTE: this demo runs on the small provided synthetic dataset (8-18 accounts/rows per
# file), so per-stage call counts here are far smaller than the production assumptions
# in the Token Math Sheet (which size each stage for the full 750-account portfolio).
# This script supplies the *measured unit cost per call*; the sheet supplies the
# *population-scale* annualization (volume x cadence) built from those measurements.
STAGE_MODEL = {
    "account_monitoring": MODEL_HAIKU, "inbound_issues": MODEL_HAIKU,
    "prioritization": MODEL_SONNET, "checkins": MODEL_SONNET,
    "quality_review": MODEL_SONNET, "intervention": MODEL_SONNET,
}

def cost_report():
    total = 0.0
    report = {}
    for stage, d in token_log.items():
        model = STAGE_MODEL.get(stage, MODEL_HAIKU)
        price = PRICING[model]
        cost = (d["input_tokens"] / 1e6) * price["in"] + (d["output_tokens"] / 1e6) * price["out"]
        total += cost
        report[stage] = {**d, "model": model, "cost_usd": round(cost, 6)}
    report["TOTAL_cost_usd"] = round(total, 6)
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run full workflow")
    ap.add_argument("--mock", action="store_true", help="force mock mode (no API calls)")
    ap.add_argument("--runs", type=int, default=5, help="number of end-to-end demo runs")
    args = ap.parse_args()

    accounts = load_csv("accounts.csv")
    usage_events = load_csv("usage_events.csv")
    tickets = load_csv("support_tickets.csv")
    checkins = load_csv("scheduled_checkins.csv")
    call_notes = load_csv("call_notes.csv")
    outputs = load_csv("junior_outputs.csv")
    standards = load_csv("quality_standards.csv")

    mock = args.mock or not os.environ.get("ANTHROPIC_API_KEY")

    run_costs = []
    for run_i in range(1, args.runs + 1):
        token_log.clear()
        scored = stage_account_monitoring(accounts, usage_events, mock)
        top = stage_prioritization(scored, mock)
        routed_issues = stage_inbound_issues(tickets, mock)
        checkin_preps = stage_checkins(checkins, call_notes, mock)
        quality_results = stage_quality_review(outputs, standards, mock)
        intervention = stage_intervention(scored, mock)

        run_result = {
            "run": run_i,
            "scored_accounts": scored,
            "priority_top5": top,
            "routed_issues": routed_issues,
            "checkin_preps": checkin_preps,
            "quality_results": quality_results,
            "intervention": json.loads(intervention),
            "token_report": cost_report(),
        }
        (OUT / f"run_{run_i}.json").write_text(json.dumps(run_result, indent=2))
        run_costs.append(run_result["token_report"]["TOTAL_cost_usd"])
        print(f"Run {run_i} complete — cost ${run_result['token_report']['TOTAL_cost_usd']:.4f} "
              f"(mock={mock})")

    avg_cost = sum(run_costs) / len(run_costs)
    summary = {
        "runs": args.runs,
        "mock_mode": mock,
        "per_run_cost_usd": run_costs,
        "avg_cost_per_run_usd": round(avg_cost, 6),
        "projected_annual_cost_usd": round(avg_cost * 250, 2),  # 250 biz days, 1 full cycle/day
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
