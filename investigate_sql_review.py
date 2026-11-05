"""Bounded live SQL-review diagnosis on a fixed, synthetic SQLite fixture.

Raw questions, candidate SQL and verdicts stay in ignored .aidebug artifacts.
This script never queries the configured application database.
"""

from collections import Counter
from datetime import date
import json
import os
from pathlib import Path
import sqlite3
import sys

from dotenv import dotenv_values


CASES = [
    ("lookup", "List the first five East orders in 2020 with order ID, date, and sales."),
    ("lookup", "Show the five most recent West orders in 2020 with order ID, date, and sales."),
    ("lookup", "List the first five East orders in 2021 with order ID, date, and sales."),
    ("lookup", "Show the five most recent West orders in 2021 with order ID, date, and sales."),
    ("lookup", "List the first five East orders in 2022 with order ID, date, and sales."),
    ("lookup", "Show the five most recent West orders in 2022 with order ID, date, and sales."),
    ("lookup", "Which East orders were placed in January 2021? Show order ID, date, and sales."),
    ("lookup", "Show order ID, date, and sales for West orders in December 2022."),
    ("forecast_grouped", "Forecast sales by region for the next 2 months."),
    ("forecast_grouped", "Forecast sales by region for the next 3 months."),
    ("forecast_grouped", "Forecast monthly revenue for each region for the next 4 months."),
    ("forecast_grouped", "Predict sales by category for the next 2 months."),
    ("forecast_grouped", "Predict monthly revenue by category for the next 3 months."),
    ("forecast_grouped", "Forecast total sales per region for the next 4 months."),
    ("forecast_grouped", "Project monthly sales for each category for the next 5 months."),
    ("forecast_grouped", "Forecast monthly order counts by region for the next 3 months."),
]


def fixture(path):
    if path.exists():
        return
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, order_date TEXT, region TEXT, category TEXT, total_sales REAL)")
        rows = []
        for index in range(36):
            year, month = 2020 + index // 12, index % 12 + 1
            for region, roffset in (("East", 0), ("West", 40)):
                for category, coffset in (("Furniture", 0), ("Technology", 20)):
                    rows.append((f"{region[0]}-{category[0]}-{year}-{month:02d}",
                                 date(year, month, 15).isoformat(), region, category,
                                 float(100 + 3 * index + 2 * month + roffset + coffset)))
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", rows)


def setup():
    key = os.getenv("OPENAI_API_KEY")
    for source in (".env", ".env.docker"):
        if not key and Path(source).is_file():
            key = dotenv_values(source).get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("No API key available; no requests made.")
    os.environ.update(OPENAI_API_KEY=key, CACHE_TTL_SECONDS="0",
                      DB_ALLOWED_TABLES="orders", DB_ALLOWED_COLUMNS="",
                      ALLOWED_LLM_COLUMNS="", LOG_LEVEL="CRITICAL")
    directory = Path(".aidebug")
    directory.mkdir(exist_ok=True)
    database = directory / "critic_fixture.db"
    fixture(database)
    os.environ["DB_FILE"] = str(database.resolve())
    return directory


def main(phase):
    if phase not in {"before", "after", "strong_after"}:
        raise SystemExit("Pass 'before', 'after', or 'strong_after'.")
    directory = setup()
    import bi
    from evaluation import SQL_CHECKS, parse_verdict

    original = bi.invoke_evaluator
    observations = []

    def diagnose(model, prompt, payload, checks, targets, **kwargs):
        if checks != SQL_CHECKS:
            return original(model, prompt, payload, checks, targets, **kwargs)
        candidate = {"candidate_sql": payload["candidate_sql"],
                     "det_issues": payload["deterministic_pre_check_issues"],
                     "raw_verdicts": []}
        invoke = model.invoke

        def capture(messages):
            response = invoke(messages)
            candidate["raw_verdicts"].append(str(response.content))
            return response

        model.invoke = capture
        try:
            try:
                verdict = original(model, prompt, payload, checks, targets, **kwargs)
                candidate["final_verdict"] = verdict
                return verdict
            except Exception as exc:
                candidate["error_kind"] = type(exc).__name__
                raise
        finally:
            model.invoke = invoke
            states = []
            for raw in candidate["raw_verdicts"]:
                try:
                    parsed = parse_verdict(raw, checks, targets,
                                           approved_schema=payload["approved_schema"])
                    states.append("approved" if parsed["passed"] else "content_rejected")
                except Exception:
                    states.append("invalid_verdict")
            candidate["raw_verdict_states"] = states
            candidate["format_repairs"] = sum(state == "invalid_verdict" for state in states[:-1])
            candidate["audit_repairs"] = sum(state == "content_rejected" for state in states[:-1])
            observations.append(candidate)

    bi.invoke_evaluator = diagnose
    output = directory / f"critic_{phase}.json"
    results = []
    for index, (category, question) in enumerate(CASES, 1):
        if phase == "strong_after" and category != "forecast_grouped":
            continue
        start = len(observations)
        response = bi.ask_bi_agent(question)
        cases = observations[start:]
        record = {"case": index, "category": category, "question": question,
                  "candidates": cases, "sql_evaluation": response.get("sql_evaluation"),
                  "evaluation_history": response.get("evaluation_history"),
                  "final_sql": response.get("sql"),
                  "outcome": response.get("evaluation", {}).get("status"),
                  "metrics": response.get("metrics")}
        results.append(record)
        output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        states = Counter(state for item in cases for state in item["raw_verdict_states"])
        print(f"{phase} {index}/{len(CASES)} {category}: "
              f"{len(cases)} SQL review attempts, {dict(states)}, "
              f"{record['outcome']}", flush=True)
        calls = record["metrics"]["calls"]
        if calls and all(call["failed"] for call in calls):
            raise SystemExit("All model calls failed; stopped live sample.")
    print(f"Saved {len(results)} local diagnostic records to {output}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
