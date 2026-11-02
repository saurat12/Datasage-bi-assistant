"""Run a bounded live latency sample against a temporary synthetic database."""

from calendar import monthrange
from datetime import date
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from dotenv import dotenv_values


QUESTIONS = {
    "lookup": [
        "List the first five East orders in 2020 with order ID, date, and sales.",
        "Show the five most recent West orders in 2020 with order ID, date, and sales.",
        "List the first five East orders in 2021 with order ID, date, and sales.",
        "Show the five most recent West orders in 2021 with order ID, date, and sales.",
        "List the first five East orders in 2022 with order ID, date, and sales.",
    ],
    "aggregation": [
        "What was total sales by region in 2020?",
        "What was total sales by region in 2021?",
        "What was total sales by region in 2022?",
        "What was monthly revenue in 2021?",
        "What was monthly revenue in 2022?",
    ],
    "forecast_single": [
        "Forecast total sales for the next 2 months.",
        "Forecast total sales for the next 3 months.",
        "Forecast total sales for the next 4 months.",
        "Forecast total sales for the next 5 months.",
        "Forecast total sales for the next 6 months.",
    ],
    "forecast_grouped": [
        "Forecast sales by region for the next 2 months.",
        "Forecast sales by region for the next 3 months.",
        "Forecast sales by region for the next 4 months.",
        "Forecast sales by region for the next 5 months.",
        "Forecast sales by region for the next 6 months.",
    ],
}


def make_fixture(path):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, order_date TEXT, region TEXT, total_sales REAL)")
        rows = []
        for index in range(36):
            year, month = 2020 + index // 12, index % 12 + 1
            for region, offset in (("East", 0), ("West", 40)):
                rows.append((f"{region[0]}-{year}-{month:02d}",
                             date(year, month, min(15, monthrange(year, month)[1])).isoformat(),
                             region, float(100 + index * 3 + offset + month * 2)))
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", rows)


def main():
    key = os.getenv("OPENAI_API_KEY")
    for env_file in (".env", ".env.docker"):
        if not key and Path(env_file).is_file():
            key = dotenv_values(env_file).get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("No API key available; no live requests made.")
    os.environ["OPENAI_API_KEY"] = key
    os.environ["CACHE_TTL_SECONDS"] = "0"
    os.environ["DB_ALLOWED_TABLES"] = "orders"
    os.environ["DB_ALLOWED_COLUMNS"] = ""
    os.environ["ALLOWED_LLM_COLUMNS"] = ""
    os.environ["LOG_LEVEL"] = "WARNING"

    output = Path(".aidebug/latency_sample.json")
    output.parent.mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="datasage_latency_") as folder:
        database = Path(folder) / "synthetic.db"
        make_fixture(database)
        os.environ["DB_FILE"] = str(database)
        from bi import ask_bi_agent

        for category, questions in QUESTIONS.items():
            for number, question in enumerate(questions, 1):
                response = ask_bi_agent(question)
                metrics = response.get("metrics", {})
                results.append({"category": category, "number": number,
                                "metrics": metrics})
                output.write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(f"{category} {number}/5: {metrics.get('latency_seconds')}s, "
                      f"{metrics.get('model_calls')} model calls, "
                      f"{metrics.get('outcome')}", flush=True)
                calls = metrics.get("calls", [])
                if calls and all(call.get("failed") for call in calls):
                    raise SystemExit("All model calls failed; stopped live sample.")
    print(f"Saved {len(results)} sanitized request timelines to {output}")


if __name__ == "__main__":
    main()
