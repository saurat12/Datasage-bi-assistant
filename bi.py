import os
import re
import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Optional, TypedDict, List, Dict, Any, Tuple
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from openai import OpenAI

from telemetry import TrackedModel, tracked_call, measure_request, mark_cache_hit
from config import get_settings
from sql_safety import looks_like_future_filter, split_statements, validate_read_only_sql
from database_gateway import DatabaseGateway
from query_planning import forecast_history_sql
from evaluation import (
    SQL_CHECKS, ANSWER_CHECKS, SQL_EVALUATOR_PROMPT, ANSWER_EVALUATOR_PROMPT,
    STRUCTURED_SUMMARY_PROMPT, parse_verdict, failed_verdict, build_evidence,
    parse_statements, check_statements, apply_answer_applicability, sql_review_rules, invoke_evaluator,
)

from forecasting import (
    ForecastResult,
    detect_forecast_intent,
    extract_horizon,
    extract_target_window,
    forecast_series,
)
from accuracy import AccuracyMetrics, quick_accuracy

load_dotenv()
settings = get_settings()
DB_FILE = settings.database_file
OPENAI_API_KEY = settings.openai_api_key.get_secret_value()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s",
)
_request_id: ContextVar[str] = ContextVar("request_id", default="startup")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


logger = logging.getLogger("datasage")
for handler in logging.getLogger().handlers:
    handler.addFilter(_RequestIdFilter())


database_gateway = DatabaseGateway(settings)
APPROVED_SCHEMA = database_gateway.schema_description()

# Separate model handles per agent role. Same underlying model here, but this
# is where you'd swap in a cheaper/faster model for the critic, or a more
# careful one for the summary, without touching the SQL agent.
_llm_options = dict(
    model=settings.openai_model,
    temperature=0,
    api_key=OPENAI_API_KEY,
    timeout=settings.request_timeout_seconds,
    max_retries=settings.openai_max_retries,
)
sql_llm = TrackedModel(ChatOpenAI(**_llm_options), "sql", settings.openai_model)
critic_llm = TrackedModel(ChatOpenAI(**_llm_options), "critic", settings.openai_model)
summary_llm = TrackedModel(ChatOpenAI(**_llm_options), "summary", settings.openai_model)
orchestrator_llm = TrackedModel(ChatOpenAI(**_llm_options), "orchestrator", settings.openai_model)
forecast_llm = TrackedModel(ChatOpenAI(**_llm_options), "forecast", settings.openai_model)
_viz_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=settings.request_timeout_seconds,
    max_retries=settings.openai_max_retries,
)

MAX_RETRIES = settings.max_sql_retries
MAX_FORECAST_RETRIES = settings.max_forecast_retries

# ---------------------------------------------------------------------------
# SQL Agent prompts — generate SQL without database tools.
# No summary responsibility (that moved to the Summary Agent).
# ---------------------------------------------------------------------------

BASE_SQL_SYSTEM_PROMPT = """
You are a senior SQL engineer working with a SQL database.

Your responsibilities:
1. Convert the user question into a correct SQL query
2. Return SQL for review and execution by a separate controlled executor
3. Never claim to have executed SQL; you have no database tools

STRICT RULES:
- Do NOT perform any DML (INSERT, UPDATE, DELETE, DROP, ALTER)
- Only generate SELECT queries
- Use only the available tables
- Always limit results to 100 rows unless aggregation is used
- If the question is ambiguous, make reasonable assumptions

COLUMN RULES:
- Match the metric: revenue uses SUM(total_sales), order counts use
  COUNT(DISTINCT order_id) when order_id exists (orders may have multiple line items).
  COUNT(*) measures rows, not necessarily distinct orders. Quantity means SUM(quantity).
- "monthly orders in <year>" means one order count per month in that year.
  Prefer a direct monthly GROUP BY rather than unnecessary nested aggregations.
- Always use SUM(total_sales) when the user asks for total sales, revenue, or sales figures
- Never use SUM(sales) or SUM(total_sale) — the correct column is total_sales
- Always use strftime('%Y-%m', order_date) for monthly aggregations
- Always use strftime('%Y', order_date) for yearly aggregations
- For any query involving time (monthly, yearly, trends, growth), always alias the date column as "period"

EXAMPLES:
Q: Region wise total sales
A: SELECT region,
          SUM(total_sales) AS total_sales
   FROM orders
   GROUP BY region
   ORDER BY total_sales DESC
   LIMIT 100;

Q: Top 5 products by sales
A: SELECT product_name, SUM(total_sales) AS total_sales FROM orders GROUP BY product_name ORDER BY total_sales DESC LIMIT 5;

Q: Year over year growth rate
A: WITH yearly_sales AS (
       SELECT strftime('%Y', order_date) AS year, SUM(total_sales) AS revenue
       FROM orders
       GROUP BY year
   )
   SELECT
       year,
       revenue,
       LAG(revenue) OVER (ORDER BY year) AS prev_year_revenue,
       ROUND(
           (revenue - LAG(revenue) OVER (ORDER BY year))
           / NULLIF(LAG(revenue) OVER (ORDER BY year), 0) * 100,
           2
       ) AS yoy_growth_percent
   FROM yearly_sales
   ORDER BY year;

If you receive a "REVISION NEEDED" note in the input, a reviewer has already
rejected your previous query. Read the stated reason carefully and fix
exactly that issue while still answering the original question.

OUTPUT FORMAT (always follow this exactly, nothing else):

SQL Query:
<query>
"""

FORECAST_SQL_ADDENDUM = """

FORECAST MODE — READ CAREFULLY:

The user is asking for a forecast, but YOU DO NOT FORECAST IN SQL.
A separate Python forecasting model (Holt-Winters / ARIMA) will generate the
future values. Your ONLY job is to return the FULL HISTORICAL TIME SERIES that
the forecasting model will train on.

CRITICAL RULES:
1. NEVER filter by future dates. NEVER use `WHERE order_date > ...`,
   `WHERE order_date >= date('now')`, `WHERE strftime(...) >= '...'`, or any
   condition that restricts to upcoming/recent periods. The data ends at the
   most recent order in the table — anything beyond that is what the model
   predicts, not what SQL returns.
2. Return ALL historical periods, oldest to newest. No date filter at all,
   unless the user explicitly asked to forecast a specific slice.
3. Phrases like "next 3 months", "next quarter", "for the upcoming year"
   refer to the FORECAST HORIZON — they are handled by the Python model. They
   are NOT a hint to filter the SQL. Ignore them when writing the WHERE clause.
4. Always GROUP BY period and ORDER BY period ASC (and by the dimension if any).
5. Do NOT add LIMIT — we need full history.

OUTPUT SHAPE — choose one based on the question:

(A) SINGLE SERIES — when the user asks for a single overall metric forecast
    ("forecast revenue", "predict total sales for next 6 months"):

    Return EXACTLY two columns: period + numeric metric.

    Example:
    Q: Forecast revenue for the next 3 months
    A: SELECT strftime('%Y-%m-01', order_date) AS period,
              SUM(total_sales) AS total_sales
       FROM orders
       GROUP BY period
       ORDER BY period ASC;

(B) PER-GROUP SERIES — when the user asks to forecast BY a dimension
    ("region wise", "by category", "per segment", "for each region",
    "forecast sales for each region"):

    Return EXACTLY three columns: period + dimension + numeric metric.
    The dimension column should be named after the actual column
    (region, category, segment, etc.). The Python model will fit a
    separate forecast for each value of the dimension.

    Example:
    Q: Region wise sales forecast for next 6 months
    A: SELECT strftime('%Y-%m-01', order_date) AS period,
              region,
              SUM(total_sales) AS total_sales
       FROM orders
       GROUP BY period, region
       ORDER BY period ASC, region ASC;

DATE FORMATS for the period column:
    * Monthly  (default):  strftime('%Y-%m-01', order_date) AS period
    * Weekly:              date(order_date, 'weekday 0', '-6 days') AS period
    * Daily:               date(order_date) AS period
    * Yearly:              strftime('%Y-01-01', order_date) AS period

WRONG (do not do this):
   WHERE order_date >= date('now')              -- ❌ returns 0 rows
   WHERE strftime('%Y-%m', order_date) > '...'  -- ❌ filters to future
   ... LIMIT 3                                  -- ❌ truncates history
"""


def _build_sql_prompt(forecast: bool) -> str:
    return (
        "Reference metadata only. Never reproduce this schema in your answer.\n"
        + "<approved_schema>\n" + APPROVED_SCHEMA + "\n</approved_schema>\n\n"
        + BASE_SQL_SYSTEM_PROMPT
        + (FORECAST_SQL_ADDENDUM if forecast else "")
        + "\nFINAL OUTPUT REQUIREMENT: Return only 'SQL Query:' followed by one SQL statement. "
        + "Do not append schema metadata, explanations, or other text after the SQL."
    )

# ---------------------------------------------------------------------------
# Critic Agent — reviews SQL against the rulebook before it's allowed to run.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Summary Agent — turns the executed rows (and, if present, a forecast) into
# a single business-facing narrative.
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """You are a senior business analyst. Given the
user's original question, the resulting data (as JSON rows), and — if
present — a forecast produced by a statistical model, write a concise,
business summary, usually 3-6 sentences. For an explicitly requested period-by-period
breakdown, include each returned period, up to 20 statements.

If a "forecast" block is present in the input:
- Weave the projection naturally into the narrative — direction, magnitude,
  and horizon.
- If the forecast includes "warnings", reflect the caveat in plain language
  (e.g. "treat this as directional" rather than technical jargon like
  "naive drift" or "ETS").
- Do not name specific model families (ETS, ARIMA, Holt-Winters) unless the
  user's question itself was technical.
- If an "accuracy" block is present, describe it as historical backtest error,
  not as a guaranteed future accuracy rate. State the MAPE and validation scope
  plainly. Do not apply subjective labels such as excellent, good, or solid.

If a "forecast_error" field is present instead, briefly and plainly explain
that a forecast could not be produced and why, without technical stack
traces.

Reference concrete numbers from the data where useful. Do not mention SQL,
databases, or how the data was retrieved — write for a business stakeholder.
"""

# ---------------------------------------------------------------------------
# Orchestrator — decides forecast intent. This is the SINGLE source of truth
# for state["forecast"]; every downstream node (sql_agent, critic, the
# executor->forecast_agent branch) reads that flag rather than re-deciding.
# ---------------------------------------------------------------------------

ORCHESTRATOR_INTENT_PROMPT = """Does this business question require a
future projection/forecast (something that hasn't happened yet), or can it
be fully answered from historical data alone?

Reply with only the single word "true" or "false"."""

_AMBIGUOUS_FUTURE_HINTS = re.compile(
    r"\bwill\b|\bgoing to\b|\bexpect(?:ed)?\b|\bshould we\b|\bare we on track\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Forecast Agent — thin LLM decision/validation layer wrapped around the
# deterministic forecasting.py engine. The statistics (ETS/ARIMA/drift,
# confidence intervals, chart building) are untouched; this layer only adds
# judgment: horizon interpretation fallback, retry-on-failure, and
# self-validation of the output before handing it to the Summary Agent.
# ---------------------------------------------------------------------------

FORECAST_HORIZON_SYSTEM_PROMPT = """You interpret time horizons in business
questions for a forecasting system. Given a question, return JSON only,
nothing else:

{"mode": "relative" or "default",
 "periods": <int, only if mode is "relative">,
 "unit": "<days|weeks|months|quarters|years>, only if mode is relative",
 "reasoning": "<one sentence>"}

Use "default" if the question doesn't specify a clear horizon (assume 12
periods). Interpret vague phrasing like "the rest of the year" or "through
year end" as your best-effort relative period count.
"""


def _llm_interpret_horizon(question: str) -> dict:
    """LLM fallback for horizon phrasing the regex-based extractor in
    forecasting.py doesn't catch (e.g. 'through end of fiscal year',
    'the rest of this quarter'). Only called when the deterministic parse
    in forecast_agent_node found nothing and a previous attempt failed."""
    try:
        resp = forecast_llm.invoke([
            ("system", FORECAST_HORIZON_SYSTEM_PROMPT),
            ("human", question),
        ])
        raw = re.sub(r"```json|```", "", resp.content.strip()).strip()
        return json.loads(raw)
    except Exception as e:
        logger.exception("Forecast horizon interpretation failed")
        return {"mode": "default"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_sql(output: str) -> Optional[str]:
    """Extract the SQL query from the SQL agent's output."""
    if "SQL Query:" in output:
        try:
            sql = output.split("SQL Query:")[1]
            if "Summary:" in sql:
                sql = sql.split("Summary:")[0]
            sql = sql.strip()
            sql = re.sub(r"```sql|```", "", sql).strip()
            return sql
        except IndexError:
            pass
    return None


def _downsample_for_chart(
    data: List[Dict[str, Any]], max_rows: int = settings.chart_sample_rows
) -> tuple[List[Dict[str, Any]], bool]:
    """Forecast queries return full, unfiltered history (by design, no
    LIMIT). Grouped forecasts especially can produce thousands of rows,
    which risks an oversized/expensive prompt or a flaky API call. Evenly
    sample down to max_rows, preserving order, rather than truncating the
    most recent data off the end."""
    if len(data) <= max_rows:
        return data, False
    step = len(data) / max_rows
    indices = sorted({int(i * step) for i in range(max_rows)})
    return [data[i] for i in indices], True


def _safe_rows_for_llm(rows: List[Dict[str, Any]], max_rows: int) -> List[Dict[str, Any]]:
    """Remove non-allowlisted fields before records leave the application."""
    allowed = settings.llm_column_allowlist
    selected = rows[:max_rows]
    if not allowed:
        return selected
    return [{key: value for key, value in row.items() if key in allowed} for row in selected]


def generate_chart_spec(
    data: List[Dict[str, Any]],
    question: Optional[str] = None,
    _retry: bool = False,
) -> tuple[Optional[dict], Optional[str]]:
    """Ask OpenAI to produce a Vega-Lite spec for the data, choosing chart
    type primarily from the user's question intent (trend vs. comparison
    vs. share-of-whole vs. relationship), falling back to data-shape rules
    when the question doesn't clearly signal intent.

    Includes one self-repair retry: if the first response isn't valid JSON
    or is missing required Vega-Lite keys, we ask again with a stricter note
    instead of silently giving up.

    Returns (spec, note). note is None on success, otherwise a human-readable
    reason there's no chart — surfaced back to the caller instead of just
    disappearing into a None.
    """
    if not data:
        logger.info("Chart skipped because result set is empty")
        return None, "No rows were returned, so there's nothing to chart."

    chart_data, was_downsampled = _downsample_for_chart(
        _safe_rows_for_llm(data, settings.chart_sample_rows)
    )
    columns = list(chart_data[0].keys())
    strict_note = ""
    if _retry:
        strict_note = (
            "\nSTRICT: your previous response was not valid, parseable JSON "
            "with the required keys. Return ONLY a single JSON object with "
            "top-level 'data', 'mark', and 'encoding' keys. No prose, no "
            "markdown fences."
        )

    question_block = ""
    if question:
        question_block = f"""
User's original question: "{question}"

CHART TYPE — decide from the QUESTION'S INTENT first, data shape second:
- Trend / "over time" / growth / change / forecast / monthly / yearly →
  line mark, period on x-axis.
- Comparison across named items ("compare", "vs", "by region", "top N",
  "which product/category/segment") → bar mark, ranked/sorted if it's a
  "top N" style question.
- Share of a whole / "percentage of", "proportion", "breakdown of total",
  "what fraction" → arc (pie) mark, but ONLY if there are roughly 7 or
  fewer categories — a pie with more slices than that is unreadable, so
  fall back to a bar mark instead and mention nothing of this reasoning
  in the output.
- Relationship / correlation between two numeric measures ("relationship
  between X and Y", "does X affect Y") → point (scatter) mark.
- Distribution / spread of a single measure ("distribution of", "how are
  X spread out") → bar mark on binned/grouped values.
- If the question doesn't clearly signal intent, fall back to the data
  shape rules below.

The question's intent wins even if the data could technically support a
different chart type — e.g. a two-column (category, value) result for
"compare sales by region" should be a bar chart, not a pie chart, unless
the question explicitly asks for a share/proportion framing.
"""

    full_data_prompt = f"""You are a data visualization expert.
Given this data (columns: {columns}), return ONLY a valid Vega-Lite v5 JSON spec.
No explanation, no markdown fences — pure JSON only.{strict_note}
{question_block}
Full row count: {len(chart_data)}{"" if not was_downsampled else f" (downsampled from {len(data)} for chart readability)"}

Rules:
- Use "period", "month", "year", "date" columns as x-axis for time series
- For the x-axis encoding, always use type "ordinal" never "temporal"
- MULTI-SERIES DATA: if there are exactly three relevant columns —
  a period/date column, a categorical dimension (e.g. region, category,
  segment), and a numeric metric — this is grouped time-series data. Use a
  line mark and add a "color" encoding on the categorical dimension so each
  group renders as its own line, e.g.
  "color": {{"field": "region", "type": "nominal"}}. Do NOT collapse the
  groups into one line.
- Always set width to 400 and height to 250
- Include a descriptive title at the top level as "title" field
- Embed the full dataset inline using the "data": {{"values": [...]}} key
- Always include "$schema": "https://vega.github.io/schema/vega-lite/v5.json"

Full data: {json.dumps(chart_data)}"""

    try:
        response = tracked_call("chart", "gpt-4o-mini", _viz_client.chat.completions.create,
            model="gpt-4o-mini",
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": full_data_prompt}]
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        spec = json.loads(raw)

        required_keys = {"data", "mark", "encoding"}
        missing = required_keys - set(spec.keys())
        if missing:
            raise ValueError(f"spec missing required keys: {missing}")

        logger.info("Visualization generated keys=%s", list(spec.keys()))
        success_note = (
            f"Chart shows a sample of {len(chart_data)} of {len(data)} rows "
            "for readability."
        ) if was_downsampled else None
        return spec, success_note

    except Exception as e:
        logger.warning("Visualization generation failed: %s", e)
        if not _retry:
            logger.info("Retrying visualization generation")
            return generate_chart_spec(data, question=question, _retry=True)
        return None, f"Chart generation failed after retry: {e}"


def _looks_like_future_filter(sql: str) -> bool:
    return looks_like_future_filter(sql)


_GROUP_KEYWORDS: List[tuple] = [
    (r"\bregion(?:s|\s*wise|\s*-?wise|\s+by\s+region)?\b|\bby\s+region\b|\bper\s+region\b|\beach\s+region\b", "region"),
    (r"\bcategor(?:y|ies)\b|\bby\s+category\b|\bper\s+category\b", "category"),
    (r"\bsub[-\s]?categor(?:y|ies)\b", "sub_category"),
    (r"\bsegment(?:s)?\b|\bby\s+segment\b|\bper\s+segment\b", "segment"),
    (r"\bstate(?:s)?\b|\bby\s+state\b|\bper\s+state\b", "state"),
    (r"\bcit(?:y|ies)\b|\bby\s+city\b|\bper\s+city\b", "city"),
    (r"\bship[-\s]?mode(?:s)?\b", "ship_mode"),
    (r"\bcustomer(?:s)?\b", "customer_id"),
]


def _detect_group_dimension(question: str) -> Optional[str]:
    if not question:
        return None
    q = question.lower()
    for pattern, col in _GROUP_KEYWORDS:
        if re.search(pattern, q):
            return col
    return None


def _build_forecast_sql(group_col: Optional[str]) -> str:
    """Canonical historical query — the deterministic backstop used only
    after the SQL/Critic agent loop has exhausted its retries."""
    if group_col:
        return (
            f"SELECT strftime('%Y-%m-01', order_date) AS period, "
            f"{group_col}, SUM(total_sales) AS total_sales "
            f"FROM orders "
            f"GROUP BY period, {group_col} "
            f"ORDER BY period ASC, {group_col} ASC;"
        )
    return (
        "SELECT strftime('%Y-%m-01', order_date) AS period, "
        "SUM(total_sales) AS total_sales "
        "FROM orders "
        "GROUP BY period "
        "ORDER BY period ASC;"
    )


def _split_statements(sql: str) -> List[str]:
    return split_statements(sql)


def _sanitize_single_statement(sql: Optional[str]) -> Optional[str]:
    """Reject multiple statements before passing SQL to the gateway."""
    if not sql:
        return sql
    statements = _split_statements(sql)
    if not statements:
        return None
    if len(statements) == 1:
        return statements[0].rstrip(";").strip() + ";"

    raise ValueError("Exactly one SQL statement is allowed.")


def _run_sql(sql: str, forecast: bool = False) -> List[Dict[str, Any]]:
    return database_gateway.execute(sql, forecast=forecast)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class BIState(TypedDict, total=False):
    question: str
    forecast: bool                       # SINGLE source of truth for intent
    group_col: Optional[str]
    sql: Optional[str]
    critic_approved: bool
    critic_reason: str
    sql_check_issues: list
    review_fallback: bool
    sql_plan_verified: bool
    fix_instructions: str
    retry_count: int
    sql_evaluation: dict
    evaluation: dict
    evaluation_history: list
    evaluation_retry_count: int
    evaluation_feedback: str
    evidence: dict
    summary_statements: list
    summary_error: Optional[str]
    rows: Optional[List[Dict[str, Any]]]
    chart_spec: Optional[dict]
    chart_note: Optional[str]
    summary: str
    error: Optional[str]
    # Forecast agent state
    forecast_result: Optional[ForecastResult]
    forecast_chart_spec: Optional[dict]
    forecast_df: Optional[pd.DataFrame]
    forecast_error: Optional[str]
    forecast_retry_count: int


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------

def orchestrator_node(state: BIState) -> BIState:
    """SINGLE place that decides state['forecast']. Every other node
    (sql_agent's prompt choice, critic's validation rules, and the
    executor -> forecast_agent branch) reads this flag rather than
    re-deciding intent independently."""
    question = state["question"]
    forecast = detect_forecast_intent(question)
    logger.info("Forecast keyword detection result=%s", forecast)
    if not forecast and _AMBIGUOUS_FUTURE_HINTS.search(question or ""):
        try:
            resp = orchestrator_llm.invoke([
                ("system", ORCHESTRATOR_INTENT_PROMPT),
                ("human", question),
            ])
            forecast = resp.content.strip().lower().startswith("true")
        except Exception:
            logger.exception("Forecast intent tiebreak failed")
            forecast = False

    group_col = _detect_group_dimension(question) if forecast else None
    logger.info("Forecast routing result=%s group=%s", forecast, group_col)
    return {**state, "forecast": forecast, "group_col": group_col, "retry_count": 0}


def sql_agent_node(state: BIState) -> BIState:
    question = state["question"]
    forecast = state["forecast"]
    retry_count = state.get("retry_count", 0)
    feedback = state.get("critic_reason")
    fix = state.get("fix_instructions")
    if forecast:
        planned_sql = forecast_history_sql(question, json.loads(APPROVED_SCHEMA))
        if planned_sql:
            return {**state, "sql": planned_sql, "sql_plan_verified": True, "error": None}

    agent_input = question
    if retry_count > 0 and feedback:
        agent_input = (
            f"{question}\n\n"
            f"REVISION NEEDED: A reviewer rejected your previous SQL. "
            f"Reason: \"{feedback}\". Required fix: \"{fix}\". "
            f"Regenerate a corrected SQL query that fixes this issue while "
            f"still answering the original question."
        )

    logger.info("SQL agent attempt=%d forecast=%s", retry_count + 1, forecast)
    try:
        response = sql_llm.invoke([
            ("system", _build_sql_prompt(forecast)),
            ("human", agent_input),
        ])
        output = response.content
    except Exception as e:
        logger.exception("SQL agent invocation failed")
        return {**state, "sql": None, "error": str(e)}

    sql = extract_sql(output)
    logger.debug("SQL generated=%r", sql)
    return {**state, "sql": sql, "sql_plan_verified": False, "error": None}


def critic_node(state: BIState) -> BIState:
    sql = state.get("sql")
    forecast = state.get("forecast", False)
    group_col = state.get("group_col")
    question = state.get("question")
    retry_count = state.get("retry_count", 0)

    if not sql:
        verdict = failed_verdict(SQL_CHECKS, "No SQL was produced. Generate a single valid SELECT query.", "sql_agent")
        return {
            **state,
            "critic_approved": False,
            "critic_reason": "No SQL was produced.",
            "sql_check_issues": ["No SQL was produced."], "review_fallback": False,
            "fix_instructions": "Generate a single valid SELECT query.",
            "retry_count": retry_count + 1,
            "sql_evaluation": verdict,
            "evaluation_history": state.get("evaluation_history", []) + [{"stage": "sql", **verdict}],
        }

    # Fast deterministic pre-checks, kept as a belt-and-suspenders layer
    # feeding into the LLM critic rather than replacing it.
    det_issues = []
    s_lower = sql.strip().lower()
    if not (s_lower.startswith("select") or s_lower.startswith("with")):
        det_issues.append("Query must be a SELECT/CTE statement, no DML.")
    if re.search(r"\b(insert|update|delete|drop|alter)\b", s_lower):
        det_issues.append("Query contains a DML/DDL keyword.")
    if forecast and _looks_like_future_filter(sql):
        det_issues.append("Forecast query filters to future/recent dates — must return full history.")
    if forecast and group_col and not re.search(rf"\b{re.escape(group_col)}\b", sql, re.IGNORECASE):
        det_issues.append(f"Question implies a per-{group_col} breakdown; SQL must SELECT and GROUP BY {group_col}.")
    if re.search(r"sum\s*\(\s*(total_sale|sales)\s*\)", s_lower):
        det_issues.append("Uses SUM(sales) or SUM(total_sale) instead of SUM(total_sales).")
    if len(_split_statements(sql)) > 1:
        det_issues.append(
            "Query contains more than one SQL statement — sqlite only "
            "accepts one at a time. Return exactly one SELECT/CTE statement."
        )

    critic_input = json.dumps({
        "question": question,
        "forecast_mode": forecast,
        "expected_group_dimension": group_col,
        "deterministic_pre_check_issues": det_issues,
        "candidate_sql": sql,
        "approved_schema": json.loads(APPROVED_SCHEMA),
        "sql_task": ("Retrieve historical training data only. The Python forecast engine handles "
                     "the requested future target; SQL must not produce predictions or filter to that target.")
                    if forecast else "Answer the historical question directly.",
    })

    try:
        verdict = invoke_evaluator(
            critic_llm, SQL_EVALUATOR_PROMPT + "\nApplicable rules:\n" + sql_review_rules(forecast),
            json.loads(critic_input), SQL_CHECKS, {"sql_agent"},
        ) if not state.get("sql_plan_verified") else {
            "passed": True, "checks": {key: True for key in SQL_CHECKS},
            "issues": [], "retry_target": None, "fix_instructions": "",
            "source": "deterministic_forecast_history_plan",
        }
    except Exception:
        logger.exception("SQL evaluation failed")
        verdict = failed_verdict(SQL_CHECKS, "SQL evaluator unavailable or returned an invalid verdict.", "sql_agent")
    if det_issues:
        verdict = {**verdict, "passed": False,
                   "checks": {**verdict["checks"], "rules_followed": False},
                   "issues": verdict["issues"] + det_issues, "retry_target": "sql_agent",
                   "fix_instructions": "; ".join(det_issues + [verdict["fix_instructions"]])}
    approved = verdict["passed"]
    reason = "; ".join(verdict["issues"])
    logger.info("SQL evaluation passed=%s issues=%s", approved, verdict["issues"])
    return {
        **state, "critic_approved": approved, "critic_reason": reason,
        "sql_check_issues": det_issues, "review_fallback": False,
        "fix_instructions": verdict["fix_instructions"], "sql_evaluation": verdict,
        "evaluation_history": state.get("evaluation_history", []) + [{"stage": "sql", **verdict}],
        "retry_count": retry_count + (0 if approved else 1),
    }


def fallback_node(state: BIState) -> BIState:
    """Keep reviewer uncertainty separate from mandatory database controls."""
    sql = state.get("sql")
    if sql and state.get("sql_check_issues") == []:
        try:
            validate_read_only_sql(sql)
            return {**state, "review_fallback": True, "critic_approved": False, "error": None}
        except ValueError:
            pass
    reason = state.get("critic_reason") or "The SQL reviewer did not approve the generated query."
    return {
        **state,
        "sql": None,
        "critic_approved": False,
        "error": "SQL review stopped before database execution. " + reason,
    }


def executor_node(state: BIState) -> BIState:
    if not state.get("critic_approved") and not state.get("review_fallback"):
        return {**state, "rows": [], "error": state.get("error") or "SQL was not approved for execution."}
    sql = state.get("sql")
    try:
        sql = _sanitize_single_statement(sql)
        if not sql:
            raise ValueError("No valid SQL statement to execute.")
        rows = _run_sql(sql, forecast=state.get("forecast", False))
        logger.info("SQL execution completed rows=%d", len(rows))
        # Persist the sanitized version so the SQL shown downstream (and
        # returned to the caller) matches what actually ran.
        return {**state, "sql": sql, "rows": rows, "error": None}
    except Exception as e:
        logger.exception("SQL execution failed")
        return {**state, "sql": sql, "rows": [], "error": str(e)}


def _build_forecast_chart_spec(result: ForecastResult) -> dict:
    """Build the forecast visualization inside the forecast-agent layer."""
    history = result.history.copy()
    forecast = result.forecast.copy()
    grouped = "group" in history.columns and "group" in forecast.columns

    history_records = []
    for _, row in history.iterrows():
        record = {
            "date": pd.Timestamp(row["date"]).isoformat(),
            "value": float(row["value"]),
            "series": "Historical",
        }
        if grouped:
            record["group"] = str(row["group"])
        history_records.append(record)

    forecast_records = []
    for _, row in forecast.iterrows():
        record = {
            "date": pd.Timestamp(row["date"]).isoformat(),
            "value": float(row["value"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
            "series": "Forecast",
        }
        if grouped:
            record["group"] = str(row["group"])
        forecast_records.append(record)

    line_records = []
    groups = sorted(forecast["group"].astype(str).unique()) if grouped else [None]
    for group in groups:
        hist_part = history[
            history["group"].astype(str) == group
        ] if grouped else history
        fc_part = forecast[
            forecast["group"].astype(str) == group
        ] if grouped else forecast
        if not hist_part.empty:
            last = hist_part.sort_values("date").iloc[-1]
            bridge = {
                "date": pd.Timestamp(last["date"]).isoformat(),
                "value": float(last["value"]),
                "series": "Forecast",
            }
            if grouped:
                bridge["group"] = group
            line_records.append(bridge)
        for _, row in fc_part.sort_values("date").iterrows():
            record = {
                "date": pd.Timestamp(row["date"]).isoformat(),
                "value": float(row["value"]),
                "series": "Forecast",
            }
            if grouped:
                record["group"] = group
            line_records.append(record)

    color = {"field": "group", "type": "nominal", "title": "Group"} if grouped else None
    history_encoding = {
        "x": {"field": "date", "type": "temporal", "title": "Date"},
        "y": {"field": "value", "type": "quantitative", "title": "Value"},
    }
    forecast_encoding = dict(history_encoding)
    band_encoding = {
        "x": history_encoding["x"],
        "y": {"field": "lower", "type": "quantitative", "title": "Value"},
        "y2": {"field": "upper"},
    }
    if color:
        history_encoding["color"] = color
        forecast_encoding["color"] = color
        band_encoding["color"] = {**color, "legend": None}

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {"labelColor": "#94a3b8", "titleColor": "#cbd5e1"},
            "legend": {"labelColor": "#cbd5e1", "titleColor": "#cbd5e1"},
        },
        "height": 340 if grouped else 320,
        "layer": [
            {
                "data": {"values": forecast_records},
                "mark": {"type": "area", "opacity": 0.15, "color": "#f5b942"},
                "encoding": band_encoding,
            },
            {
                "data": {"values": history_records},
                "mark": {
                    "type": "line",
                    "color": "#7dd3fc",
                    "strokeWidth": 2.2,
                    "point": {"filled": True, "size": 45},
                },
                "encoding": history_encoding,
            },
            {
                "data": {"values": line_records},
                "mark": {
                    "type": "line", "color": "#f5b942",
                    "strokeWidth": 2.4, "strokeDash": [5, 4],
                    "point": {"filled": True, "size": 55},
                },
                "encoding": forecast_encoding,
            },
        ],
    }


def forecast_agent_node(state: BIState) -> BIState:
    """Real agent node — only ever reached when state['forecast'] is True
    (see route_after_executor). Wraps the deterministic forecasting.py
    engine (ETS/ARIMA/drift, confidence intervals, chart building — all
    untouched) with an LLM decision/validation layer:
      1. Deterministic horizon/window parsing first (cheap, handles most
         phrasing already).
      2. LLM horizon reinterpretation, but only as a retry after the
         deterministic parse + forecast_series() failed once.
      3. Self-validation of the output (flags naive-drift / implausible
         negative bounds) before handing off to the Summary Agent.
    """
    # Defensive guard: graph routing should already make this impossible, but
    # never generate a forecast unless the orchestrator explicitly classified
    # the question as forecast-related.
    if not state.get("forecast", False):
        return {
            **state,
            "forecast_result": None,
            "forecast_df": None,
            "forecast_error": None,
        }

    question = state["question"]
    rows = state.get("rows") or []
    retry_count = state.get("forecast_retry_count", 0)

    if state.get("error"):
        # SQL execution already failed upstream; nothing to forecast on.
        return {**state, "forecast_error": "No data available to forecast on."}

    if not rows:
        logger.info("Forecast skipped because SQL returned no rows")
        return {
            **state,
            "forecast_error": "No historical data was returned to forecast on.",
            "forecast_retry_count": retry_count + 1,
        }

    df = pd.DataFrame(rows)

    target = extract_target_window(question)
    periods, unit = extract_horizon(question, default=12)
    horizon_label = f"next {periods} {unit}"

    # Only reach for the LLM once the deterministic parse has already been
    # tried and failed (retry_count > 0) AND there's no clear target window.
    if target is None and retry_count > 0:
        interpretation = _llm_interpret_horizon(question)
        logger.info("Forecast horizon interpretation=%s", interpretation)
        if interpretation.get("mode") == "relative" and interpretation.get("periods"):
            periods = interpretation["periods"]
            unit = interpretation.get('unit', unit)
            horizon_label = f"next {periods} {unit}"

    try:
        if target is not None:
            result = forecast_series(df, periods=12, target_window=target)
        else:
            result = forecast_series(df, periods=periods, horizon_label=horizon_label, horizon_unit=unit)
    except ValueError as e:
        logger.warning("Forecast generation rejected input: %s", e)
        return {
            **state,
            "forecast_error": str(e),
            "forecast_retry_count": retry_count + 1,
        }
    except Exception as e:
        logger.exception("Unexpected forecast generation failure")
        return {
            **state,
            "forecast_error": f"Forecasting failed unexpectedly: {e}",
            "forecast_retry_count": retry_count + 1,
        }

    # Self-validation: flag low-confidence output rather than presenting it
    # with the same authority as a well-fitted seasonal model.
    warnings: List[str] = []
    if result.method == "Naive drift":
        warnings.append(
            "Used a simple trend projection due to insufficient history for "
            "a seasonal model — treat this forecast as low-confidence."
        )
    try:
        forecast_df = result.forecast
        if "lower" in forecast_df.columns and not forecast_df.empty:
            numeric_cols = df.select_dtypes(include="number")
            if not numeric_cols.empty and (numeric_cols >= 0).all().all():
                if (forecast_df["lower"] < 0).any():
                    warnings.append(
                        "The lower confidence bound dips below zero for a "
                        "metric that has historically always been positive "
                        "— treat that lower bound as not practically meaningful."
                    )
    except Exception as e:
        logger.warning("Forecast validation check skipped: %s", e)

    result.meta["warnings"] = warnings
    forecast_chart_spec = _build_forecast_chart_spec(result)
    logger.info("Forecast succeeded method=%s warnings=%s", result.method, warnings)
    return {
        **state,
        "forecast_result": result,
        "forecast_chart_spec": forecast_chart_spec,
        "forecast_df": df,
        "forecast_error": None,
    }


def accuracy_agent_node(state: BIState) -> BIState:
    """Backtest the same historical series used by the forecast agent.

    This node never creates a second user-facing forecast. ``quick_accuracy``
    refits the shared forecasting engine on historical windows solely to
    estimate out-of-sample error.
    """
    if not state.get("forecast", False):
        return state

    result = state.get("forecast_result")
    df = state.get("forecast_df")

    if result is None or df is None or df.empty:
        return state

    try:
        accuracy_df = df
        group_col = result.meta.get("group_col")
        if group_col and group_col in df.columns:
            date_candidates = [
                c for c in df.columns
                if c != group_col and not pd.api.types.is_numeric_dtype(df[c])
            ]
            numeric_candidates = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
            if date_candidates and numeric_candidates:
                accuracy_df = df.groupby(
                    date_candidates[0], as_index=False
                )[numeric_candidates[0]].sum()

        metrics = quick_accuracy(accuracy_df, horizon=3)
        if metrics is not None:
            result.meta["accuracy"] = metrics
            logger.info(
                "Forecast backtest mape=%.1f splits=%d points=%d",
                metrics.mape,
                metrics.n_splits,
                metrics.n_points,
            )
    except Exception as e:
        # Accuracy is supplementary; a scoring failure must not discard a
        # successfully generated forecast.
        logger.warning("Forecast backtest skipped: %s", e)

    return {**state, "forecast_result": result}


def summary_node(state: BIState) -> BIState:
    question = state["question"]
    rows = state.get("rows") or []
    sql = state.get("sql")
    error = state.get("error")
    forecast_result = state.get("forecast_result")
    forecast_error = state.get("forecast_error")

    if error:
        summary = f"⚠️ {error}"
        return {**state, "summary": summary}
    if state.get("review_fallback"):
        return {**state, "summary": data_only_summary(state)}

    summary_input: Dict[str, Any] = {
        "question": question,
        "row_count": len(rows),
        "sample_rows": _safe_rows_for_llm(rows, settings.summary_sample_rows),
    }

    if state.get("forecast", False) and forecast_result is not None:
        try:
            forecast_stats = forecast_result.forecast.describe().to_dict()
        except Exception:
            forecast_stats = None
        summary_input["forecast"] = {
            "monthly_values": forecast_result.forecast.head(20).to_dict(orient="records"),
            "method": forecast_result.method,
            "horizon": forecast_result.meta.get("target_window")
                       or f"{forecast_result.meta.get('periods')} periods",
            "warnings": forecast_result.meta.get("warnings", []),
            "forecast_stats": forecast_stats,
        }
        accuracy = forecast_result.meta.get("accuracy")
        if isinstance(accuracy, AccuracyMetrics):
            summary_input["forecast"]["accuracy"] = {
                "metric": "historical walk-forward backtest error",
                "mape_percent": accuracy.mape,
                "mae": accuracy.mae,
                "rmse": accuracy.rmse,
                "smape_percent": accuracy.smape,
                "validation_points": accuracy.n_points,
                "validation_splits": accuracy.n_splits,
                "validation_scope": "Aggregated across groups" if forecast_result.meta.get("group_col") else "Single historical series",
            }
    elif forecast_error:
        summary_input["forecast_error"] = forecast_error

    evidence = build_evidence(
        _safe_rows_for_llm(rows, len(rows)), settings.summary_sample_rows,
        summary_input.get("forecast") or {"error": forecast_error},
        executed_sql=sql,
    )
    if len(rows) <= settings.summary_sample_rows:
        # The complete query result already contains the requested metrics.
        # Extra aggregates encourage unrelated totals/averages in small breakdowns.
        evidence["facts"] = {key: fact for key, fact in evidence["facts"].items()
                             if not key.startswith("returned_rows.")}
    try:
        resp = summary_llm.invoke([
            ("system", SUMMARY_SYSTEM_PROMPT + STRUCTURED_SUMMARY_PROMPT),
            ("human", json.dumps({"question": question, "evidence": evidence,
                                 "revision_instructions": state.get("evaluation_feedback", "")}, default=str)),
        ])
        statements = parse_statements(resp.content)
        summary = " ".join(item["text"] for item in statements).replace("$", r"\$")
        summary_error = None
    except Exception:
        logger.exception("Summary generation failed")
        summary, statements = "", []
        summary_error = "Summary generation failed or returned invalid structured statements."
    return {**state, "summary": summary, "summary_statements": statements,
            "summary_error": summary_error, "evidence": evidence}


def data_only_summary(state: BIState) -> str:
    result = state.get("forecast_result")
    if result is not None and state.get("sql_plan_verified") and not state.get("forecast_error"):
        summary = "Showing the statistical forecast below, based on the historical data. These are estimates, not actual future orders or sales. The AI-written interpretation could not be verified."
        warnings = result.meta.get("warnings", [])
        if warnings:
            summary += " " + " ".join(warnings)
        accuracy = result.meta.get("accuracy")
        if isinstance(accuracy, AccuracyMetrics):
            summary += (f" Historical backtest error was {accuracy.mape:.2f}% MAPE across "
                        f"{accuracy.n_splits} validation splits ({accuracy.n_points} points) "
                        "on the single historical series; this is not guaranteed future accuracy.")
        return summary
    if not state.get("rows"):
        return "The generated query returned no matching rows. Automated review could not confirm that it fully answers your question; the SQL is available for inspection."
    return "Showing the retrieved query results below. Automated review could not confirm a reliable interpretation, so no AI-generated analysis is included. You can inspect the SQL and data."


def answer_evaluation_node(state: BIState) -> BIState:
    if state.get("error"):
        verdict = failed_verdict(ANSWER_CHECKS, "No executable approved result was available.", "sql_agent")
        return {**state, "evaluation": {**verdict, "status": "blocked"},
                "evaluation_history": state.get("evaluation_history", []) + [{"stage": "answer", **verdict}]}
    if state.get("review_fallback"):
        verdict = failed_verdict(ANSWER_CHECKS, "SQL semantic review was inconclusive; showing query results only.", "sql_agent")
        return {**state, "summary": data_only_summary(state),
                "evaluation": {**verdict, "status": "data_only"},
                "evaluation_history": state.get("evaluation_history", []) + [{"stage": "answer", **verdict}]}
    issues = ([state["summary_error"]] if state.get("summary_error") else
              check_statements(state.get("summary_statements", []), state["evidence"]))
    try:
        if issues:
            verdict = failed_verdict(ANSWER_CHECKS, "; ".join(issues), "summary_agent")
        else:
            verdict = invoke_evaluator(
                critic_llm, ANSWER_EVALUATOR_PROMPT, {
                    "question": state["question"], "executed_sql": state.get("sql"),
                    "evidence": state["evidence"], "statements": state["summary_statements"],
                    "forecast_requested": state.get("forecast", False),
                }, ANSWER_CHECKS, {"sql_agent", "summary_agent"},
            )
            verdict = apply_answer_applicability(verdict, state.get("forecast", False))
    except Exception:
        logger.exception("Answer evaluation failed")
        verdict = failed_verdict(ANSWER_CHECKS, "Answer evaluator unavailable or returned an invalid verdict.", "summary_agent")
    retries = state.get("evaluation_retry_count", 0)
    status = "passed" if verdict["passed"] else (
        "retrying" if retries < settings.max_evaluation_retries else "blocked")
    result = {**state, "evaluation": {**verdict, "status": status},
              "evaluation_history": state.get("evaluation_history", []) + [{"stage": "answer", **verdict}],
              "evaluation_feedback": verdict["fix_instructions"]}
    if status == "retrying":
        result["evaluation_retry_count"] = retries + 1
        if verdict["retry_target"] == "sql_agent":
            result.update(sql=None, rows=[], critic_approved=False, retry_count=1,
                          review_fallback=False, sql_check_issues=[],
                          critic_reason="; ".join(verdict["issues"]),
                          fix_instructions=verdict["fix_instructions"], error=None,
                          forecast_result=None, forecast_df=None, forecast_error=None,
                          forecast_retry_count=0, forecast_chart_spec=None,
                          chart_spec=None, chart_note=None, summary="", summary_statements=[])
    if status == "blocked":
        result["evaluation"]["status"] = "data_only"
        result["summary"] = data_only_summary(state)
    logger.info("Answer evaluation status=%s revisions=%d", status, retries)
    return result


def route_after_evaluation(state: BIState) -> str:
    verdict = state["evaluation"]
    if verdict["status"] == "retrying":
        return verdict["retry_target"]
    return "viz" if verdict["passed"] else "end"


def viz_node(state: BIState) -> BIState:
    # The forecast agent already produces one combined history + forecast
    # visualization. Do not create a second historical-results chart.
    if state.get("forecast", False):
        return {**state, "chart_spec": None, "chart_note": None}

    rows = state.get("rows") or []
    question = state.get("question")
    chart_spec, chart_note = generate_chart_spec(rows, question=question)
    return {**state, "chart_spec": chart_spec, "chart_note": chart_note}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_critic(state: BIState) -> str:
    if state.get("critic_approved"):
        return "proceed"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        logger.warning("SQL retries exhausted; routing to fallback")
        return "fallback"
    return "retry"


def route_after_executor(state: BIState) -> str:
    """The ONLY place forecast_agent gets reached from. Reads the single
    state['forecast'] flag set once by orchestrator_node — non-forecast
    questions never invoke forecast_agent_node at all (not a no-op call,
    a graph edge that's never taken)."""
    if state.get("review_fallback") or state.get("error"):
        return "summary"
    if state.get("forecast"):
        return "forecast"
    return "summary"


def route_after_forecast(state: BIState) -> str:
    """Allow the forecast agent one self-retry (e.g. LLM horizon
    reinterpretation) before giving up and letting the Summary Agent
    explain the failure to the user."""
    if state.get("forecast_error") and state.get("forecast_retry_count", 0) <= MAX_FORECAST_RETRIES:
        return "retry"
    if state.get("forecast_result") is not None:
        return "accuracy"
    return "summary"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_graph():
    graph = StateGraph(BIState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("critic", critic_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("executor", executor_node)
    graph.add_node("forecast_agent", forecast_agent_node)
    graph.add_node("accuracy_agent", accuracy_agent_node)
    graph.add_node("summary_agent", summary_node)
    graph.add_node("viz_agent", viz_node)
    graph.add_node("answer_evaluator", answer_evaluation_node)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "sql_agent")
    graph.add_edge("sql_agent", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retry": "sql_agent", "proceed": "executor", "fallback": "fallback"},
    )
    graph.add_edge("fallback", "executor")

    # Forecast questions detour through forecast_agent; everything else
    # goes straight to summary. This edge is the structural guarantee that
    # forecast_agent only ever runs for forecast-related questions.
    graph.add_conditional_edges(
        "executor",
        route_after_executor,
        {"forecast": "forecast_agent", "summary": "summary_agent"},
    )

    graph.add_conditional_edges(
        "forecast_agent",
        route_after_forecast,
        {
            "retry": "forecast_agent",
            "accuracy": "accuracy_agent",
            "summary": "summary_agent",
        },
    )
    graph.add_edge("accuracy_agent", "summary_agent")

    graph.add_edge("summary_agent", "answer_evaluator")
    graph.add_conditional_edges("answer_evaluator", route_after_evaluation, {
        "sql_agent": "sql_agent", "summary_agent": "summary_agent", "viz": "viz_agent", "end": END,
    })
    graph.add_edge("viz_agent", END)

    return graph.compile()


_compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_response_cache: Dict[str, Tuple[float, dict]] = {}


@measure_request
def ask_bi_agent(question: str) -> dict:
    """
    Run the multi-agent BI pipeline on a question.

    Orchestrator (decides forecast intent) -> SQL Agent -> Critic Agent
    (bounded retry loop; stop on rejection) -> Executor ->
    [Forecast Agent, only if forecast-related] -> Summary Agent -> Answer Evaluator
    (bounded SQL/summary revisions) -> Viz Agent, only after a verified answer.

    Parameters
    ----------
    question : the user's natural-language question.
    Returns
    -------
    dict with keys:
        summary             : str
        sql                 : str | None
        chart_spec          : dict | None
        chart_note          : str | None
        rows                : list[dict] | None
        forecast_used       : bool
        forecast            : dict | None
        forecast_error      : str | None
        evaluation          : dict with final verdict, checks, issues, and status
        sql_evaluation      : dict with the most recent SQL verdict
        evaluation_history  : list of stage verdicts across revisions
    """
    try:
        normalized_question = question.strip()
        cached = _response_cache.get(normalized_question)
        if cached and time.monotonic() - cached[0] < settings.cache_ttl_seconds:
            logger.info("Returning cached response")
            mark_cache_hit()
            return cached[1]

        token = _request_id.set(str(uuid.uuid4()))
        try:
            final_state = _compiled_graph.invoke(
                {"question": question},
                {"recursion_limit": (settings.max_evaluation_retries + 1)
                 * (2 * MAX_RETRIES + MAX_FORECAST_RETRIES + 15) + 10},
            )
        finally:
            _request_id.reset(token)
        evaluation = final_state.get("evaluation", {"passed": False, "status": "blocked"})
        verified = evaluation.get("passed") is True
        statistical_fallback = (evaluation.get("status") == "data_only"
                                and final_state.get("sql_plan_verified")
                                and not final_state.get("forecast_error"))
        forecast_result = final_state.get("forecast_result") if verified or statistical_fallback else None

        forecast_payload = None
        if final_state.get("forecast", False) and forecast_result is not None:
            accuracy = forecast_result.meta.get("accuracy")
            meta = {
                key: value
                for key, value in forecast_result.meta.items()
                if key != "accuracy"
            }
            if isinstance(accuracy, AccuracyMetrics):
                split_label = "split" if accuracy.n_splits == 1 else "splits"
                meta["accuracy"] = {
                    "mae": accuracy.mae,
                    "rmse": accuracy.rmse,
                    "mape": accuracy.mape,
                    "smape": accuracy.smape,
                    "n_points": accuracy.n_points,
                    "n_splits": accuracy.n_splits,
                    "summary": (
                        f"Historical forecast error: {accuracy.mape:.1f}% MAPE "
                        f"across {accuracy.n_splits} validation {split_label} "
                        f"({accuracy.n_points} forecast points)."
                    ),
                }
            forecast_payload = {
                "rows": json.loads(forecast_result.forecast.to_json(orient="records", date_format="iso")),
                "chart_spec": final_state.get("forecast_chart_spec"),
                "method": forecast_result.method,
                "meta": meta,
            }

        response = {
            "summary": final_state.get("summary"),
            "sql": final_state.get("sql"),
            "chart_spec": final_state.get("chart_spec"),
            "chart_note": final_state.get("chart_note"),
            "rows": final_state.get("rows") if verified or evaluation.get("status") == "data_only" else [],
            "evaluation": evaluation,
            "sql_evaluation": final_state.get("sql_evaluation"),
            "evaluation_history": final_state.get("evaluation_history", []),
            "forecast_used": bool(forecast_payload) or (bool(final_state.get("forecast")) and evaluation.get("status") != "data_only"),
            "forecast": forecast_payload,
            "forecast_error": final_state.get("forecast_error"),
        }
        if verified:
            _response_cache[normalized_question] = (time.monotonic(), response)
        return response

    except Exception:
        logger.exception("BI request failed")
        return {
            "evaluation": {"passed": False, "status": "blocked"},
            "summary": "The assistant is temporarily unavailable. Please try again shortly.",
            "sql": None,
            "chart_spec": None,
            "chart_note": None,
            "rows": None,
            "forecast_used": False,
            "forecast": None,
            "forecast_error": None,
        }
