"""Streamlit dashboard for DataSage answers."""

import hmac
import html
import logging
from numbers import Real

import streamlit as st

st.set_page_config(page_title="DataSage", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")
logger = logging.getLogger("datasage.app")

try:
    from config import get_settings
    from bi import ask_bi_agent
    settings = get_settings()
except Exception:
    logger.exception("DataSage startup failed")
    st.error("DataSage could not start. Check the container logs and deployment settings.")
    st.stop()


def require_authentication() -> None:
    if not settings.auth_required:
        return
    expected = settings.app_password.get_secret_value()
    if not expected:
        st.error("Authentication is enabled but no application password is configured.")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    st.title("DataSage")
    password = st.text_input("Application password", type="password")
    if st.button("Sign in"):
        st.session_state.authenticated = hmac.compare_digest(password, expected)
        if not st.session_state.authenticated:
            st.error("Incorrect password.")
        else:
            st.rerun()
    st.stop()


require_authentication()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root { --bg:#071226; --panel:#14233d; --line:#30415e; --text:#f5f8ff;
        --muted:#aebdd2; --accent:#ffbd4a; }
html,body,[class*="css"],.stApp { font-family:'DM Sans',sans-serif;
    color:var(--text); background:var(--bg); }
.stApp { background:linear-gradient(180deg,#09162c,#071226 60%); }
[data-testid="stHeader"],#MainMenu,footer { display:none; }
.block-container { max-width:1580px; padding-top:1.2rem; padding-bottom:3rem; }
h1,h2,h3,p,label { color:var(--text); }
.ds-header { display:flex; justify-content:space-between; align-items:center; gap:1rem;
    border-bottom:1px solid var(--line); padding:.2rem 0 1.1rem; }
.ds-brand { font-size:1.8rem; font-weight:800; letter-spacing:-.05em; }
.ds-mark { color:var(--accent); letter-spacing:-.2em; padding-right:.6rem; }
.ds-tab { color:var(--accent); font-weight:700; border-bottom:3px solid var(--accent);
    padding:.7rem 0; }
.ds-source { color:#90e7ab; font-size:.9rem; }
.ds-intro { padding:1.5rem 0 .75rem; }
.ds-eyebrow { color:var(--accent); font-size:.76rem; font-weight:800;
    letter-spacing:.13em; text-transform:uppercase; }
.ds-intro h1 { font-size:clamp(1.8rem,3vw,2.5rem); margin:.25rem 0;
    letter-spacing:-.04em; }
.ds-intro p,.ds-question { color:var(--muted); }
[data-testid="stForm"] { background:var(--panel); border:1px solid var(--accent)!important;
    border-radius:14px; padding:.45rem .8rem!important; }
[data-testid="stForm"] input { background:transparent; color:var(--text);
    border:0; font-size:1.04rem; }
[data-testid="stFormSubmitButton"] button { background:var(--accent); color:#172238;
    border:0; border-radius:10px; font-weight:800; }
.ds-section { display:flex; align-items:baseline; justify-content:space-between;
    margin:1.5rem 0 .6rem; gap:1rem; }
.ds-section h2 { font-size:1.35rem; margin:0; }
.ds-section span { color:var(--muted); font-size:.85rem; }
.ds-question { font-size:.94rem; margin-bottom:.7rem; }
[data-testid="stMetric"] { background:linear-gradient(135deg,#1a2e4d,var(--panel));
    border:1px solid var(--line); border-radius:13px; padding:1rem; min-height:105px; }
[data-testid="stMetricValue"] { color:var(--text); font-weight:800; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.ds-panel-title) {
    background:linear-gradient(145deg,#192b48,#112039); border:1px solid var(--line);
    border-radius:14px; }
.ds-panel-title { font-size:1.13rem; font-weight:800; margin-bottom:.5rem; }
.ds-empty { margin-top:1.3rem; padding:2.5rem 1rem; text-align:center;
    background:var(--panel); border:1px solid var(--line); border-radius:14px; }
.ds-empty h2 { margin:0 0 .5rem; }
.ds-empty p { color:var(--muted); margin:0; }
[data-testid="stExpander"] { background:var(--panel)!important;
    border:1px solid var(--line)!important; border-radius:12px!important; }
[data-testid="stExpander"] summary { color:var(--text)!important; font-weight:700; }
@media(max-width:760px) { .block-container { padding:.8rem .8rem 2rem; }
    .ds-tab { display:none; } .ds-brand { font-size:1.35rem; }
    .ds-source { font-size:.75rem; } }
</style>
""", unsafe_allow_html=True)


def render_chart(spec: dict) -> None:
    try:
        st.vega_lite_chart(spec, width="stretch")
    except Exception:
        logger.exception("Chart rendering failed")
        st.warning("The result was generated, but its chart could not be displayed.")


def _toggle_metrics(key: str) -> None:
    st.session_state[key] = not st.session_state.get(key, False)


def render_metrics(metrics: dict | None) -> None:
    if not metrics:
        return
    key = f"metrics_visible_{metrics['request_id']}"
    visible = st.session_state.get(key, False)
    st.button("Hide request details" if visible else "Show request details",
              key=f"metrics_button_{metrics['request_id']}",
              on_click=_toggle_metrics, args=(key,))
    if not visible:
        return
    with st.container(border=True):
        st.write(f"Response time: {metrics['latency_seconds']:.2f} seconds")
        st.write(f"Tokens: {metrics['input_tokens']:,} input / {metrics['output_tokens']:,} output")
        cost = metrics.get("estimated_cost_usd")
        st.write(f"Estimated API cost: ${cost:.6f}" if cost is not None
                 else "Estimated API cost: unavailable (missing usage or pricing)")
        st.write(f"Model calls: {metrics['model_calls']}")
        if metrics.get("cache_hit"):
            st.caption("Cached answer; no new API calls.")
        if metrics.get("calls"):
            st.dataframe(metrics["calls"], width="stretch")
        st.caption("Estimates cover reported API usage only. Unreported SDK retries and infrastructure costs are excluded.")


def _metric_cards(rows: list[dict]) -> None:
    """Display only numeric values returned by a single-row query."""
    if len(rows) != 1:
        return
    values = [(key, value) for key, value in rows[0].items()
              if isinstance(value, Real) and not isinstance(value, bool)]
    if not values:
        return
    st.markdown('<div class="ds-section"><h2>Key figures</h2><span>From this query</span></div>',
                unsafe_allow_html=True)
    for column, (key, value) in zip(st.columns(min(len(values), 4)), values[:4]):
        number = f"{value:,}" if isinstance(value, int) else f"{value:,.2f}"
        column.metric(key.replace("_", " ").title(), number)


def _render_status(status: str | None) -> None:
    if status == "passed":
        st.success("Verified by automated review")
    elif status == "qualified":
        st.warning("Automated verification was unavailable. This answer has not been independently verified.")
    elif status == "data_only":
        st.info("The AI interpretation was not verified. Inspect the available results below.")
    elif status == "blocked":
        st.warning("This request could not be verified.")


def render_answer(message: dict) -> None:
    """Render the latest answer as a data-driven dashboard."""
    forecast = message.get("forecast_used", False)
    chart = message.get("forecast_chart") if forecast else message.get("chart_spec")
    panel_title = "Result" if message.get("evaluation_status") in {"data_only", "blocked"} else "AI insight"
    rows = message.get("rows") or []
    forecast_rows = message.get("forecast_rows") or []
    st.markdown('<div class="ds-section"><h2>Analysis</h2><span>'
                + ("Forecast" if forecast else "Historical data") + '</span></div>',
                unsafe_allow_html=True)
    if message.get("question"):
        st.markdown('<div class="ds-question">' + html.escape(message["question"])
                    + '</div>', unsafe_allow_html=True)
    _render_status(message.get("evaluation_status"))
    if not forecast:
        _metric_cards(rows)
    if chart:
        visual, insight = st.columns([1.8, 1], gap="large")
        with visual:
            with st.container(border=True):
                st.markdown('<div class="ds-panel-title">Visual overview</div>',
                            unsafe_allow_html=True)
                render_chart(chart)
                if message.get("chart_note"):
                    st.caption(message["chart_note"])
        with insight:
            with st.container(border=True):
                st.markdown(f'<div class="ds-panel-title">{panel_title}</div>',
                            unsafe_allow_html=True)
                st.markdown(message.get("content") or "No interpretation available.")
    else:
        with st.container(border=True):
            st.markdown(f'<div class="ds-panel-title">{panel_title}</div>',
                        unsafe_allow_html=True)
            st.markdown(message.get("content") or "No interpretation available.")
        if message.get("chart_note"):
            st.caption(message["chart_note"])
    if forecast_rows:
        st.markdown('<div class="ds-section"><h2>Forecast values</h2></div>',
                    unsafe_allow_html=True)
        st.dataframe(forecast_rows, width="stretch", hide_index=True)
    if rows:
        with st.expander("View data", expanded=message.get("evaluation_status") == "data_only"):
            st.dataframe(rows, width="stretch", hide_index=True)
    if message.get("sql"):
        with st.expander("View SQL"):
            st.code(message["sql"], language="sql")
    render_metrics(message.get("metrics"))


if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("""
<div class="ds-header">
  <div class="ds-brand"><span class="ds-mark">▂▅▇</span>DataSage</div>
  <div class="ds-tab">Dashboard</div>
  <div class="ds-source">● Data source ready</div>
</div>
<div class="ds-intro">
  <div class="ds-eyebrow">Executive analytics</div>
  <h1>Ask your data. See the evidence.</h1>
  <p>Explore your connected dataset with reviewed SQL, clear answers, and visual results.</p>
</div>
""", unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True):
    prompt_column, submit_column = st.columns([8, 1], vertical_alignment="bottom")
    with prompt_column:
        question = st.text_input("Business question", label_visibility="collapsed",
                                 placeholder="Ask DataSage about your business data...")
    with submit_column:
        submitted = st.form_submit_button("Ask", width="stretch")

if submitted and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Analyzing your data..."):
            # Forecast routing remains owned by the pipeline.
            result = ask_bi_agent(question)
    except Exception:
        logger.exception("DataSage request failed")
        result = {"summary": "The assistant is temporarily unavailable. Please try again shortly.",
                  "evaluation": {"status": "blocked"}}
    forecast = result.get("forecast") or {}
    summary = result.get("summary") or "No answer was available."
    st.session_state.messages.append({
        "role": "assistant", "question": question, "content": summary,
        "metrics": result.get("metrics"), "sql": result.get("sql"),
        "chart_spec": result.get("chart_spec"), "chart_note": result.get("chart_note"),
        "rows": result.get("rows"),
        "evaluation_status": result.get("evaluation", {}).get("status"),
        "forecast_used": result.get("forecast_used", False),
        "forecast_chart": forecast.get("chart_spec"), "forecast_rows": forecast.get("rows"),
    })

answers = [msg for msg in st.session_state.messages if msg.get("role") == "assistant"]
if answers:
    render_answer(answers[-1])
    if len(answers) > 1:
        with st.expander(f"Previous questions ({len(answers) - 1})"):
            for previous in reversed(answers[:-1]):
                st.markdown(f"**{html.escape(previous.get('question', 'Question'))}**")
                st.markdown(previous.get("content") or "No answer was available.")
                st.divider()
else:
    st.markdown("""
    <div class="ds-empty"><h2>Your analysis starts here</h2>
    <p>Ask a question above to see a chart, an explanation, and the data behind it.</p></div>
    """, unsafe_allow_html=True)
