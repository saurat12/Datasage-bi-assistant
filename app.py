"""Streamlit dashboard for DataSage answers."""

import hmac
import html
import logging

import streamlit as st

st.set_page_config(page_title="DataSage", page_icon="✦", layout="wide",
                   initial_sidebar_state="expanded")
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
:root { --bg:#08132a; --nav:#0b1832; --card:#152643; --card2:#1a2b4c;
        --border:#304361; --text:#f5f8ff; --muted:#b7c5dc;
        --teal:#52e4dd; --cyan:#67b9ef; --purple:#a18aff; }
html,body,.stApp,[class*="css"] { font-family:'DM Sans',sans-serif; color:var(--text); }
.stApp { background:radial-gradient(circle at 56% 38%,#142753 0%,#0a1732 45%,#071128 100%); }
[data-testid="stHeader"],#MainMenu,footer { display:none; }
[data-testid="stSidebar"] { background:linear-gradient(155deg,#0a1832,#0b1630 65%,#10274d);
    border-right:1px solid #293955; min-width:270px; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:1.3rem .85rem; }
[data-testid="stSidebar"] hr { border-color:#2a3c5c; }
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
    border-radius:14px; padding:.72rem .9rem; margin:.18rem 0;
    color:#d9e4f5; font-size:1rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background:linear-gradient(105deg,#227d80,#174c69); color:white;
    box-shadow:inset 0 0 0 1px #44c9c5; }
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] { display:none; }
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display:none; }
.block-container { max-width:1650px; padding:1.4rem 2rem 3rem; }
h1,h2,h3,p,label { color:var(--text); }
.ds-brand { display:flex; align-items:center; gap:.7rem; padding:.2rem .45rem 1.6rem;
    font-size:1.75rem; font-weight:800; letter-spacing:-.05em; }
.ds-brand svg { width:46px; height:43px; flex:0 0 auto; }
.ds-sidebar-foot { margin-top:3rem; border-top:1px solid #2a3c5c;
    padding:1rem .6rem; color:#96aac8; font-size:.82rem; line-height:1.5; }
.ds-top { display:flex; justify-content:space-between; align-items:center; gap:1rem;
    margin:.15rem 0 1.45rem; }
.ds-top h1 { font-size:clamp(1.9rem,2.8vw,2.55rem); line-height:1.1;
    margin:0; font-weight:800; letter-spacing:-.055em; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.ds-chat-marker),
[data-testid="stVerticalBlockBorderWrapper"]:has(.ds-chart-marker) {
    background:linear-gradient(135deg,#1a294b,#111d3a 65%,#1e2151);
    border:1px solid #34496f; border-radius:22px;
    box-shadow:0 14px 40px #030b1c55; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.ds-chat-marker) { min-height:420px; }
.ds-chat-marker,.ds-chart-marker { height:0; }
.ds-chat-heading { display:flex; align-items:center; gap:.7rem; color:#eaf4ff;
    font-size:1.05rem; font-weight:700; margin:.1rem 0 1rem; }
.ds-glow-icon { display:inline-flex; width:38px; height:38px; border-radius:12px;
    align-items:center; justify-content:center; color:var(--teal); font-size:1.4rem;
    background:linear-gradient(135deg,#1d6876,#24446e); border:1px solid #34899d; }
.ds-question-wrap { display:flex; justify-content:flex-start; margin:.2rem 0 1rem; }
.ds-question-bubble { display:inline-block; max-width:94%; padding:.83rem 1rem;
    background:linear-gradient(125deg,#344164,#34375f); border-radius:16px 16px 16px 5px;
    border-bottom:1px solid #9eadd3; font-size:1rem; }
.ds-insight-bar { display:flex; align-items:center; gap:.75rem; margin:.1rem 0 .7rem;
    font-size:1.08rem; font-weight:800; }
.ds-answer-box { background:linear-gradient(110deg,#183f56,#252a65);
    border:1px solid #35577e; border-radius:16px; padding:1.1rem 1.2rem .8rem;
    margin-bottom:1rem; }
.ds-answer-copy { line-height:1.65; margin:.35rem 0; }
.ds-empty { padding:1.3rem .3rem 1.8rem; min-height:145px; }
.ds-empty h2 { font-size:1.45rem; margin:0 0 .4rem; }
.ds-empty p { color:var(--muted); margin:0; line-height:1.55; }
[data-testid="stChatInput"] { background:linear-gradient(105deg,#203c55,#252b57);
    border:1px solid #3c8aab!important; border-radius:15px; padding:.4rem .75rem!important;
    box-shadow:inset 0 1px 0 #78cad331; }
[data-testid="stChatInput"] textarea { background:transparent; color:var(--text);
    font-size:1rem; border:0; }
[data-testid="stChatInputSubmitButton"] button { border:0; border-radius:11px;
    color:#081b2a; font-weight:800; background:linear-gradient(135deg,#79e7f1,#44d9cc); }
.ds-chart-head { display:flex; justify-content:space-between; align-items:flex-start;
    gap:1rem; margin:.1rem 0 .6rem; }
.ds-chart-head strong { display:block; font-size:1.15rem; }
.ds-chart-head small { color:var(--muted); font-size:.84rem; }
.ds-chart-icon { color:var(--teal); font-size:1.4rem; padding-right:.45rem; }
.ds-lower { color:var(--muted); margin:.7rem 0; font-size:.85rem; }
[data-testid="stExpander"] { background:#14243f!important; border:1px solid #354b69!important;
    border-radius:12px!important; }
[data-testid="stExpander"] summary { color:var(--text)!important; font-weight:700; }
@media(max-width:900px) { .block-container { padding:1rem .8rem 2rem; } }
</style>
""", unsafe_allow_html=True)


def render_chart(spec: dict) -> None:
    try:
        styled = {**spec, "background": "transparent"}
        config = dict(spec.get("config", {}))
        config["axis"] = {**config.get("axis", {}), "labelColor": "#c5d6ef",
                          "titleColor": "#e8f2ff", "gridColor": "#344a6a",
                          "domainColor": "#526789", "tickColor": "#526789"}
        config["legend"] = {**config.get("legend", {}), "labelColor": "#d4e0f2",
                            "titleColor": "#e8f2ff"}
        styled["config"] = config
        st.vega_lite_chart(styled, width="stretch")
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


def build_followup_context(message: dict | None) -> dict | None:
    """Carry only the latest verified insight into the next question."""
    if (not message or message.get("evaluation_status") != "passed"
            or message.get("context_version") is None):
        return None
    return {"question": message.get("interpreted_question") or message["question"],
            "summary": message["content"], "passed": True,
            "database_version": message["context_version"]}


def render_conversation(answers: list[dict]) -> None:
    with st.container(border=True):
        st.markdown('<div class="ds-chat-marker"></div><div class="ds-chat-heading">'
                    '<span class="ds-glow-icon">✧</span>Ask DataSage</div>',
                    unsafe_allow_html=True)
        if answers:
            for index, message in enumerate(answers):
                if index:
                    st.divider()
                st.markdown('<div class="ds-question-wrap"><div class="ds-question-bubble">'
                            + html.escape(message.get("question") or "") + '</div></div>',
                            unsafe_allow_html=True)
                if message.get("interpreted_question"):
                    st.caption("Interpreted as: " + message["interpreted_question"])
                answer = html.escape(message.get("content") or "No interpretation available.")
                st.markdown('<div class="ds-answer-box"><div class="ds-insight-bar">'
                            '<span class="ds-glow-icon">✦</span>DataSage insight</div>'
                            '<div class="ds-answer-copy">' + answer.replace("\n", "<br>")
                            + '</div></div>', unsafe_allow_html=True)
                if (index == len(answers) - 1
                        and message.get("evaluation_status") == "data_only"
                        and message.get("rows")):
                    st.button("View returned data", key="view_returned_data",
                              on_click=lambda: st.session_state.update(navigation="Data"))
        else:
            st.markdown('<div class="ds-empty"><h2>Ask a question about your data</h2>'
                        '<p>Explore trends, comparisons, and forecasts. Your latest '
                        'answer will appear here.</p></div>',
                        unsafe_allow_html=True)


def render_question_input(answers: list[dict]) -> None:
    question = st.chat_input("Ask a question...")
    if not question or not question.strip():
        return
    question = question.strip()
    available_context = build_followup_context(answers[-1] if answers else None)
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Analyzing your data..."):
            result = ask_bi_agent(question, context=available_context)
    except Exception:
        logger.exception("DataSage request failed")
        result = {"summary": "The assistant is temporarily unavailable. Please try again shortly.",
                  "evaluation": {"status": "blocked"}}
    follow_up = result.get("followup_requested", False)
    forecast = result.get("forecast") or {}
    st.session_state.messages.append({
        "role": "assistant", "question": question,
        "content": result.get("summary") or "No answer was available.",
        "metrics": result.get("metrics"),
        "sql": None if follow_up else result.get("sql"),
        "chart_spec": None if follow_up else result.get("chart_spec"),
        "chart_note": None if follow_up else result.get("chart_note"),
        "rows": result.get("rows"),
        "evaluation_status": result.get("evaluation", {}).get("status"),
        "forecast_used": result.get("forecast_used", False),
        "forecast_requested": result.get("forecast_requested", result.get("forecast_used", False)),
        "forecast_chart": None if follow_up else forecast.get("chart_spec"),
        "forecast_rows": forecast.get("rows"), "forecast_meta": forecast.get("meta"),
        "context_version": result.get("context_version"),
        "interpreted_question": result.get("interpreted_question"),
        "followup_requested": follow_up,
    })
    st.rerun()


def render_chart_panel(message: dict | None) -> None:
    if not message or message.get("followup_requested"):
        return
    chart = message.get("forecast_chart") if message.get("forecast_used") else message.get("chart_spec")
    if not chart:
        if message.get("evaluation_status") == "data_only" and message.get("rows"):
            st.info(message.get("chart_note") or
                    "No chart is available for these results. Open the Data tab to inspect the values.")
        return
    with st.container(border=True):
        st.markdown('<div class="ds-chart-marker"></div><div class="ds-chart-head">'
                    '<div><strong><span class="ds-chart-icon">↗</span>Visual overview</strong>'
                    '<small>Based on the latest returned values</small></div></div>',
                    unsafe_allow_html=True)
        render_chart(chart)
        if message.get("chart_note"):
            st.caption(message["chart_note"])


if "messages" not in st.session_state:
    st.session_state.messages = []
if st.session_state.get("navigation") == "Insights":
    st.session_state.navigation = "Ask DataSage"

with st.sidebar:
    st.markdown('''<div class="ds-brand"><svg viewBox="0 0 56 52" aria-hidden="true">
      <defs><linearGradient id="ds-logo" x1="0" x2="1"><stop stop-color="#46e5d0"/>
      <stop offset="1" stop-color="#7b91fb"/></linearGradient></defs>
      <path d="M3 42c13-1 17-25 30-29 12-4 18 8 20 21-11-10-18-10-24-3C18 44 12 47 3 42Z"
      fill="url(#ds-logo)"/><path d="M24 42c11-22 17-34 26-34 4 0 6 3 5 8-6-5-11 6-17 17-4 8-8 12-14 9Z"
      fill="#66e6e2" opacity=".78"/></svg>DataSage</div>''', unsafe_allow_html=True)
    navigation = st.radio("Navigation", ["Ask DataSage", "Data", "SQL", "Settings"],
                          label_visibility="collapsed", key="navigation")
    st.markdown('<div class="ds-sidebar-foot">Ask questions in plain language.<br>'
                'Answers are checked against the available data.</div>', unsafe_allow_html=True)

answers = [item for item in st.session_state.messages if item.get("role") == "assistant"]
latest = answers[-1] if answers else None
st.markdown('<div class="ds-top"><h1>' + html.escape(navigation) + '</h1></div>',
            unsafe_allow_html=True)

if navigation == "Ask DataSage":
    render_conversation(answers)
    render_chart_panel(latest)
    if latest:
        render_metrics(latest.get("metrics"))
    render_question_input(answers)
elif navigation == "Data":
    if not latest:
        st.info("Run an analysis to inspect its returned values.")
    else:
        values = (latest.get("forecast_rows") if latest.get("forecast_requested")
                  else latest.get("rows")) or []
        if values:
            st.dataframe(values, width="stretch", hide_index=True)
        else:
            st.info("No values were returned for the latest question.")
elif navigation == "SQL":
    if not latest:
        st.info("Run an analysis to inspect its SQL query.")
    elif latest.get("forecast_requested") or latest.get("followup_requested"):
        st.info("SQL is not shown for forecast or follow-up questions.")
    elif latest.get("sql"):
        st.caption("Query for the latest question: " + latest.get("question", ""))
        st.code(latest["sql"], language="sql")
    else:
        st.info("No approved SQL query is available for the latest question.")
else:
    st.markdown("### Application settings")
    st.write(f"Answer model: {settings.openai_model}")
    st.write("Authentication: " + ("required" if settings.auth_required else "not required"))
    st.write(f"Maximum returned rows: {settings.sql_row_limit}")
    st.caption("Deployment settings are configured through environment variables.")
