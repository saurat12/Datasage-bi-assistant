import streamlit as st

from bi import ask_bi_agent

st.set_page_config(
    page_title="DataSage",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #08111f;
    --panel: rgba(15, 23, 39, 0.78);
    --panel-2: rgba(20, 30, 50, 0.88);
    --text: #edf2f7;
    --muted: #94a3b8;
    --border: rgba(255,255,255,0.08);
    --accent: #f5b942;
    --accent-soft: rgba(245, 185, 66, 0.14);
    --user-bg: linear-gradient(180deg, #1a2440, #141d33);
    --assistant-bg: linear-gradient(180deg, #10192a, #0d1524);
    --shadow: 0 16px 45px rgba(0,0,0,0.35);
    --radius-lg: 24px;
    --radius-md: 16px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background:
        radial-gradient(circle at top left, rgba(245,185,66,0.12), transparent 22%),
        radial-gradient(circle at top right, rgba(73,127,255,0.10), transparent 20%),
        linear-gradient(180deg, #08111f 0%, #060c17 100%);
    color: var(--text);
}

.stApp { background: transparent; }

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 1140px;
    padding-top: 1.2rem;
    padding-bottom: 9rem;
}

.datasage-shell {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
    border: 1px solid var(--border);
    border-radius: 28px;
    box-shadow: var(--shadow);
    overflow: hidden;
    backdrop-filter: blur(14px);
}

.datasage-header {
    position: sticky;
    top: 0;
    z-index: 20;
    padding: 1.2rem 1.5rem 1rem 1.5rem;
    background: linear-gradient(180deg, rgba(10,16,28,0.95), rgba(10,16,28,0.72));
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.datasage-toprow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.datasage-brand {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.datasage-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    line-height: 1;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.4px;
}

.datasage-title span { color: var(--accent); }

.datasage-caption {
    margin: 0;
    font-size: 0.92rem;
    color: var(--muted);
}

.datasage-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(245,185,66,0.24);
    background: rgba(245,185,66,0.08);
    color: #f8d27b;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.datasage-body { padding: 1rem 1rem 1.2rem 1rem; }

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
}

[data-testid="stChatMessageContent"] {
    border-radius: 18px !important;
    padding: 1rem 1.1rem !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 10px 26px rgba(0,0,0,0.18);
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: var(--assistant-bg) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--user-bg) !important;
    border: 1px solid rgba(245,185,66,0.16) !important;
}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
    color: #dbe4f0 !important;
    font-size: 0.98rem !important;
    line-height: 1.7 !important;
}

[data-testid="stExpander"] {
    background: linear-gradient(180deg, #0f1727, #0c1322) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    margin-top: 0.8rem !important;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    font-size: 0.78rem !important;
    color: #9fb0c7 !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

.stCode {
    background: #09111d !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    font-size: 0.86rem !important;
}

[data-testid="stChatInput"] {
    position: fixed;
    left: 50%;
    bottom: 1.15rem;
    transform: translateX(-50%);
    width: min(980px, calc(100% - 2rem));
    background: rgba(12, 18, 32, 0.82) !important;
    border: 1px solid rgba(245,185,66,0.22) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(16px);
    box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}

[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    font-size: 0.98rem !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stChatInput"] textarea::placeholder { color: #7f8da3 !important; }

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(245,185,66,0.5) !important;
    box-shadow: 0 0 0 4px rgba(245,185,66,0.08), 0 18px 50px rgba(0,0,0,0.35) !important;
}

/* Forecast control row, sits just above the chat input */
.forecast-control {
    position: fixed;
    left: 50%;
    bottom: 5.6rem;
    transform: translateX(-50%);
    width: min(980px, calc(100% - 2rem));
    z-index: 25;
    display: flex;
    justify-content: flex-end;
    pointer-events: none;
}
.forecast-control > div {
    background: rgba(12,18,32,0.78);
    border: 1px solid rgba(245,185,66,0.22);
    backdrop-filter: blur(14px);
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    pointer-events: auto;
    box-shadow: 0 10px 28px rgba(0,0,0,0.3);
}
.forecast-control [data-testid="stCheckbox"] label {
    color: #f8d27b !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}
.forecast-control [data-testid="stCheckbox"] label p { color: #f8d27b !important; }

.forecast-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    background: rgba(245,185,66,0.14);
    border: 1px solid rgba(245,185,66,0.32);
    color: #f8d27b;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.6rem;
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.25); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.4); }

@media (max-width: 768px) {
    .block-container { padding-top: 0.8rem; padding-bottom: 9.5rem; }
    .datasage-header { padding: 1rem 0.9rem; }
    .datasage-title { font-size: 1.65rem; }
    .datasage-caption { font-size: 0.84rem; }
    [data-testid="stChatMessageContent"] { padding: 0.9rem 0.95rem !important; border-radius: 16px !important; }
    .forecast-control { bottom: 5.2rem; }
}
</style>

<div class="datasage-shell">
    <div class="datasage-header">
        <div class="datasage-toprow">
            <div class="datasage-brand">
                <p class="datasage-title">Data<span>Sage</span></p>
                <p class="datasage-caption">Turning your data into actionable insights</p>
            </div>
            <div class="datasage-badge">● BI Copilot</div>
        </div>
    </div>
    <div class="datasage-body">
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_chart(spec: dict):
    """Render a Vega-Lite spec directly using st.vega_lite_chart."""
    try:
        col1, col2, col3 = st.columns([0.08, 1, 0.08])
        with col2:
            st.vega_lite_chart(spec, use_container_width=True)
    except Exception as e:
        print("RENDER ERROR:", e)
        st.warning(f"Could not render chart: {e}")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
# ---------------------------------------------------------------------------
# Render history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("forecast_used"):
            st.markdown('<div class="forecast-pill">🔮 FORECAST</div>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("sql") and not msg.get("forecast_used"):
            with st.expander("SQL Query"):
                st.code(msg["sql"], language="sql")
        if msg.get("chart_spec"):
            render_chart(msg["chart_spec"])
        if msg.get("forecast_chart"):
            render_chart(msg["forecast_chart"])
        if msg.get("forecast_accuracy"):
            st.caption(f"📏 {msg['forecast_accuracy']}")


# ---------------------------------------------------------------------------
# Chat input + main loop
# ---------------------------------------------------------------------------

question = st.chat_input("Ask a question about your data...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            # Forecast intent is owned exclusively by the orchestrator in
            # bi.py; the presentation layer never forces or re-derives it.
            result = ask_bi_agent(question)

        forecast_used = result.get("forecast_used", False)
        forecast = result.get("forecast")
        forecast_error = result.get("forecast_error")

        if forecast_used:
            st.markdown('<div class="forecast-pill">🔮 FORECAST</div>', unsafe_allow_html=True)
        st.markdown(result["summary"])

        if result.get("sql") and not forecast_used:
            with st.expander("SQL Query"):
                st.code(result["sql"], language="sql")

        # Forecast responses have their own combined history + forecast chart.
        if result.get("chart_spec") and not forecast_used:
            render_chart(result["chart_spec"])

        if forecast is not None:
            st.markdown("---")
            if forecast.get("chart_spec"):
                render_chart(forecast["chart_spec"])
            accuracy = forecast.get("meta", {}).get("accuracy")
            if accuracy is not None:
                st.caption(f"📏 {accuracy['summary']}")
        elif forecast_error:
            st.markdown("---")
            st.markdown(forecast_error)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["summary"],
        "sql": result.get("sql"),
        "chart_spec": result.get("chart_spec"),
        "forecast_used": forecast_used,
        "forecast_chart": forecast.get("chart_spec") if forecast else None,
        "forecast_accuracy": (
            forecast.get("meta", {}).get("accuracy", {}).get("summary")
            if forecast else None
        ),
    })

st.markdown("</div></div>", unsafe_allow_html=True)
