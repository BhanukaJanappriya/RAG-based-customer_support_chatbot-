"""Production usage analytics — read-only view of data/analytics.db.

Auto-discovered by Streamlit as a second page alongside
ui/streamlit_app.py. Reflects live chatbot usage (queries logged by
app/api/routes.py), not the offline DS experiment results in
experiments/dashboard.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import sqlite3

import pandas as pd
import streamlit as st

from app.config import settings

st.set_page_config(page_title="Usage Analytics", page_icon="\U0001F4CA", layout="wide")

st.title("Usage Analytics")
st.caption("Live production chatbot activity, logged from the /chat endpoint.")

db_path = settings.analytics_path
if not db_path.exists():
    st.info("No queries logged yet. Chat with the bot on the main page to generate data.")
    st.stop()

with sqlite3.connect(db_path) as conn:
    df = pd.read_sql_query("SELECT * FROM queries ORDER BY timestamp", conn)

if df.empty:
    st.info("No queries logged yet. Chat with the bot on the main page to generate data.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# ---------------------------------------------------------------------------
# Time range filter
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Filters")
    time_range = st.radio(
        "Time range",
        ["Today", "Last 7 days", "Last 30 days", "All time"],
        index=3,
    )

now = pd.Timestamp.now(tz="UTC")
if time_range == "Today":
    cutoff = now.normalize()
elif time_range == "Last 7 days":
    cutoff = now - pd.Timedelta(days=7)
elif time_range == "Last 30 days":
    cutoff = now - pd.Timedelta(days=30)
else:
    cutoff = None

filtered = df if cutoff is None else df[df["timestamp"] >= cutoff]

if filtered.empty:
    st.info(f"No queries logged in the selected range ({time_range}).")
    st.stop()

# ---------------------------------------------------------------------------
# KPI summary
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total queries", len(filtered))
col2.metric("Avg latency", f"{filtered['latency_ms'].mean() / 1000:.1f}s")
col3.metric("p95 latency", f"{filtered['latency_ms'].quantile(0.95) / 1000:.1f}s")
col4.metric("Refusal rate", f"{filtered['is_refusal'].mean():.0%}")
col5.metric("Active sessions", filtered["session_id"].nunique())

# ---------------------------------------------------------------------------
# Activity over time
# ---------------------------------------------------------------------------
st.subheader("Queries over time")
by_day = filtered.set_index("timestamp").resample("D").size().rename("queries")
st.line_chart(by_day)

st.subheader("Average latency over time")
latency_by_day = (
    filtered.set_index("timestamp")
    .resample("D")["latency_ms"]
    .mean()
    .rename("avg_latency_ms")
)
st.line_chart(latency_by_day)

# ---------------------------------------------------------------------------
# Top questions / sources
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top questions")
    top_questions = filtered["query"].value_counts().head(10).rename("count")
    st.dataframe(top_questions, width="stretch")

with col_right:
    st.subheader("Most retrieved sources")
    source_counts: dict = {}
    for sources_json in filtered["sources_json"]:
        for src in json.loads(sources_json):
            name = src.get("source", "unknown")
            source_counts[name] = source_counts.get(name, 0) + 1
    if source_counts:
        source_df = pd.Series(source_counts, name="retrievals").sort_values(ascending=False)
        st.bar_chart(source_df)
    else:
        st.write("No sources retrieved yet.")

# ---------------------------------------------------------------------------
# Refusal analysis — surfaces possible knowledge base gaps
# ---------------------------------------------------------------------------
st.subheader("Refused queries")
st.caption("Questions the bot couldn't answer — useful for spotting knowledge base gaps.")
refusals = (
    filtered[filtered["is_refusal"] == 1][["timestamp", "session_id", "query"]]
    .sort_values("timestamp", ascending=False)
)
if refusals.empty:
    st.write("No refusals in this range.")
else:
    st.dataframe(refusals, width="stretch")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.download_button(
    "Download filtered data as CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="usage_analytics.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Session drill-down
# ---------------------------------------------------------------------------
st.subheader("Session drill-down")
session_ids = sorted(filtered["session_id"].unique().tolist())
selected_session = st.selectbox("Session ID", session_ids)
session_df = filtered[filtered["session_id"] == selected_session].sort_values("timestamp")
st.dataframe(
    session_df[["timestamp", "query", "latency_ms", "num_sources", "is_refusal"]],
    width="stretch",
)

# ---------------------------------------------------------------------------
# Recent queries
# ---------------------------------------------------------------------------
st.subheader("Recent queries")
st.dataframe(
    filtered[["timestamp", "session_id", "query", "latency_ms", "num_sources", "is_refusal"]]
    .sort_values("timestamp", ascending=False)
    .head(50),
    width="stretch",
)
