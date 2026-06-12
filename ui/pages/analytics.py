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

df["timestamp"] = pd.to_datetime(df["timestamp"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total queries", len(df))
col2.metric("Avg latency", f"{df['latency_ms'].mean() / 1000:.1f}s")
col3.metric("Refusal rate", f"{df['is_refusal'].mean():.0%}")
col4.metric("Active sessions", df["session_id"].nunique())

st.subheader("Queries over time")
by_day = df.set_index("timestamp").resample("D").size().rename("queries")
st.line_chart(by_day)

st.subheader("Top questions")
top_questions = df["query"].value_counts().head(10).rename("count")
st.dataframe(top_questions, use_container_width=True)

st.subheader("Most retrieved sources")
source_counts: dict = {}
for sources_json in df["sources_json"]:
    for src in json.loads(sources_json):
        name = src.get("source", "unknown")
        source_counts[name] = source_counts.get(name, 0) + 1
if source_counts:
    source_df = pd.Series(source_counts, name="retrievals").sort_values(ascending=False)
    st.bar_chart(source_df)
else:
    st.write("No sources retrieved yet.")

st.subheader("Recent queries")
st.dataframe(
    df[["timestamp", "session_id", "query", "latency_ms", "num_sources", "is_refusal"]]
    .sort_values("timestamp", ascending=False)
    .head(50),
    use_container_width=True,
)
