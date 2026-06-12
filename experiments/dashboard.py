"""Standalone results dashboard for the DS experiments workstream.

Reads the JSON results + plots already produced by the embedding,
retrieval, and prompt ablation scripts and renders them as tables/charts.
Does not touch the production app or ChromaDB — read-only.

Run with:
    streamlit run experiments/dashboard.py --server.port 8502
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

st.set_page_config(page_title="DS Experiments Dashboard", page_icon="\U0001F9EA", layout="wide")

st.title("DS Experiments Dashboard")
st.caption("Read-only view of embedding, retrieval, and prompt ablation results. Does not affect the production chatbot.")

tab_emb, tab_ret, tab_prompt, tab_findings = st.tabs(
    ["1. Embedding Model", "2. Chunking & Retrieval", "3. Prompt Template", "Recommendations"]
)

# ── Tab 1: Embedding comparison ──────────────────────────────────────────
with tab_emb:
    summary_path = RESULTS / "embedding_comparison_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        rows = data if isinstance(data, list) else data.get("results", [])
        df = pd.DataFrame(rows)
        cols = ["model_name", "dim", "ndcg_at_5", "ndcg_at_5_ci_lower", "ndcg_at_5_ci_upper", "mrr", "mean_query_latency_ms"]
        df = df[[c for c in cols if c in df.columns]].sort_values("ndcg_at_5", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No embedding comparison results found. Run `run_embedding_comparison.py` first.")

    plot_path = PLOTS / "embedding_pareto.png"
    if plot_path.exists():
        st.image(str(plot_path), caption="Embedding model Pareto plot (quality vs. latency)")

# ── Tab 2: Retrieval ablation ────────────────────────────────────────────
with tab_ret:
    summary_path = RESULTS / "retrieval" / "retrieval_ablation_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())

        st.subheader("Chunking strategies")
        df_chunk = pd.DataFrame(data.get("chunking_results", []))
        df_chunk = df_chunk.sort_values("ndcg_at_5", ascending=False)
        st.dataframe(df_chunk, use_container_width=True, hide_index=True)
        st.caption(f"Best chunking strategy: **{data.get('best_chunking', 'n/a')}**")

        st.subheader("Retrieval methods")
        df_method = pd.DataFrame(data.get("retrieval_results", []))
        st.dataframe(df_method, use_container_width=True, hide_index=True)
    else:
        st.info("No retrieval ablation results found. Run `run_retrieval_ablation.py` first.")

    plot_path = PLOTS / "retrieval_comparison.png"
    if plot_path.exists():
        st.image(str(plot_path), caption="Retrieval strategy comparison")

# ── Tab 3: Prompt ablation ───────────────────────────────────────────────
with tab_prompt:
    summary_path = RESULTS / "prompts" / "prompt_ablation_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        df = pd.DataFrame([r for r in data if "error" not in r])
        if "_score" not in df.columns:
            df["_score"] = (
                0.4 * df.get("faithfulness_heuristic_mean", 0)
                + 0.3 * df.get("citation_rate", 0)
                + 0.3 * df.get("true_refusal_rate", 0)
            )
        cols = [
            "template_name", "faithfulness_heuristic_mean", "citation_rate",
            "true_refusal_rate", "false_refusal_rate", "mean_latency_ms", "_score",
        ]
        df = df[[c for c in cols if c in df.columns]].sort_values("_score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No prompt ablation results found. Run `run_prompt_ablation.py` first.")

    plot_path = PLOTS / "prompt_comparison.png"
    if plot_path.exists():
        st.image(str(plot_path), caption="Prompt template comparison")

# ── Tab 4: Findings / recommendations ────────────────────────────────────
with tab_findings:
    findings_path = ROOT / "findings.md"
    if findings_path.exists():
        st.markdown(findings_path.read_text(encoding="utf-8"))
    else:
        st.info("findings.md not found.")
