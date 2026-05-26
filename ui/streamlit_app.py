"""Streamlit chat interface for the RAG customer support chatbot."""
# Ensure the project root is importable when this script is run directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import uuid
from typing import Generator, List

import requests
import streamlit as st

from app.config import settings

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Support Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str, "sources": list}]
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stream_chat(query: str, session_id: str) -> Generator[str, None, None]:
    """Consume the FastAPI SSE stream and yield raw text tokens.

    As a side-effect, populates ``st.session_state.last_sources`` once
    the sources frame arrives, so the UI can render them after the stream.

    Args:
        query: The user's question.
        session_id: Current conversation session identifier.

    Yields:
        Partial response text tokens as they arrive.
    """
    url = f"{settings.streamlit_api_url}/chat"
    sources: List[dict] = []

    with requests.post(
        url,
        json={"query": query, "session_id": session_id},
        stream=True,
        timeout=180,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line or not raw_line.startswith(b"data: "):
                continue
            event = json.loads(raw_line[6:])
            if event["type"] == "token":
                yield event["content"]
            elif event["type"] == "sources":
                sources = event["content"]
            elif event["type"] == "error":
                raise RuntimeError(event["content"])

    # Runs after generator is fully exhausted by st.write_stream
    st.session_state.last_sources = sources


def fetch_health() -> dict:
    """Fetch the /health endpoint; return empty dict on failure."""
    try:
        resp = requests.get(f"{settings.streamlit_api_url}/health", timeout=4)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.caption("Powered by Llama 3.2 1B · ChromaDB · all-MiniLM-L6-v2")

    st.divider()

    # System status
    health = fetch_health()
    if health:
        st.success("API connected", icon="✅")
        st.caption(f"**Model:** {health.get('llm_model', '—')}")
        st.caption(f"**Chunks in store:** {health.get('document_count', '—')}")
    else:
        st.error("API unreachable — is the FastAPI server running?", icon="🔴")

    st.divider()

    # Session controls
    st.subheader("Session")
    st.caption(f"ID: `{st.session_state.session_id[:12]}…`")
    if st.button("🗑️ New conversation", use_container_width=True):
        # Clear history on the server too
        try:
            requests.delete(
                f"{settings.streamlit_api_url}/chat/{st.session_state.session_id}",
                timeout=4,
            )
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    st.divider()

    # Retrieved chunks debug panel
    st.subheader("Retrieved Context")
    if st.session_state.last_sources:
        for i, chunk in enumerate(st.session_state.last_sources, start=1):
            label = f"Chunk {i} · {chunk['source']} p.{chunk['page']}"
            with st.expander(label, expanded=False):
                st.text(chunk["content"])
    else:
        st.caption("Ask a question to see which chunks were retrieved.")


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title("Customer Support Assistant")
st.caption(
    "Ask anything about our products or policies. "
    "Answers are grounded in the uploaded knowledge base."
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])} chunk(s))", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}** — p.{src['page']}")
                    st.code(src["content"], language=None)

# Chat input
if prompt := st.chat_input("Type your question here…"):
    # Append and display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the assistant response
    with st.chat_message("assistant"):
        try:
            response_text = st.write_stream(
                stream_chat(prompt, st.session_state.session_id)
            )
            retrieved_sources = st.session_state.last_sources

            if retrieved_sources:
                with st.expander(
                    f"📄 Sources ({len(retrieved_sources)} chunk(s))", expanded=False
                ):
                    for src in retrieved_sources:
                        st.markdown(f"**{src['source']}** — p.{src['page']}")
                        st.code(src["content"], language=None)

        except requests.exceptions.ConnectionError:
            response_text = (
                "Cannot reach the API server. "
                "Run `uvicorn app.api.main:app --reload` and refresh."
            )
            retrieved_sources = []
            st.error(response_text)
        except Exception as exc:
            response_text = f"Error: {exc}"
            retrieved_sources = []
            st.error(response_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": retrieved_sources,
        }
    )
