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
# Custom CSS — thinking animation + chat polish
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes thinking-pulse {
        0%   { opacity: 1; }
        50%  { opacity: 0.35; }
        100% { opacity: 1; }
    }
    .thinking-box {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        background: #f0f2f6;
        border-left: 3px solid #4a90d9;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #444;
        animation: thinking-pulse 1.4s ease-in-out infinite;
        margin-bottom: 6px;
    }
    .thinking-box .dot-row {
        display: flex;
        gap: 4px;
    }
    .thinking-box .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #4a90d9;
        animation: thinking-pulse 1.4s ease-in-out infinite;
    }
    .thinking-box .dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-box .dot:nth-child(3) { animation-delay: 0.4s; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _thinking_html(label: str) -> str:
    return (
        f'<div class="thinking-box">'
        f'<span>{label}</span>'
        f'<div class="dot-row">'
        f'<div class="dot"></div><div class="dot"></div><div class="dot"></div>'
        f'</div></div>'
    )


def stream_chat(
    query: str,
    session_id: str,
    thinking_slot,
) -> Generator[str, None, None]:
    """Consume the FastAPI SSE stream and yield raw text tokens.

    Displays animated thinking steps in ``thinking_slot`` until the first
    token arrives, then clears the slot so the streamed text takes over.

    Args:
        query: The user's question.
        session_id: Current conversation session identifier.
        thinking_slot: A ``st.empty()`` placeholder for the thinking indicator.

    Yields:
        Partial response text tokens as they arrive.
    """
    url = f"{settings.streamlit_api_url}/chat"
    sources: List[dict] = []
    first_token = True

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

            if event["type"] == "thinking":
                thinking_slot.markdown(
                    _thinking_html(event["content"]),
                    unsafe_allow_html=True,
                )
            elif event["type"] == "token":
                if first_token:
                    thinking_slot.empty()
                    first_token = False
                yield event["content"]
            elif event["type"] == "sources":
                sources = event["content"]
            elif event["type"] == "error":
                thinking_slot.empty()
                raise RuntimeError(event["content"])

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
    st.caption("Powered by Llama 3.2 · ChromaDB · all-MiniLM-L6-v2")

    st.divider()

    health = fetch_health()
    if health:
        st.success("API connected", icon="✅")
        st.caption(f"**Model:** {health.get('llm_model', '—')}")
        st.caption(f"**Chunks in store:** {health.get('document_count', '—')}")
    else:
        st.error("API unreachable — is the FastAPI server running?", icon="🔴")

    st.divider()

    st.subheader("Session")
    st.caption(f"ID: `{st.session_state.session_id[:12]}…`")
    if st.button("🗑️ New conversation", use_container_width=True):
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
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Thinking indicator — cleared automatically when first token arrives
        thinking_slot = st.empty()
        thinking_slot.markdown(_thinking_html("Thinking…"), unsafe_allow_html=True)

        try:
            response_text = st.write_stream(
                stream_chat(prompt, st.session_state.session_id, thinking_slot)
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
            thinking_slot.empty()
            response_text = (
                "Cannot reach the API server. "
                "Run `uvicorn app.api.main:app --reload` and refresh."
            )
            retrieved_sources = []
            st.error(response_text)
        except Exception as exc:
            thinking_slot.empty()
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
