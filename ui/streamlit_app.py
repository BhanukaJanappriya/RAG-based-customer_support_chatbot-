"""Streamlit chat interface for the RAG customer support chatbot."""
# Ensure the project root is importable when this script is run directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import uuid
from typing import List

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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── User messages: right side ─────────────────────────────────────── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
    }

    /* Push user bubble text to the right */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
        > div:last-child {
        align-items: flex-end;
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
        [data-testid="stMarkdownContainer"] p {
        background: #1a73e8;
        color: #ffffff;
        border-radius: 18px 4px 18px 18px;
        padding: 10px 16px;
        display: inline-block;
        max-width: 100%;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Assistant bubble ───────────────────────────────────────────────── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
        > div:last-child {
        align-items: flex-start;
        max-width: 80%;
    }

    /* ── Typing cursor blink ────────────────────────────────────────────── */
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0; }
    }
    .cursor {
        display: inline-block;
        width: 9px;
        height: 1.1em;
        background: #444;
        vertical-align: text-bottom;
        margin-left: 2px;
        border-radius: 1px;
        animation: blink 0.85s step-start infinite;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
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

def fetch_health() -> dict:
    try:
        r = requests.get(f"{settings.streamlit_api_url}/health", timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def render_response(prompt: str, session_id: str) -> tuple[str, List[dict]]:
    """Connect to the SSE stream, show thinking steps and stream the reply.

    Uses st.status() so the user can watch every thinking step live and
    re-open the panel after the response is done. The reply text is
    rendered character-by-character with a blinking cursor.

    Returns:
        (full response text, list of source dicts)
    """
    url = f"{settings.streamlit_api_url}/chat"
    sources: List[dict] = []
    full_text = ""
    first_token = True

    # Thinking panel — user can expand / collapse at any time
    status = st.status("Thinking…", expanded=True)
    # Streaming text slot — lives below the status widget
    text_slot = st.empty()

    try:
        with requests.post(
            url,
            json={"query": prompt, "session_id": session_id},
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line or not raw_line.startswith(b"data: "):
                    continue
                event = json.loads(raw_line[6:])

                if event["type"] == "thinking":
                    with status:
                        st.markdown(f"🔍 {event['content']}")

                elif event["type"] == "token":
                    if first_token:
                        # Transition the status panel to "generating" state
                        status.update(
                            label="Generating response…",
                            state="running",
                            expanded=True,
                        )
                        with status:
                            st.markdown("✍️ Writing answer…")
                        first_token = False

                    full_text += event["content"]
                    # Show text with a blinking cursor while streaming
                    text_slot.markdown(
                        full_text + '<span class="cursor"></span>',
                        unsafe_allow_html=True,
                    )

                elif event["type"] == "sources":
                    sources = event["content"]

                elif event["type"] == "done":
                    # Final render — remove cursor, collapse status
                    text_slot.markdown(full_text)
                    status.update(
                        label="Response complete",
                        state="complete",
                        expanded=False,
                    )

                elif event["type"] == "error":
                    status.update(label="Error", state="error", expanded=True)
                    with status:
                        st.error(event["content"])
                    raise RuntimeError(event["content"])

    except requests.exceptions.ConnectionError:
        status.update(label="Connection failed", state="error", expanded=True)
        with status:
            st.error(
                "Cannot reach the API server. "
                "Run `uvicorn app.api.main:app --reload` and refresh."
            )
        full_text = "⚠️ API server is unreachable."

    except Exception as exc:
        if "RuntimeError" not in type(exc).__name__:
            status.update(label="Error", state="error", expanded=True)
            with status:
                st.error(str(exc))
        full_text = full_text or f"⚠️ {exc}"

    st.session_state.last_sources = sources
    return full_text, sources


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
            with st.expander(f"Chunk {i} · {chunk['source']} p.{chunk['page']}"):
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
            with st.expander(f"📄 Sources ({len(msg['sources'])} chunk(s))"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}** — p.{src['page']}")
                    st.code(src["content"], language=None)

# Chat input
if prompt := st.chat_input("Type your question here…"):
    # User message — rendered on the RIGHT via CSS
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message — rendered on the LEFT
    with st.chat_message("assistant"):
        response_text, retrieved_sources = render_response(
            prompt, st.session_state.session_id
        )

        if retrieved_sources:
            with st.expander(f"📄 Sources ({len(retrieved_sources)} chunk(s))"):
                for src in retrieved_sources:
                    st.markdown(f"**{src['source']}** — p.{src['page']}")
                    st.code(src["content"], language=None)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "sources": retrieved_sources}
    )
