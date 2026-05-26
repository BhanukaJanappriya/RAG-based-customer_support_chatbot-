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
# WhatsApp-style CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>

    /* ── App background ─────────────────────────────────────────────────── */
    .stApp {
        background-color: #000000;
    }

    /* ── Chat message base reset ─────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 2px 16px !important;
        gap: 8px !important;
        align-items: flex-end !important;
    }

    /* ── Avatars — small circles ─────────────────────────────────────────── */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        width: 30px !important;
        height: 30px !important;
        min-width: 30px !important;
        border-radius: 50% !important;
        overflow: hidden !important;
        flex-shrink: 0 !important;
    }

    /* ── ASSISTANT messages — LEFT side (white bubble) ──────────────────── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        flex-direction: row;
        justify-content: flex-start;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
        > div:last-child {
        background: #ffffff;
        border-radius: 0px 16px 16px 16px;
        padding: 8px 12px 8px 12px;
        max-width: 62%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.13);
        align-items: flex-start;
    }

    /* Remove extra margins inside assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
        > div:last-child p {
        margin: 0 !important;
        line-height: 1.45;
    }

    /* ── USER messages — RIGHT side (green bubble) ───────────────────────── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        justify-content: flex-start;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
        > div:last-child {
        background: #dcf8c6;
        border-radius: 16px 0px 16px 16px;
        padding: 8px 12px 8px 12px;
        max-width: 62%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.13);
        align-items: flex-end;
    }

    /* Remove extra margins inside user bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
        > div:last-child p {
        margin: 0 !important;
        line-height: 1.45;
    }

    /* ── Remove stray Streamlit default backgrounds ──────────────────────── */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        background: transparent !important;
    }

    /* ── Typing cursor blink ─────────────────────────────────────────────── */
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0; }
    }
    .cursor {
        display: inline-block;
        width: 8px;
        height: 1em;
        background: #333;
        vertical-align: text-bottom;
        margin-left: 2px;
        border-radius: 1px;
        animation: blink 0.85s step-start infinite;
    }

    /* ── Chat input bar ──────────────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        background: #f0f0f0;
        border-radius: 24px !important;
    }

    /* ── Title area ──────────────────────────────────────────────────────── */
    h1 { margin-bottom: 4px !important; }

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
    """Stream a response from the API with live thinking steps and cursor effect.

    Returns:
        (full response text, list of source dicts)
    """
    url = f"{settings.streamlit_api_url}/chat"
    sources: List[dict] = []
    full_text = ""
    first_token = True

    # Thinking panel — user can expand / collapse at any time
    status = st.status("Thinking…", expanded=True)
    # Streaming text appears below the thinking panel inside the bubble
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
                        status.update(
                            label="Generating…",
                            state="running",
                            expanded=True,
                        )
                        with status:
                            st.markdown("✍️ Writing answer…")
                        first_token = False

                    full_text += event["content"]
                    text_slot.markdown(
                        full_text + '<span class="cursor"></span>',
                        unsafe_allow_html=True,
                    )

                elif event["type"] == "sources":
                    sources = event["content"]

                elif event["type"] == "done":
                    text_slot.markdown(full_text)
                    status.update(
                        label="Done",
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
            st.error("Cannot reach the API server. Run `uvicorn app.api.main:app --reload`.")
        full_text = "API server is unreachable."

    except Exception as exc:
        if "RuntimeError" not in type(exc).__name__:
            status.update(label="Error", state="error", expanded=True)
            with status:
                st.error(str(exc))
        full_text = full_text or f"Error: {exc}"

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
if prompt := st.chat_input("Type a message…"):
    # User message — LEFT side
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message — RIGHT side
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
