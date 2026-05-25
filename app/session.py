"""In-memory conversation history store, keyed by session_id.

For a single-worker deployment this is fine. If you scale to multiple
uvicorn workers, swap this for a Redis-backed store.
"""

import logging
from collections import defaultdict
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

# Capped at 10 turns (20 messages) to avoid unbounded context growth
_MAX_TURNS = 10
_sessions: Dict[str, List[BaseMessage]] = defaultdict(list)


def get_history(session_id: str) -> List[BaseMessage]:
    """Return the last _MAX_TURNS turns for a session."""
    history = _sessions[session_id]
    max_msgs = _MAX_TURNS * 2
    return history[-max_msgs:] if len(history) > max_msgs else list(history)


def add_to_history(session_id: str, user_query: str, assistant_response: str) -> None:
    """Append a completed turn to the session history."""
    _sessions[session_id].append(HumanMessage(content=user_query))
    _sessions[session_id].append(AIMessage(content=assistant_response))
    logger.debug(f"Session {session_id}: {len(_sessions[session_id])} messages stored")


def clear_history(session_id: str) -> None:
    """Wipe history for a session (e.g., when the user starts a new chat)."""
    _sessions[session_id] = []
    logger.info(f"Cleared history for session {session_id}")


def list_sessions() -> List[str]:
    """Return all active session IDs (useful for monitoring)."""
    return list(_sessions.keys())
