"""Lightweight local usage logging for the production chatbot.

Each completed chat turn is recorded to a SQLite file so the Streamlit
analytics page can show basic usage stats (query volume, latency, refusal
rate, popular questions/sources). Logging failures are caught and logged,
never raised — analytics must never break the chat response.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import List

from app.config import settings
from app.generation.prompt import REFUSAL_PHRASE

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    response_length INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    num_sources INTEGER NOT NULL,
    sources_json TEXT NOT NULL,
    is_refusal INTEGER NOT NULL
)
"""


def init_db() -> None:
    """Create the analytics database and table if they don't exist."""
    db_path = settings.analytics_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_SCHEMA)


def log_query(
    session_id: str,
    query: str,
    response: str,
    latency_ms: float,
    sources: List[dict],
) -> None:
    """Record a completed chat turn for analytics.

    Args:
        session_id: Session identifier.
        query: User question.
        response: Full assistant response text.
        latency_ms: End-to-end time from request start to stream completion.
        sources: Source payload from the ``sources`` SSE event
            (``[{"source": ..., "page": ..., ...}, ...]``).
    """
    try:
        init_db()
        with sqlite3.connect(settings.analytics_path) as conn:
            conn.execute(
                "INSERT INTO queries "
                "(timestamp, session_id, query, response_length, latency_ms, "
                "num_sources, sources_json, is_refusal) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    session_id,
                    query,
                    len(response),
                    latency_ms,
                    len(sources),
                    json.dumps(sources),
                    int(response.strip() == REFUSAL_PHRASE),
                ),
            )
    except Exception:
        logger.exception(f"Failed to log analytics for session {session_id}")
