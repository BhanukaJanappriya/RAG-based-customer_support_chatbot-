"""Unit tests for app.analytics (local SQLite usage logging)."""

import json
import sqlite3

import pytest

from app import analytics
from app.config import settings
from app.generation.prompt import REFUSAL_PHRASE


@pytest.fixture(autouse=True)
def temp_db_path(tmp_path, monkeypatch):
    """Point settings.analytics_db_path at a temp file for each test."""
    db_file = tmp_path / "analytics.db"
    monkeypatch.setattr(settings, "analytics_db_path", str(db_file))
    return db_file


class TestInitDb:
    def test_creates_db_file(self, temp_db_path):
        analytics.init_db()
        assert temp_db_path.exists()

    def test_creates_queries_table(self, temp_db_path):
        analytics.init_db()
        with sqlite3.connect(temp_db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        assert ("queries",) in tables


class TestLogQuery:
    def test_inserts_row(self, temp_db_path):
        sources = [{"source": "handbook.pdf", "page": "3"}]
        analytics.log_query("session-1", "What is the return policy?", "30 days.", 123.4, sources)

        with sqlite3.connect(temp_db_path) as conn:
            rows = conn.execute("SELECT * FROM queries").fetchall()
        assert len(rows) == 1

    def test_records_refusal(self, temp_db_path):
        analytics.log_query("session-1", "What's the weather?", REFUSAL_PHRASE, 50.0, [])

        with sqlite3.connect(temp_db_path) as conn:
            row = conn.execute("SELECT is_refusal FROM queries").fetchone()
        assert row[0] == 1

    def test_non_refusal_not_flagged(self, temp_db_path):
        analytics.log_query("session-1", "What is the return policy?", "30 days.", 50.0, [])

        with sqlite3.connect(temp_db_path) as conn:
            row = conn.execute("SELECT is_refusal FROM queries").fetchone()
        assert row[0] == 0

    def test_sources_round_trip(self, temp_db_path):
        sources = [{"source": "handbook.pdf", "page": "3"}, {"source": "faq.md", "page": "1"}]
        analytics.log_query("session-1", "q", "a", 50.0, sources)

        with sqlite3.connect(temp_db_path) as conn:
            row = conn.execute("SELECT num_sources, sources_json FROM queries").fetchone()
        assert row[0] == 2
        assert json.loads(row[1]) == sources

    def test_does_not_raise_on_db_error(self, tmp_path, monkeypatch):
        """Logging must never break the chat response, even if the DB write fails."""
        # A regular file where a directory is expected makes mkdir(parents=True) fail.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        monkeypatch.setattr(settings, "analytics_db_path", str(blocker / "analytics.db"))
        analytics.log_query("session-1", "q", "a", 50.0, [])  # should not raise
