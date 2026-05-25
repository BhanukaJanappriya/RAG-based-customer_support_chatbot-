"""Unit tests for the retrieval layer."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.generation.chain import format_context


# ---------------------------------------------------------------------------
# format_context tests (pure function, no mocking needed)
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_empty_list_returns_fallback(self):
        result = format_context([])
        assert "No relevant context" in result

    def test_single_doc_includes_source_and_page(self):
        doc = Document(
            page_content="Refunds are processed within 5 business days.",
            metadata={"source": "policy.pdf", "page": 4},
        )
        result = format_context([doc])
        assert "policy.pdf" in result
        assert "p.4" in result
        assert "Refunds are processed" in result

    def test_multiple_docs_separated_by_divider(self):
        docs = [
            Document(page_content="Section A.", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="Section B.", metadata={"source": "b.pdf", "page": 2}),
        ]
        result = format_context(docs)
        assert "---" in result
        assert "a.pdf" in result
        assert "b.pdf" in result

    def test_missing_metadata_defaults_gracefully(self):
        doc = Document(page_content="Some content.", metadata={})
        result = format_context([doc])
        assert "unknown" in result
        assert "?" in result


# ---------------------------------------------------------------------------
# retrieve() tests (vector store mocked)
# ---------------------------------------------------------------------------

class TestRetrieve:
    @patch("app.retrieval.retriever.get_vector_store")
    def test_filters_below_threshold(self, mock_get_store):
        """Chunks below similarity_threshold must be excluded."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        high_doc = Document(page_content="relevant", metadata={"source": "a.pdf", "page": 1})
        low_doc = Document(page_content="noise", metadata={"source": "b.pdf", "page": 1})

        mock_store.similarity_search_with_relevance_scores.return_value = [
            (high_doc, 0.85),
            (low_doc, 0.10),  # below default threshold of 0.3
        ]

        from app.retrieval.retriever import retrieve
        results = retrieve("test query")

        assert len(results) == 1
        assert results[0].page_content == "relevant"

    @patch("app.retrieval.retriever.get_vector_store")
    def test_returns_empty_when_all_below_threshold(self, mock_get_store):
        """All chunks below threshold → empty list, no exception."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="noise", metadata={}), 0.05),
        ]

        from app.retrieval.retriever import retrieve
        results = retrieve("anything")
        assert results == []

    @patch("app.retrieval.retriever.get_vector_store")
    def test_respects_top_k_override(self, mock_get_store):
        """The top_k parameter must be passed through to the vector store."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.similarity_search_with_relevance_scores.return_value = []

        from app.retrieval.retriever import retrieve
        retrieve("q", top_k=7)

        mock_store.similarity_search_with_relevance_scores.assert_called_once_with(
            query="q", k=7
        )
