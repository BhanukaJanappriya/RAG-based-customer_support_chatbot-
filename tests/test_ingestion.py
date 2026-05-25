"""Unit tests for the ingestion layer (loader + chunker)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_directory


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def test_long_document_is_split(self):
        """A document longer than chunk_size must produce multiple chunks."""
        doc = Document(
            page_content="word " * 300,  # ~1500 chars, well above 512
            metadata={"source": "test.pdf", "page": 1},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) > 1

    def test_short_document_stays_one_chunk(self):
        """A document shorter than chunk_size should remain a single chunk."""
        doc = Document(
            page_content="Short content.",
            metadata={"source": "test.pdf", "page": 1},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) == 1

    def test_source_metadata_preserved(self):
        """All chunks must inherit the source filename from the parent document."""
        doc = Document(
            page_content="word " * 300,
            metadata={"source": "handbook.pdf", "page": 3, "file_type": "pdf"},
        )
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "handbook.pdf"
            assert chunk.metadata["file_type"] == "pdf"

    def test_chunk_index_assigned(self):
        """Every chunk must have a monotonically increasing chunk_index."""
        docs = [
            Document(page_content="word " * 200, metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="word " * 200, metadata={"source": "b.pdf", "page": 1}),
        ]
        chunks = chunk_documents(docs)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_document_list(self):
        """Chunking an empty list should return an empty list without error."""
        assert chunk_documents([]) == []

    def test_chunk_size_respected(self):
        """No chunk (excluding separator artefacts) should exceed chunk_size + overlap."""
        from app.config import settings

        doc = Document(
            page_content="a" * 2000,
            metadata={"source": "big.pdf", "page": 1},
        )
        chunks = chunk_documents([doc])
        # Allow a small tolerance for separator handling
        for chunk in chunks:
            assert len(chunk.page_content) <= settings.chunk_size + settings.chunk_overlap + 10


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestLoadDirectory:
    def test_raises_on_missing_directory(self):
        """Should raise FileNotFoundError for non-existent paths."""
        with pytest.raises(FileNotFoundError):
            load_directory(Path("/totally/nonexistent/directory/xyz"))

    @patch("app.ingestion.loader.PyPDFLoader")
    def test_pdf_files_are_loaded(self, mock_loader_cls, tmp_path):
        """PDF files in the directory should be picked up and loaded."""
        (tmp_path / "doc.pdf").touch()

        mock_doc = Document(
            page_content="PDF content", metadata={"source": "doc.pdf", "page": 0}
        )
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_cls.return_value = mock_loader

        docs = load_directory(tmp_path)
        assert any(d.metadata["source"] == "doc.pdf" for d in docs)

    @patch("app.ingestion.loader.TextLoader")
    def test_markdown_files_are_loaded(self, mock_loader_cls, tmp_path):
        """Markdown files in the directory should be picked up and loaded."""
        (tmp_path / "readme.md").write_text("# Hello", encoding="utf-8")

        mock_doc = Document(
            page_content="# Hello", metadata={}
        )
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_cls.return_value = mock_loader

        docs = load_directory(tmp_path)
        assert any(d.metadata.get("file_type") == "markdown" for d in docs)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """An empty directory should produce an empty document list."""
        docs = load_directory(tmp_path)
        assert docs == []
