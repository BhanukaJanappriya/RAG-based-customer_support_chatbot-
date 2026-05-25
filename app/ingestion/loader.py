"""Document loading utilities for PDF and Markdown files."""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(file_path: Path) -> List[Document]:
    """Load a PDF and return one Document per page, each tagged with source metadata.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Documents with ``source``, ``page``, and ``file_type`` metadata.
    """
    logger.info(f"Loading PDF: {file_path.name}")
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_type"] = "pdf"
        # PyPDFLoader uses 0-based pages; shift to 1-based for human display
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
    logger.info(f"  → {len(docs)} pages from {file_path.name}")
    return docs


def load_markdown(file_path: Path) -> List[Document]:
    """Load a Markdown file as a single Document.

    Args:
        file_path: Absolute or relative path to the .md file.

    Returns:
        List with a single Document containing the full file content.
    """
    logger.info(f"Loading Markdown: {file_path.name}")
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_type"] = "markdown"
        doc.metadata.setdefault("page", 1)
    logger.info(f"  → {len(docs)} document(s) from {file_path.name}")
    return docs


def load_directory(data_dir: Path) -> List[Document]:
    """Recursively load all PDFs and Markdown files from a directory.

    Args:
        data_dir: Root directory to scan for documents.

    Returns:
        Combined list of Documents from all supported file types.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_files = sorted(data_dir.glob("**/*.pdf"))
    md_files = sorted(data_dir.glob("**/*.md"))

    logger.info(
        f"Found {len(pdf_files)} PDF(s) and {len(md_files)} Markdown file(s) in {data_dir}"
    )

    documents: List[Document] = []

    for pdf_path in pdf_files:
        try:
            documents.extend(load_pdf(pdf_path))
        except Exception as exc:
            logger.error(f"Failed to load {pdf_path.name}: {exc}")

    for md_path in md_files:
        try:
            documents.extend(load_markdown(md_path))
        except Exception as exc:
            logger.error(f"Failed to load {md_path.name}: {exc}")

    logger.info(f"Total raw documents loaded: {len(documents)}")
    return documents
