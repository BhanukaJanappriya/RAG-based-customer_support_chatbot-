"""Text chunking using recursive character splitting."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks suitable for embedding.

    Uses a hierarchy of separators so that the splitter prefers to break
    at paragraph boundaries, then sentences, then words, and only falls
    back to character-level splitting as a last resort.

    Args:
        documents: Raw Documents produced by the loader.

    Returns:
        Chunked Documents with an additional ``chunk_index`` metadata field.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        # Prefer semantic boundaries: paragraphs → sentences → words → chars
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata.setdefault("page", 1)

    logger.info(
        f"Chunked {len(documents)} document(s) → {len(chunks)} chunks "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks
