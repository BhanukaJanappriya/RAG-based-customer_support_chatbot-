"""ChromaDB vector store management."""

import logging
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.ingestion.embedder import get_embedding_model

logger = logging.getLogger(__name__)


def get_vector_store() -> Chroma:
    """Return a Chroma instance connected to the local persistent collection.

    Creates the collection if it does not already exist. The embedding
    function is the same singleton used during ingestion, guaranteeing
    that query vectors live in the same space as stored document vectors.

    Returns:
        A ready-to-use ``Chroma`` vector store.
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(documents: List[Document]) -> int:
    """Add chunks to the vector store with idempotent upsert semantics.

    Each document receives a stable ID derived from its source filename and
    chunk index.  Existing documents with the same IDs are deleted before
    re-insertion, preventing duplicates on re-ingestion.

    Args:
        documents: Chunked Documents (must have ``source`` and ``chunk_index``
            in their metadata).

    Returns:
        Number of chunks written to the store.
    """
    vector_store = get_vector_store()

    ids = [
        f"{doc.metadata.get('source', 'unknown')}__chunk_{doc.metadata.get('chunk_index', i)}"
        for i, doc in enumerate(documents)
    ]

    # Delete stale copies before upserting so metadata changes propagate
    try:
        existing = vector_store.get(ids=ids)
        if existing["ids"]:
            vector_store.delete(ids=existing["ids"])
            logger.info(f"Removed {len(existing['ids'])} stale chunk(s) before re-ingestion")
    except Exception as exc:
        logger.debug(f"Pre-delete step skipped: {exc}")

    vector_store.add_documents(documents=documents, ids=ids)
    logger.info(f"Stored {len(documents)} chunks in '{settings.chroma_collection_name}'")
    return len(documents)


def get_collection_count() -> int:
    """Return the total number of chunks currently persisted in the collection.

    Returns:
        Chunk count, or -1 if the collection cannot be reached.
    """
    try:
        store = get_vector_store()
        return store._collection.count()
    except Exception as exc:
        logger.warning(f"Could not fetch collection count: {exc}")
        return -1
