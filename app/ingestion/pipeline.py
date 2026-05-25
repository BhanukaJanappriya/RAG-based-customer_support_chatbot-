"""Full ingestion pipeline: load → chunk → embed → persist."""

import logging
from pathlib import Path
from typing import Dict, Optional

from app.config import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_directory
from app.retrieval.vector_store import add_documents, get_collection_count

logger = logging.getLogger(__name__)


def run_ingestion(data_dir: Optional[Path] = None) -> Dict[str, int]:
    """Execute the full document ingestion pipeline.

    The pipeline is idempotent: re-running with the same documents deletes
    existing chunks (matched by stable ID) before re-inserting, so no
    duplicates accumulate.

    Args:
        data_dir: Directory to scan for documents. Defaults to ``settings.raw_data_path``.

    Returns:
        Dictionary with pipeline statistics::

            {
                "documents_loaded": int,
                "chunks_created": int,
                "chunks_stored": int,
                "total_in_store": int,
            }
    """
    source_dir = data_dir or settings.raw_data_path
    logger.info(f"Starting ingestion from: {source_dir}")

    documents = load_directory(source_dir)

    if not documents:
        logger.warning(
            "No documents found. Drop PDF or Markdown files into the "
            f"'{source_dir}' directory and re-run."
        )
        return {
            "documents_loaded": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "total_in_store": get_collection_count(),
        }

    chunks = chunk_documents(documents)
    stored = add_documents(chunks)
    total = get_collection_count()

    stats: Dict[str, int] = {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": stored,
        "total_in_store": total,
    }
    logger.info(f"Ingestion complete — {stats}")
    return stats
