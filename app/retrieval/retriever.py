"""Semantic retrieval with similarity-threshold filtering."""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: Optional[int] = None) -> List[Document]:
    """Retrieve the most relevant document chunks for a user query.

    Performs cosine-similarity search in ChromaDB and drops any chunks
    whose relevance score falls below ``settings.similarity_threshold``.
    Dropping low-confidence chunks prevents the LLM from hallucinating
    on tangentially related content.

    Args:
        query: Natural-language question from the user.
        top_k: Maximum number of candidates to retrieve before filtering.
            Defaults to ``settings.retrieval_top_k``.

    Returns:
        Filtered list of Documents ordered by descending relevance.
    """
    k = top_k if top_k is not None else settings.retrieval_top_k
    vector_store = get_vector_store()

    results: List[Tuple[Document, float]] = (
        vector_store.similarity_search_with_relevance_scores(query=query, k=k)
    )

    filtered = [
        doc
        for doc, score in results
        if score >= settings.similarity_threshold
    ]

    logger.info(
        f"Retrieval: {len(filtered)}/{len(results)} chunks passed threshold "
        f"{settings.similarity_threshold} — scores: {[round(s,3) for _,s in results]} "
        f"for query='{query[:60]}'"
    )
    return filtered
