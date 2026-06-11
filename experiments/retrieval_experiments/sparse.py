"""BM25 sparse retrieval implementation.

Uses ``rank_bm25`` (BM25Okapi variant) over the tokenised corpus.
BM25 is the standard sparse baseline in IR — it rewards exact term
matches and IDF weighting. It outperforms dense retrieval for
queries with rare or specific terminology (product names, error codes,
policy specifics) where the embedding space may not discriminate well.

Install: pip install rank-bm25
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenisation with lowercasing.

    Does not do stemming or stopword removal — BM25Okapi handles IDF
    weighting for common words already.
    """
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Retriever:
    """Sparse BM25 retriever over a fixed document corpus.

    Args:
        documents: Indexed document chunks.
        k1: BM25Okapi k1 parameter (term frequency saturation). Default 1.5.
        b: BM25Okapi b parameter (document length normalisation). Default 0.75.

    Example::

        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("what is the return policy", top_k=4)
    """

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for sparse retrieval. "
                "Install with: pip install rank-bm25"
            )

        self.documents = documents
        self._corpus_tokens = [_tokenise(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens, k1=k1, b=b)
        self._chunk_ids = [
            f"{doc.metadata.get('source', 'doc')}__chunk_{doc.metadata.get('chunk_index', i)}"
            for i, doc in enumerate(documents)
        ]
        logger.info(f"BM25 index built over {len(documents)} chunks (k1={k1}, b={b})")

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        return_scores: bool = False,
    ) -> list[Document] | list[tuple[Document, float]]:
        """Retrieve top-k documents by BM25 score.

        Args:
            query: User query string.
            top_k: Number of documents to return.
            return_scores: If True, return (document, score) tuples.

        Returns:
            List of Documents or (Document, score) tuples, ranked by BM25 score.
        """
        query_tokens = _tokenise(query)
        if not query_tokens:
            logger.warning(f"Empty query tokens for: {query!r}")
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        if return_scores:
            return [(self.documents[i], float(scores[i])) for i in top_indices]
        return [self.documents[i] for i in top_indices]

    def retrieve_with_ids(self, query: str, top_k: int = 4) -> list[tuple[str, str, float]]:
        """Retrieve (chunk_id, text, score) triples — useful for RRF fusion.

        Args:
            query: User query string.
            top_k: Number of results.

        Returns:
            List of (chunk_id, page_content, bm25_score) tuples.
        """
        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            (self._chunk_ids[i], self.documents[i].page_content, float(scores[i]))
            for i in top_indices
        ]

    def get_doc_id(self, index: int) -> str:
        return self._chunk_ids[index]

    def __len__(self) -> int:
        return len(self.documents)
