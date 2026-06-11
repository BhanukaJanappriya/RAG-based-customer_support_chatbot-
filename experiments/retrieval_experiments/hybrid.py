"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines dense (embedding) and sparse (BM25) ranked lists using RRF.

Why RRF over weighted sum (Cormack, Clarke & Buettcher, 2009):
- RRF is hyperparameter-free beyond the constant k=60 (insensitive in [10, 100]).
- Weighted sum requires tuning α (dense weight) — adds an experiment axis we
  want to avoid. With a 50-sample eval set, α tuning overfits.
- RRF degrades gracefully when one signal is absent (0 score = no contribution).
- Empirically matches or beats tuned weighted sum on most IR benchmarks.

Hybrid retrieval is expected to outperform either signal alone on our corpus
because: dense handles paraphrase/semantic queries well; BM25 handles
policy-specific terms (exact dollar amounts, URLs, exact phrases) well.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from experiments.retrieval_experiments.sparse import BM25Retriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum_{r in rankings} 1 / (k + rank(d, r))
    where rank is 1-based and documents absent from a ranking score 0.

    Args:
        rankings: List of ranked lists of document IDs.
            Each list is ordered from best (index 0) to worst.
        k: RRF constant. Insensitive in [10, 100]; default 60 per paper.

    Returns:
        List of (doc_id, rrf_score) sorted by descending RRF score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Dense + BM25 hybrid retriever using RRF fusion.

    Args:
        chroma_store: Chroma vector store (dense retrieval).
        bm25_retriever: Pre-built BM25 retriever (sparse retrieval).
        dense_top_k: Candidates to fetch from dense retriever.
        bm25_top_k: Candidates to fetch from BM25 retriever.
        rrf_k: RRF constant (default 60).
        final_top_k: Final number of results after fusion.
        query_prefix: Optional query prefix for models that require it (BGE, E5).

    Example::

        hybrid = HybridRetriever(chroma_store, bm25_retriever)
        results = hybrid.retrieve("how do I get a refund?")
    """

    def __init__(
        self,
        chroma_store: Chroma,
        bm25_retriever: BM25Retriever,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        rrf_k: int = 60,
        final_top_k: int = 4,
        query_prefix: str = "",
    ) -> None:
        self.chroma_store = chroma_store
        self.bm25_retriever = bm25_retriever
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self.query_prefix = query_prefix

        # Build a lookup from chunk_id → Document for result assembly
        self._id_to_doc: dict[str, Document] = {}
        for doc in bm25_retriever.documents:
            cid = f"{doc.metadata.get('source', 'doc')}__chunk_{doc.metadata.get('chunk_index', 0)}"
            self._id_to_doc[cid] = doc

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve top documents via hybrid RRF.

        Args:
            query: User query string.

        Returns:
            Top ``final_top_k`` documents ranked by RRF score.
        """
        prefixed_query = self.query_prefix + query if self.query_prefix else query

        # Dense ranking
        dense_results = self.chroma_store.similarity_search_with_relevance_scores(
            query=prefixed_query, k=self.dense_top_k
        )
        dense_ids = [
            f"{doc.metadata.get('source', 'doc')}__chunk_{doc.metadata.get('chunk_index', i)}"
            for i, (doc, _) in enumerate(dense_results)
        ]
        # Update id→doc with dense results
        for (doc, _), cid in zip(dense_results, dense_ids):
            self._id_to_doc[cid] = doc

        # Sparse (BM25) ranking
        bm25_results = self.bm25_retriever.retrieve_with_ids(query, top_k=self.bm25_top_k)
        bm25_ids = [cid for cid, _, _ in bm25_results]

        # RRF fusion
        fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=self.rrf_k)

        # Assemble final results
        results: list[Document] = []
        for cid, _ in fused[: self.final_top_k]:
            if cid in self._id_to_doc:
                results.append(self._id_to_doc[cid])
            else:
                logger.debug(f"chunk {cid} in fusion result but not in id_to_doc")

        logger.debug(
            f"Hybrid RRF: {len(dense_ids)} dense + {len(bm25_ids)} BM25 → "
            f"{len(fused)} fused → {len(results)} final"
        )
        return results

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Like ``retrieve`` but returns (Document, rrf_score) pairs."""
        prefixed_query = self.query_prefix + query if self.query_prefix else query

        dense_results = self.chroma_store.similarity_search_with_relevance_scores(
            query=prefixed_query, k=self.dense_top_k
        )
        dense_ids = [
            f"{doc.metadata.get('source', 'doc')}__chunk_{doc.metadata.get('chunk_index', i)}"
            for i, (doc, _) in enumerate(dense_results)
        ]
        for (doc, _), cid in zip(dense_results, dense_ids):
            self._id_to_doc[cid] = doc

        bm25_results = self.bm25_retriever.retrieve_with_ids(query, top_k=self.bm25_top_k)
        bm25_ids = [cid for cid, _, _ in bm25_results]

        fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=self.rrf_k)

        output = []
        for cid, score in fused[: self.final_top_k]:
            if cid in self._id_to_doc:
                output.append((self._id_to_doc[cid], score))

        return output
