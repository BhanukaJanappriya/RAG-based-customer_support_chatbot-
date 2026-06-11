"""Cross-encoder reranking for retrieval results.

Uses a cross-encoder model (ms-marco-MiniLM-L-6-v2) to rerank the fused
candidate set from hybrid retrieval.

Why cross-encoder vs bi-encoder for reranking:
- Bi-encoder (the dense retrieval model) computes query and document embeddings
  independently — fast at scale, but misses fine-grained query-document
  interactions.
- Cross-encoder concatenates (query, document) and scores them jointly, capturing
  exact-match signals and token-level interactions. ~10-100x slower but much
  more accurate at the top of the ranked list.
- Two-stage (bi-encoder retrieve → cross-encoder rerank) is standard practice:
  use bi-encoder to get top-50, rerank to top-4.

Install: sentence-transformers is already in requirements.txt
Model download: ~50MB on first use.
"""

from __future__ import annotations

import logging
import time

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks candidate documents using a cross-encoder model.

    Args:
        model_name: HuggingFace model ID. Default is the MiniLM cross-encoder
            fine-tuned on MS-MARCO passage ranking.
        device: "cpu" or "cuda".
        batch_size: Number of (query, doc) pairs per forward pass.

    Example::

        reranker = CrossEncoderReranker()
        candidates = hybrid_retriever.retrieve(query, top_k=20)
        top4 = reranker.rerank(query, candidates, top_k=4)
    """

    _MODEL_CACHE: dict[str, "CrossEncoder"] = {}

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = self._load_model(model_name, device)

    @classmethod
    def _load_model(cls, model_name: str, device: str):
        cache_key = f"{model_name}::{device}"
        if cache_key not in cls._MODEL_CACHE:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "It is already in requirements.txt — run `pip install -r requirements.txt`."
                )
            logger.info(f"Loading cross-encoder: {model_name}")
            cls._MODEL_CACHE[cache_key] = CrossEncoder(model_name, device=device)
            logger.info("Cross-encoder loaded")
        return cls._MODEL_CACHE[cache_key]

    def rerank(
        self,
        query: str,
        candidates: list[Document],
        top_k: int = 4,
    ) -> list[Document]:
        """Rerank candidates by cross-encoder score and return top-k.

        Args:
            query: User query string.
            candidates: Candidate documents from a first-stage retriever.
            top_k: Number of documents to return after reranking.

        Returns:
            Top-k documents sorted by cross-encoder score (best first).
        """
        if not candidates:
            return []

        t0 = time.perf_counter()
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        latency_ms = (time.perf_counter() - t0) * 1000

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.debug(
            f"Cross-encoder reranked {len(candidates)} → {top_k} "
            f"in {latency_ms:.1f}ms (top score: {ranked[0][1]:.3f})"
        )
        return [doc for doc, _ in ranked[:top_k]]

    def rerank_with_scores(
        self,
        query: str,
        candidates: list[Document],
        top_k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Like ``rerank`` but returns (Document, score) tuples."""
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        ranked = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
