"""Hypothetical Document Embeddings (HyDE) retrieval.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels", 2022.

HyDE generates a hypothetical document that *would* answer the query,
then uses that document's embedding (instead of the query embedding)
for retrieval. The intuition: the embedding of a relevant-looking answer
is closer to real relevant documents than the embedding of a question.

Expected behaviour on this corpus:
- May improve recall on ambiguous questions by expanding the query vocabulary.
- May hurt precision if the hypothetical document contains hallucinated
  policy details that match irrelevant chunks.
- Adds LLM latency (~2-8s on CPU) per query — only worthwhile if retrieval
  metrics improve substantially.

Known limitation: HyDE can degrade on short corpora where hallucinated
content misleads the embedding. Measure, don't assume.
"""

from __future__ import annotations

import logging
import time

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_HYDE_SYSTEM_PROMPT = """\
You are a customer support knowledge base. Write a short, factual passage
(2-4 sentences) that would directly answer the following question.
Write as if it were an excerpt from an official support document.
Do NOT include phrases like "I think" or "probably".
"""


class HyDERetriever:
    """Retriever that uses HyDE: generate a hypothetical answer, embed it, search.

    Args:
        chroma_store: Chroma vector store for dense retrieval.
        llm_model: Ollama model tag for hypothesis generation.
        ollama_base_url: Ollama server URL.
        top_k: Number of documents to retrieve.
        temperature: LLM temperature for hypothesis generation (slightly higher
            than generation temperature to produce varied hypotheses).
        query_prefix: Optional prefix for embedding models that require it.

    Example::

        hyde = HyDERetriever(chroma_store)
        docs = hyde.retrieve("what is the return period?")
    """

    def __init__(
        self,
        chroma_store,
        llm_model: str = "llama3.2:latest",
        ollama_base_url: str = "http://localhost:11434",
        top_k: int = 4,
        temperature: float = 0.3,
        query_prefix: str = "",
    ) -> None:
        self.chroma_store = chroma_store
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url
        self.top_k = top_k
        self.temperature = temperature
        self.query_prefix = query_prefix

    def _generate_hypothesis(self, query: str) -> str:
        """Generate a hypothetical document for the query via Ollama."""
        from experiments.utils.ollama_client import chat
        response = chat(
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}"},
            ],
            model=self.llm_model,
            base_url=self.ollama_base_url,
            temperature=self.temperature,
            max_tokens=200,
        )
        logger.debug(f"HyDE hypothesis: {response[:100]}...")
        return response.strip()

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve using the embedding of a generated hypothesis.

        Args:
            query: User query string.

        Returns:
            Top-k documents retrieved using the hypothesis embedding.
        """
        t0 = time.perf_counter()
        hypothesis = self._generate_hypothesis(query)
        hypothesis_time_ms = (time.perf_counter() - t0) * 1000

        search_text = self.query_prefix + hypothesis if self.query_prefix else hypothesis
        t1 = time.perf_counter()
        results = self.chroma_store.similarity_search(search_text, k=self.top_k)
        retrieval_time_ms = (time.perf_counter() - t1) * 1000

        logger.debug(
            f"HyDE: hypothesis_gen={hypothesis_time_ms:.0f}ms, "
            f"retrieval={retrieval_time_ms:.0f}ms"
        )
        return results

    def retrieve_with_hypothesis(self, query: str) -> tuple[list[Document], str]:
        """Like ``retrieve`` but also returns the generated hypothesis.

        Useful for error analysis: inspect what hypothesis was generated
        for failed retrievals.

        Args:
            query: User query string.

        Returns:
            Tuple of (retrieved_documents, hypothesis_text).
        """
        hypothesis = self._generate_hypothesis(query)
        search_text = self.query_prefix + hypothesis if self.query_prefix else hypothesis
        results = self.chroma_store.similarity_search(search_text, k=self.top_k)
        return results, hypothesis
