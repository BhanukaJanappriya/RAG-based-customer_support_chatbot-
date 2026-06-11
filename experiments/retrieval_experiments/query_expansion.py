"""Query expansion and multi-query retrieval.

Two strategies:
1. Multi-query: Generate N rephrasings of the original query, retrieve for
   each, and union the results (deduplication by chunk ID, re-rank by hit
   count across rephrasings as a proxy for relevance).

2. Query expansion: Expand the original query with synonyms / related terms
   extracted by the LLM, then retrieve with the expanded query string.

Expected behaviour:
- Multi-query is the more principled approach: it handles paraphrase variance
  (a query phrased differently might not retrieve the same chunks as the
  original). The downside is N * retrieval_latency.
- Query expansion is simpler but may introduce noise if the LLM adds
  terms that match irrelevant chunks.

Both add LLM latency (1-3s per call on CPU). Only adopt if the retrieval
metric gain justifies the latency cost for your SLA.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_MULTI_QUERY_PROMPT = """\
Generate {n} alternative phrasings of the following customer support question.
Each phrasing should preserve the meaning but use different words.
Return ONLY a JSON array of strings, e.g. ["phrasing 1", "phrasing 2"].
Do NOT include the original question.

Original question: {query}
"""

_EXPANSION_PROMPT = """\
Expand the following customer support question with related terms and synonyms.
Return a single expanded query string (not a list). Keep it under 50 words.

Original question: {query}
"""


class MultiQueryRetriever:
    """Retrieve using N rephrasings of the query, union and re-rank by hit count.

    Args:
        chroma_store: Chroma vector store.
        bm25_retriever: Optional BM25 retriever (if hybrid mode).
        llm_model: Ollama model for rephrasing.
        ollama_base_url: Ollama server URL.
        n_rephrasings: Number of additional phrasings to generate.
        top_k_per_query: Candidates to fetch per rephrasing.
        final_top_k: Documents to return after union and re-ranking.
        temperature: LLM temperature (higher = more diverse phrasings).
        query_prefix: Embedding model query prefix.

    Example::

        retriever = MultiQueryRetriever(chroma_store, n_rephrasings=3)
        docs = retriever.retrieve("how long does refund take")
    """

    def __init__(
        self,
        chroma_store,
        bm25_retriever=None,
        llm_model: str = "llama3.2:latest",
        ollama_base_url: str = "http://localhost:11434",
        n_rephrasings: int = 3,
        top_k_per_query: int = 4,
        final_top_k: int = 4,
        temperature: float = 0.4,
        query_prefix: str = "",
    ) -> None:
        self.chroma_store = chroma_store
        self.bm25_retriever = bm25_retriever
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url
        self.n_rephrasings = n_rephrasings
        self.top_k_per_query = top_k_per_query
        self.final_top_k = final_top_k
        self.temperature = temperature
        self.query_prefix = query_prefix

    def _generate_rephrasings(self, query: str) -> list[str]:
        """Generate N alternative phrasings via Ollama."""
        from experiments.utils.ollama_client import chat, parse_json_response
        prompt = _MULTI_QUERY_PROMPT.format(n=self.n_rephrasings, query=query)
        try:
            response = chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_model,
                base_url=self.ollama_base_url,
                temperature=self.temperature,
                max_tokens=200,
            )
            rephrasings = parse_json_response(response)
            if isinstance(rephrasings, list):
                return [str(r).strip() for r in rephrasings if r][:self.n_rephrasings]
        except Exception as exc:
            logger.warning(f"Rephrasing generation failed: {exc}")
        return []

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve using original + N rephrasings, re-rank by hit count.

        Args:
            query: Original user query.

        Returns:
            Top ``final_top_k`` documents ranked by frequency of appearance
            across all query variants (a proxy for cross-query relevance).
        """
        all_queries = [query] + self._generate_rephrasings(query)
        logger.debug(f"Multi-query ({len(all_queries)} variants): {all_queries}")

        # Collect all retrieved docs with a hit count per chunk ID
        id_to_doc: dict[str, Document] = {}
        hit_counts: Counter = Counter()

        for q in all_queries:
            pq = self.query_prefix + q if self.query_prefix else q
            results = self.chroma_store.similarity_search(pq, k=self.top_k_per_query)
            for doc in results:
                cid = (
                    f"{doc.metadata.get('source', 'doc')}__chunk_"
                    f"{doc.metadata.get('chunk_index', 0)}"
                )
                id_to_doc[cid] = doc
                hit_counts[cid] += 1

        # Re-rank by hit count (ties broken by first-appearance order)
        ranked_ids = [cid for cid, _ in hit_counts.most_common()]
        results_docs = [id_to_doc[cid] for cid in ranked_ids[: self.final_top_k]]

        logger.debug(
            f"Multi-query union: {len(id_to_doc)} unique docs → top {len(results_docs)}"
        )
        return results_docs


class QueryExpansionRetriever:
    """Retriever that uses LLM-expanded query for dense retrieval.

    Args:
        chroma_store: Chroma vector store.
        llm_model: Ollama model for query expansion.
        ollama_base_url: Ollama server URL.
        top_k: Number of documents to return.
        temperature: LLM temperature.
        query_prefix: Embedding model query prefix.

    Example::

        retriever = QueryExpansionRetriever(chroma_store)
        docs = retriever.retrieve("refund time")
    """

    def __init__(
        self,
        chroma_store,
        llm_model: str = "llama3.2:latest",
        ollama_base_url: str = "http://localhost:11434",
        top_k: int = 4,
        temperature: float = 0.2,
        query_prefix: str = "",
    ) -> None:
        self.chroma_store = chroma_store
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url
        self.top_k = top_k
        self.temperature = temperature
        self.query_prefix = query_prefix

    def _expand_query(self, query: str) -> str:
        from experiments.utils.ollama_client import chat
        prompt = _EXPANSION_PROMPT.format(query=query)
        try:
            expanded = chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_model,
                base_url=self.ollama_base_url,
                temperature=self.temperature,
                max_tokens=100,
            )
            expanded = expanded.strip().replace("\n", " ")
            logger.debug(f"Expanded: {query!r} → {expanded!r}")
            return expanded
        except Exception as exc:
            logger.warning(f"Query expansion failed: {exc}. Using original.")
            return query

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve using the LLM-expanded query.

        Args:
            query: Original user query.

        Returns:
            Top-k documents for the expanded query.
        """
        expanded = self._expand_query(query)
        search_text = self.query_prefix + expanded if self.query_prefix else expanded
        return self.chroma_store.similarity_search(search_text, k=self.top_k)

    def retrieve_with_expansion(self, query: str) -> tuple[list[Document], str]:
        """Like ``retrieve`` but also returns the expanded query string."""
        expanded = self._expand_query(query)
        search_text = self.query_prefix + expanded if self.query_prefix else expanded
        results = self.chroma_store.similarity_search(search_text, k=self.top_k)
        return results, expanded
