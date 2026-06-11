"""Deterministic retrieval metrics — no LLM required.

Implements Precision@k, Recall@k, MRR@k, nDCG@k, Hit Rate, and a simple
keyword-based Context Relevance score. All functions operate on ranked lists
of document IDs / texts and a set of gold-relevant document IDs.

Design decisions:
- Per-query scores are returned alongside the aggregate so callers can run
  bootstrap CI (experiments/utils/bootstrap.py) on the raw per-query arrays.
- nDCG uses a binary relevance scale (relevant=1, irrelevant=0) because our
  gold set has binary labels; graded relevance requires a richer annotation effort.
- MRR stops at the first relevant hit — consistent with IR literature where
  MRR measures "how quickly does the system surface *a* relevant document".
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ── Individual per-query metrics ─────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant.

    Args:
        retrieved: Ranked list of retrieved doc IDs (position 0 = best).
        relevant: Set of gold-relevant doc IDs.
        k: Cut-off rank.

    Returns:
        P@k in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved.

    Args:
        retrieved: Ranked list of retrieved doc IDs.
        relevant: Set of gold-relevant doc IDs.
        k: Cut-off rank.

    Returns:
        R@k in [0, 1]. Returns 0 when ``relevant`` is empty.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant document.

    Args:
        retrieved: Ranked list of retrieved doc IDs.
        relevant: Set of gold-relevant doc IDs.

    Returns:
        1/rank of first hit, or 0.0 if no relevant doc is found.
    """
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank k (binary relevance).

    DCG = sum_{i=1}^{k} rel_i / log2(i+1)
    Normalised by IDCG = DCG of a perfect ranking.

    Args:
        retrieved: Ranked list of retrieved doc IDs.
        relevant: Set of gold-relevant doc IDs.
        k: Cut-off rank.

    Returns:
        nDCG@k in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(i + 2)  # i+2 because enumerate starts at 0
        for i, doc in enumerate(top_k)
        if doc in relevant
    )

    # Ideal DCG: min(|relevant|, k) hits at the top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary: 1 if any of the top-k docs is relevant, else 0.

    Equivalent to Recall@k > 0. Useful for measuring "did the system find
    *something* useful" when the query has multiple valid answers.

    Args:
        retrieved: Ranked list of retrieved doc IDs.
        relevant: Set of gold-relevant doc IDs.
        k: Cut-off rank.

    Returns:
        1.0 or 0.0.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    return 1.0 if top_k & relevant else 0.0


def context_relevance_keyword(
    query: str,
    retrieved_texts: list[str],
    min_overlap: int = 1,
) -> float:
    """Keyword-overlap proxy for context relevance (no LLM required).

    Measures the fraction of retrieved chunks that share at least
    ``min_overlap`` content words (>3 chars) with the query. This is a
    fast, cheap signal — not a substitute for semantic relevance.

    Args:
        query: User query string.
        retrieved_texts: List of retrieved document texts (not IDs).
        min_overlap: Minimum number of shared content words to count as relevant.

    Returns:
        Fraction of chunks with sufficient keyword overlap, in [0, 1].
    """
    if not retrieved_texts:
        return 0.0

    query_words = {
        w.lower().strip(".,?!") for w in query.split() if len(w) > 3
    }
    if not query_words:
        return 0.0

    relevant_count = 0
    for text in retrieved_texts:
        chunk_words = {w.lower().strip(".,?!") for w in text.split() if len(w) > 3}
        if len(query_words & chunk_words) >= min_overlap:
            relevant_count += 1

    return relevant_count / len(retrieved_texts)


# ── Aggregate across a query set ────────────────────────────────────────────

def compute_retrieval_metrics(
    queries: list[str],
    retrieved_ids_per_query: list[list[str]],
    relevant_ids_per_query: list[set[str]],
    k_values: list[int] | None = None,
) -> dict[str, dict]:
    """Compute all retrieval metrics for a set of queries.

    Args:
        queries: List of query strings (used only for context_relevance).
        retrieved_ids_per_query: For each query, a ranked list of retrieved doc IDs.
        relevant_ids_per_query: For each query, a set of gold-relevant doc IDs.
        k_values: List of k cut-offs. Defaults to [1, 3, 5, 10].

    Returns:
        Nested dict:
        ``{metric_name: {"mean": float, "per_query": [float, ...]}}``

        Metric names follow the pattern ``{metric}_{suffix}`` e.g.
        ``precision_at_5``, ``ndcg_at_5``, ``mrr``, ``hit_rate_at_5``.

    Raises:
        ValueError: If the three input lists have different lengths.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    n = len(queries)
    if not (n == len(retrieved_ids_per_query) == len(relevant_ids_per_query)):
        raise ValueError(
            f"Input lengths differ: queries={n}, "
            f"retrieved={len(retrieved_ids_per_query)}, "
            f"relevant={len(relevant_ids_per_query)}"
        )

    # Collectors: metric_name → per-query list
    per_query: dict[str, list[float]] = {}

    for k in k_values:
        per_query[f"precision_at_{k}"] = []
        per_query[f"recall_at_{k}"] = []
        per_query[f"ndcg_at_{k}"] = []
        per_query[f"hit_rate_at_{k}"] = []

    per_query["mrr"] = []
    per_query["context_relevance"] = []

    for query, retrieved, relevant in zip(
        queries, retrieved_ids_per_query, relevant_ids_per_query
    ):
        for k in k_values:
            per_query[f"precision_at_{k}"].append(precision_at_k(retrieved, relevant, k))
            per_query[f"recall_at_{k}"].append(recall_at_k(retrieved, relevant, k))
            per_query[f"ndcg_at_{k}"].append(ndcg_at_k(retrieved, relevant, k))
            per_query[f"hit_rate_at_{k}"].append(hit_rate_at_k(retrieved, relevant, k))

        per_query["mrr"].append(reciprocal_rank(retrieved, relevant))

        # Context relevance uses text, so we need a placeholder here;
        # callers that have text pass it separately via retrieved_texts_per_query.
        per_query["context_relevance"].append(0.0)

    results: dict[str, dict] = {}
    for name, values in per_query.items():
        arr = np.array(values, dtype=float)
        results[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "per_query": values,
        }

    logger.info(
        "Retrieval metrics over %d queries: MRR=%.3f, nDCG@5=%.3f, P@5=%.3f",
        n,
        results["mrr"]["mean"],
        results.get("ndcg_at_5", {}).get("mean", float("nan")),
        results.get("precision_at_5", {}).get("mean", float("nan")),
    )
    return results


def compute_retrieval_metrics_with_text(
    queries: list[str],
    retrieved_ids_per_query: list[list[str]],
    retrieved_texts_per_query: list[list[str]],
    relevant_ids_per_query: list[set[str]],
    k_values: list[int] | None = None,
) -> dict[str, dict]:
    """Like ``compute_retrieval_metrics`` but also computes keyword context relevance.

    Args:
        queries: Query strings.
        retrieved_ids_per_query: Ranked doc IDs per query.
        retrieved_texts_per_query: Corresponding raw texts per query.
        relevant_ids_per_query: Gold-relevant doc IDs per query.
        k_values: k cut-offs.

    Returns:
        Same format as ``compute_retrieval_metrics`` with correct
        ``context_relevance`` values.
    """
    results = compute_retrieval_metrics(
        queries, retrieved_ids_per_query, relevant_ids_per_query, k_values
    )

    cr_scores = [
        context_relevance_keyword(q, texts)
        for q, texts in zip(queries, retrieved_texts_per_query)
    ]
    arr = np.array(cr_scores, dtype=float)
    results["context_relevance"] = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "per_query": cr_scores,
    }
    return results


# ── Relevance label utilities ─────────────────────────────────────────────────

def build_relevant_ids_from_text_match(
    gold_answers: list[str],
    candidate_texts: list[str],
    candidate_ids: list[str],
    match_fn: Callable[[str, str], bool] | None = None,
) -> list[set[str]]:
    """Build relevant-ID sets by matching gold answers against candidate texts.

    When you don't have pre-labelled relevant doc IDs, use this to derive
    pseudo-relevance labels: a chunk is "relevant" if it contains enough
    content words from the gold answer.

    Args:
        gold_answers: Ground-truth answer strings, one per query.
        candidate_texts: All available document chunk texts.
        candidate_ids: Corresponding IDs (parallel to ``candidate_texts``).
        match_fn: Optional custom relevance function(gold, chunk) → bool.
            Defaults to word-overlap with threshold=0.2.

    Returns:
        List of relevant-ID sets, one per query.
    """
    if match_fn is None:
        def match_fn(gold: str, chunk: str) -> bool:
            gold_words = {w.lower() for w in gold.split() if len(w) > 3}
            chunk_words = {w.lower() for w in chunk.split() if len(w) > 3}
            if not gold_words:
                return False
            overlap = len(gold_words & chunk_words) / len(gold_words)
            return overlap >= 0.2

    relevant_sets = []
    for gold in gold_answers:
        rel = {
            cid
            for cid, ctext in zip(candidate_ids, candidate_texts)
            if match_fn(gold, ctext)
        }
        relevant_sets.append(rel)
    return relevant_sets
