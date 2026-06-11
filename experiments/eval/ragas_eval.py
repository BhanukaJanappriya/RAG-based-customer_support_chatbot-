"""RAGAS-based end-to-end RAG evaluation using a local LLM as judge.

This module wraps RAGAS 0.2.x to run Faithfulness, AnswerRelevancy,
ContextPrecision, and ContextRecall against Llama 3.2 1B running locally
via Ollama.

⚠️  KNOWN LIMITATIONS (document before citing results):
1. Judge == generator bias: Llama 3.2 1B both answers and judges. It is
   likely to score its own outputs more favourably than a stronger judge
   would. Treat LLM-as-judge scores as relative (config A vs. B), not
   absolute quality estimates.
2. Small-model faithfulness: Llama 3.2 1B has limited instruction-following
   fidelity. RAGAS faithfulness relies on the judge decomposing claims and
   verifying them against context; small models do this imperfectly.
3. Slow on CPU: Each sample requires multiple LLM calls. Budget ~30-60s per
   sample on CPU. Run only on finalists (~10-20 samples), not the full 50.
4. Context window: 3.2 1B has a 128k context window, but quality degrades
   on complex multi-hop questions with long contexts.

Mitigation: Always report retrieval metrics (precision, nDCG) alongside
RAGAS scores. Use RAGAS only to distinguish configurations that are close
on retrieval metrics.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Guard: RAGAS is an optional dependency
try:
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning(
        "RAGAS not installed. Install with: pip install ragas>=0.2.0\n"
        "LLM-as-judge evaluation will be unavailable."
    )

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def _require_ragas() -> None:
    if not RAGAS_AVAILABLE:
        raise ImportError(
            "RAGAS is required for LLM-as-judge evaluation. "
            "Install with: pip install ragas>=0.2.0"
        )
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("langchain-ollama and langchain-huggingface are required.")


def build_ragas_dataset(
    questions: list[str],
    responses: list[str],
    retrieved_contexts_per_query: list[list[str]],
    ground_truths: Optional[list[str]] = None,
) -> "EvaluationDataset":
    """Build a RAGAS EvaluationDataset from experiment outputs.

    Args:
        questions: User queries.
        responses: LLM-generated answers (one per query).
        retrieved_contexts_per_query: Retrieved chunk texts per query.
        ground_truths: Reference answers (required for ContextRecall).

    Returns:
        RAGAS EvaluationDataset ready for ``evaluate()``.
    """
    _require_ragas()

    if ground_truths is None:
        ground_truths = [""] * len(questions)

    samples = [
        SingleTurnSample(
            user_input=q,
            response=r,
            retrieved_contexts=ctxs,
            reference=gt,
        )
        for q, r, ctxs, gt in zip(
            questions, responses, retrieved_contexts_per_query, ground_truths
        )
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_evaluation(
    questions: list[str],
    responses: list[str],
    retrieved_contexts_per_query: list[list[str]],
    ground_truths: Optional[list[str]] = None,
    llm_model: str = "llama3.2:latest",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ollama_base_url: str = "http://localhost:11434",
    metrics: Optional[list] = None,
    batch_size: int = 5,
) -> dict[str, float | list[float]]:
    """Run RAGAS evaluation with a local LLM judge.

    Args:
        questions: User queries (evaluation examples).
        responses: LLM answers to evaluate.
        retrieved_contexts_per_query: Retrieved chunk texts, one list per query.
        ground_truths: Reference answers. Required for ContextRecall; optional
            for Faithfulness and AnswerRelevancy.
        llm_model: Ollama model tag for the judge LLM.
        embedding_model: HuggingFace model for AnswerRelevancy embeddings.
        ollama_base_url: Ollama server URL.
        metrics: RAGAS metric instances. Defaults to all four core metrics.
        batch_size: Number of samples per RAGAS batch (lower = less OOM risk).

    Returns:
        Dict mapping metric name → mean score and per-sample scores.

    Raises:
        ImportError: If RAGAS or LangChain packages are not installed.
    """
    _require_ragas()

    if metrics is None:
        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]

    # Wire up local LLM and embeddings
    judge_llm = LangchainLLMWrapper(
        ChatOllama(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0.0,   # deterministic judging
            num_predict=512,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    dataset = build_ragas_dataset(
        questions, responses, retrieved_contexts_per_query, ground_truths
    )

    logger.info(
        f"Running RAGAS evaluation: {len(questions)} samples, "
        f"{len(metrics)} metrics, judge={llm_model}"
    )
    logger.warning(
        "⚠️  RAGAS judge == generator (both use %s). Scores measure relative "
        "quality between configs, not absolute quality.",
        llm_model,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        batch_size=batch_size,
        show_progress=True,
    )

    # Convert result to plain dict
    scores: dict[str, float | list[float]] = {}
    result_df = result.to_pandas()

    metric_cols = [c for c in result_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    for col in metric_cols:
        per_sample = result_df[col].tolist()
        scores[col] = {
            "mean": float(result_df[col].mean()),
            "per_sample": [float(v) if v is not None else float("nan") for v in per_sample],
        }
        logger.info(f"RAGAS {col}: mean={scores[col]['mean']:.3f}")

    return scores


def faithfulness_heuristic(
    response: str,
    context_texts: list[str],
) -> float:
    """Keyword-overlap heuristic for faithfulness — no LLM required.

    A cheap proxy: measures what fraction of content words in the response
    appear in the retrieved context. This does NOT detect grammatically
    fluent hallucinations and should not replace RAGAS faithfulness.
    Use only when Ollama is unavailable or as a quick sanity check.

    Args:
        response: LLM-generated answer.
        context_texts: Retrieved chunk texts.

    Returns:
        Fraction of response content words found in context, in [0, 1].
    """
    if not response or not context_texts:
        return 0.0

    context_all = " ".join(context_texts).lower()
    context_words = set(context_all.split())

    response_words = [w.lower().strip(".,?!") for w in response.split() if len(w) > 4]
    if not response_words:
        return 1.0

    covered = sum(1 for w in response_words if w in context_words)
    return covered / len(response_words)


def answer_contains_refusal(response: str) -> bool:
    """Return True if the response is a well-formed refusal for out-of-scope queries.

    Checks for the exact production refusal phrase and common variants.

    Args:
        response: LLM-generated answer.

    Returns:
        True if the response is a refusal.
    """
    refusal_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "i don't have information",
        "not enough information",
        "outside the scope",
        "i cannot answer",
        "i can't answer",
        "not available in",
        "not covered in",
    ]
    lowered = response.lower().strip()
    return any(phrase in lowered for phrase in refusal_phrases)


def compute_refusal_accuracy(
    responses: list[str],
    categories: list[str],
) -> dict[str, float]:
    """Compute refusal accuracy on out-of-scope and standard questions.

    A good system should:
    - Refuse out-of-scope questions (high true-refusal rate).
    - NOT refuse standard questions (low false-refusal rate).

    Args:
        responses: LLM-generated answers.
        categories: Category label per response (e.g., "out_of_scope", "standard").

    Returns:
        Dict with ``true_refusal_rate``, ``false_refusal_rate``, and ``f1``.
    """
    out_of_scope_responses = [
        r for r, c in zip(responses, categories) if c == "out_of_scope"
    ]
    standard_responses = [
        r for r, c in zip(responses, categories) if c == "standard"
    ]

    true_refusals = sum(
        1 for r in out_of_scope_responses if answer_contains_refusal(r)
    )
    false_refusals = sum(
        1 for r in standard_responses if answer_contains_refusal(r)
    )

    true_refusal_rate = (
        true_refusals / len(out_of_scope_responses)
        if out_of_scope_responses else float("nan")
    )
    false_refusal_rate = (
        false_refusals / len(standard_responses)
        if standard_responses else float("nan")
    )

    # F1 treating "correctly refusing out-of-scope" as the positive class
    precision = true_refusal_rate
    recall = true_refusal_rate  # same numerator/denominator when treating TP only
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "true_refusal_rate": true_refusal_rate,
        "false_refusal_rate": false_refusal_rate,
        "f1_refusal": f1,
        "n_out_of_scope": len(out_of_scope_responses),
        "n_standard": len(standard_responses),
    }


def compute_citation_rate(responses: list[str]) -> float:
    """Fraction of responses containing at least one [Source: ...] citation.

    Args:
        responses: LLM-generated answers.

    Returns:
        Citation rate in [0, 1].
    """
    import re
    citation_pattern = re.compile(r"\[Source:", re.IGNORECASE)
    cited = sum(1 for r in responses if citation_pattern.search(r))
    return cited / len(responses) if responses else 0.0
