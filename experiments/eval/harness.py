"""Main evaluation harness — pluggable retrieval, reusable metrics pipeline.

Any retrieval configuration (embedding model, chunking strategy, retrieval
method, prompt variant) can be evaluated by passing a ``RetrievalFn``
callable. The harness runs it against the gold dataset, computes all metrics
with bootstrap CIs, logs to MLflow, and saves a structured JSON result.

Usage::

    from experiments.eval.harness import EvalHarness, RetrievalResult
    from experiments.eval.dataset import load_gold_dataset

    def my_retrieval_fn(query: str) -> RetrievalResult:
        docs = my_retriever.retrieve(query)
        return RetrievalResult(
            doc_ids=[d.metadata["chunk_id"] for d in docs],
            doc_texts=[d.page_content for d in docs],
        )

    harness = EvalHarness(
        dataset=load_gold_dataset(),
        config_name="my_experiment",
        experiment_name="retrieval_ablation",
    )
    results = harness.run(my_retrieval_fn)
    harness.save_results(results, "experiments/results/my_experiment.json")
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.eval.dataset import EvalDataset, load_gold_dataset
from experiments.eval.metrics import (
    compute_retrieval_metrics_with_text,
    build_relevant_ids_from_text_match,
)
from experiments.utils.bootstrap import bootstrap_ci_all, compare_configs
from experiments.utils.mlflow_utils import (
    init_mlflow,
    log_run,
    log_metrics_with_ci,
    save_and_log_json,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Output of a single retrieval call."""

    doc_ids: list[str]     # ranked, position 0 = most relevant
    doc_texts: list[str]   # parallel to doc_ids
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


RetrievalFn = Callable[[str], RetrievalResult]


@dataclass
class GenerationResult:
    """Output of a single RAG generation call."""

    response: str
    retrieved_doc_ids: list[str]
    retrieved_doc_texts: list[str]
    latency_ms: float = 0.0


GenerationFn = Callable[[str], GenerationResult]


@dataclass
class HarnessResults:
    """Full results for one evaluated configuration."""

    config_name: str
    config_params: dict
    n_queries: int
    retrieval_metrics: dict[str, dict]   # name → {mean, std, per_query}
    retrieval_metrics_ci: dict[str, dict] # name → {point, lower, upper, std_err}
    generation_metrics: dict = field(default_factory=dict)
    latency_stats: dict = field(default_factory=dict)
    error_analysis: list[dict] = field(default_factory=list)
    comparison_vs_baseline: dict = field(default_factory=dict)
    timestamp: str = ""


class EvalHarness:
    """Reusable evaluation harness for retrieval and generation experiments.

    Args:
        dataset: EvalDataset to evaluate against.
        config_name: Short identifier for this configuration (used in filenames).
        experiment_name: MLflow experiment name (e.g., "embedding_comparison").
        mlflow_tracking_uri: Path to local MLflow tracking store.
        k_values: List of k cut-offs for ranked metrics.
        baseline_results: Optional baseline HarnessResults for comparison.
    """

    def __init__(
        self,
        dataset: Optional[EvalDataset] = None,
        config_name: str = "unnamed",
        experiment_name: str = "rag_experiments",
        mlflow_tracking_uri: str = "file:./mlruns",
        k_values: Optional[list[int]] = None,
        baseline_results: Optional[HarnessResults] = None,
    ) -> None:
        self.dataset = dataset or load_gold_dataset()
        self.config_name = config_name
        self.experiment_name = experiment_name
        self.k_values = k_values or [1, 3, 5, 10]
        self.baseline_results = baseline_results
        self._mlflow_client = init_mlflow(experiment_name, mlflow_tracking_uri)

    def run_retrieval_eval(
        self,
        retrieval_fn: RetrievalFn,
        config_params: dict | None = None,
        use_standard_only: bool = True,
        n_bootstrap: int = 1000,
    ) -> HarnessResults:
        """Evaluate a retrieval function against the dataset.

        Args:
            retrieval_fn: Callable(query: str) → RetrievalResult.
            config_params: Hyperparameters to log (for MLflow).
            use_standard_only: If True, evaluate only on "standard" category
                questions (not out-of-scope/ambiguous). Retrieval metrics
                are only meaningful for answerable questions.
            n_bootstrap: Bootstrap resamples for CI.

        Returns:
            HarnessResults with all metrics, CIs, and error analysis.
        """
        params = config_params or {}
        eval_dataset = self.dataset.filter_standard() if use_standard_only else self.dataset
        n = len(eval_dataset)
        logger.info(f"Eval: {self.config_name} on {n} queries (standard_only={use_standard_only})")

        all_retrieved_ids: list[list[str]] = []
        all_retrieved_texts: list[list[str]] = []
        latencies: list[float] = []

        for pair in eval_dataset.pairs:
            t0 = time.perf_counter()
            result = retrieval_fn(pair.question)
            latency_ms = (time.perf_counter() - t0) * 1000

            all_retrieved_ids.append(result.doc_ids)
            all_retrieved_texts.append(result.doc_texts)
            latencies.append(result.latency_ms or latency_ms)

        # Build pseudo-relevance labels from gold answers via keyword overlap
        all_candidate_ids = list({cid for ids in all_retrieved_ids for cid in ids})
        all_candidate_texts = []
        for ids in all_retrieved_ids:
            for cid, ctext in zip(
                all_retrieved_ids[all_retrieved_ids.index(ids)],
                all_retrieved_texts[all_retrieved_texts.index(all_retrieved_texts[all_retrieved_ids.index(ids)])],
            ):
                pass  # handled inline below

        # Flatten unique id→text mapping
        id_to_text: dict[str, str] = {}
        for ids, texts in zip(all_retrieved_ids, all_retrieved_texts):
            for cid, ctext in zip(ids, texts):
                id_to_text[cid] = ctext

        relevant_ids_per_query = build_relevant_ids_from_text_match(
            gold_answers=eval_dataset.ground_truths,
            candidate_texts=list(id_to_text.values()),
            candidate_ids=list(id_to_text.keys()),
        )

        # Compute metrics
        metrics = compute_retrieval_metrics_with_text(
            queries=eval_dataset.questions,
            retrieved_ids_per_query=all_retrieved_ids,
            retrieved_texts_per_query=all_retrieved_texts,
            relevant_ids_per_query=relevant_ids_per_query,
            k_values=self.k_values,
        )

        # Bootstrap CIs
        per_query_metrics = {name: m["per_query"] for name, m in metrics.items()}
        metrics_ci = bootstrap_ci_all(per_query_metrics, n_bootstrap=n_bootstrap)

        # Latency stats
        lat_arr = np.array(latencies)
        latency_stats = {
            "mean_ms": float(lat_arr.mean()),
            "median_ms": float(np.median(lat_arr)),
            "p95_ms": float(np.percentile(lat_arr, 95)),
            "max_ms": float(lat_arr.max()),
        }

        # Error analysis: bottom 10 queries by nDCG@5
        ndcg5_scores = metrics.get("ndcg_at_5", {}).get("per_query", [])
        error_analysis = self._error_analysis(
            pairs=eval_dataset.pairs,
            retrieved_ids=all_retrieved_ids,
            relevant_ids=relevant_ids_per_query,
            per_query_ndcg5=ndcg5_scores,
            top_n=10,
        )

        # Baseline comparison
        comparison = {}
        if self.baseline_results is not None:
            baseline_pq = {
                name: m["per_query"]
                for name, m in self.baseline_results.retrieval_metrics.items()
            }
            comparison = compare_configs(
                baseline_scores=baseline_pq,
                challenger_scores=per_query_metrics,
                primary_metric="ndcg_at_5",
            )

        import datetime
        results = HarnessResults(
            config_name=self.config_name,
            config_params=params,
            n_queries=n,
            retrieval_metrics=metrics,
            retrieval_metrics_ci=metrics_ci,
            latency_stats=latency_stats,
            error_analysis=error_analysis,
            comparison_vs_baseline=comparison,
            timestamp=datetime.datetime.now().isoformat(),
        )

        # Log to MLflow
        with log_run(
            params={**params, "config_name": self.config_name, "n_queries": n},
            run_name=self.config_name,
        ):
            log_metrics_with_ci(metrics_ci)
            mlflow_lat = {f"latency_{k}": v for k, v in latency_stats.items()}
            import mlflow
            mlflow.log_metrics(mlflow_lat)

        return results

    def run_generation_eval(
        self,
        generation_fn: GenerationFn,
        config_params: dict | None = None,
        include_ragas: bool = False,
        ragas_kwargs: dict | None = None,
    ) -> dict:
        """Evaluate a full RAG generation function (retrieval + generation).

        Args:
            generation_fn: Callable(query: str) → GenerationResult.
            config_params: Hyperparameters for logging.
            include_ragas: Whether to run the RAGAS LLM-as-judge evaluation.
                Slow on CPU — only recommended for finalist configurations.
            ragas_kwargs: Extra kwargs passed to ``run_ragas_evaluation``.

        Returns:
            Dict with generation metrics (faithfulness heuristic, refusal
            accuracy, citation rate, and optionally RAGAS scores).
        """
        from experiments.eval.ragas_eval import (
            faithfulness_heuristic,
            compute_refusal_accuracy,
            compute_citation_rate,
            run_ragas_evaluation,
        )

        all_responses: list[str] = []
        all_retrieved_texts: list[list[str]] = []
        categories: list[str] = []

        for pair in self.dataset.pairs:
            result = generation_fn(pair.question)
            all_responses.append(result.response)
            all_retrieved_texts.append(result.retrieved_doc_texts)
            categories.append(pair.category)

        # Fast heuristic metrics (no LLM)
        faithfulness_scores = [
            faithfulness_heuristic(r, ctxs)
            for r, ctxs in zip(all_responses, all_retrieved_texts)
        ]
        refusal_metrics = compute_refusal_accuracy(all_responses, categories)
        citation_rate = compute_citation_rate(all_responses)

        gen_metrics = {
            "faithfulness_heuristic": {
                "mean": float(np.mean(faithfulness_scores)),
                "per_sample": faithfulness_scores,
            },
            "citation_rate": citation_rate,
            **refusal_metrics,
        }

        # Optional: RAGAS LLM-as-judge (slow)
        if include_ragas:
            logger.info("Running RAGAS evaluation (this is slow on CPU)...")
            ragas_scores = run_ragas_evaluation(
                questions=self.dataset.questions,
                responses=all_responses,
                retrieved_contexts_per_query=all_retrieved_texts,
                ground_truths=self.dataset.ground_truths,
                **(ragas_kwargs or {}),
            )
            gen_metrics["ragas"] = ragas_scores

        return gen_metrics

    def save_results(self, results: HarnessResults, path: str | Path) -> None:
        """Save HarnessResults to JSON and log as MLflow artifact.

        Args:
            results: Results to save.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config_name": results.config_name,
            "config_params": results.config_params,
            "n_queries": results.n_queries,
            "timestamp": results.timestamp,
            "retrieval_metrics": {
                name: {
                    "mean": m["mean"],
                    "std": m["std"],
                    "ci": results.retrieval_metrics_ci.get(name, {}),
                }
                for name, m in results.retrieval_metrics.items()
            },
            "latency_stats": results.latency_stats,
            "generation_metrics": results.generation_metrics,
            "comparison_vs_baseline": results.comparison_vs_baseline,
            "error_analysis": results.error_analysis,
        }

        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved results to {path}")

    @staticmethod
    def _error_analysis(
        pairs,
        retrieved_ids: list[list[str]],
        relevant_ids: list[set[str]],
        per_query_ndcg5: list[float],
        top_n: int = 10,
    ) -> list[dict]:
        """Identify the worst-performing queries for failure mode analysis."""
        indexed = sorted(
            enumerate(per_query_ndcg5), key=lambda x: x[1]
        )[:top_n]

        analysis = []
        for idx, score in indexed:
            pair = pairs[idx]
            retrieved = retrieved_ids[idx]
            relevant = relevant_ids[idx]

            analysis.append({
                "id": pair.id,
                "question": pair.question,
                "category": pair.category,
                "ndcg_at_5": score,
                "n_retrieved": len(retrieved),
                "n_relevant_found": len(set(retrieved[:5]) & relevant),
                "n_relevant_total": len(relevant),
                "failure_mode": _classify_failure(
                    retrieved[:5], relevant, pair.ground_truth
                ),
            })
        return analysis


def _classify_failure(
    top5_retrieved: list[str],
    relevant_ids: set[str],
    ground_truth: str,
) -> str:
    """Heuristic classification of retrieval failure mode."""
    if not relevant_ids:
        return "no_relevant_docs_in_index"

    hits = set(top5_retrieved) & relevant_ids
    if not hits:
        if not top5_retrieved:
            return "nothing_retrieved"
        return "low_lexical_overlap"  # retrieved docs exist but don't match

    if len(hits) < len(relevant_ids):
        return "partial_recall"  # found some but not all relevant

    return "ranking_error"  # relevant docs retrieved but ranked poorly


def load_results(path: str | Path) -> dict:
    """Load previously saved HarnessResults from JSON.

    Args:
        path: Path to the JSON file saved by ``save_results``.

    Returns:
        Raw dict (not a HarnessResults object — use for reporting only).
    """
    with Path(path).open() as f:
        return json.load(f)


def summarise_all_results(results_dir: str | Path) -> list[dict]:
    """Load all results JSON files and return a sorted summary table.

    Args:
        results_dir: Directory containing ``*.json`` result files.

    Returns:
        List of summary dicts sorted by nDCG@5 descending.
    """
    results_dir = Path(results_dir)
    summaries = []

    for p in sorted(results_dir.glob("*.json")):
        try:
            data = load_results(p)
            ci = data.get("retrieval_metrics", {}).get("ndcg_at_5", {}).get("ci", {})
            summaries.append({
                "config": data.get("config_name", p.stem),
                "ndcg_at_5": ci.get("point", float("nan")),
                "ndcg_at_5_lower": ci.get("lower", float("nan")),
                "ndcg_at_5_upper": ci.get("upper", float("nan")),
                "mrr": data.get("retrieval_metrics", {}).get("mrr", {}).get("ci", {}).get("point", float("nan")),
                "mean_latency_ms": data.get("latency_stats", {}).get("mean_ms", float("nan")),
                "n_queries": data.get("n_queries", 0),
            })
        except Exception as exc:
            logger.warning(f"Failed to load {p}: {exc}")

    summaries.sort(key=lambda x: x["ndcg_at_5"], reverse=True)
    return summaries
