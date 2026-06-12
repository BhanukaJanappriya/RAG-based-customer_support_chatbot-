"""Retrieval strategy ablation experiment.

# HYPOTHESIS
# Dense retrieval is the weakest baseline — it struggles with exact-term queries
# (prices, URLs, email addresses) where vocabulary overlap matters.
# BM25 alone will outperform dense on these but fail on semantic paraphrase queries.
# Hybrid RRF will dominate both by combining their strengths.
# Cross-encoder reranking on top of hybrid will add marginal gains at the cost
# of significant latency (cross-encoder forward pass per candidate).
# HyDE and multi-query are the highest-risk strategies: their gains depend on
# Llama 3.2 1B's instruction-following quality for rephrasing/hypothesis tasks.
#
# Expected Pareto winner: hybrid_rrf or hybrid_rrf_reranked depending on the
# latency budget. Pure BM25 will likely beat dense-only for this corpus (factual
# customer support text with specific numbers, URLs, policy terms).
"""

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from experiments.eval.dataset import load_gold_dataset
from experiments.eval.harness import EvalHarness, RetrievalResult, HarnessResults, summarise_all_results
from experiments.retrieval_experiments.chunking import chunk_recursive, chunk_fixed, chunk_sentence_window, expand_with_window
from experiments.retrieval_experiments.sparse import BM25Retriever
from experiments.retrieval_experiments.hybrid import HybridRetriever
from experiments.retrieval_experiments.reranking import CrossEncoderReranker
from experiments.retrieval_experiments.hyde import HyDERetriever
from experiments.retrieval_experiments.query_expansion import MultiQueryRetriever
from experiments.utils.mlflow_utils import init_mlflow, log_run, log_metrics_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(Path(config_path).parent / "base.yaml") as f:
        base = yaml.safe_load(f)
    base.update(cfg)
    return base


def load_documents(data_dir: str) -> list[Document]:
    data_path = Path(data_dir)
    docs = []
    for fp in data_path.rglob("*"):
        if fp.suffix in {".txt", ".md"}:
            docs.extend(TextLoader(str(fp), encoding="utf-8").load())
        elif fp.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(fp)).load())
    return docs


def apply_chunking_strategy(
    docs: list[Document],
    strategy_cfg: dict,
    embedding_model=None,
) -> tuple[list[Document], Optional[dict]]:
    """Apply a chunking strategy and return (chunks, window_map).

    window_map is only set for sentence_window strategy.
    """
    strategy = strategy_cfg["strategy"]
    if strategy in {"fixed", "recursive"}:
        chunk_fn = chunk_fixed if strategy == "fixed" else chunk_recursive
        chunks = chunk_fn(docs, strategy_cfg["chunk_size"], strategy_cfg["chunk_overlap"])
        return chunks, None
    elif strategy == "semantic":
        from experiments.retrieval_experiments.chunking import chunk_semantic
        return chunk_semantic(
            docs,
            embedding_model,
            breakpoint_threshold_type=strategy_cfg.get("breakpoint_threshold_type", "percentile"),
            breakpoint_threshold_amount=strategy_cfg.get("breakpoint_threshold_amount", 95),
        ), None
    elif strategy == "sentence_window":
        chunks, window_map = chunk_sentence_window(
            docs, window_size=strategy_cfg.get("window_size", 3)
        )
        return chunks, window_map
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def build_chroma_store(
    chunks: list[Document],
    embedding_model_id: str,
    persist_dir: str,
    passage_prefix: str = "",
) -> tuple[Chroma, str]:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    indexed = chunks
    if passage_prefix:
        indexed = [
            Document(page_content=passage_prefix + c.page_content, metadata=c.metadata)
            for c in chunks
        ]

    collection_name = f"ret_{uuid.uuid4().hex[:8]}"
    ids = [
        f"{c.metadata.get('source', 'doc')}__chunk_{c.metadata.get('chunk_index', i)}"
        for i, c in enumerate(indexed)
    ]

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(documents=indexed, ids=ids)
    return store, collection_name


def doc_to_result(doc: Document, store_ids: list[str] | None = None) -> tuple[str, str]:
    cid = (
        f"{doc.metadata.get('source', 'doc')}__chunk_"
        f"{doc.metadata.get('chunk_index', 0)}"
    )
    return cid, doc.page_content


def make_dense_fn(store: Chroma, top_k: int, query_prefix: str = "") -> callable:
    def fn(query: str) -> RetrievalResult:
        pq = query_prefix + query if query_prefix else query
        t0 = time.perf_counter()
        results = store.similarity_search_with_relevance_scores(pq, k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        ids, texts = [], []
        for doc, _ in results:
            cid, text = doc_to_result(doc)
            ids.append(cid)
            texts.append(text)
        return RetrievalResult(doc_ids=ids, doc_texts=texts, latency_ms=lat)
    return fn


def make_bm25_fn(bm25: BM25Retriever, top_k: int) -> callable:
    def fn(query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        results = bm25.retrieve_with_ids(query, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        ids = [r[0] for r in results]
        texts = [r[1] for r in results]
        return RetrievalResult(doc_ids=ids, doc_texts=texts, latency_ms=lat)
    return fn


def make_hybrid_fn(hybrid: HybridRetriever, top_k: int) -> callable:
    def fn(query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        docs = hybrid.retrieve(query)[:top_k]
        lat = (time.perf_counter() - t0) * 1000
        ids = [doc_to_result(d)[0] for d in docs]
        texts = [d.page_content for d in docs]
        return RetrievalResult(doc_ids=ids, doc_texts=texts, latency_ms=lat)
    return fn


def make_reranked_fn(
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    top_k: int,
) -> callable:
    def fn(query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        candidates = hybrid.retrieve(query)
        docs = reranker.rerank(query, candidates, top_k=top_k)
        lat = (time.perf_counter() - t0) * 1000
        ids = [doc_to_result(d)[0] for d in docs]
        texts = [d.page_content for d in docs]
        return RetrievalResult(doc_ids=ids, doc_texts=texts, latency_ms=lat)
    return fn


def make_hyde_fn(hyde: HyDERetriever, top_k: int) -> callable:
    def fn(query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        docs = hyde.retrieve(query)[:top_k]
        lat = (time.perf_counter() - t0) * 1000
        ids = [doc_to_result(d)[0] for d in docs]
        texts = [d.page_content for d in docs]
        return RetrievalResult(doc_ids=ids, doc_texts=texts, latency_ms=lat)
    return fn


def make_multiquery_fn(mq: MultiQueryRetriever, top_k: int) -> callable:
    def fn(query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        docs = mq.retrieve(query)[:top_k]
        lat = (time.perf_counter() - t0) * 1000
        ids = [doc_to_result(d)[0] for d in docs]
        texts = [d.page_content for d in docs]
        return RetrievalResult(doc_ids=ids, doc_texts=texts, latency_ms=lat)
    return fn


def run_phase_chunking(cfg: dict, docs: list[Document], dataset, results_dir: Path) -> None:
    """Phase 1: Ablate chunking strategies with dense retrieval."""
    logger.info("\n" + "="*60 + "\nPHASE 1: CHUNKING STRATEGY ABLATION\n" + "="*60)
    embedding_model_id = cfg["embedding_model"]
    eval_cfg = cfg["evaluation"]
    baseline_results_obj = None
    phase_results = []

    semantic_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    for strategy_cfg in cfg["chunking_strategies"]:
        name = strategy_cfg["name"]
        is_baseline = strategy_cfg.get("is_baseline", False)
        logger.info(f"\n--- Chunking: {name} ---")

        chunks, _ = apply_chunking_strategy(docs, strategy_cfg, embedding_model=semantic_embeddings)
        store, coll_name = build_chroma_store(
            chunks, embedding_model_id, cfg["chroma"]["experiment_persist_dir"]
        )
        bm25 = BM25Retriever(chunks)
        retrieval_fn = make_dense_fn(store, cfg["eval"]["top_k"])

        harness = EvalHarness(
            dataset=dataset,
            config_name=f"chunk_{name}",
            experiment_name=cfg["experiment_name"],
            mlflow_tracking_uri=cfg["experiment"]["mlflow_tracking_uri"],
            k_values=eval_cfg["k_values"],
            baseline_results=baseline_results_obj,
        )
        result = harness.run_retrieval_eval(
            retrieval_fn=retrieval_fn,
            config_params={**strategy_cfg},
            n_bootstrap=eval_cfg["bootstrap_n"],
        )
        if is_baseline:
            baseline_results_obj = result
        harness.save_results(result, results_dir / f"chunk_{name}.json")

        ndcg5 = result.retrieval_metrics_ci.get("ndcg_at_5", {}).get("point", float("nan"))
        phase_results.append({"name": name, "ndcg_at_5": ndcg5, **result.latency_stats})

        try:
            store._client.delete_collection(coll_name)
        except Exception:
            pass

    phase_results.sort(key=lambda x: x["ndcg_at_5"], reverse=True)
    logger.info("\n=== CHUNKING LEADERBOARD ===")
    for r in phase_results:
        logger.info(f"  {r['name']:<30} nDCG@5={r['ndcg_at_5']:.3f}  lat={r.get('mean_ms', 0):.1f}ms")
    return phase_results


def run_phase_retrieval(
    cfg: dict,
    docs: list[Document],
    best_chunking_cfg: dict,
    dataset,
    results_dir: Path,
) -> list[dict]:
    """Phase 2: Ablate retrieval methods using the best chunking config."""
    logger.info("\n" + "="*60 + "\nPHASE 2: RETRIEVAL METHOD ABLATION\n" + "="*60)
    embedding_model_id = cfg["embedding_model"]
    eval_cfg = cfg["evaluation"]
    llm_cfg = cfg["baseline"]
    baseline_results_obj = None
    phase_results = []

    # Build shared chunks and indexes for this phase
    semantic_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    chunks, window_map = apply_chunking_strategy(docs, best_chunking_cfg, embedding_model=semantic_embeddings)
    store, coll_name = build_chroma_store(
        chunks, embedding_model_id, cfg["chroma"]["experiment_persist_dir"]
    )
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(store, bm25, dense_top_k=10, bm25_top_k=10, rrf_k=60, final_top_k=cfg["eval"]["top_k"])
    reranker = CrossEncoderReranker()

    for method_cfg in cfg["retrieval_methods"]:
        name = method_cfg["name"]
        method = method_cfg["method"]
        is_baseline = method_cfg.get("is_baseline", False)
        top_k = method_cfg.get("final_top_k", method_cfg.get("top_k", 4))
        logger.info(f"\n--- Retrieval: {name} ---")

        try:
            if method == "dense":
                retrieval_fn = make_dense_fn(store, cfg["eval"]["top_k"])
            elif method == "bm25":
                retrieval_fn = make_bm25_fn(bm25, top_k)
            elif method == "hybrid_rrf":
                retrieval_fn = make_hybrid_fn(hybrid, top_k)
            elif method == "hybrid_rrf_reranked":
                retrieval_fn = make_reranked_fn(hybrid, reranker, top_k)
            elif method == "hyde":
                hyde = HyDERetriever(
                    store,
                    llm_model=method_cfg.get("llm_model", llm_cfg["llm_model"]),
                    ollama_base_url=llm_cfg["ollama_base_url"],
                    top_k=top_k,
                    temperature=method_cfg.get("temperature", 0.3),
                )
                retrieval_fn = make_hyde_fn(hyde, top_k)
            elif method == "multi_query":
                mq = MultiQueryRetriever(
                    store,
                    llm_model=method_cfg.get("llm_model", llm_cfg["llm_model"]),
                    ollama_base_url=llm_cfg["ollama_base_url"],
                    n_rephrasings=method_cfg.get("n_rephrasings", 3),
                    top_k_per_query=method_cfg.get("top_k_per_query", 4),
                    final_top_k=top_k,
                    temperature=method_cfg.get("temperature", 0.3),
                )
                retrieval_fn = make_multiquery_fn(mq, top_k)
            else:
                logger.warning(f"Unknown method {method!r}, skipping")
                continue

            harness = EvalHarness(
                dataset=dataset,
                config_name=f"ret_{name}",
                experiment_name=cfg["experiment_name"],
                mlflow_tracking_uri=cfg["experiment"]["mlflow_tracking_uri"],
                k_values=eval_cfg["k_values"],
                baseline_results=baseline_results_obj,
            )
            result = harness.run_retrieval_eval(
                retrieval_fn=retrieval_fn,
                config_params={**method_cfg, "chunking": best_chunking_cfg["name"]},
                n_bootstrap=eval_cfg["bootstrap_n"],
            )
            if is_baseline:
                baseline_results_obj = result
            harness.save_results(result, results_dir / f"ret_{name}.json")

            ndcg5 = result.retrieval_metrics_ci.get("ndcg_at_5", {}).get("point", float("nan"))
            phase_results.append({
                "name": name, "ndcg_at_5": ndcg5, **result.latency_stats,
                "beats_baseline": result.comparison_vs_baseline.get("beats_baseline", False),
            })

        except Exception as exc:
            logger.error(f"FAILED {name}: {exc}", exc_info=True)
            phase_results.append({"name": name, "error": str(exc)})

    try:
        store._client.delete_collection(coll_name)
    except Exception:
        pass

    valid = [r for r in phase_results if "error" not in r]
    valid.sort(key=lambda x: x["ndcg_at_5"], reverse=True)
    logger.info("\n=== RETRIEVAL METHOD LEADERBOARD ===")
    for r in valid:
        marker = " <- BEST" if r == valid[0] else (" <- BASELINE" if r["name"] == "dense_only" else "")
        beats = " [beats baseline]" if r.get("beats_baseline") else ""
        logger.info(f"  {r['name']:<35} nDCG@5={r['ndcg_at_5']:.3f}  lat={r.get('mean_ms', 0):.1f}ms{beats}{marker}")
    return phase_results


def plot_retrieval_comparison(phase_results: list[dict], output_path: str) -> None:
    """Bar chart comparing retrieval methods by nDCG@5."""
    valid = [r for r in phase_results if "error" not in r and "ndcg_at_5" in r]
    if not valid:
        return
    valid.sort(key=lambda x: x["ndcg_at_5"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_style("whitegrid")

    names = [r["name"] for r in valid]
    ndcg5 = [r["ndcg_at_5"] for r in valid]
    lats = [r.get("mean_ms", 0) for r in valid]
    colors = ["#e74c3c" if "dense_only" in n else "#3498db" for n in names]

    ax1.barh(names, ndcg5, color=colors)
    ax1.set_xlabel("nDCG@5", fontsize=11)
    ax1.set_title("Retrieval Quality (nDCG@5)", fontsize=12)
    ax1.axvline(x=next((r["ndcg_at_5"] for r in valid if "dense_only" in r["name"]), 0),
                color="red", linestyle="--", alpha=0.5, label="Dense baseline")

    ax2.barh(names, lats, color=colors)
    ax2.set_xlabel("Mean Query Latency (ms)", fontsize=11)
    ax2.set_title("Query Latency", fontsize=12)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Retrieval comparison plot saved to {output_path}")


def run(config_path: str, data_dir: str | None = None, phases: str = "all") -> None:
    cfg = load_config(config_path)
    exp_cfg = cfg["experiment"]
    data_dir = data_dir or exp_cfg["data_dir"]
    results_dir = Path(exp_cfg["results_dir"]) / "retrieval"
    plots_dir = Path(exp_cfg["plots_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    init_mlflow(cfg["experiment_name"], exp_cfg["mlflow_tracking_uri"])
    dataset = load_gold_dataset(exp_cfg["gold_qa_path"])
    docs = load_documents(data_dir)
    logger.info(f"Loaded {len(docs)} documents from {data_dir}")

    run_chunking = phases in {"all", "chunking"}
    run_retrieval = phases in {"all", "retrieval"}

    chunking_results = []
    if run_chunking:
        chunking_results = run_phase_chunking(cfg, docs, dataset, results_dir)

    # Pick best chunking config for retrieval phase
    if chunking_results:
        best_chunking_name = max(chunking_results, key=lambda r: r.get("ndcg_at_5", 0))["name"]
        best_chunking_cfg = next(
            c for c in cfg["chunking_strategies"] if c["name"] == best_chunking_name
        )
        logger.info(f"Best chunking for retrieval phase: {best_chunking_name}")
    else:
        # Default to production baseline if chunking phase not run
        best_chunking_cfg = next(
            c for c in cfg["chunking_strategies"] if c.get("is_baseline")
        )

    retrieval_results = []
    if run_retrieval:
        retrieval_results = run_phase_retrieval(
            cfg, docs, best_chunking_cfg, dataset, results_dir
        )
        plot_retrieval_comparison(
            retrieval_results, str(plots_dir / "retrieval_comparison.png")
        )

    # Save combined summary
    summary = {
        "best_chunking": best_chunking_cfg.get("name"),
        "chunking_results": chunking_results,
        "retrieval_results": retrieval_results,
    }
    with (results_dir / "retrieval_ablation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    if retrieval_results:
        valid = [r for r in retrieval_results if "error" not in r]
        if valid:
            best = max(valid, key=lambda r: r.get("ndcg_at_5", 0))
            print(f"\nBest retrieval config: {best['name']} (nDCG@5={best['ndcg_at_5']:.3f})")
            print(f"  Update experiments/config/prompts.yaml -> retrieval_method: \"{best['name']}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval ablation")
    parser.add_argument("--config", default="experiments/config/retrieval.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--phases", default="all", choices=["all", "chunking", "retrieval"])
    args = parser.parse_args()
    run(args.config, args.data_dir, args.phases)
