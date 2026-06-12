"""Embedding model comparison experiment.

# HYPOTHESIS
# Different embedding models will show a clear quality-speed Pareto frontier:
# - all-MiniLM-L6-v2 (baseline) will be fast but lower quality
# - BAAI/bge-base-en-v1.5 will score highest on retrieval metrics
#   (MTEB SOTA at its size tier) but take ~3x longer to query
# - BAAI/bge-small-en-v1.5 will match or beat all-MiniLM-L6-v2 quality
#   at similar speed (BGE training is more aligned with retrieval tasks)
# - all-mpnet-base-v2 will be the best-balanced option if bge-base is too slow
#
# Expected primary failure mode: all-MiniLM tends to struggle with questions
# that use different vocabulary from the document (paraphrase gap); BGE models
# use hard-negative mining which reduces this.
"""

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from experiments.eval.dataset import load_gold_dataset
from experiments.eval.harness import EvalHarness, RetrievalResult
from experiments.utils.mlflow_utils import init_mlflow, log_run, log_metrics_with_ci, save_and_log_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        base = yaml.safe_load(open(Path(config_path).parent / "base.yaml"))
        emb = yaml.safe_load(f)
    base.update(emb)
    return base


def load_and_chunk_documents(data_dir: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Load all .txt/.md/.pdf files and chunk them."""
    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    data_path = Path(data_dir)
    docs = []
    for fp in data_path.rglob("*"):
        if fp.suffix == ".txt":
            loader = TextLoader(str(fp), encoding="utf-8")
            docs.extend(loader.load())
        elif fp.suffix in {".md"}:
            loader = TextLoader(str(fp), encoding="utf-8")
            docs.extend(loader.load())
        elif fp.suffix == ".pdf":
            loader = PyPDFLoader(str(fp))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def build_temp_vector_store(
    chunks: list[Document],
    embedding_model_id: str,
    query_prefix: str = "",
    passage_prefix: str = "",
    device: str = "cpu",
    persist_dir: str = "./chroma_experiments",
) -> tuple[Chroma, str]:
    """Index chunks into a temporary isolated ChromaDB collection.

    Returns (Chroma instance, collection_name).
    """
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Prepend passage prefix to chunk content if required (E5, BGE models)
    indexed_chunks = chunks
    if passage_prefix:
        indexed_chunks = [
            Document(
                page_content=passage_prefix + c.page_content,
                metadata=c.metadata,
            )
            for c in chunks
        ]

    collection_name = f"exp_{uuid.uuid4().hex[:8]}"
    ids = [
        f"{c.metadata.get('source', 'doc')}__chunk_{c.metadata.get('chunk_index', i)}"
        for i, c in enumerate(indexed_chunks)
    ]

    t0 = time.perf_counter()
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(documents=indexed_chunks, ids=ids)
    index_time_s = time.perf_counter() - t0

    logger.info(
        f"Indexed {len(indexed_chunks)} chunks with {embedding_model_id} "
        f"in {index_time_s:.1f}s → collection={collection_name}"
    )
    return store, collection_name, index_time_s


def make_retrieval_fn(
    store: Chroma,
    top_k: int = 10,
    query_prefix: str = "",
    similarity_threshold: float = 0.0,
) -> callable:
    """Return a RetrievalFn that queries the given Chroma store."""
    def retrieval_fn(query: str) -> RetrievalResult:
        prefixed_query = query_prefix + query if query_prefix else query
        t0 = time.perf_counter()
        results = store.similarity_search_with_relevance_scores(
            query=prefixed_query, k=top_k
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        filtered = [
            (doc, score) for doc, score in results
            if score >= similarity_threshold
        ]

        doc_ids = [
            f"{doc.metadata.get('source', 'doc')}__chunk_{doc.metadata.get('chunk_index', i)}"
            for i, (doc, _) in enumerate(filtered)
        ]
        doc_texts = [doc.page_content for doc, _ in filtered]

        return RetrievalResult(
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            latency_ms=latency_ms,
        )
    return retrieval_fn


def measure_memory_mb(store: Chroma) -> float:
    """Estimate in-memory size of the embedding matrix in MB."""
    try:
        n = store._collection.count()
        # Peek at embedding dim
        sample = store._collection.peek(limit=1)
        if sample and sample.get("embeddings") and sample["embeddings"]:
            dim = len(sample["embeddings"][0])
            return (n * dim * 4) / (1024 ** 2)  # float32
    except Exception:
        pass
    return float("nan")


def plot_pareto(results: list[dict], output_path: str) -> None:
    """Generate quality vs. latency Pareto frontier plot."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.set_style("whitegrid")

    names = [r["model_name"] for r in results]
    x = [r["mean_query_latency_ms"] for r in results]
    y = [r["ndcg_at_5"] for r in results]
    y_err_low = [r["ndcg_at_5"] - r.get("ndcg_at_5_ci_lower", r["ndcg_at_5"]) for r in results]
    y_err_high = [r.get("ndcg_at_5_ci_upper", r["ndcg_at_5"]) - r["ndcg_at_5"] for r in results]

    colors = ["red" if r.get("is_baseline") else "steelblue" for r in results]

    for xi, yi, yl, yh, name, color in zip(x, y, y_err_low, y_err_high, names, colors):
        ax.errorbar(xi, yi, yerr=[[yl], [yh]], fmt="o", color=color, markersize=10,
                    capsize=5, elinewidth=1.5, zorder=3)
        ax.annotate(name, (xi, yi), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Mean Query Latency (ms)", fontsize=12)
    ax.set_ylabel("nDCG@5 (95% CI)", fontsize=12)
    ax.set_title("Embedding Model Pareto Frontier: Quality vs. Speed", fontsize=13)

    baseline = next((r for r in results if r.get("is_baseline")), None)
    if baseline:
        ax.axhline(baseline["ndcg_at_5"], color="red", linestyle="--", alpha=0.4,
                   label=f"Baseline ({baseline['model_name']})")
        ax.legend(fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Pareto plot saved to {output_path}")


def run(config_path: str, data_dir: str | None = None) -> None:
    cfg = load_config(config_path)
    exp_cfg = cfg["experiment"]
    eval_cfg = cfg["evaluation"]

    data_dir = data_dir or exp_cfg["data_dir"]
    results_dir = Path(exp_cfg["results_dir"])
    plots_dir = Path(exp_cfg["plots_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    init_mlflow(cfg["experiment_name"], exp_cfg["mlflow_tracking_uri"])
    dataset = load_gold_dataset(exp_cfg["gold_qa_path"])

    chunk_cfg = cfg["chunking"]
    chunks = load_and_chunk_documents(
        data_dir, chunk_cfg["chunk_size"], chunk_cfg["chunk_overlap"]
    )
    logger.info(f"Loaded {len(chunks)} chunks from {data_dir}")

    query_prefixes = cfg.get("query_prefixes", {})
    passage_prefixes = cfg.get("passage_prefixes", {})

    all_results = []
    baseline_results_obj = None

    for model_cfg in cfg["models"]:
        model_id = model_cfg["model_id"]
        model_name = model_cfg["name"]
        is_baseline = model_cfg.get("is_baseline", False)

        logger.info(f"\n{'='*60}\nEvaluating: {model_name} ({model_id})\n{'='*60}")

        query_prefix = query_prefixes.get(model_id, "")
        passage_prefix = passage_prefixes.get(model_id, "")

        try:
            store, collection_name, index_time_s = build_temp_vector_store(
                chunks=chunks,
                embedding_model_id=model_id,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
                persist_dir=cfg["chroma"]["experiment_persist_dir"],
            )

            mem_mb = measure_memory_mb(store)
            retrieval_fn = make_retrieval_fn(
                store=store,
                top_k=cfg["retrieval"]["top_k"],
                query_prefix=query_prefix,
                similarity_threshold=cfg["retrieval"]["similarity_threshold"],
            )

            harness = EvalHarness(
                dataset=dataset,
                config_name=model_name,
                experiment_name=cfg["experiment_name"],
                mlflow_tracking_uri=exp_cfg["mlflow_tracking_uri"],
                k_values=eval_cfg["k_values"],
                baseline_results=baseline_results_obj,
            )

            with log_run(
                params={
                    "model_id": model_id,
                    "model_dim": model_cfg.get("dim"),
                    "is_baseline": is_baseline,
                    "chunk_size": chunk_cfg["chunk_size"],
                    "chunk_overlap": chunk_cfg["chunk_overlap"],
                },
                run_name=model_name,
                tags={"experiment": "embedding_comparison"},
            ):
                result_obj = harness.run_retrieval_eval(
                    retrieval_fn=retrieval_fn,
                    config_params=model_cfg,
                    n_bootstrap=eval_cfg["bootstrap_n"],
                )
                import mlflow
                mlflow.log_metrics({
                    "index_time_s": index_time_s,
                    "memory_mb": mem_mb if not np.isnan(mem_mb) else -1,
                })

            if is_baseline:
                baseline_results_obj = result_obj

            ndcg5_ci = result_obj.retrieval_metrics_ci.get("ndcg_at_5", {})
            mrr_ci = result_obj.retrieval_metrics_ci.get("mrr", {})
            row = {
                "model_name": model_name,
                "model_id": model_id,
                "dim": model_cfg.get("dim"),
                "is_baseline": is_baseline,
                "ndcg_at_5": ndcg5_ci.get("point", float("nan")),
                "ndcg_at_5_ci_lower": ndcg5_ci.get("lower", float("nan")),
                "ndcg_at_5_ci_upper": ndcg5_ci.get("upper", float("nan")),
                "mrr": mrr_ci.get("point", float("nan")),
                "mean_query_latency_ms": result_obj.latency_stats.get("mean_ms", float("nan")),
                "index_time_s": index_time_s,
                "memory_mb": mem_mb,
            }
            # Add all metric means
            for name, m in result_obj.retrieval_metrics.items():
                if name not in row:
                    row[name] = m["mean"]

            all_results.append(row)
            harness.save_results(result_obj, results_dir / f"emb_{model_name}.json")

            # Cleanup temp collection
            try:
                store._client.delete_collection(collection_name)
            except Exception:
                pass

        except Exception as exc:
            logger.error(f"FAILED for {model_name}: {exc}", exc_info=True)
            all_results.append({"model_name": model_name, "error": str(exc)})

    # Save summary
    summary_path = results_dir / "embedding_comparison_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    # Pareto plot
    valid = [r for r in all_results if "error" not in r]
    if valid:
        plot_pareto(valid, str(plots_dir / "embedding_pareto.png"))

    # Print leaderboard
    valid.sort(key=lambda x: x.get("ndcg_at_5", 0), reverse=True)
    print("\n=== EMBEDDING COMPARISON LEADERBOARD ===")
    print(f"{'Rank':<5} {'Model':<30} {'nDCG@5':<10} {'MRR':<10} {'Latency(ms)':<14} {'Memory(MB)':<12}")
    print("-" * 85)
    for rank, r in enumerate(valid, 1):
        marker = " <- BEST" if rank == 1 else (" <- BASELINE" if r.get("is_baseline") else "")
        print(
            f"{rank:<5} {r['model_name']:<30} "
            f"{r.get('ndcg_at_5', float('nan')):<10.3f} "
            f"{r.get('mrr', float('nan')):<10.3f} "
            f"{r.get('mean_query_latency_ms', float('nan')):<14.1f} "
            f"{r.get('memory_mb', float('nan')):<12.1f}"
            f"{marker}"
        )

    best = valid[0]["model_name"] if valid else "unknown"
    print(f"\nRecommendation: Use '{best}' as the embedding model in subsequent experiments.")
    print(f"Update experiments/config/retrieval.yaml → embedding_model: \"{valid[0]['model_id']}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run embedding model comparison")
    parser.add_argument(
        "--config", default="experiments/config/embeddings.yaml",
        help="Path to embeddings.yaml config"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override data directory from config"
    )
    args = parser.parse_args()
    run(args.config, args.data_dir)
