"""Prompt engineering ablation experiment.

# HYPOTHESIS
# The baseline prompt (zero-shot, soft citation enforcement) likely has:
# - Low citation rate: Llama 3.2 1B omits [Source:] citations ~30-50% of the time
# - Moderate refusal accuracy: ~70-80% on out-of-scope questions
#   (small models sometimes hallucinate answers for OOS queries)
#
# Expected gains:
# - Few-shot (2-3 examples) will improve citation rate and refusal accuracy
#   most reliably (in-context demonstration > instruction alone for small models)
# - CoT may help with multi-hop questions but adds latency and may hurt on
#   simple questions (over-reasoning)
# - Strict citation enforcement improves citation rate but risks formatting
#   artefacts that confuse downstream parsing
# - The biggest failure mode is false refusals on ambiguous questions —
#   track this separately from out-of-scope refusals

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from experiments.eval.dataset import load_gold_dataset, EvalDataset
from experiments.eval.harness import EvalHarness, GenerationResult, GenerationFn
from experiments.eval.ragas_eval import (
    faithfulness_heuristic,
    compute_refusal_accuracy,
    compute_citation_rate,
    answer_contains_refusal,
)
from experiments.prompt_experiments.templates import build_template, get_all_templates
from experiments.utils.mlflow_utils import init_mlflow, log_run, save_and_log_json
from experiments.utils.bootstrap import bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(Path(config_path).parent / "base.yaml") as f:
        base = yaml.safe_load(f)
    base.update(cfg)
    return base


def build_retrieval_fn(cfg: dict):
    """Build the retrieval function using the best config from retrieval ablation."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model_id = cfg.get("embedding_model", cfg["baseline"]["embedding_model"])
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(
        collection_name=cfg["baseline"]["chroma_collection_name"] if "chroma_collection_name" in cfg.get("baseline", {}) else cfg["chroma"]["production_collection"],
        embedding_function=embeddings,
        persist_directory=cfg["chroma"]["production_persist_dir"],
        collection_metadata={"hnsw:space": "cosine"},
    )
    top_k = cfg["baseline"]["retrieval_top_k"]

    def retrieve(query: str) -> list:
        return store.similarity_search(query, k=top_k)

    return retrieve, store


def make_generation_fn(
    template,
    llm,
    retrieve_fn,
) -> GenerationFn:
    """Create a GenerationFn that uses a specific prompt template."""
    chain = template | llm | StrOutputParser()

    def gen_fn(query: str) -> GenerationResult:
        t0 = time.perf_counter()
        docs = retrieve_fn(query)
        from app.generation.chain import format_context
        context = format_context(docs)

        response = chain.invoke({
            "context": context,
            "chat_history": [],
            "question": query,
        })
        lat = (time.perf_counter() - t0) * 1000

        ids = [
            f"{doc.metadata.get('source', 'doc')}__chunk_{doc.metadata.get('chunk_index', i)}"
            for i, doc in enumerate(docs)
        ]
        texts = [doc.page_content for doc in docs]
        return GenerationResult(
            response=response,
            retrieved_doc_ids=ids,
            retrieved_doc_texts=texts,
            latency_ms=lat,
        )

    return gen_fn


def evaluate_template(
    template_name: str,
    template,
    dataset: EvalDataset,
    llm,
    retrieve_fn,
    cfg: dict,
) -> dict:
    """Run full evaluation for one prompt template."""
    logger.info(f"\n--- Template: {template_name} ---")
    gen_fn = make_generation_fn(template, llm, retrieve_fn)

    responses = []
    retrieved_texts_all = []
    categories = []
    latencies = []

    for pair in dataset.pairs:
        result = gen_fn(pair.question)
        responses.append(result.response)
        retrieved_texts_all.append(result.retrieved_doc_texts)
        categories.append(pair.category)
        latencies.append(result.latency_ms)

    # Compute metrics
    faithfulness_scores = [
        faithfulness_heuristic(r, ctxs)
        for r, ctxs in zip(responses, retrieved_texts_all)
    ]
    refusal_metrics = compute_refusal_accuracy(responses, categories)
    citation_rate = compute_citation_rate(responses)

    lat_arr = np.array(latencies)
    faith_ci = bootstrap_ci(faithfulness_scores, n_bootstrap=cfg["evaluation"]["bootstrap_n"])

    results = {
        "template_name": template_name,
        "faithfulness_heuristic_mean": faith_ci["point"],
        "faithfulness_heuristic_ci": faith_ci,
        "citation_rate": citation_rate,
        "mean_latency_ms": float(lat_arr.mean()),
        "n_evaluated": len(dataset),
        **refusal_metrics,
        "per_response": [
            {
                "id": pair.id,
                "question": pair.question,
                "category": pair.category,
                "response_snippet": resp[:150],
                "faithfulness": faith,
                "has_citation": "[Source:" in resp,
                "is_refusal": answer_contains_refusal(resp),
                "latency_ms": lat,
            }
            for pair, resp, faith, lat in zip(
                dataset.pairs, responses, faithfulness_scores, latencies
            )
        ],
    }

    # Error analysis: false refusals (standard Q refused) and missed refusals (OOS answered)
    false_refusals = [
        p.question for p, r in zip(dataset.pairs, responses)
        if p.category == "standard" and answer_contains_refusal(r)
    ]
    missed_refusals = [
        p.question for p, r in zip(dataset.pairs, responses)
        if p.category == "out_of_scope" and not answer_contains_refusal(r)
    ]
    results["false_refusals"] = false_refusals
    results["missed_refusals"] = missed_refusals

    logger.info(
        f"  faithfulness={faith_ci['point']:.3f}, "
        f"citation_rate={citation_rate:.2%}, "
        f"true_refusal={refusal_metrics['true_refusal_rate']:.2%}, "
        f"false_refusal={refusal_metrics['false_refusal_rate']:.2%}, "
        f"lat={lat_arr.mean():.0f}ms"
    )
    return results


def plot_prompt_comparison(all_results: list[dict], output_path: str) -> None:
    """Grouped bar chart comparing key metrics across templates."""
    valid = [r for r in all_results if "error" not in r]
    if not valid:
        return

    metrics = ["faithfulness_heuristic_mean", "citation_rate", "true_refusal_rate"]
    metric_labels = ["Faithfulness (heuristic)", "Citation Rate", "True Refusal Rate"]
    names = [r["template_name"] for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.set_style("whitegrid")

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [r.get(metric, float("nan")) for r in valid]
        colors = ["#e74c3c" if "baseline" in n else "#3498db" for n in names]
        bars = ax.barh(names, values, color=colors)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, 1.05)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                        f"{val:.2f}", va="center", fontsize=8)

    plt.suptitle("Prompt Template Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Prompt comparison plot saved to {output_path}")


def run(config_path: str, data_dir: str | None = None) -> None:
    cfg = load_config(config_path)
    exp_cfg = cfg["experiment"]
    llm_cfg = cfg["llm"]

    results_dir = Path(exp_cfg["results_dir"]) / "prompts"
    plots_dir = Path(exp_cfg["plots_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    init_mlflow(cfg["experiment_name"], exp_cfg["mlflow_tracking_uri"])

    # Load full dataset (standard + adversarial) for prompt eval
    dataset = load_gold_dataset(exp_cfg["gold_qa_path"])
    logger.info(f"Dataset: {dataset.stats()}")

    # Build LLM and retrieval function (uses production ChromaDB)
    llm = ChatOllama(
        model=llm_cfg["model"],
        base_url=llm_cfg["base_url"],
        temperature=llm_cfg["temperature"],
        num_predict=llm_cfg["max_tokens"],
    )

    try:
        retrieve_fn, _ = build_retrieval_fn(cfg)
    except Exception as exc:
        logger.error(
            f"Could not connect to production ChromaDB: {exc}\n"
            "Make sure you have run `python scripts/ingest.py` first."
        )
        raise

    templates = get_all_templates(cfg["templates"])
    all_results = []
    baseline_result = None

    for tpl_cfg in cfg["templates"]:
        name = tpl_cfg["name"]
        is_baseline = tpl_cfg.get("is_baseline", False)
        template = templates[name]

        try:
            with log_run(
                params={**tpl_cfg},
                run_name=name,
                tags={"experiment": "prompt_ablation"},
            ):
                result = evaluate_template(
                    template_name=name,
                    template=template,
                    dataset=dataset,
                    llm=llm,
                    retrieve_fn=retrieve_fn,
                    cfg=cfg,
                )
                import mlflow
                mlflow.log_metrics({
                    k: v for k, v in result.items()
                    if isinstance(v, (int, float)) and not np.isnan(float(v))
                })

            if is_baseline:
                baseline_result = result

            all_results.append(result)
            with (results_dir / f"prompt_{name}.json").open("w") as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as exc:
            logger.error(f"FAILED {name}: {exc}", exc_info=True)
            all_results.append({"template_name": name, "error": str(exc)})

    # Save summary
    summary_path = results_dir / "prompt_ablation_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot
    plot_prompt_comparison(all_results, str(plots_dir / "prompt_comparison.png"))

    # Print leaderboard
    valid = [r for r in all_results if "error" not in r]
    # Sort by composite score: 0.4*faithfulness + 0.3*citation_rate + 0.3*true_refusal
    for r in valid:
        r["_score"] = (
            0.4 * r.get("faithfulness_heuristic_mean", 0)
            + 0.3 * r.get("citation_rate", 0)
            + 0.3 * r.get("true_refusal_rate", 0)
        )
    valid.sort(key=lambda x: x["_score"], reverse=True)

    print("\n=== PROMPT TEMPLATE LEADERBOARD ===")
    print(f"{'Rank':<5} {'Template':<30} {'Faithful':<10} {'Citation':<10} {'TrueRefusal':<13} {'FalseRefusal':<13} {'Score':<8}")
    print("-" * 95)
    for rank, r in enumerate(valid, 1):
        print(
            f"{rank:<5} {r['template_name']:<30} "
            f"{r.get('faithfulness_heuristic_mean', float('nan')):<10.3f} "
            f"{r.get('citation_rate', float('nan')):<10.2%} "
            f"{r.get('true_refusal_rate', float('nan')):<13.2%} "
            f"{r.get('false_refusal_rate', float('nan')):<13.2%} "
            f"{r['_score']:<8.3f}"
        )

    if valid:
        best = valid[0]
        print(f"\nRecommendation: Use template '{best['template_name']}'")
        print("Copy the template body from experiments/prompt_experiments/templates.py")
        print("into app/generation/prompt.py (SYSTEM_TEMPLATE variable).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt engineering ablation")
    parser.add_argument("--config", default="experiments/config/prompts.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    run(args.config, args.data_dir)
