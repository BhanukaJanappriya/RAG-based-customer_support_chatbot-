# Experiments

Research codebase for systematic RAG improvement. Sits alongside the production app and feeds validated improvements back via `INTEGRATION.md`.

## Setup

```bash
# From project root, with venv active:
pip install -r experiments/requirements.txt
```

Requires Ollama running with `llama3.2:latest` pulled, and documents ingested into ChromaDB (`python scripts/ingest.py`).

## Running Experiments (in order)

### 1. Embedding model comparison
```bash
python experiments/embedding_experiments/run_embedding_comparison.py \
    --config experiments/config/embeddings.yaml
```
Outputs: `experiments/results/emb_*.json`, `experiments/plots/embedding_pareto.png`

Then update `embedding_model` in `experiments/config/retrieval.yaml` with the winner.

### 2. Retrieval strategy ablation
```bash
# Both phases (chunking + retrieval methods):
python experiments/retrieval_experiments/run_retrieval_ablation.py \
    --config experiments/config/retrieval.yaml

# Chunking phase only:
python experiments/retrieval_experiments/run_retrieval_ablation.py \
    --config experiments/config/retrieval.yaml --phases chunking

# Retrieval methods only (faster, uses best chunking from first run):
python experiments/retrieval_experiments/run_retrieval_ablation.py \
    --config experiments/config/retrieval.yaml --phases retrieval
```
Outputs: `experiments/results/retrieval/`, `experiments/plots/retrieval_comparison.png`

Then update `retrieval_method` in `experiments/config/prompts.yaml` with the winner.

### 3. Prompt engineering ablation
```bash
python experiments/prompt_experiments/run_prompt_ablation.py \
    --config experiments/config/prompts.yaml
```
Outputs: `experiments/results/prompts/`, `experiments/plots/prompt_comparison.png`

### 4. View results in MLflow UI
```bash
mlflow ui --backend-store-uri experiments/mlruns --port 5000
# Open http://localhost:5000
```

### 5. Generate synthetic QA (optional, for scale)
```python
from experiments.eval.dataset import generate_synthetic_qa, save_dataset
# After ingesting documents, call with your chunk list
```

## File Structure

```
experiments/
├── config/          # YAML configs for each workstream
├── eval/            # Evaluation framework (metrics, dataset, RAGAS, harness)
├── utils/           # Bootstrap CI, MLflow helpers, Ollama client
├── embedding_experiments/  # Embedding model sweep
├── retrieval_experiments/  # Chunking + retrieval method ablation
├── prompt_experiments/     # Prompt template comparison
├── data/            # gold_qa.json + synthetic QA output
├── results/         # JSON results (git-ignored by default)
├── plots/           # PNG plots (git-ignored by default)
├── mlruns/          # MLflow tracking store (git-ignored)
├── findings.md      # Template — fill in after experiments
└── INTEGRATION.md   # How to plug winners into production
```

## Key Design Decisions

- **Primary metric**: nDCG@5 (balances precision and rank; standard in IR).
- **Statistics**: Bootstrap percentile CI (n=1000); Wilcoxon signed-rank for pairwise tests.
- **LLM-as-judge**: RAGAS with Llama 3.2 1B. Only on finalist configs. Judge==generator bias documented.
- **RRF over weighted sum**: Hyperparameter-free (Cormack et al., 2009).
- **Evaluation dataset**: 50 hand-curated gold pairs. Standard questions only for retrieval metrics; full set for refusal/generation metrics.
