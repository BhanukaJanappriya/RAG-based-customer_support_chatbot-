"""MLflow experiment tracking helpers.

All experiments use a local file backend at ``experiments/mlruns/``.
No server process required — just a file path.

Usage pattern in experiment scripts::

    from experiments.utils.mlflow_utils import init_mlflow, log_run

    mlflow = init_mlflow("embedding_comparison", tracking_uri="file:./mlruns")
    with log_run(config, tags={"phase": "embedding_sweep"}):
        # ... run experiment ...
        mlflow.log_metrics(results)
        mlflow.log_artifact("results/embedding_comparison.json")
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# MLflow >=3 deprecates the local filesystem tracking backend by default;
# this project intentionally uses "file:./mlruns" with no server process.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow

logger = logging.getLogger(__name__)


def init_mlflow(
    experiment_name: str,
    tracking_uri: str = "file:./mlruns",
) -> mlflow.MlflowClient:
    """Initialise MLflow with a local file backend.

    Args:
        experiment_name: Human-readable experiment name (creates if absent).
        tracking_uri: Path to local mlruns directory.

    Returns:
        An MLflowClient for low-level access if needed.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking → {tracking_uri} / experiment={experiment_name!r}")
    return mlflow.MlflowClient(tracking_uri=tracking_uri)


@contextmanager
def log_run(
    params: dict[str, Any],
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
):
    """Context manager wrapping an mlflow.start_run().

    Logs all ``params`` as MLflow parameters on entry. Callers log metrics
    and artifacts inside the ``with`` block.

    Args:
        params: Flat dict of hyperparameters to log.
        run_name: Optional descriptive name shown in the MLflow UI.
        tags: Optional key-value tags attached to the run.

    Yields:
        The active MLflow run object.
    """
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        # Flatten nested dicts for MLflow's flat param store
        flat_params = _flatten_dict(params)
        # MLflow param values must be strings ≤ 500 chars
        mlflow.log_params(
            {k: str(v)[:500] for k, v in flat_params.items()}
        )
        yield run


def log_metrics_with_ci(metrics_with_ci: dict[str, dict[str, float]]) -> None:
    """Log point estimates and CI bounds as separate MLflow metrics.

    Naming convention: ``{metric}_mean``, ``{metric}_ci_lower``,
    ``{metric}_ci_upper``, ``{metric}_std_err``.

    Args:
        metrics_with_ci: Output of ``bootstrap_ci_all``.
    """
    flat: dict[str, float] = {}
    for metric, ci in metrics_with_ci.items():
        flat[f"{metric}_mean"] = ci["point"]
        flat[f"{metric}_ci_lower"] = ci["lower"]
        flat[f"{metric}_ci_upper"] = ci["upper"]
        flat[f"{metric}_std_err"] = ci["std_err"]
    mlflow.log_metrics(flat)


def save_and_log_json(data: Any, path: str | Path, artifact_path: str | None = None) -> None:
    """Write data to a JSON file and log it as an MLflow artifact.

    Args:
        data: JSON-serialisable object.
        path: Local file path to write.
        artifact_path: Subdirectory within the MLflow run's artifact store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
    logger.info(f"Saved + logged artifact: {path}")


def load_best_run(
    experiment_name: str,
    metric: str = "ndcg_at_5_mean",
    tracking_uri: str = "file:./mlruns",
) -> dict[str, Any] | None:
    """Return params + metrics of the best run for a given experiment.

    Args:
        experiment_name: Name of the MLflow experiment.
        metric: Metric to maximise.
        tracking_uri: Path to mlruns directory.

    Returns:
        Dict with ``run_id``, ``params``, ``metrics``, or None if no runs exist.
    """
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experiment {experiment_name!r} not found in {tracking_uri}")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        return None

    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "params": best.data.params,
        "metrics": best.data.metrics,
    }


def _flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Recursively flatten a nested dict with dot-separated keys."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, prefix=key, sep=sep))
        else:
            items[key] = v
    return items
