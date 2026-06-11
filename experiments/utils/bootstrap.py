"""Bootstrap confidence intervals and statistical significance tests.

All experiments report 95% CI via percentile bootstrap (n=1000).
Pairwise comparisons use the Wilcoxon signed-rank test (non-parametric,
appropriate for small eval sets where normality cannot be assumed).
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

RNG = np.random.default_rng(42)


def bootstrap_ci(
    data: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute a bootstrap confidence interval for a scalar statistic.

    Uses the percentile method (not BCa) — simpler, defensible for n>=30,
    and avoids the acceleration constant estimation that can fail on small
    samples.

    Args:
        data: Observed per-query metric values (one float per eval example).
        stat_fn: Aggregation function; defaults to np.mean.
        n_bootstrap: Number of resamples. 1000 is standard.
        alpha: Two-sided significance level (0.05 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: ``point``, ``lower``, ``upper``, ``std_err``.
    """
    arr = np.array(data, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)

    point = float(stat_fn(arr))

    bootstrap_stats = np.array(
        [stat_fn(rng.choice(arr, size=n, replace=True)) for _ in range(n_bootstrap)]
    )

    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    std_err = float(np.std(bootstrap_stats, ddof=1))

    return {"point": point, "lower": lower, "upper": upper, "std_err": std_err}


def bootstrap_ci_all(
    per_query_metrics: dict[str, list[float]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Apply ``bootstrap_ci`` to every metric in a per-query metrics dict.

    Args:
        per_query_metrics: Mapping of metric_name → list of per-query values.
        n_bootstrap: Resamples per metric.
        alpha: CI width.
        seed: Base seed; each metric gets seed+i to keep results independent.

    Returns:
        Mapping of metric_name → CI dict (point, lower, upper, std_err).
    """
    return {
        name: bootstrap_ci(values, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed + i)
        for i, (name, values) in enumerate(per_query_metrics.items())
    }


def wilcoxon_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Wilcoxon signed-rank test for paired per-query metric values.

    Preferred over paired t-test because metric distributions (P@k, nDCG)
    are bounded and often non-normal on small eval sets.

    Args:
        scores_a: Per-query metric values for configuration A.
        scores_b: Per-query metric values for configuration B.
        alternative: ``"two-sided"``, ``"greater"`` (A > B), or ``"less"``.

    Returns:
        Dict with ``statistic``, ``p_value``, and ``significant`` (bool at α=0.05).

    Raises:
        ValueError: If sequences have different lengths.
    """
    a, b = np.array(scores_a, dtype=float), np.array(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Paired test requires equal lengths: {len(a)} != {len(b)}")

    diff = a - b
    if np.all(diff == 0):
        logger.warning("All differences are zero — test is undefined; returning p=1.0")
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    result = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
    }


def compare_configs(
    baseline_scores: dict[str, list[float]],
    challenger_scores: dict[str, list[float]],
    primary_metric: str = "ndcg_at_5",
) -> dict:
    """One-stop comparison between two configurations.

    Args:
        baseline_scores: Per-query metrics for the baseline config.
        challenger_scores: Per-query metrics for the challenger config.
        primary_metric: Metric used for the significance test.

    Returns:
        Summary dict with CIs for both configs and significance test result.
    """
    baseline_ci = bootstrap_ci_all(baseline_scores)
    challenger_ci = bootstrap_ci_all(challenger_scores)

    sig = {}
    if primary_metric in baseline_scores and primary_metric in challenger_scores:
        sig = wilcoxon_test(
            baseline_scores[primary_metric],
            challenger_scores[primary_metric],
            alternative="two-sided",
        )

    delta = (
        challenger_ci[primary_metric]["point"] - baseline_ci[primary_metric]["point"]
        if primary_metric in baseline_ci and primary_metric in challenger_ci
        else None
    )

    return {
        "baseline": baseline_ci,
        "challenger": challenger_ci,
        "primary_metric": primary_metric,
        "delta": delta,
        "significance": sig,
        "beats_baseline": (
            delta is not None and delta > 0 and sig.get("significant", False)
        ),
    }
