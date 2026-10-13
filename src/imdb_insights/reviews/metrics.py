"""Metrics for 1-10 rating prediction, with bootstrap confidence intervals.

Pure NumPy/pandas so they run (and are tested) without the ML dependency group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POSITIVE_FROM, NEGATIVE_TO = 7, 4  # Stanford's polarity definition; 5-6 are neutral
DECISION = 5.5  # a predicted rating above this counts as a positive review


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:  # (np.std can be 1e-16 for a constant array)
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1, 10)
    error = y_pred - y_true
    polar = (y_true >= POSITIVE_FROM) | (y_true <= NEGATIVE_TO)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": _spearman(y_true, y_pred),
        "within_1": float(np.mean(np.abs(np.rint(y_pred) - y_true) <= 1)),
        "polarity_accuracy": (
            float(np.mean((y_pred[polar] > DECISION) == (y_true[polar] >= POSITIVE_FROM)))
            if polar.any()
            else float("nan")
        ),
        "bias": float(np.mean(error)),  # > 0: the model rates too high on average
    }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci_metrics: tuple[str, ...] = ("mae", "polarity_accuracy"),
) -> dict[str, float]:
    """Point metrics plus 95% bootstrap intervals (``<metric>_lo`` / ``<metric>_hi``)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    result = point_metrics(y_true, y_pred) | {"n": int(len(y_true))}
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in ci_metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        boot = point_metrics(y_true[idx], y_pred[idx])
        for m in ci_metrics:
            samples[m].append(boot[m])
    for m, values in samples.items():
        result[f"{m}_lo"], result[f"{m}_hi"] = (
            float(v) for v in np.nanpercentile(values, [2.5, 97.5])
        )
    return result


def per_rating_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    """MAE for each true rating: shows whether a model squashes the extremes."""
    df = pd.DataFrame({"true": y_true, "error": np.abs(np.clip(y_pred, 1, 10) - y_true)})
    return {int(k): round(float(v), 3) for k, v in df.groupby("true")["error"].mean().items()}
