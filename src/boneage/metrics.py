"""Regression metrics and deterministic bootstrap confidence intervals."""

from __future__ import annotations

from math import isfinite
from statistics import NormalDist

import numpy as np


def _validated_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.size == 0 or truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must be non-empty arrays with equal shape")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite values")
    return truth, prediction


def binomial_rate_upper_bound(
    events: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Return a one-sided Wilson upper confidence bound for an event rate."""
    if isinstance(events, bool) or not isinstance(events, (int, np.integer)):
        raise TypeError("events must be an integer")
    if isinstance(total, bool) or not isinstance(total, (int, np.integer)):
        raise TypeError("total must be an integer")
    if total <= 0 or events < 0 or events > total:
        raise ValueError("Require 0 <= events <= total and total > 0")
    if not isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and strictly between 0 and 1")

    rate = events / total
    z_score = NormalDist().inv_cdf(confidence)
    z_squared = z_score**2
    denominator = 1.0 + z_squared / total
    center = rate + z_squared / (2.0 * total)
    radius = z_score * np.sqrt(
        rate * (1.0 - rate) / total + z_squared / (4.0 * total**2)
    )
    return float(min(1.0, (center + radius) / denominator))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth, prediction = _validated_arrays(y_true, y_pred)
    absolute_error = np.abs(truth - prediction)
    over_24_count = int((absolute_error > 24).sum())
    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "bias_months": float((prediction - truth).mean()),
        "median_ae": float(np.median(absolute_error)),
        "p90_ae": float(np.percentile(absolute_error, 90)),
        "p95_ae": float(np.percentile(absolute_error, 95)),
        "p97_5_ae": float(np.percentile(absolute_error, 97.5)),
        "max_ae": float(absolute_error.max()),
        "over_24_count": over_24_count,
        "over_24_rate": float(over_24_count / absolute_error.size),
        "over_24_rate_upper_95": binomial_rate_upper_bound(
            over_24_count, absolute_error.size
        ),
        "over_36_count": int((absolute_error > 36).sum()),
        "over_48_count": int((absolute_error > 48).sum()),
        "over_60_count": int((absolute_error > 60).sum()),
        "within_6_months": float((absolute_error <= 6).mean()),
        "within_12_months": float((absolute_error <= 12).mean()),
        "within_18_months": float((absolute_error <= 18).mean()),
    }


def bootstrap_mae_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    samples: int = 2_000,
    seed: int = 20260924,
) -> tuple[float, float]:
    truth, prediction = _validated_arrays(y_true, y_pred)
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(truth), size=len(truth))
        values[index] = np.abs(truth[selected] - prediction[selected]).mean()
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
