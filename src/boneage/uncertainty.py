"""Prediction-interval calibration for bone-age regression."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    """Return the split-conformal finite-sample quantile for absolute residuals."""
    residuals = np.asarray(values, dtype=np.float64).reshape(-1)
    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Calibration residuals must be non-empty and finite")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    if (residuals < 0).any():
        raise ValueError("Calibration residuals must be non-negative")
    rank = min(residuals.size, math.ceil((residuals.size + 1) * coverage))
    return float(np.partition(residuals, rank - 1)[rank - 1])


def calibrate_oof_interval(run_dir: Path, coverage: float = 0.90) -> dict[str, Any]:
    """Calibrate a symmetric interval from leakage-safe OOF predictions."""
    path = run_dir / "oof_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"OOF predictions are missing: {path}")
    frame = pd.read_csv(path)
    required = {"image_id", "target_months", "prediction_months"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {missing}")
    if frame.empty or frame["image_id"].duplicated().any():
        raise ValueError("OOF predictions must be non-empty with unique image IDs")
    target = frame["target_months"].to_numpy(dtype=np.float64)
    prediction = frame["prediction_months"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("OOF targets and predictions must be finite")
    radius = finite_sample_quantile(np.abs(target - prediction), coverage)
    return {
        "method": "symmetric_split_conformal_oof",
        "coverage": float(coverage),
        "radius_months": radius,
        "calibration_rows": int(len(frame)),
    }


def prediction_bounds(
    prediction: np.ndarray,
    radius_months: float,
    *,
    minimum_months: float = 1.0,
    maximum_months: float = 228.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a calibrated symmetric radius and clamp to the supported target range."""
    values = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(radius_months) or radius_months < 0:
        raise ValueError("radius_months must be finite and non-negative")
    if minimum_months >= maximum_months:
        raise ValueError("minimum_months must be below maximum_months")
    lower = np.where(
        np.isfinite(values), np.clip(values - radius_months, minimum_months, maximum_months), np.nan
    )
    upper = np.where(
        np.isfinite(values), np.clip(values + radius_months, minimum_months, maximum_months), np.nan
    )
    return lower, upper
