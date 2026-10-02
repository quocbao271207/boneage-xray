from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boneage.uncertainty import (
    calibrate_oof_interval,
    finite_sample_quantile,
    prediction_bounds,
)


def test_finite_sample_quantile_uses_conformal_rank() -> None:
    values = np.arange(1, 11, dtype=float)
    assert finite_sample_quantile(values, 0.80) == 9.0


def test_prediction_bounds_preserve_missing_and_clamp() -> None:
    lower, upper = prediction_bounds(np.array([3.0, 227.0, np.nan]), 5.0)
    np.testing.assert_allclose(lower[:2], [1.0, 222.0])
    np.testing.assert_allclose(upper[:2], [8.0, 228.0])
    assert np.isnan(lower[2]) and np.isnan(upper[2])


def test_calibrate_oof_interval_rejects_duplicate_ids(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "image_id": [1, 1],
            "target_months": [10.0, 12.0],
            "prediction_months": [9.0, 11.0],
        }
    ).to_csv(tmp_path / "oof_predictions.csv", index=False)
    with pytest.raises(ValueError, match="unique image IDs"):
        calibrate_oof_interval(tmp_path)
