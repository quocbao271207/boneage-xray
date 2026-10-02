import numpy as np
import pytest

from boneage.metrics import binomial_rate_upper_bound, bootstrap_mae_ci, compute_metrics


def test_compute_metrics() -> None:
    metrics = compute_metrics(np.array([120, 150, 100]), np.array([120, 140, 110]))
    assert metrics["mae"] == pytest.approx(20 / 3)
    assert metrics["bias_months"] == 0.0
    assert metrics["median_ae"] == 10.0
    assert metrics["p95_ae"] == 10.0
    assert metrics["p97_5_ae"] == 10.0
    assert metrics["max_ae"] == 10.0
    assert metrics["over_24_count"] == 0
    assert metrics["over_24_rate"] == 0.0
    assert metrics["over_24_rate_upper_95"] == pytest.approx(0.4741955741575521)
    assert metrics["within_6_months"] == pytest.approx(1 / 3)
    assert metrics["within_12_months"] == 1.0


def test_compute_metrics_uses_strict_catastrophic_thresholds() -> None:
    metrics = compute_metrics(np.zeros(5), np.array([0, 24, 25, 48, 61]))
    assert metrics["over_24_count"] == 3
    assert metrics["over_24_rate"] == pytest.approx(3 / 5)
    assert metrics["over_36_count"] == 2
    assert metrics["over_48_count"] == 1
    assert metrics["over_60_count"] == 1


def test_bootstrap_ci_is_deterministic() -> None:
    truth = np.arange(20, dtype=float)
    prediction = truth + np.linspace(-3, 3, 20)
    assert bootstrap_mae_ci(truth, prediction, samples=100, seed=7) == bootstrap_mae_ci(
        truth, prediction, samples=100, seed=7
    )


def test_binomial_upper_bound_quantifies_zero_observed_events() -> None:
    bound = binomial_rate_upper_bound(0, 100, confidence=0.95)
    assert bound == pytest.approx(0.026342720384717877)
    assert binomial_rate_upper_bound(100, 100) == 1.0


@pytest.mark.parametrize(
    ("events", "total", "confidence", "error"),
    [
        (-1, 10, 0.95, ValueError),
        (11, 10, 0.95, ValueError),
        (0, 0, 0.95, ValueError),
        (0, 10, 0.0, ValueError),
        (0, 10, 1.0, ValueError),
        (0, 10, np.nan, ValueError),
        (1.5, 10, 0.95, TypeError),
        (1, True, 0.95, TypeError),
    ],
)
def test_binomial_upper_bound_rejects_invalid_inputs(
    events: object,
    total: object,
    confidence: float,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        binomial_rate_upper_bound(events, total, confidence=confidence)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [([], []), ([1, 2], [1]), ([1, np.nan], [1, 2]), ([1, 2], [1, np.inf])],
)
def test_metrics_reject_invalid_arrays(truth, prediction) -> None:
    with pytest.raises(ValueError):
        compute_metrics(np.asarray(truth), np.asarray(prediction))
