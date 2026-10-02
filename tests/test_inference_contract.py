import numpy as np
import pytest

from boneage.infer import validate_predictions


def test_inference_contract_accepts_valid_predictions() -> None:
    output = validate_predictions(
        np.array([101, 102]), np.array([12.5, 144.0]), expected_rows=2, lower=1, upper=228
    )
    assert list(output.columns) == ["Case ID", "prediction_months"]
    assert output["Case ID"].tolist() == [101, 102]


@pytest.mark.parametrize(
    ("ids", "predictions", "message"),
    [
        ([1], [10, 20], "row count"),
        ([1, 2], [10, np.nan], "non-finite"),
        ([1, 2], [0, 20], "outside"),
        ([1, 1], [10, 20], "duplicate"),
    ],
)
def test_inference_contract_rejects_invalid_output(ids, predictions, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_predictions(
            np.asarray(ids), np.asarray(predictions), expected_rows=2, lower=1, upper=228
        )
