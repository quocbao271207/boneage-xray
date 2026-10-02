import pandas as pd
import pytest

from boneage.data_contracts import validate_manifest


def valid_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": pd.Series([1, 2], dtype="int64"),
            "split": ["train", "test"],
            "sex": pd.Series([1, 0], dtype="int8"),
            "bone_age_months": pd.Series([120.0, None], dtype="float32"),
            "image_path": ["a.png", "b.png"],
            "width": pd.Series([500, 500], dtype="int32"),
            "height": pd.Series([500, 500], dtype="int32"),
            "file_bytes": pd.Series([1_000, 1_100], dtype="int64"),
            "sha256": ["a" * 64, "b" * 64],
            "phash": ["0" * 16, "f" * 16],
            "duplicate_group_id": [None, None],
            "source_version": ["v2", "v2"],
        }
    )


def test_validate_manifest_accepts_canonical_schema() -> None:
    assert validate_manifest(valid_manifest()) is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.assign(split=["train", "validation"]), "split"),
        (lambda frame: frame.assign(sex=[1, 2]), "sex"),
        (lambda frame: frame.assign(bone_age_months=[0.0, None]), "1..228"),
        (lambda frame: frame.assign(image_id=[1, 1]), "unique"),
        (lambda frame: frame.assign(sha256=["bad", "b" * 64]), "SHA-256"),
    ],
)
def test_validate_manifest_rejects_contract_violations(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_manifest(mutation(valid_manifest()))


def test_validate_manifest_rejects_wrong_dtype() -> None:
    frame = valid_manifest()
    frame["image_id"] = frame["image_id"].astype(str)
    with pytest.raises(ValueError, match="image_id must be an integer"):
        validate_manifest(frame)


def test_validate_manifest_checks_expected_counts() -> None:
    with pytest.raises(ValueError, match="Expected 2 train rows"):
        validate_manifest(valid_manifest(), expected_train_rows=2, expected_test_rows=1)
