"""Canonical manifest validation for the RSNA Bone Age dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api import types as pdt

REQUIRED_COLUMNS = {
    "image_id",
    "split",
    "sex",
    "bone_age_months",
    "image_path",
    "width",
    "height",
    "file_bytes",
    "sha256",
    "phash",
    "duplicate_group_id",
    "source_version",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(
    df: pd.DataFrame,
    *,
    expected_train_rows: int | None = None,
    expected_test_rows: int | None = None,
    check_paths: bool = False,
) -> None:
    """Raise ``ValueError`` when the canonical manifest contract is violated."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    _require(not missing, f"Manifest is missing required columns: {missing}")
    _require(len(df) > 0, "Manifest must not be empty")

    _require(pdt.is_integer_dtype(df["image_id"]), "image_id must be an integer dtype")
    _require(pdt.is_integer_dtype(df["sex"]), "sex must be an integer dtype")
    _require(pdt.is_numeric_dtype(df["bone_age_months"]), "bone_age_months must be numeric")
    for column in ("width", "height", "file_bytes"):
        _require(pdt.is_integer_dtype(df[column]), f"{column} must be an integer dtype")

    _require(not df["image_id"].isna().any(), "image_id contains nulls")
    _require(df["image_id"].is_unique, "image_id must be globally unique")
    _require((df["image_id"] > 0).all(), "image_id must be positive")
    _require(
        set(df["split"].unique()) == {"train", "test"}, "split must contain train and test only"
    )
    _require(set(df["sex"].unique()).issubset({0, 1}), "sex must contain only 0 or 1")
    _require(
        (df[["width", "height", "file_bytes"]] > 0).all().all(), "image metrics must be positive"
    )

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    _require(not train["bone_age_months"].isna().any(), "Train rows require bone_age_months")
    _require(test["bone_age_months"].isna().all(), "Test rows must not contain bone_age_months")
    train_ages = train["bone_age_months"].to_numpy(dtype=float)
    _require(np.isfinite(train_ages).all(), "Train ages must be finite")
    _require(
        ((train_ages >= 1) & (train_ages <= 228)).all(), "Train ages must be within 1..228 months"
    )

    _require(df["image_path"].notna().all(), "image_path contains nulls")
    _require(df["image_path"].is_unique, "image_path must be unique")
    _require(
        df["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}", case=False).all(), "Invalid SHA-256"
    )
    _require(
        df["phash"].astype(str).str.fullmatch(r"[0-9a-f]{16}", case=False).all(), "Invalid pHash"
    )
    _require(df["source_version"].notna().all(), "source_version contains nulls")

    if expected_train_rows is not None:
        _require(
            len(train) == expected_train_rows,
            f"Expected {expected_train_rows} train rows, got {len(train)}",
        )
    if expected_test_rows is not None:
        _require(
            len(test) == expected_test_rows,
            f"Expected {expected_test_rows} test rows, got {len(test)}",
        )
    if check_paths:
        absent = [path for path in df["image_path"] if not Path(path).is_file()]
        _require(not absent, f"Missing image files, first entries: {absent[:5]}")
