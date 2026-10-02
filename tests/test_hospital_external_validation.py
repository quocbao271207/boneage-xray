import cv2
import numpy as np
import pandas as pd
import torch

from analysis.hospital_external_validation import (
    SAFE_MANIFEST_COLUMNS,
    _predict_variant,
    assess_mask_quality,
    evaluate_external_gate,
)
from boneage.xai import EnsembleContract


def test_safe_manifest_contract_excludes_direct_identifiers() -> None:
    forbidden = {"patient_id", "patient_name", "birth_date", "study_date", "sop_instance_uid"}
    assert SAFE_MANIFEST_COLUMNS.isdisjoint(forbidden)


def test_mask_quality_accepts_centered_hand_shape() -> None:
    mask = np.zeros((240, 180), dtype=np.uint8)
    cv2.ellipse(mask, (90, 120), (45, 75), 0, 0, 360, 1, -1)
    cv2.rectangle(mask, (75, 175), (105, 239), 1, -1)
    quality = assess_mask_quality(mask)
    assert quality["flags"] == []


def test_mask_quality_rejects_marker_sized_and_border_masks() -> None:
    marker = np.zeros((240, 180), dtype=np.uint8)
    cv2.rectangle(marker, (145, 190), (170, 225), 1, -1)
    assert "implausible_area" in assess_mask_quality(marker)["flags"]

    border = np.zeros((240, 180), dtype=np.uint8)
    cv2.rectangle(border, (0, 30), (100, 230), 1, -1)
    assert "unsafe_edge_contact" in assess_mask_quality(border)["flags"]


def test_predict_variant_keeps_invalid_rows_nan_without_runtime_warning() -> None:
    frame = pd.DataFrame({"sex_valid": [False], "sex": [np.nan], "image_path": [""]})
    contract = EnsembleContract(
        backbone="unused",
        preprocess_mode="P0",
        resolution=64,
        dataset_version="unused",
    )

    mean, standard_deviation = _predict_variant(
        frame,
        "image_path",
        [object()],  # No model method is reached because the only row is invalid.
        contract,
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert np.isnan(mean[0])
    assert np.isnan(standard_deviation[0])


def test_external_gate_uses_raw_metrics_without_standardization_gate(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "oof_metrics.json").write_text('{"mae": 7.0}', encoding="utf-8")
    metrics = {
        "raw": {
            "mae": 8.0,
            "p95_ae": 20.0,
            "over_24_rate_upper_95": 0.04,
            "by_sex": {"F": {"mae": 7.5}, "M": {"mae": 8.5}},
        },
        "standardized": {
            "mae": 99.0,
            "p95_ae": 99.0,
            "over_24_rate_upper_95": 1.0,
            "by_sex": {},
        },
    }
    config = {
        "hospital_input_variant": "raw",
        "run_dir": str(run_dir),
        "external_gates": {
            "max_internal_external_mae_gap_months": 2.0,
            "max_p95_months": 24.0,
            "max_over_24_rate_upper_95": 0.05,
            "max_subgroup_mae_gap_months": 3.0,
            "min_standardization_pass_rate": 0.99,
        },
    }

    decision = evaluate_external_gate(
        {"dicom_total": 100, "standardization_status": {"passed": 0}},
        metrics,
        config,
    )

    assert decision["passed"] is True
    assert decision["observed"]["input_variant"] == "raw"
    assert "standardization_pass_rate" not in decision["checks"]
