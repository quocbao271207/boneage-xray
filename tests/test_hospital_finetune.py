from pathlib import Path

import pandas as pd

from analysis.hospital_finetune import _paired_comparison, build_patient_split


def test_patient_split_has_no_cross_partition_leakage(tmp_path: Path) -> None:
    rows = []
    for patient in range(50):
        for image in range(2):
            path = tmp_path / f"{patient}_{image}.png"
            path.touch()
            rows.append(
                {
                    "case_key": f"case_{patient}_{image}",
                    "patient_key": f"patient_{patient}",
                    "study_key": f"study_{patient}",
                    "sex": patient % 2,
                    "sex_valid": True,
                    "report_bone_age_months": 12.0 + (patient % 18) * 12,
                    "standardized_image_path": str(path),
                    "standardization_status": "passed",
                }
            )
    split = build_patient_split(pd.DataFrame(rows), seed=7)
    assert set(split["partition"]) == {"train", "validation", "test"}
    assert split.groupby("patient_key")["partition"].nunique().max() == 1
    counts = split.groupby("partition")["patient_key"].nunique()
    assert counts["train"] == 30
    assert counts["validation"] == 10
    assert counts["test"] == 10


def test_raw_split_reuses_reference_patients_and_includes_failed_masks(
    tmp_path: Path,
) -> None:
    rows = []
    for patient in range(50):
        raw_path = tmp_path / f"raw_{patient}.png"
        raw_path.touch()
        rows.append(
            {
                "case_key": f"case_{patient}",
                "patient_key": f"patient_{patient}",
                "study_key": f"study_{patient}",
                "sex": patient % 2,
                "sex_valid": True,
                "report_bone_age_months": 12.0 + (patient % 18) * 12,
                "raw_image_path": str(raw_path),
                "standardization_status": (
                    "passed" if patient < 30 else "needs_mask_review"
                ),
            }
        )
    reference = pd.DataFrame(
        {
            "patient_key": [f"patient_{patient}" for patient in range(10)],
            "partition": ["test"] * 5 + ["validation"] * 5,
        }
    )
    split = build_patient_split(
        pd.DataFrame(rows),
        seed=7,
        image_path_column="raw_image_path",
        require_standardization_passed=False,
        reference_split=reference,
    )
    assert len(split) == 50
    assert set(split["input_variant"]) == {"raw"}
    assert split.loc[split["patient_key"].eq("patient_0"), "partition"].item() == "test"
    assert (
        split.loc[split["patient_key"].eq("patient_7"), "partition"].item()
        == "validation"
    )
    assert split.groupby("patient_key")["partition"].nunique().max() == 1


def test_paired_comparison_uses_only_common_studies(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    pd.DataFrame(
        {
            "study_key": ["a", "b", "baseline_only"],
            "target_months": [10.0, 20.0, 30.0],
            "prediction_months": [15.0, 25.0, 30.0],
        }
    ).to_csv(baseline_path, index=False)
    candidate = pd.DataFrame(
        {
            "study_key": ["a", "b", "candidate_only"],
            "target_months": [10.0, 20.0, 40.0],
            "prediction_months": [11.0, 21.0, 40.0],
        }
    )
    comparison = _paired_comparison(
        baseline_path,
        candidate,
        baseline_name="standardized_gradual",
        seed=11,
    )
    assert comparison["study_count"] == 2
    assert comparison["mae_delta_candidate_minus_baseline"] == -4.0
    assert comparison["candidate_better"] is True
