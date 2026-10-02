import pandas as pd

from analysis.audit_standardized_outputs import (
    unexpected_border_contact_ids,
    validate_metadata_overrides,
)


def test_validate_metadata_overrides_accepts_exact_recorded_geometry() -> None:
    metadata = pd.DataFrame(
        {
            "image_id": [7049],
            "rotation_degrees": [90.0],
            "rotation_override_degrees": [90.0],
            "skip_forearm_limit": [True],
        }
    )
    config = {
        "rotation_overrides": {7049: 90.0},
        "skip_forearm_limit_ids": [7049],
    }

    assert validate_metadata_overrides(metadata, config) == []


def test_validate_metadata_overrides_rejects_drift_and_missing_bypass() -> None:
    metadata = pd.DataFrame(
        {
            "image_id": [7049],
            "rotation_degrees": [100.0],
            "rotation_override_degrees": [90.0],
            "skip_forearm_limit": [False],
        }
    )
    config = {
        "rotation_overrides": {7049: 90.0},
        "skip_forearm_limit_ids": [7049],
    }

    errors = validate_metadata_overrides(metadata, config)

    assert any("applied rotation" in error for error in errors)
    assert any("skip_forearm_limit=True" in error for error in errors)


def test_border_contact_gate_allows_only_explicit_ids() -> None:
    quality = pd.DataFrame(
        {
            "image_id": [1, 7049, 9],
            "border_nonuniform_fraction": [0.0, 0.03, 0.01],
        }
    )

    assert unexpected_border_contact_ids(quality, {7049}) == [9]
