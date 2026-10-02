import pandas as pd
import pytest

from analysis.audit_hand_orientation import orientation_gate_errors, rotation_reverses_axis


@pytest.mark.parametrize(
    ("axis_angle", "rotation_override", "expected"),
    [
        (80.0, 90.0, True),
        (80.0, -80.0, False),
        (-20.0, 200.0, True),
        (-20.0, 20.0, False),
    ],
)
def test_rotation_reverses_axis_from_exact_override(
    axis_angle: float, rotation_override: float, expected: bool
) -> None:
    assert rotation_reverses_axis(axis_angle, rotation_override) is expected


def test_orientation_gate_blocks_reversal_and_mask_holes_only() -> None:
    audit = pd.DataFrame(
        {
            "image_id": [1, 2, 3],
            "flag_reverse_after_config": [True, False, False],
            "flag_mask_holes": [False, True, False],
            "flag_incomplete_or_nonstandard_pose": [False, False, True],
        }
    )

    errors = orientation_gate_errors(audit)

    assert errors == [
        "reversed hands remain after configuration: [1]",
        "retained masks contain enclosed holes: [2]",
    ]


def test_orientation_gate_allows_review_only_nonstandard_pose() -> None:
    audit = pd.DataFrame(
        {
            "image_id": [3],
            "flag_reverse_after_config": [False],
            "flag_mask_holes": [False],
            "flag_incomplete_or_nonstandard_pose": [True],
        }
    )

    assert orientation_gate_errors(audit) == []
