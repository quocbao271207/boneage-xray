import pandas as pd
import pytest

from analysis.render_paired_delta_review import (
    select_delta_cases,
    select_largest_candidate_errors,
    select_transition_cases,
)


def test_select_delta_cases_orders_both_directions() -> None:
    frame = pd.DataFrame(
        {
            "image_id": [1, 2, 3, 4],
            "absolute_error_delta": [-5.0, -2.0, 1.0, 3.0],
        }
    )

    regressions, improvements = select_delta_cases(frame, 2)

    assert regressions["image_id"].tolist() == [4, 3]
    assert improvements["image_id"].tolist() == [1, 2]


@pytest.mark.parametrize("count", [0, -1])
def test_select_delta_cases_requires_positive_count(count: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        select_delta_cases(
            pd.DataFrame({"image_id": [1], "absolute_error_delta": [0.0]}), count
        )


def test_select_transition_cases_orders_new_and_fixed_crossings() -> None:
    frame = pd.DataFrame(
        {
            "image_id": [1, 2, 3, 4],
            "absolute_error_delta": [5.0, 10.0, -4.0, -12.0],
            "tail_transition": [
                "new_over_24",
                "new_over_24",
                "fixed_below_24",
                "fixed_below_24",
            ],
        }
    )

    newly_over = select_transition_cases(frame, "new_over_24", 2)
    fixed = select_transition_cases(frame, "fixed_below_24", 2)

    assert newly_over["image_id"].tolist() == [2, 1]
    assert fixed["image_id"].tolist() == [4, 3]


def test_select_largest_candidate_errors_orders_by_candidate_error() -> None:
    frame = pd.DataFrame(
        {
            "image_id": [1, 2, 3],
            "candidate_absolute_error": [12.0, 40.0, 25.0],
        }
    )

    selected = select_largest_candidate_errors(frame, 2)

    assert selected["image_id"].tolist() == [2, 3]
