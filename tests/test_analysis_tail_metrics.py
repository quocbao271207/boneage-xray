import numpy as np
import pandas as pd
import pytest

from analysis.analyze_oof_tail import grouped_metrics, quality_gate_scenarios
from analysis.compare_paired_runs import (
    comparison_decision,
    subgroup_comparison,
    tail_transition_labels,
    transition_quality_summary,
    transition_summary,
)
from analysis.compare_paired_runs import metrics as paired_metrics


def test_paired_metrics_include_catastrophic_rate_upper_bound() -> None:
    frame = pd.DataFrame(
        {
            "target_months": [10.0, 20.0, 30.0],
            "prediction_months": [10.0, 20.0, 30.0],
        }
    )

    result = paired_metrics(frame, "prediction_months")

    assert result["rows"] == 3
    assert result["over_24_count"] == 0
    assert result["over_24_rate_upper_95"] == pytest.approx(0.4741955741575521)


def test_grouped_tail_metrics_include_group_specific_upper_bounds() -> None:
    frame = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "image_id": [1, 2, 3, 4],
            "absolute_error": [0.0, 25.0, 0.0, 0.0],
            "signed_error": [0.0, 25.0, 0.0, 0.0],
            "over_24_months": [False, True, False, False],
        }
    )

    result = grouped_metrics(frame, "group").set_index("group")

    assert result.loc[0, "over_24_rate_upper_95"] > result.loc[0, "over_24_rate"]
    assert result.loc[1, "over_24_rate"] == 0.0
    assert result.loc[1, "over_24_rate_upper_95"] > 0.0


def test_paired_subgroups_include_tail_rates_bounds_and_maximum() -> None:
    frame = pd.DataFrame(
        {
            "group": [0, 0, 1],
            "target_months": [10.0, 20.0, 30.0],
            "baseline_prediction": [40.0, 20.0, 30.0],
            "candidate_prediction": [10.0, 20.0, 60.0],
        }
    )

    result = subgroup_comparison(frame, "group").set_index("group")

    assert result.loc[0, "baseline_over_24_rate"] == 0.5
    assert result.loc[0, "candidate_over_24_rate"] == 0.0
    assert result.loc[1, "candidate_over_24_rate_upper_95"] > 0.0
    assert result.loc[1, "candidate_max_ae"] == 30.0


def test_quality_gate_scenarios_measure_tail_capture_and_coverage() -> None:
    frame = pd.DataFrame(
        {
            "absolute_error": [30.0, 26.0, 2.0, 1.0],
            "over_24_months": [True, True, False, False],
            "flag_incomplete_or_nonstandard_pose": [True, False, False, False],
            "flag_crop_risk": [False, False, False, False],
            "flag_off_center": [False, False, False, False],
            "flag_small_hand": [False, False, False, False],
            "flag_low_contrast": [False, False, False, False],
            "flag_rotated": [False, False, False, False],
            "flag_blur_review": [False, False, False, False],
        }
    )

    result = quality_gate_scenarios(frame).set_index("scenario")

    assert result.loc["accept_all", "coverage"] == 1.0
    assert result.loc["nonstandard_pose", "coverage"] == 0.75
    assert result.loc["nonstandard_pose", "tail_capture_rate"] == 0.5
    assert result.loc["nonstandard_pose", "accepted_over_24_count"] == 1


def test_comparison_decision_requires_control_for_ambiguous_result() -> None:
    decision = comparison_decision(
        {"mae": -0.1},
        {
            "mae_delta_ci_95": [-0.3, 0.1],
            "p95_delta_ci_95": [-0.2, 0.2],
            "over_24_rate_delta_ci_95": [-0.01, 0.01],
        },
    )

    assert decision["mae_ci_includes_zero"] is True
    assert decision["requires_fresh_control"] is True


def test_comparison_decision_accepts_clear_improvement_without_control() -> None:
    decision = comparison_decision(
        {"mae": -0.4},
        {
            "mae_delta_ci_95": [-0.6, -0.2],
            "p95_delta_ci_95": [-0.5, 0.1],
            "over_24_rate_delta_ci_95": [-0.02, 0.0],
        },
    )

    assert decision["requires_fresh_control"] is False
    assert decision["promote_screen"] is True


def test_tail_transition_labels_use_strict_24_month_threshold() -> None:
    labels = tail_transition_labels(
        np.array([24.0, 25.0, 25.0, 0.0]),
        np.array([25.0, 24.0, 26.0, 0.0]),
    )

    assert labels.tolist() == [
        "new_over_24",
        "fixed_below_24",
        "both_over_24",
        "both_within_24",
    ]


def test_transition_summary_groups_threshold_churn() -> None:
    frame = pd.DataFrame(
        {
            "sex": [0, 0, 1],
            "tail_transition": ["new_over_24", "new_over_24", "fixed_below_24"],
            "target_months": [100.0, 120.0, 140.0],
            "baseline_absolute_error": [20.0, 22.0, 30.0],
            "candidate_absolute_error": [25.0, 28.0, 20.0],
            "absolute_error_delta": [5.0, 6.0, -10.0],
        }
    )

    result = transition_summary(frame, "sex").set_index(["sex", "tail_transition"])

    assert result.loc[(0, "new_over_24"), "rows"] == 2
    assert result.loc[(0, "new_over_24"), "mean_target_months"] == 110.0
    assert result.loc[(0, "new_over_24"), "mean_ae_delta"] == 5.5
    assert result.loc[(1, "fixed_below_24"), "mean_ae_delta"] == -10.0


def test_transition_quality_summary_reports_flag_counts_and_rates() -> None:
    frame = pd.DataFrame(
        {
            "tail_transition": ["new_over_24", "new_over_24", "fixed_below_24"],
            "flag_low_contrast": [True, False, False],
            "flag_small_hand": [True, True, False],
        }
    )

    result = transition_quality_summary(frame).set_index("tail_transition")

    assert result.loc["new_over_24", "rows"] == 2
    assert result.loc["new_over_24", "flag_low_contrast_count"] == 1
    assert result.loc["new_over_24", "flag_low_contrast_rate"] == 0.5
    assert result.loc["fixed_below_24", "flag_small_hand_rate"] == 0.0
