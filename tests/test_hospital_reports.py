import pytest

from analysis.hospital_reports import (
    numeric_identifier_tokens,
    parse_acquisition_date,
    parse_bone_age_label,
    parse_patient_age_months,
)


@pytest.mark.parametrize(
    ("phrase", "midpoint", "lower", "upper"),
    [
        ("Tuổi xương tương ứng với # 6 tuổi", 72, 72, 72),
        ("Tuổi xương tương ứng với # 5-6 tuổi", 66, 60, 72),
        ("Tuổi xương tương ứng với # 8 tháng tuổi", 8, 8, 8),
        ("Tuổi xương tương ứng với # 10-12 tháng tuổi", 11, 10, 12),
        ("Tuổi xương tương ứng với # 7 tuổi 6 tháng", 90, 90, 90),
    ],
)
def test_parse_bone_age_label(
    phrase: str, midpoint: float, lower: float, upper: float
) -> None:
    label = parse_bone_age_label(phrase)
    assert label is not None
    assert label.midpoint_months == midpoint
    assert label.lower_months == lower
    assert label.upper_months == upper


def test_conflicting_report_labels_are_rejected() -> None:
    assert (
        parse_bone_age_label(
            "Tuổi xương tương ứng với # 5 tuổi\nTuổi xương tương ứng với # 6 tuổi"
        )
        is None
    )


def test_parse_report_date_and_identifier_tokens() -> None:
    assert parse_acquisition_date("Thời gian chụp: 21/07/2025 10:41") == "20250721"
    assert numeric_identifier_tokens("Mã 25134310, điện thoại 0902565114") == {
        "25134310",
        "0902565114",
    }


@pytest.mark.parametrize(("source", "expected"), [("006Y", 72), ("018M", 18)])
def test_parse_patient_age_months(source: str, expected: float) -> None:
    assert parse_patient_age_months(source) == expected
