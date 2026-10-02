"""Parse de-identified bone-age labels from hospital radiology report text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class BoneAgeLabel:
    midpoint_months: float
    lower_months: float
    upper_months: float
    precision: str


def fold_vietnamese(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "\n".join(" ".join(line.lower().split()) for line in ascii_text.splitlines())


def _valid_label(lower: float, upper: float, precision: str) -> BoneAgeLabel | None:
    if not (0 <= lower <= upper <= 240):
        return None
    return BoneAgeLabel(
        midpoint_months=(lower + upper) / 2.0,
        lower_months=lower,
        upper_months=upper,
        precision=precision,
    )


def _parse_age_fragment(fragment: str) -> BoneAgeLabel | None:
    fragment = fragment.replace("–", "-").replace("—", "-")
    combined = re.search(r"(\d+)\s*tuoi\s*(\d+)\s*thang", fragment)
    if combined:
        months = float(int(combined.group(1)) * 12 + int(combined.group(2)))
        return _valid_label(months, months, "exact")

    year_range = re.search(r"(\d+)\s*(?:-|den)\s*(\d+)\s*tuoi", fragment)
    if year_range:
        return _valid_label(
            float(int(year_range.group(1)) * 12),
            float(int(year_range.group(2)) * 12),
            "range",
        )
    month_range = re.search(r"(\d+)\s*(?:-|den)\s*(\d+)\s*thang", fragment)
    if month_range:
        return _valid_label(
            float(month_range.group(1)),
            float(month_range.group(2)),
            "range",
        )
    months = re.search(r"(\d+)\s*thang", fragment)
    if months:
        value = float(months.group(1))
        return _valid_label(value, value, "exact")
    years = re.search(r"(\d+)\s*tuoi", fragment)
    if years:
        value = float(int(years.group(1)) * 12)
        return _valid_label(value, value, "exact")
    return None


def parse_bone_age_label(text: str) -> BoneAgeLabel | None:
    """Return one unambiguous report label without retaining the source phrase."""
    folded = fold_vietnamese(text)
    fragments = re.findall(
        r"tuoi xuong tuong ung voi\s*#?\s*([^\n(]{0,60})",
        folded,
    )
    labels = [label for fragment in fragments if (label := _parse_age_fragment(fragment))]
    unique = {
        (label.midpoint_months, label.lower_months, label.upper_months, label.precision): label
        for label in labels
    }
    return next(iter(unique.values())) if len(unique) == 1 else None


def parse_acquisition_date(text: str) -> str | None:
    folded = fold_vietnamese(text)
    match = re.search(
        r"thoi gian chup\s*:?\s*(\d{1,2})/(\d{1,2})/(\d{4})",
        folded,
    )
    if not match:
        return None
    day, month, year = (int(value) for value in match.groups())
    try:
        return date(year, month, day).strftime("%Y%m%d")
    except ValueError:
        return None


def numeric_identifier_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d{6,12}\b", text))


def parse_patient_age_months(value: object) -> float | None:
    match = re.fullmatch(r"\s*(\d{1,3})([DWMY])\s*", str(value).upper())
    if not match:
        return None
    amount = int(match.group(1))
    factor = {"D": 1 / 30.4375, "W": 7 / 30.4375, "M": 1.0, "Y": 12.0}[match.group(2)]
    months = amount * factor
    return float(months) if 0 <= months <= 240 else None
