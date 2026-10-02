"""Analyze absolute-error tails, age bias and image-quality associations in OOF data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boneage.metrics import binomial_rate_upper_bound

QUALITY_FLAGS = (
    "flag_low_contrast",
    "flag_light_background",
    "flag_off_center",
    "flag_rotated",
    "flag_small_hand",
    "flag_crop_risk",
    "flag_blur_review",
    "candidate_blur_exclusion",
    "flag_incomplete_or_nonstandard_pose",
    "configured_flip",
)

QUALITY_GATE_SCENARIOS = {
    "accept_all": (),
    "nonstandard_pose": ("flag_incomplete_or_nonstandard_pose",),
    "structural": (
        "flag_incomplete_or_nonstandard_pose",
        "flag_crop_risk",
        "flag_off_center",
    ),
    "structural_or_small": (
        "flag_incomplete_or_nonstandard_pose",
        "flag_crop_risk",
        "flag_off_center",
        "flag_small_hand",
    ),
    "all_review_flags": (
        "flag_incomplete_or_nonstandard_pose",
        "flag_crop_risk",
        "flag_off_center",
        "flag_small_hand",
        "flag_low_contrast",
        "flag_rotated",
        "flag_blur_review",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/runs/standardized_v1/oof_predictions.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/manifest_standardized_v1.parquet"),
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=Path("reports/eda-quality/quality_metrics.csv"),
    )
    parser.add_argument(
        "--orientation-audit",
        type=Path,
        default=Path("reports/eda-quality/standardized_v2_orientation_audit.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/model-analysis/standardized_v1_tail"),
    )
    return parser.parse_args()


def grouped_metrics(data: pd.DataFrame, group: str) -> pd.DataFrame:
    """Aggregate metrics needed by the tail-risk promotion gate."""
    grouped = data.groupby(group, dropna=False).agg(
        rows=("image_id", "size"),
        mae=("absolute_error", "mean"),
        bias_months=("signed_error", "mean"),
        median_ae=("absolute_error", "median"),
        p90_ae=("absolute_error", lambda values: values.quantile(0.90)),
        p95_ae=("absolute_error", lambda values: values.quantile(0.95)),
        over_24_count=("over_24_months", "sum"),
        max_ae=("absolute_error", "max"),
    )
    grouped["over_24_rate"] = grouped["over_24_count"] / grouped["rows"]
    grouped["over_24_rate_upper_95"] = [
        binomial_rate_upper_bound(int(events), int(total))
        for events, total in zip(grouped["over_24_count"], grouped["rows"], strict=True)
    ]
    return grouped.reset_index()


def quality_gate_scenarios(data: pd.DataFrame) -> pd.DataFrame:
    """Measure whether fixed image-quality review flags capture catastrophic errors."""
    total_tail = int(data["over_24_months"].sum())
    rows: list[dict[str, float | int | str]] = []
    for scenario, flags in QUALITY_GATE_SCENARIOS.items():
        review = pd.Series(False, index=data.index)
        for flag in flags:
            review |= data[flag].fillna(False).astype(bool)
        accepted = data[~review]
        rejected = data[review]
        accepted_rows = len(accepted)
        accepted_tail = int(accepted["over_24_months"].sum())
        rejected_tail = int(rejected["over_24_months"].sum())
        rows.append(
            {
                "scenario": scenario,
                "review_flags": ",".join(flags),
                "accepted_rows": accepted_rows,
                "reviewed_rows": len(rejected),
                "coverage": float(accepted_rows / len(data)),
                "accepted_mae": (
                    float(accepted["absolute_error"].mean()) if accepted_rows else np.nan
                ),
                "accepted_p95": (
                    float(accepted["absolute_error"].quantile(0.95))
                    if accepted_rows
                    else np.nan
                ),
                "accepted_over_24_count": accepted_tail,
                "accepted_over_24_rate": (
                    float(accepted_tail / accepted_rows) if accepted_rows else np.nan
                ),
                "accepted_over_24_rate_upper_95": (
                    binomial_rate_upper_bound(accepted_tail, accepted_rows)
                    if accepted_rows
                    else np.nan
                ),
                "reviewed_over_24_count": rejected_tail,
                "tail_capture_rate": (
                    float(rejected_tail / total_tail) if total_tail else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    manifest = pd.read_parquet(args.manifest)[["image_id", "sex"]]
    quality = pd.read_csv(args.quality).drop(columns=["path"], errors="ignore")
    quality = quality.rename(columns={"id": "image_id"})
    orientation = pd.read_csv(args.orientation_audit)[
        [
            "image_id",
            "flag_incomplete_or_nonstandard_pose",
            "configured_flip",
            "flag_mask_holes",
        ]
    ]
    data = predictions.merge(manifest, on="image_id", validate="one_to_one")
    data = data.merge(quality, on="image_id", how="left", validate="one_to_one")
    data = data.merge(orientation, on="image_id", how="left", validate="one_to_one")
    if len(data) != len(predictions):
        raise RuntimeError("Merged tail-analysis rows differ from OOF predictions")

    data["signed_error"] = data["prediction_months"] - data["target_months"]
    data["absolute_error"] = data["signed_error"].abs()
    data["over_24_months"] = data["absolute_error"] > 24.0
    data["age_band_start"] = (data["target_months"] // 24 * 24).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped_metrics(data, "age_band_start").to_csv(
        args.output_dir / "by_age_band.csv", index=False
    )
    grouped_metrics(data, "fold").to_csv(args.output_dir / "by_fold.csv", index=False)
    grouped_metrics(data, "sex").to_csv(args.output_dir / "by_sex.csv", index=False)

    association_rows: list[dict[str, float | int | str]] = []
    for flag in QUALITY_FLAGS:
        selected = data[data[flag].fillna(False).astype(bool)]
        if selected.empty:
            continue
        association_rows.append(
            {
                "flag": flag,
                "rows": len(selected),
                "mae": float(selected["absolute_error"].mean()),
                "over_24_count": int(selected["over_24_months"].sum()),
                "over_24_rate": float(selected["over_24_months"].mean()),
                "over_24_rate_upper_95": binomial_rate_upper_bound(
                    int(selected["over_24_months"].sum()), len(selected)
                ),
            }
        )
    pd.DataFrame(association_rows).sort_values(
        "over_24_rate", ascending=False
    ).to_csv(args.output_dir / "quality_flag_associations.csv", index=False)
    quality_gate_scenarios(data).to_csv(
        args.output_dir / "quality_gate_scenarios.csv", index=False
    )

    top_columns = [
        "image_id",
        "target_months",
        "prediction_months",
        "signed_error",
        "absolute_error",
        "fold",
        "sex",
        "age_band_start",
        "quality_flags",
        "flag_incomplete_or_nonstandard_pose",
        "configured_flip",
    ]
    data.nlargest(100, "absolute_error")[top_columns].to_csv(
        args.output_dir / "top_100_errors.csv", index=False
    )
    data[data["over_24_months"]][top_columns].sort_values(
        "absolute_error", ascending=False
    ).to_csv(args.output_dir / "over_24_month_errors.csv", index=False)

    quantiles = data["absolute_error"].quantile(
        [0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1.0]
    )
    over_24_count = int(data["over_24_months"].sum())
    summary = {
        "rows": int(len(data)),
        "mae": float(data["absolute_error"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(data["signed_error"])))),
        "mean_bias_months": float(data["signed_error"].mean()),
        "absolute_error_quantiles": {
            str(key): float(value) for key, value in quantiles.items()
        },
        "over_24_count": over_24_count,
        "over_24_rate": float(data["over_24_months"].mean()),
        "over_24_rate_upper_95": binomial_rate_upper_bound(over_24_count, len(data)),
        "over_36_count": int((data["absolute_error"] > 36.0).sum()),
        "over_48_count": int((data["absolute_error"] > 48.0).sum()),
        "over_60_count": int((data["absolute_error"] > 60.0).sum()),
        "over_predictions": int((data["signed_error"] > 0).sum()),
        "under_predictions": int((data["signed_error"] < 0).sum()),
        "tail_over_predictions": int(
            ((data["signed_error"] > 24.0) & data["over_24_months"]).sum()
        ),
        "tail_under_predictions": int(
            ((data["signed_error"] < -24.0) & data["over_24_months"]).sum()
        ),
        "maximum_error": data.nlargest(1, "absolute_error")[
            ["image_id", "target_months", "prediction_months", "absolute_error"]
        ].iloc[0].to_dict(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
