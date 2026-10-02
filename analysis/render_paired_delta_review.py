"""Render raw, baseline and candidate images for the largest paired error changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deltas",
        type=Path,
        default=Path(
            "artifacts/runs/standardized_v2_224_screen/"
            "comparison_v1_fold0_interim/per_image_deltas.csv"
        ),
    )
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        default=Path("artifacts/data/manifest.parquet"),
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=Path("artifacts/data/manifest_standardized_v1.parquet"),
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path("artifacts/data/manifest_standardized_v2.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/model-analysis/standardized_v2_fold0_delta_review"),
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--review-label", default="paired-fold")
    return parser.parse_args()


def select_delta_cases(data: pd.DataFrame, count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the largest paired regressions and improvements."""
    if count <= 0:
        raise ValueError("count must be positive")
    required = {"image_id", "absolute_error_delta"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing paired-delta columns: {sorted(missing)}")
    regressions = data.nlargest(count, "absolute_error_delta").copy()
    improvements = data.nsmallest(count, "absolute_error_delta").copy()
    return regressions, improvements


def select_transition_cases(
    data: pd.DataFrame, transition: str, count: int
) -> pd.DataFrame:
    """Select the largest error changes for one tail-threshold transition."""
    if count <= 0:
        raise ValueError("count must be positive")
    required = {"image_id", "absolute_error_delta", "tail_transition"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing paired-delta columns: {sorted(missing)}")
    subset = data.loc[data["tail_transition"] == transition].copy()
    ascending = transition == "fixed_below_24"
    return subset.sort_values("absolute_error_delta", ascending=ascending).head(count)


def select_largest_candidate_errors(data: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select the candidate's largest absolute errors regardless of transition."""
    if count <= 0:
        raise ValueError("count must be positive")
    required = {"image_id", "candidate_absolute_error"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing paired-delta columns: {sorted(missing)}")
    return data.nlargest(count, "candidate_absolute_error").copy()


def manifest_paths(path: Path, column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)[["image_id", "image_path"]].copy()
    if frame["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image IDs in {path}")
    return frame.rename(columns={"image_path": column})


def load_image(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Review image not found: {image_path}")
    with Image.open(image_path) as image:
        return image.convert("L").copy()


def render_cases(data: pd.DataFrame, output: Path, heading: str) -> None:
    figure, axes = plt.subplots(len(data), 3, figsize=(13, 3.8 * len(data)), squeeze=False)
    for row_index, item in enumerate(data.to_dict("records")):
        raw = load_image(str(item["raw_path"]))
        baseline = load_image(str(item["baseline_path"]))
        candidate = load_image(str(item["candidate_path"]))
        images = (raw, baseline, candidate)
        titles = (
            f"ID {int(item['image_id'])} raw\ntarget {item['target_months']:.0f} months",
            (
                f"v1 pred {item['baseline_prediction']:.1f}\n"
                f"AE {item['baseline_absolute_error']:.1f}"
            ),
            (
                f"v2 pred {item['candidate_prediction']:.1f}\n"
                f"AE {item['candidate_absolute_error']:.1f}; "
                f"delta {item['absolute_error_delta']:+.1f}"
            ),
        )
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            axes[row_index, column].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row_index, column].set_title(title, fontsize=10)
            axes[row_index, column].axis("off")

    figure.suptitle(heading, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    deltas = pd.read_csv(args.deltas)
    regressions, improvements = select_delta_cases(deltas, args.count)
    paths = manifest_paths(args.raw_manifest, "raw_path")
    paths = paths.merge(
        manifest_paths(args.baseline_manifest, "baseline_path"),
        on="image_id",
        validate="one_to_one",
    )
    paths = paths.merge(
        manifest_paths(args.candidate_manifest, "candidate_path"),
        on="image_id",
        validate="one_to_one",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[pd.DataFrame] = []
    for label, filename, cases, heading in (
        (
            "regression",
            "regressions.png",
            regressions,
            f"Largest {args.review_label} regressions: raw vs v1 vs v2",
        ),
        (
            "improvement",
            "improvements.png",
            improvements,
            f"Largest {args.review_label} improvements: raw vs v1 vs v2",
        ),
        (
            "new_over_24",
            "new_over_24.png",
            select_transition_cases(deltas, "new_over_24", args.count),
            f"New {args.review_label} errors above 24 months: raw vs v1 vs v2",
        ),
        (
            "fixed_below_24",
            "fixed_below_24.png",
            select_transition_cases(deltas, "fixed_below_24", args.count),
            f"{args.review_label} errors fixed below 24 months: raw vs v1 vs v2",
        ),
        (
            "largest_candidate_error",
            "largest_candidate_errors.png",
            select_largest_candidate_errors(deltas, args.count),
            f"Largest {args.review_label} candidate errors: raw vs v1 vs v2",
        ),
    ):
        if cases.empty:
            continue
        review = cases.merge(paths, on="image_id", validate="one_to_one")
        render_cases(review, args.output_dir / filename, heading)
        selected.append(review.assign(review_group=label))
    pd.concat(selected, ignore_index=True).to_csv(
        args.output_dir / "selected_cases.csv", index=False
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
