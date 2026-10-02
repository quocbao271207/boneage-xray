"""Reproducible inventory and image-level EDA for the Kaggle RSNA Bone Age dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def image_metrics(item: tuple[str, Path]) -> dict[str, Any]:
    split, path = item
    try:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
            source_mode = image.mode
            source_format = image.format
            preview = image.convert("L")
            preview.thumbnail((64, 64), Image.Resampling.BILINEAR)
            pixels = np.asarray(preview, dtype=np.float32)
        return {
            "split": split,
            "id": int(path.stem),
            "relative_path": path.as_posix(),
            "file_bytes": len(payload),
            "sha256": digest,
            "format": source_format,
            "mode": source_mode,
            "width": width,
            "height": height,
            "megapixels": width * height / 1_000_000,
            "aspect_width_height": width / height,
            "intensity_mean_preview": float(pixels.mean()),
            "intensity_std_preview": float(pixels.std()),
            "intensity_p01_preview": float(np.quantile(pixels, 0.01)),
            "intensity_p50_preview": float(np.quantile(pixels, 0.50)),
            "intensity_p99_preview": float(np.quantile(pixels, 0.99)),
            "near_black_fraction_preview": float((pixels <= 5).mean()),
            "near_white_fraction_preview": float((pixels >= 250).mean()),
            "error": None,
        }
    # A dataset audit must report any per-file decoder/runtime failure and continue.
    except Exception as exc:  # noqa: BLE001
        return {
            "split": split,
            "id": path.stem,
            "relative_path": path.as_posix(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def describe(series: pd.Series) -> dict[str, float]:
    result = series.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {str(key): round(float(value), 6) for key, value in result.items()}


def ids_from_paths(paths: list[Path]) -> tuple[set[int], list[str]]:
    ids: set[int] = set()
    invalid: list[str] = []
    for path in paths:
        try:
            ids.add(int(path.stem))
        except ValueError:
            invalid.append(path.name)
    return ids, invalid


def tabular_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    age = train["boneage"]
    return {
        "train": {
            "rows": len(train),
            "columns": train.columns.tolist(),
            "dtypes": {name: str(dtype) for name, dtype in train.dtypes.items()},
            "missing_by_column": train.isna().sum().astype(int).to_dict(),
            "duplicate_rows": int(train.duplicated().sum()),
            "duplicate_ids": int(train["id"].duplicated().sum()),
            "age_months": describe(age),
            "age_unique_values": int(age.nunique()),
            "sex_counts": {
                str(key): int(value) for key, value in train["male"].value_counts().items()
            },
            "sex_age_months": {
                str(key): {name: round(float(value), 6) for name, value in row.items()}
                for key, row in train.groupby("male")["boneage"]
                .agg(["count", "mean", "std", "min", "median", "max"])
                .to_dict(orient="index")
                .items()
            },
        },
        "test": {
            "rows": len(test),
            "columns": test.columns.tolist(),
            "dtypes": {name: str(dtype) for name, dtype in test.dtypes.items()},
            "missing_by_column": test.isna().sum().astype(int).to_dict(),
            "duplicate_rows": int(test.duplicated().sum()),
            "duplicate_ids": int(test["Case ID"].duplicated().sum()),
            "sex_counts": {
                str(key): int(value) for key, value in test["Sex"].value_counts().items()
            },
        },
        "train_test_id_overlap": len(set(train["id"]) & set(test["Case ID"])),
    }


def plot_age_distribution(train: pd.DataFrame, output_dir: Path) -> None:
    bins = np.arange(0, 241, 12)
    fig, ax = plt.subplots(figsize=(10, 5))
    for male, label, color in [(False, "Female", "#d95f8d"), (True, "Male", "#2f6db0")]:
        ax.hist(
            train.loc[train["male"] == male, "boneage"],
            bins=bins,
            alpha=0.65,
            label=label,
            color=color,
        )
    ax.set(title="Training labels by sex", xlabel="Bone age (months)", ylabel="Images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "age_distribution_by_sex.png", dpi=160)
    plt.close(fig)


def plot_image_distributions(metrics: pd.DataFrame, output_dir: Path) -> None:
    valid = metrics[metrics["error"].isna()].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fields = [
        ("megapixels", "Megapixels"),
        ("aspect_width_height", "Width / height"),
        ("intensity_mean_preview", "Mean intensity (64 px preview)"),
        ("intensity_std_preview", "Intensity std. dev. (64 px preview)"),
    ]
    colors = {"train": "#2f6db0", "test": "#df8f2d"}
    for ax, (field, label) in zip(axes.flat, fields, strict=True):
        for split in ("train", "test"):
            ax.hist(
                valid.loc[valid["split"] == split, field],
                bins=40,
                alpha=0.55,
                density=True,
                label=split,
                color=colors[split],
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
    axes[0, 0].legend()
    fig.suptitle("Image geometry and intensity distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "image_distributions.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_root / "boneage-training-dataset.csv"
    test_csv = data_root / "boneage-test-dataset.csv"
    train_dir = data_root / "boneage-training-dataset" / "boneage-training-dataset"
    test_dir = data_root / "boneage-test-dataset" / "boneage-test-dataset"

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    paths_by_split = {
        "train": sorted(train_dir.glob("*.png"), key=lambda path: int(path.stem)),
        "test": sorted(test_dir.glob("*.png"), key=lambda path: int(path.stem)),
    }
    work = [(split, path) for split, paths in paths_by_split.items() for path in paths]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(image_metrics, work))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "image_metrics.csv", index=False)

    train_image_ids, train_invalid_names = ids_from_paths(paths_by_split["train"])
    test_image_ids, test_invalid_names = ids_from_paths(paths_by_split["test"])
    expected_train_ids = set(train["id"])
    expected_test_ids = set(test["Case ID"])
    valid = metrics[metrics["error"].isna()].copy()

    duplicates: defaultdict[str, list[str]] = defaultdict(list)
    for row in valid.itertuples(index=False):
        duplicates[row.sha256].append(row.relative_path)
    duplicate_groups = [items for items in duplicates.values() if len(items) > 1]

    image_columns = [
        "file_bytes",
        "width",
        "height",
        "megapixels",
        "aspect_width_height",
        "intensity_mean_preview",
        "intensity_std_preview",
        "near_black_fraction_preview",
        "near_white_fraction_preview",
    ]
    image_summary: dict[str, Any] = {}
    for split in ("train", "test"):
        subset = valid[valid["split"] == split]
        dimensions = Counter(
            zip(subset["width"].astype(int), subset["height"].astype(int), strict=True)
        )
        image_summary[split] = {
            "files": len(paths_by_split[split]),
            "bytes": int(sum(path.stat().st_size for path in paths_by_split[split])),
            "format_counts": {
                str(key): int(value) for key, value in subset["format"].value_counts().items()
            },
            "mode_counts": {
                str(key): int(value) for key, value in subset["mode"].value_counts().items()
            },
            "unique_dimensions": len(dimensions),
            "top_dimensions": [
                {"width": width, "height": height, "count": count}
                for (width, height), count in dimensions.most_common(15)
            ],
            "portrait": int((subset["height"] > subset["width"]).sum()),
            "square": int((subset["height"] == subset["width"]).sum()),
            "landscape": int((subset["height"] < subset["width"]).sum()),
            "metrics": {column: describe(subset[column]) for column in image_columns},
        }

    train_metrics = valid[valid["split"] == "train"].merge(train, on="id", validate="one_to_one")
    correlations = {}
    for method in ("pearson", "spearman"):
        correlations[method] = {
            column: round(
                float(train_metrics[["boneage", column]].corr(method=method).iloc[0, 1]), 6
            )
            for column in image_columns
        }

    summary = {
        "data_root": str(data_root),
        "tabular": tabular_summary(train, test),
        "file_contract": {
            "train_missing_images": sorted(expected_train_ids - train_image_ids),
            "train_extra_images": sorted(train_image_ids - expected_train_ids),
            "test_missing_images": sorted(expected_test_ids - test_image_ids),
            "test_extra_images": sorted(test_image_ids - expected_test_ids),
            "invalid_train_filenames": train_invalid_names,
            "invalid_test_filenames": test_invalid_names,
            "corrupt_or_unreadable_images": metrics.loc[
                metrics["error"].notna(), ["relative_path", "error"]
            ].to_dict(orient="records"),
            "exact_duplicate_groups": duplicate_groups,
        },
        "images": image_summary,
        "potential_confounders": {
            "age_vs_image_metric_correlations": correlations,
            "image_metric_means_by_train_sex": train_metrics.groupby("male")[image_columns]
            .mean()
            .round(6)
            .to_dict(orient="index"),
        },
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_age_distribution(train, output_dir)
    plot_image_distributions(metrics, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "images_scanned": len(metrics),
                "errors": int(metrics["error"].notna().sum()),
                "exact_duplicate_groups": len(duplicate_groups),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
