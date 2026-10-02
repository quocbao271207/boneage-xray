"""Build a reproducible mask-standardized image dataset and manifest."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

from .build_manifest import compute_image_metrics, sha256_file
from .data_contracts import validate_manifest
from .hand_standardization import standardize_hand


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_mask_path(image_id: int, mask_dirs: list[Path]) -> Path | None:
    """Return the first available same-ID mask according to configured priority."""
    for directory in mask_dirs:
        candidate = directory / f"{image_id}.png"
        if candidate.is_file():
            return candidate
    return None


def _valid_cached_image(path: Path, target_size: int) -> bool:
    if not path.is_file():
        return False
    cached = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cached is not None and cached.shape == (target_size, target_size)


def standardize_record(task: dict[str, Any]) -> dict[str, Any]:
    """Standardize one image; designed as a picklable process-pool worker."""
    image_id = int(task["image_id"])
    source_path = Path(str(task["source_path"]))
    mask_path = Path(str(task["mask_path"]))
    output_path = Path(str(task["output_path"]))
    target_size = int(task["target_size"])
    resume = bool(task["resume"])
    background_mode = str(task.get("background_mode", "black"))
    intensity_mode = str(task.get("intensity_mode", "normalized_clahe"))
    flip_orientation = bool(task.get("flip_orientation", False))
    rotation_override_degrees = task.get("rotation_override_degrees")
    if rotation_override_degrees is not None:
        rotation_override_degrees = float(rotation_override_degrees)
    forearm_extension_fraction = float(task.get("forearm_extension_fraction", 0.24))
    skip_forearm_limit = bool(task.get("skip_forearm_limit", False))

    rotation_degrees: float | None = None
    if not (resume and _valid_cached_image(output_path, target_size)):
        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Unable to decode source image_id={image_id}: {source_path}")
        if mask is None:
            raise RuntimeError(f"Unable to decode hand mask image_id={image_id}: {mask_path}")
        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for image_id={image_id}: "
                f"image={image.shape}, mask={mask.shape}"
            )
        result = standardize_hand(
            image,
            mask,
            target_size=target_size,
            background_mode=background_mode,
            intensity_mode=intensity_mode,
            flip_orientation=flip_orientation,
            rotation_override_degrees=rotation_override_degrees,
            forearm_extension_fraction=forearm_extension_fraction,
            skip_forearm_limit=skip_forearm_limit,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.png")
        if not cv2.imwrite(str(temporary), result.image):
            raise OSError(f"Unable to write standardized image_id={image_id}: {temporary}")
        os.replace(temporary, output_path)
        rotation_degrees = result.rotation_degrees

    width, height, file_bytes, digest, perceptual = compute_image_metrics(str(output_path))
    return {
        "image_id": image_id,
        "image_path": str(output_path),
        "mask_path": str(mask_path),
        "mask_source": mask_path.parent.name,
        "background_mode": background_mode,
        "intensity_mode": intensity_mode,
        "flip_orientation": flip_orientation,
        "rotation_override_degrees": rotation_override_degrees,
        "forearm_extension_fraction": forearm_extension_fraction,
        "skip_forearm_limit": skip_forearm_limit,
        "rotation_degrees": rotation_degrees,
        "width": width,
        "height": height,
        "file_bytes": file_bytes,
        "sha256": digest,
        "phash": perceptual,
    }


def build_standardized_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_manifest_path = Path(str(config["source_manifest_path"]))
    source_splits_path = Path(str(config["source_splits_path"]))
    output_dir = Path(str(config["output_dir"]))
    mask_dirs = [Path(str(path)) for path in config["mask_dirs"]]
    target_size = int(config.get("target_size", 768))
    workers = int(config.get("workers", min(6, max(1, os.cpu_count() or 1))))
    resume = bool(config.get("resume", True))
    excluded_ids = {int(value) for value in config.get("exclude_ids", [])}
    mask_overrides = {
        int(image_id): Path(str(path))
        for image_id, path in config.get("mask_overrides", {}).items()
    }
    background_mode = str(config.get("background_mode", "black"))
    intensity_mode = str(config.get("intensity_mode", "normalized_clahe"))
    flip_orientation_ids = {
        int(image_id) for image_id in config.get("flip_orientation_ids", [])
    }
    rotation_overrides = {
        int(image_id): float(value)
        for image_id, value in config.get("rotation_overrides", {}).items()
    }
    conflicting_orientation_overrides = flip_orientation_ids & set(rotation_overrides)
    if conflicting_orientation_overrides:
        raise ValueError(
            "Image IDs cannot use both flip_orientation and an exact rotation override: "
            f"{sorted(conflicting_orientation_overrides)}"
        )
    forearm_extension_overrides = {
        int(image_id): float(value)
        for image_id, value in config.get("forearm_extension_overrides", {}).items()
    }
    skip_forearm_limit_ids = {
        int(image_id) for image_id in config.get("skip_forearm_limit_ids", [])
    }

    if target_size < 64:
        raise ValueError("target_size must be at least 64")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not mask_dirs:
        raise ValueError("At least one mask directory is required")

    manifest = pd.read_parquet(source_manifest_path)
    validate_manifest(manifest, check_paths=True)
    unknown_exclusions = excluded_ids - set(manifest["image_id"].astype(int))
    if unknown_exclusions:
        raise ValueError(f"Excluded IDs are absent from the manifest: {sorted(unknown_exclusions)}")
    unknown_overrides = set(mask_overrides) - set(manifest["image_id"].astype(int))
    if unknown_overrides:
        raise ValueError(
            f"Mask override IDs are absent from the manifest: {sorted(unknown_overrides)}"
        )
    unknown_orientation_overrides = flip_orientation_ids - set(
        manifest["image_id"].astype(int)
    )
    if unknown_orientation_overrides:
        raise ValueError(
            "Orientation override IDs are absent from the manifest: "
            f"{sorted(unknown_orientation_overrides)}"
        )
    unknown_rotation_overrides = set(rotation_overrides) - set(
        manifest["image_id"].astype(int)
    )
    if unknown_rotation_overrides:
        raise ValueError(
            "Rotation override IDs are absent from the manifest: "
            f"{sorted(unknown_rotation_overrides)}"
        )
    unknown_forearm_overrides = set(forearm_extension_overrides) - set(
        manifest["image_id"].astype(int)
    )
    if unknown_forearm_overrides:
        raise ValueError(
            "Forearm override IDs are absent from the manifest: "
            f"{sorted(unknown_forearm_overrides)}"
        )
    unknown_skip_forearm_ids = skip_forearm_limit_ids - set(
        manifest["image_id"].astype(int)
    )
    if unknown_skip_forearm_ids:
        raise ValueError(
            "Skip-forearm-limit IDs are absent from the manifest: "
            f"{sorted(unknown_skip_forearm_ids)}"
        )

    tasks: list[dict[str, Any]] = []
    missing_masks: list[int] = []
    for row in manifest.itertuples(index=False):
        image_id = int(row.image_id)
        if image_id in excluded_ids:
            excluded_output = output_dir / str(row.split) / f"{image_id}.png"
            if excluded_output.is_file():
                excluded_output.unlink()
            continue
        mask_path = mask_overrides.get(image_id) or resolve_mask_path(image_id, mask_dirs)
        if mask_path is None:
            missing_masks.append(image_id)
            continue
        if not mask_path.is_file():
            raise FileNotFoundError(f"Configured mask override does not exist: {mask_path}")
        tasks.append(
            {
                "image_id": image_id,
                "source_path": str(row.image_path),
                "mask_path": str(mask_path),
                "output_path": str(output_dir / str(row.split) / f"{image_id}.png"),
                "target_size": target_size,
                "resume": resume,
                "background_mode": background_mode,
                "intensity_mode": intensity_mode,
                "flip_orientation": image_id in flip_orientation_ids,
                "rotation_override_degrees": rotation_overrides.get(image_id),
                "forearm_extension_fraction": forearm_extension_overrides.get(
                    image_id, 0.24
                ),
                "skip_forearm_limit": image_id in skip_forearm_limit_ids,
            }
        )
    if missing_masks:
        raise FileNotFoundError(
            f"Missing masks for {len(missing_masks)} non-excluded images: {missing_masks[:20]}"
        )
    if not tasks:
        raise ValueError("No images remain after exclusions")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(standardize_record, tasks, chunksize=8),
                total=len(tasks),
                desc="Standardizing images",
            )
        )
    metrics = pd.DataFrame(results).set_index("image_id")
    metadata_output = Path(str(config["metadata_output"]))
    if resume and metadata_output.is_file() and metrics["rotation_degrees"].isna().any():
        previous_metadata = pd.read_csv(metadata_output).set_index("image_id")
        if "rotation_degrees" in previous_metadata:
            metrics["rotation_degrees"] = metrics["rotation_degrees"].fillna(
                previous_metadata["rotation_degrees"]
            )
    standardized = manifest[~manifest["image_id"].isin(excluded_ids)].copy()
    standardized = standardized.set_index("image_id")
    if set(standardized.index) != set(metrics.index):
        raise RuntimeError("Standardized output IDs do not match the retained manifest IDs")
    for column in ("image_path", "width", "height", "file_bytes", "sha256", "phash"):
        standardized[column] = metrics[column]
    version = str(config.get("version", "standardized-v1"))
    standardized["source_version"] = standardized["source_version"].astype(str) + f"+{version}"
    standardized = standardized.reset_index()
    standardized = standardized.astype(
        {
            "image_id": "int64",
            "bone_age_months": "float32",
            "sex": "int8",
            "width": "int32",
            "height": "int32",
            "file_bytes": "int32",
        }
    )
    validate_manifest(standardized, check_paths=True)

    splits = pd.read_csv(source_splits_path)
    retained_train_ids = set(
        standardized.loc[standardized["split"] == "train", "image_id"].astype(int)
    )
    standardized_splits = splits[splits["image_id"].isin(retained_train_ids)].copy()
    if set(standardized_splits["image_id"].astype(int)) != retained_train_ids:
        raise ValueError("Filtered split file does not cover every retained training image")
    if standardized_splits["image_id"].duplicated().any():
        raise ValueError("Filtered split file contains duplicate image IDs")

    manifest_output = Path(str(config["manifest_output"]))
    splits_output = Path(str(config["splits_output"]))
    for path in (manifest_output, splits_output, metadata_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_temporary = manifest_output.with_suffix(manifest_output.suffix + ".tmp")
    splits_temporary = splits_output.with_suffix(splits_output.suffix + ".tmp")
    metadata_temporary = metadata_output.with_suffix(metadata_output.suffix + ".tmp")
    standardized.to_parquet(manifest_temporary, index=False)
    standardized_splits.to_csv(splits_temporary, index=False)
    metrics.reset_index().to_csv(metadata_temporary, index=False)
    manifest_temporary.replace(manifest_output)
    splits_temporary.replace(splits_output)
    metadata_temporary.replace(metadata_output)

    source_counts = metrics["mask_source"].value_counts().sort_index().to_dict()
    audit = {
        "version": version,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "manifest_sha256": sha256_file(manifest_output),
        "target_size": target_size,
        "background_mode": background_mode,
        "intensity_mode": intensity_mode,
        "flip_orientation_ids": sorted(flip_orientation_ids),
        "rotation_overrides": {
            str(key): value for key, value in sorted(rotation_overrides.items())
        },
        "forearm_extension_overrides": {
            str(key): value for key, value in sorted(forearm_extension_overrides.items())
        },
        "skip_forearm_limit_ids": sorted(skip_forearm_limit_ids),
        "total_rows": len(standardized),
        "train_rows": int((standardized["split"] == "train").sum()),
        "test_rows": int((standardized["split"] == "test").sum()),
        "excluded_ids": sorted(excluded_ids),
        "mask_source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "missing_masks": [],
    }
    atomic_json(Path(str(config["audit_output"])), audit)
    return standardized, metrics.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standardization.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.overwrite:
        config["resume"] = False
    manifest, _ = build_standardized_dataset(config)
    print(
        json.dumps(
            {
                "rows": len(manifest),
                "train_rows": int((manifest["split"] == "train").sum()),
                "test_rows": int((manifest["split"] == "test").sum()),
                "manifest_output": str(config["manifest_output"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
