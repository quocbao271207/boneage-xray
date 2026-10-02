"""Verify standardized images against their immutable manifest and summarize backgrounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def validate_metadata_overrides(
    metadata: pd.DataFrame, config: dict[str, Any]
) -> list[str]:
    """Validate that audited per-image geometry overrides reached generated metadata."""
    errors: list[str] = []
    indexed = metadata.set_index("image_id")
    rotation_overrides = {
        int(image_id): float(value)
        for image_id, value in config.get("rotation_overrides", {}).items()
    }
    skip_forearm_ids = {
        int(image_id) for image_id in config.get("skip_forearm_limit_ids", [])
    }

    required_rotation_columns = {"rotation_degrees", "rotation_override_degrees"}
    if rotation_overrides and not required_rotation_columns.issubset(metadata.columns):
        errors.append("metadata is missing exact rotation-override columns")
    else:
        for image_id, expected in rotation_overrides.items():
            if image_id not in indexed.index:
                errors.append(f"rotation override image_id={image_id} is absent from metadata")
                continue
            applied = float(indexed.at[image_id, "rotation_degrees"])
            recorded = float(indexed.at[image_id, "rotation_override_degrees"])
            if not np.isclose(applied, expected, atol=1e-6):
                errors.append(
                    f"image_id={image_id} applied rotation {applied} != override {expected}"
                )
            if not np.isclose(recorded, expected, atol=1e-6):
                errors.append(
                    f"image_id={image_id} recorded rotation override {recorded} != {expected}"
                )

    if skip_forearm_ids and "skip_forearm_limit" not in metadata.columns:
        errors.append("metadata is missing skip_forearm_limit")
    else:
        for image_id in skip_forearm_ids:
            if image_id not in indexed.index:
                errors.append(f"skip-forearm image_id={image_id} is absent from metadata")
                continue
            if not bool(indexed.at[image_id, "skip_forearm_limit"]):
                errors.append(f"image_id={image_id} did not record skip_forearm_limit=True")
    return errors


def unexpected_border_contact_ids(
    quality: pd.DataFrame, allowed_ids: set[int]
) -> list[int]:
    """Return nonuniform-border outputs not covered by an explicit fidelity exception."""
    observed = set(
        quality.loc[quality["border_nonuniform_fraction"] > 0, "image_id"].astype(int)
    )
    return sorted(observed - allowed_ids)


def audit_image(task: dict[str, Any]) -> dict[str, Any]:
    """Decode one output and independently recompute integrity/background metrics."""
    path = Path(str(task["image_path"]))
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"image_id": int(task["image_id"]), "decode_ok": False}
    border_width = max(1, round(min(image.shape) * 0.01))
    border = np.concatenate(
        (
            image[:border_width, :].ravel(),
            image[-border_width:, :].ravel(),
            image[:, :border_width].ravel(),
            image[:, -border_width:].ravel(),
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    background_value = int(np.median(border))
    return {
        "image_id": int(task["image_id"]),
        "decode_ok": True,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "sha256": digest,
        "hash_matches_manifest": digest == str(task["sha256"]),
        "background_value": background_value,
        "border_min": int(border.min()),
        "border_max": int(border.max()),
        "border_nonuniform_fraction": float(np.mean(border != background_value)),
        "zero_fraction": float(np.mean(image == 0)),
        "saturated_fraction": float(np.mean(image == 255)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/standardization_v2.yaml"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("artifacts/data/standardized_v2_quality.csv"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/data/standardized_v2_quality_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    manifest = pd.read_parquet(Path(config["manifest_output"]))
    splits = pd.read_csv(Path(config["splits_output"]))
    metadata = pd.read_csv(Path(config["metadata_output"]))
    generation_audit = json.loads(Path(config["audit_output"]).read_text(encoding="utf-8"))
    source = pd.read_parquet(Path(config["source_manifest_path"]))
    excluded_ids = {int(value) for value in config.get("exclude_ids", [])}
    expected_ids = set(source["image_id"].astype(int)) - excluded_ids

    manifest_ids = set(manifest["image_id"].astype(int))
    metadata_ids = set(metadata["image_id"].astype(int))
    train_ids = set(
        manifest.loc[manifest["split"] == "train", "image_id"].astype(int)
    )
    split_ids = set(splits["image_id"].astype(int))
    structural_errors: list[str] = []
    if manifest["image_id"].duplicated().any():
        structural_errors.append("manifest contains duplicate image IDs")
    if metadata["image_id"].duplicated().any():
        structural_errors.append("metadata contains duplicate image IDs")
    if splits["image_id"].duplicated().any():
        structural_errors.append("split table contains duplicate image IDs")
    if manifest_ids != expected_ids:
        structural_errors.append("manifest IDs differ from retained source IDs")
    if metadata_ids != expected_ids:
        structural_errors.append("metadata IDs differ from retained source IDs")
    if split_ids != train_ids:
        structural_errors.append("split IDs differ from retained training IDs")
    structural_errors.extend(validate_metadata_overrides(metadata, config))

    tasks = manifest[["image_id", "image_path", "sha256"]].to_dict("records")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(
            tqdm(
                executor.map(audit_image, tasks, chunksize=16),
                total=len(tasks),
                desc="Auditing standardized outputs",
            )
        )
    quality = pd.DataFrame(rows).sort_values("image_id")
    decoded = quality[quality["decode_ok"]].copy()
    target_size = int(config["target_size"])
    bad_dimensions = decoded[
        (decoded["width"] != target_size) | (decoded["height"] != target_size)
    ]
    bad_hashes = decoded[~decoded["hash_matches_manifest"]]
    undecodable = quality[~quality["decode_ok"]]
    allowed_border_contact_ids = {
        int(image_id) for image_id in config.get("allowed_border_contact_ids", [])
    }
    unexpected_border_ids = unexpected_border_contact_ids(
        decoded, allowed_border_contact_ids
    )
    if not undecodable.empty:
        structural_errors.append(f"{len(undecodable)} images cannot be decoded")
    if not bad_dimensions.empty:
        structural_errors.append(f"{len(bad_dimensions)} images have wrong dimensions")
    if not bad_hashes.empty:
        structural_errors.append(f"{len(bad_hashes)} image hashes differ from manifest")
    if unexpected_border_ids:
        structural_errors.append(
            f"unexpected nonuniform output borders: {unexpected_border_ids}"
        )

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.csv_output, index=False)
    zero_quantiles = decoded["zero_fraction"].quantile([0.5, 0.9, 0.95, 0.99, 1.0])
    summary = {
        "status": "passed" if not structural_errors else "failed",
        "rows": int(len(manifest)),
        "train_rows": int((manifest["split"] == "train").sum()),
        "test_rows": int((manifest["split"] == "test").sum()),
        "metadata_rows": int(len(metadata)),
        "split_rows": int(len(splits)),
        "generation_audit_rows": int(generation_audit["total_rows"]),
        "rotation_override_count": int(len(config.get("rotation_overrides", {}))),
        "skip_forearm_limit_count": int(len(config.get("skip_forearm_limit_ids", []))),
        "wrong_dimension_count": int(len(bad_dimensions)),
        "hash_mismatch_count": int(len(bad_hashes)),
        "undecodable_count": int(len(undecodable)),
        "uniform_border_count": int(
            (decoded["border_nonuniform_fraction"] == 0).sum()
        ),
        "allowed_border_contact_ids": sorted(allowed_border_contact_ids),
        "unexpected_border_contact_ids": unexpected_border_ids,
        "black_background_count": int((decoded["background_value"] == 0).sum()),
        "background_value_quantiles": {
            str(key): float(value)
            for key, value in decoded["background_value"]
            .quantile([0.0, 0.5, 0.9, 1.0])
            .items()
        },
        "zero_fraction_quantiles": {
            str(key): float(value) for key, value in zero_quantiles.items()
        },
        "structural_errors": structural_errors,
        "csv_output": str(args.csv_output),
    }
    args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if structural_errors:
        raise RuntimeError("; ".join(structural_errors))


if __name__ == "__main__":
    main()
