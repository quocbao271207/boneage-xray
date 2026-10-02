"""Build a validated, versioned manifest from the immutable Kaggle download."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from .data_contracts import validate_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    pixels = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
    )
    low_frequency = cv2.dct(pixels)[:8, :8]
    median = float(np.median(low_frequency[1:, :]))
    bits = low_frequency > median
    bits[0, 0] = False
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def compute_image_metrics(path_value: str) -> tuple[int, int, int, str, str]:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_hash = perceptual_hash(image)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unable to decode {path}") from exc
    return width, height, path.stat().st_size, digest, image_hash


def _csv_sha_guard(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"CSV checksum mismatch for {path}: expected {expected}, got {actual}")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_manifest(config: dict[str, object]) -> pd.DataFrame:
    train_csv = Path(str(config["train_csv"]))
    test_csv = Path(str(config["test_csv"]))
    _csv_sha_guard(train_csv, str(config["train_csv_sha256"]))
    _csv_sha_guard(test_csv, str(config["test_csv_sha256"]))

    train = pd.read_csv(train_csv).rename(columns={"id": "image_id", "boneage": "bone_age_months"})
    if list(train.columns) != ["image_id", "bone_age_months", "male"]:
        raise ValueError(f"Unexpected train schema: {train.columns.tolist()}")
    train["split"] = "train"
    train["sex"] = train.pop("male").astype("int8")
    train["image_path"] = train["image_id"].map(
        lambda image_id: str(Path(str(config["train_img_dir"])) / f"{image_id}.png")
    )

    test = pd.read_csv(test_csv).rename(columns={"Case ID": "image_id"})
    if list(test.columns) != ["image_id", "Sex"]:
        raise ValueError(f"Unexpected test schema: {test.columns.tolist()}")
    test["split"] = "test"
    test["sex"] = test.pop("Sex").map({"F": 0, "M": 1})
    if test["sex"].isna().any():
        raise ValueError("Test Sex contains values other than F/M")
    test["sex"] = test["sex"].astype("int8")
    test["bone_age_months"] = np.nan
    test["image_path"] = test["image_id"].map(
        lambda image_id: str(Path(str(config["test_img_dir"])) / f"{image_id}.png")
    )

    manifest = pd.concat([train, test], ignore_index=True)
    workers = min(6, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        metrics = list(
            tqdm(
                executor.map(compute_image_metrics, manifest["image_path"], chunksize=32),
                total=len(manifest),
                desc="Auditing images",
            )
        )
    metric_frame = pd.DataFrame(
        metrics,
        columns=["width", "height", "file_bytes", "sha256", "phash"],
    )
    manifest = pd.concat([manifest, metric_frame], axis=1)
    manifest["duplicate_group_id"] = None
    manifest["source_version"] = str(config["source_version"])
    manifest = manifest[
        [
            "image_id",
            "bone_age_months",
            "split",
            "sex",
            "image_path",
            "width",
            "height",
            "file_bytes",
            "sha256",
            "phash",
            "duplicate_group_id",
            "source_version",
        ]
    ]
    manifest = manifest.astype(
        {
            "image_id": "int64",
            "bone_age_months": "float32",
            "sex": "int8",
            "width": "int32",
            "height": "int32",
            "file_bytes": "int32",
        }
    )
    validate_manifest(
        manifest,
        expected_train_rows=int(config["expected_train_rows"]),
        expected_test_rows=int(config["expected_test_rows"]),
        check_paths=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    manifest = build_manifest(config)
    output = Path(str(config["manifest_out"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    manifest.to_parquet(temporary, index=False)
    temporary.replace(output)
    audit = {
        "total_rows": len(manifest),
        "train_rows": int((manifest["split"] == "train").sum()),
        "test_rows": int((manifest["split"] == "test").sum()),
        "unreadable_images": 0,
        "exact_sha256_duplicate_groups": int((manifest["sha256"].value_counts() > 1).sum()),
        "manifest_sha256": sha256_file(output),
        "source_version": str(config["source_version"]),
    }
    _atomic_json(Path(str(config["audit_out"])), audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
