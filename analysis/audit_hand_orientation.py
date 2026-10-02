"""Rank masks whose PCA-aligned hand probably has the wrist above the fingers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from boneage.hand_standardization import (
    hand_orientation_degrees,
    keep_largest_component,
    rotate_bound,
)
from boneage.standardize_dataset import resolve_mask_path


def rotation_reverses_axis(
    axis_angle_degrees: float, rotation_override_degrees: float
) -> bool:
    """Return whether an exact rotation selects the opposite PCA-axis direction."""
    automatic_rotation = -axis_angle_degrees
    delta = (rotation_override_degrees - automatic_rotation) % 360.0
    return 90.0 <= delta < 270.0


def orientation_gate_errors(audit: pd.DataFrame) -> list[str]:
    """Return structural orientation failures that must block downstream training."""
    errors: list[str] = []
    reversed_ids = audit.loc[audit["flag_reverse_after_config"], "image_id"].astype(int)
    hole_ids = audit.loc[audit["flag_mask_holes"], "image_id"].astype(int)
    if not reversed_ids.empty:
        errors.append(f"reversed hands remain after configuration: {reversed_ids.tolist()}")
    if not hole_ids.empty:
        errors.append(f"retained masks contain enclosed holes: {hole_ids.tolist()}")
    return errors


def finger_likeness(mask: np.ndarray, start_fraction: float, end_fraction: float) -> float:
    """Score separation into several finger-like runs near one longitudinal end."""
    occupied_rows = np.flatnonzero(mask.any(axis=1))
    if len(occupied_rows) < 20:
        return 0.0
    top = int(occupied_rows[0])
    span = int(occupied_rows[-1] - top + 1)
    start = top + round(span * start_fraction)
    end = top + round(span * end_fraction)
    runs: list[int] = []
    fragmentations: list[float] = []
    for row in mask[start:end]:
        columns = np.flatnonzero(row)
        if len(columns) < 2:
            continue
        runs.append(1 + int(np.count_nonzero(np.diff(columns) > 1)))
        fragmentations.append(float(columns[-1] - columns[0] + 1) / len(columns))
    if not runs:
        return 0.0
    return float(np.percentile(runs, 75) + np.mean(fragmentations))


def enclosed_hole_fraction(mask: np.ndarray) -> float:
    """Return the enclosed hole area relative to hand-mask foreground area."""
    contours, hierarchy = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    foreground_area = float(np.count_nonzero(mask))
    if hierarchy is None or foreground_area == 0:
        return 0.0
    hole_area = sum(
        cv2.contourArea(contour)
        for contour, relationship in zip(contours, hierarchy[0], strict=True)
        if relationship[3] >= 0
    )
    return float(hole_area / foreground_area)


def audit_mask(task: dict[str, Any]) -> dict[str, Any]:
    mask_path = Path(str(task["mask_path"]))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Unable to decode mask: {mask_path}")
    clean = keep_largest_component(mask)
    angle = hand_orientation_degrees(clean)
    aligned = rotate_bound(clean, -angle, cv2.INTER_NEAREST)
    aligned = (aligned > 0).astype(np.uint8)
    top_score = finger_likeness(aligned, 0.04, 0.35)
    bottom_score = finger_likeness(aligned, 0.65, 0.96)
    rotation_override = task.get("rotation_override_degrees")
    configured_flip = bool(task["configured_flip"])
    if rotation_override is not None:
        configured_flip = rotation_reverses_axis(angle, float(rotation_override))
    post_top_score = bottom_score if configured_flip else top_score
    post_bottom_score = top_score if configured_flip else bottom_score
    post_reverse_score = post_bottom_score - post_top_score
    hole_fraction = enclosed_hole_fraction(clean)
    return {
        "image_id": int(task["image_id"]),
        "mask_path": str(mask_path),
        "mask_source": mask_path.parent.name,
        "axis_angle_degrees": angle,
        "top_finger_score": top_score,
        "bottom_finger_score": bottom_score,
        "reverse_score": bottom_score - top_score,
        "configured_flip": configured_flip,
        "rotation_override_degrees": rotation_override,
        "post_top_finger_score": post_top_score,
        "post_bottom_finger_score": post_bottom_score,
        "post_reverse_score": post_reverse_score,
        "enclosed_hole_fraction": hole_fraction,
        "flag_reverse_after_config": post_reverse_score >= 2.5,
        "flag_incomplete_or_nonstandard_pose": post_top_score < 3.0,
        "flag_mask_holes": hole_fraction > 0.005,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/standardization_v2.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/eda-quality/standardized_v2_orientation_audit.csv"),
    )
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = pd.read_parquet(Path(config["source_manifest_path"]))
    excluded_ids = {int(value) for value in config.get("exclude_ids", [])}
    mask_dirs = [Path(path) for path in config["mask_dirs"]]
    mask_overrides = {
        int(image_id): Path(path)
        for image_id, path in config.get("mask_overrides", {}).items()
    }
    flip_ids = {int(image_id) for image_id in config.get("flip_orientation_ids", [])}
    rotation_overrides = {
        int(image_id): float(value)
        for image_id, value in config.get("rotation_overrides", {}).items()
    }

    tasks: list[dict[str, Any]] = []
    for image_id in manifest["image_id"].astype(int):
        if image_id in excluded_ids:
            continue
        mask_path = mask_overrides.get(image_id) or resolve_mask_path(image_id, mask_dirs)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for image_id={image_id}")
        tasks.append(
            {
                "image_id": image_id,
                "mask_path": str(mask_path),
                "configured_flip": image_id in flip_ids,
                "rotation_override_degrees": rotation_overrides.get(image_id),
            }
        )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(
            tqdm(
                executor.map(audit_mask, tasks, chunksize=16),
                total=len(tasks),
                desc="Auditing hand direction",
            )
        )
    audit = pd.DataFrame(rows).sort_values(
        ["flag_reverse_after_config", "post_reverse_score"],
        ascending=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    print(args.output)
    print(
        audit.loc[
            audit[
                [
                    "flag_reverse_after_config",
                    "flag_incomplete_or_nonstandard_pose",
                    "flag_mask_holes",
                ]
            ].any(axis=1)
        ]
        .head(30)
        .to_string(index=False)
    )
    gate_errors = orientation_gate_errors(audit)
    if gate_errors:
        raise RuntimeError("; ".join(gate_errors))


if __name__ == "__main__":
    main()
