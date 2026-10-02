"""Render original-versus-standardized hand X-rays for visual review."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from boneage.hand_standardization import standardize_hand
from boneage.standardize_dataset import resolve_mask_path

DEFAULT_IDS = (1381, 1489, 1514, 1737, 1729, 7049)


def resolve_target_size(argument: int | None, config: dict[str, object]) -> int:
    """Use an explicit CLI size, otherwise inherit the dataset configuration."""
    return int(argument if argument is not None else config.get("target_size", 768))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/boneage-training-dataset/boneage-training-dataset"),
    )
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/standardization_v2.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--target-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    target_size = resolve_target_size(args.target_size, config)
    mask_dirs = (
        [args.mask_dir.resolve()]
        if args.mask_dir is not None
        else [Path(path).resolve() for path in config["mask_dirs"]]
    )
    mask_overrides = {
        int(image_id): Path(path).resolve()
        for image_id, path in config.get("mask_overrides", {}).items()
    }
    flip_orientation_ids = {
        int(image_id) for image_id in config.get("flip_orientation_ids", [])
    }
    rotation_overrides = {
        int(image_id): float(value)
        for image_id, value in config.get("rotation_overrides", {}).items()
    }
    forearm_extension_overrides = {
        int(image_id): float(value)
        for image_id, value in config.get("forearm_extension_overrides", {}).items()
    }
    skip_forearm_limit_ids = {
        int(image_id) for image_id in config.get("skip_forearm_limit_ids", [])
    }
    background_mode = str(config.get("background_mode", "black"))
    intensity_mode = str(config.get("intensity_mode", "normalized_clahe"))

    results = []
    for image_id in args.ids:
        image_path = image_dir / f"{image_id}.png"
        mask_path = mask_overrides.get(image_id) or resolve_mask_path(image_id, mask_dirs)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for ID {image_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image or mask for ID {image_id}")
        result = standardize_hand(
            image,
            mask,
            target_size=target_size,
            background_mode=background_mode,
            intensity_mode=intensity_mode,
            flip_orientation=image_id in flip_orientation_ids,
            rotation_override_degrees=rotation_overrides.get(image_id),
            forearm_extension_fraction=forearm_extension_overrides.get(image_id, 0.24),
            skip_forearm_limit=image_id in skip_forearm_limit_ids,
        )
        output_path = output_dir / f"{image_id}_standardized.png"
        if not cv2.imwrite(str(output_path), result.image):
            raise OSError(f"Could not write {output_path}")
        border = np.concatenate(
            (
                result.image[0],
                result.image[-1],
                result.image[:, 0],
                result.image[:, -1],
            )
        )
        results.append(
            {
                "id": image_id,
                "image_path": image_path,
                "output_path": output_path,
                "rotation_degrees": result.rotation_degrees,
                "mask_source": mask_path.parent.name,
                "target_size": target_size,
                "background_mode": background_mode,
                "intensity_mode": intensity_mode,
                "border_min": int(border.min()),
                "border_max": int(border.max()),
                "zero_pixel_count": int(np.count_nonzero(result.image == 0)),
                "skip_forearm_limit": image_id in skip_forearm_limit_ids,
                "rotation_override_degrees": rotation_overrides.get(image_id),
            }
        )

    fig, axes = plt.subplots(len(results), 2, figsize=(10, 5 * len(results)), squeeze=False)
    for row_index, item in enumerate(results):
        original = cv2.imread(str(item["image_path"]), cv2.IMREAD_GRAYSCALE)
        standardized = cv2.imread(str(item["output_path"]), cv2.IMREAD_GRAYSCALE)
        axes[row_index, 0].imshow(original, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 0].set_title(f"ID {item['id']} — ảnh gốc")
        axes[row_index, 1].imshow(standardized, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].set_title(
            f"chuẩn hóa — xoay {item['rotation_degrees']:.1f}°; "
            f"nền={background_mode}; tone={intensity_mode}; mask={item['mask_source']}"
        )
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle(
        "Ảnh gốc ↔ ảnh chuẩn hóa từ chính pixel gốc (không sinh thêm chi tiết)", fontsize=15
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    montage_path = output_dir / "original_vs_standardized.png"
    fig.savefig(montage_path, dpi=160, facecolor="white")
    plt.close(fig)

    pd.DataFrame(results).drop(columns=["image_path", "output_path"]).to_csv(
        output_dir / "standardization_metrics.csv", index=False
    )
    print(montage_path)


if __name__ == "__main__":
    main()
