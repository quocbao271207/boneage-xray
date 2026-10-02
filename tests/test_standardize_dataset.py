from pathlib import Path

import cv2
import numpy as np

from boneage.standardize_dataset import resolve_mask_path, standardize_record


def test_resolve_mask_path_uses_configured_priority(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    primary.mkdir()
    fallback.mkdir()
    (fallback / "17.png").write_bytes(b"fallback")
    assert resolve_mask_path(17, [primary, fallback]) == fallback / "17.png"
    (primary / "17.png").write_bytes(b"primary")
    assert resolve_mask_path(17, [primary, fallback]) == primary / "17.png"


def test_standardize_record_writes_square_adaptive_background_image(tmp_path: Path) -> None:
    image = np.full((160, 120), 25, dtype=np.uint8)
    mask = np.zeros_like(image)
    cv2.ellipse(mask, (60, 75), (35, 55), 10, 0, 360, 255, -1)
    cv2.rectangle(mask, (48, 110), (72, 159), 255, -1)
    gradient = np.tile(np.linspace(80, 190, image.shape[1], dtype=np.uint8), (160, 1))
    image[mask > 0] = gradient[mask > 0]
    cv2.putText(image, "L", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    image_path = tmp_path / "7.png"
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    mask_path = mask_dir / "7.png"
    output_path = tmp_path / "output" / "7.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)

    result = standardize_record(
        {
            "image_id": 7,
            "source_path": image_path,
            "mask_path": mask_path,
            "output_path": output_path,
            "target_size": 128,
            "resume": False,
            "background_mode": "adaptive_uniform",
            "intensity_mode": "preserve",
            "flip_orientation": False,
            "rotation_override_degrees": 90.0,
            "skip_forearm_limit": True,
        }
    )
    output = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert output is not None
    assert output.shape == (128, 128)
    assert output[0, 0] > 0
    assert result["mask_source"] == "masks"
    assert result["intensity_mode"] == "preserve"
    assert result["rotation_override_degrees"] == 90.0
    assert result["rotation_degrees"] == 90.0
    assert result["skip_forearm_limit"] is True
    assert result["width"] == 128
    assert result["height"] == 128
