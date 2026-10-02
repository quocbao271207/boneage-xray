import cv2
import numpy as np
import pytest

from boneage.hand_standardization import (
    estimate_background_tone,
    hand_orientation_degrees,
    keep_largest_component,
    limit_forearm_length,
    standardize_hand,
)


def synthetic_hand(angle: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((260, 220), 35, dtype=np.uint8)
    mask = np.zeros_like(image)
    cv2.ellipse(mask, (110, 155), (42, 82), angle, 0, 360, 255, -1)
    cv2.rectangle(mask, (94, 205), (126, 259), 255, -1)
    image[mask > 0] = 120
    for offset in (-24, -12, 0, 12, 24):
        cv2.line(image, (110, 175), (110 + offset, 75), 215, 5)
    return image, mask


def test_standardize_hand_aligns_and_removes_outside_marker() -> None:
    image, mask = synthetic_hand(angle=28)
    cv2.putText(image, "L", (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3)
    result = standardize_hand(image, mask, target_size=256)
    assert result.image.shape == (256, 256)
    assert result.mask.shape == (256, 256)
    assert abs(hand_orientation_degrees(result.mask)) < 1.0
    assert np.all(result.image[result.mask == 0] == 0)
    assert result.image[result.mask > 0].max() > 0


def test_standardize_hand_adaptive_background_removes_marker_without_black_frame() -> None:
    image, mask = synthetic_hand(angle=18)
    cv2.putText(image, "L", (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3)
    result = standardize_hand(
        image,
        mask,
        target_size=256,
        background_mode="adaptive_uniform",
    )
    outside = result.image[result.mask == 0]
    assert outside.size > 0
    assert outside.min() == outside.max()
    assert outside[0] > 0
    assert result.image.max() > outside[0]


def test_standardize_hand_preserves_source_tone_and_uses_matching_background() -> None:
    image, mask = synthetic_hand()
    result = standardize_hand(
        image,
        mask,
        target_size=256,
        background_mode="adaptive_uniform",
        intensity_mode="preserve",
    )
    outside = result.image[result.mask == 0]
    foreground = result.image[result.mask > 0]
    assert outside.min() == outside.max() == 35
    assert np.count_nonzero(foreground == 120) > 100
    assert np.count_nonzero(foreground == 215) > 10


def test_standardize_hand_rejects_unknown_intensity_mode() -> None:
    image, mask = synthetic_hand()
    with pytest.raises(ValueError, match="intensity_mode"):
        standardize_hand(image, mask, intensity_mode="invented")


def test_standardize_hand_orientation_override_selects_opposite_axis() -> None:
    image, mask = synthetic_hand(angle=28)
    normal = standardize_hand(image, mask, target_size=256)
    flipped = standardize_hand(image, mask, target_size=256, flip_orientation=True)
    rotation_delta = (flipped.rotation_degrees - normal.rotation_degrees) % 360
    assert rotation_delta == pytest.approx(180.0, abs=15.0)


def test_standardize_hand_exact_rotation_override_skips_residual_deskew() -> None:
    image, mask = synthetic_hand(angle=28)

    result = standardize_hand(
        image,
        mask,
        target_size=256,
        rotation_override_degrees=90.0,
        skip_forearm_limit=True,
    )

    assert result.rotation_degrees == 90.0
    assert abs(hand_orientation_degrees(result.mask)) > 1.0


def test_standardize_hand_rejects_conflicting_rotation_overrides() -> None:
    image, mask = synthetic_hand()

    with pytest.raises(ValueError, match="cannot both be enabled"):
        standardize_hand(
            image,
            mask,
            rotation_override_degrees=90.0,
            flip_orientation=True,
        )


def test_estimate_background_tone_ignores_marker_outliers() -> None:
    image = np.full((80, 80), 37, dtype=np.uint8)
    mask = np.zeros_like(image)
    cv2.circle(mask, (40, 40), 20, 1, -1)
    image[mask > 0] = 150
    image[:8, :8] = 255
    assert estimate_background_tone(image, mask) == 37


def test_estimate_background_tone_uses_local_cassette_not_bright_outer_frame() -> None:
    image = np.full((160, 160), 240, dtype=np.uint8)
    image[25:135, 25:135] = 31
    mask = np.zeros_like(image)
    cv2.ellipse(mask, (80, 80), (25, 48), 0, 0, 360, 1, -1)
    image[mask > 0] = 150
    assert estimate_background_tone(image, mask) == 31


def test_standardize_hand_rejects_shape_mismatch() -> None:
    image = np.zeros((100, 80), dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="identical dimensions"):
        standardize_hand(image, mask)


def test_standardize_hand_rejects_empty_mask() -> None:
    image = np.tile(np.arange(80, dtype=np.uint8), (100, 1))
    mask = np.zeros_like(image)
    with pytest.raises(ValueError, match="empty or too small"):
        standardize_hand(image, mask)


def test_limit_forearm_length_removes_far_distal_artifact() -> None:
    image = np.full((320, 220), 100, dtype=np.uint8)
    mask = np.zeros_like(image)
    cv2.ellipse(mask, (110, 125), (70, 90), 0, 0, 360, 1, -1)
    cv2.rectangle(mask, (90, 185), (130, 319), 1, -1)
    cv2.circle(mask, (165, 305), 22, 1, -1)
    limited_image, limited_mask = limit_forearm_length(image, mask)
    assert limited_mask[305, 165] == 0
    assert limited_mask[220, 110] == 1
    assert limited_image[305, 165] == 0


def test_standardize_hand_can_skip_forearm_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    image, mask = synthetic_hand()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("forearm limiter should be skipped")

    monkeypatch.setattr(
        "boneage.hand_standardization.limit_forearm_length",
        fail_if_called,
    )
    result = standardize_hand(image, mask, skip_forearm_limit=True)
    assert result.image.shape == (768, 768)


def test_keep_largest_component_removes_detached_object() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 10), (65, 90), 1, -1)
    cv2.circle(mask, (90, 85), 6, 1, -1)
    cleaned = keep_largest_component(mask)
    assert cleaned[50, 40] == 1
    assert cleaned[85, 90] == 0
