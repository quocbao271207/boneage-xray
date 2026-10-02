"""Mask-guided standardization for hand radiographs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StandardizedHand:
    image: np.ndarray
    mask: np.ndarray
    rotation_degrees: float


VALID_BACKGROUND_MODES = {"black", "adaptive_uniform"}
VALID_INTENSITY_MODES = {"normalized_clahe", "preserve"}


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Remove detached foreground objects while preserving the main hand silhouette."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise ValueError("Hand mask is empty or too small")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def hand_orientation_degrees(mask: np.ndarray) -> float:
    """Return the signed angle of the hand axis relative to image vertical."""
    y_coordinates, x_coordinates = np.nonzero(mask)
    if len(x_coordinates) < 10:
        raise ValueError("Hand mask is empty or too small")
    points = np.column_stack([x_coordinates, y_coordinates]).astype(np.float32)
    _, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
    axis_x, axis_y = eigenvectors[0]
    angle = math.degrees(math.atan2(float(axis_x), float(axis_y)))
    return float(((angle + 90) % 180) - 90)


def normalize_hand_intensity(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize and locally enhance only pixels belonging to the hand."""
    foreground = image[mask > 0].astype(np.float32)
    if foreground.size < 10:
        raise ValueError("Hand mask is empty or too small")
    low, high = np.percentile(foreground, [1.0, 99.5])
    if high <= low:
        raise ValueError("Masked hand region has no usable intensity range")
    normalized = np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    normalized = np.rint(normalized * 255).astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(normalized)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0, sigmaY=1.0)
    # A mild unsharp mask improves perceived focus without inventing anatomy.
    return cv2.addWeighted(enhanced, 1.22, blurred, -0.22, 0)


def estimate_background_tone(image: np.ndarray, mask: np.ndarray) -> int:
    """Estimate the local hand background while ignoring cassette/marker outliers."""
    binary_mask = (mask > 0).astype(np.uint8)
    outside = (binary_mask == 0).astype(np.uint8)
    distance_from_hand = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    ring_width = max(3, round(min(image.shape) * 0.04))
    local_ring = (binary_mask == 0) & (distance_from_hand <= ring_width)
    background = image[local_ring].astype(np.float32)
    if background.size < 10:
        background = image[binary_mask == 0].astype(np.float32)
    if background.size < 10:
        border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
        background = border.astype(np.float32)
    low, high = np.percentile(background, [10.0, 90.0])
    central = background[(background >= low) & (background <= high)]
    usable = central if central.size >= 10 else background
    return int(np.clip(np.rint(np.median(usable)), 0, 255))


def rotate_bound(
    image: np.ndarray,
    angle_degrees: float,
    interpolation: int,
    border_value: int = 0,
) -> np.ndarray:
    """Rotate without clipping corners and expand the output canvas as needed."""
    height, width = image.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    output_width = max(1, math.ceil(height * sine + width * cosine))
    output_height = max(1, math.ceil(height * cosine + width * sine))
    matrix[0, 2] += output_width / 2.0 - center[0]
    matrix[1, 2] += output_height / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _crop_to_mask(
    image: np.ndarray, mask: np.ndarray, margin_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Rotated hand mask is empty")
    x, y, width, height = cv2.boundingRect(points)
    margin = round(max(width, height) * margin_fraction)
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(image.shape[1], x + width + margin)
    bottom = min(image.shape[0], y + height + margin)
    return image[top:bottom, left:right], mask[top:bottom, left:right]


def limit_forearm_length(
    image: np.ndarray,
    mask: np.ndarray,
    extension_fraction: float = 0.24,
    background_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a consistent distal forearm length and remove attached edge artifacts."""
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) < 20:
        return image, mask
    top = int(rows[0])
    bottom = int(rows[-1])
    span = bottom - top + 1
    widths = mask.sum(axis=1).astype(np.float32)
    smooth_widths = np.convolve(widths, np.ones(11, dtype=np.float32) / 11, mode="same")

    peak_start = top + round(span * 0.25)
    peak_end = top + round(span * 0.68)
    if peak_end <= peak_start:
        return image, mask
    palm_row = peak_start + int(np.argmax(smooth_widths[peak_start:peak_end]))
    palm_width = float(smooth_widths[palm_row])
    if palm_width <= 0:
        return image, mask

    threshold = palm_width * 0.56
    wrist_row = None
    required_run = max(5, round(span * 0.012))
    below_threshold = smooth_widths <= threshold
    for row in range(palm_row + 1, bottom - required_run + 2):
        if below_threshold[row : row + required_run].all():
            wrist_row = row
            break
    anatomical_wrist_cap = top + round(span * 0.68)
    wrist_row = (
        anatomical_wrist_cap
        if wrist_row is None
        else min(wrist_row, anatomical_wrist_cap)
    )

    wrist_band = mask[wrist_row : wrist_row + required_run]
    wrist_columns = np.flatnonzero(wrist_band.any(axis=0))
    if len(wrist_columns) >= 2:
        wrist_center = (float(wrist_columns[0]) + float(wrist_columns[-1])) / 2.0
        wrist_width = float(wrist_columns[-1] - wrist_columns[0] + 1)
        corridor_half_width = max(8, round(wrist_width * 0.72))
        corridor_left = max(0, round(wrist_center - corridor_half_width))
        corridor_right = min(mask.shape[1], round(wrist_center + corridor_half_width + 1))
        limited_region = np.zeros_like(mask[wrist_row:, :])
        limited_region[:, corridor_left:corridor_right] = mask[
            wrist_row:, corridor_left:corridor_right
        ]
        mask = mask.copy()
        mask[wrist_row:, :] = limited_region

    cutoff = min(bottom + 1, wrist_row + round((wrist_row - top) * extension_fraction))
    limited_image = image.copy()
    limited_mask = mask.copy()
    limited_image[cutoff:, :] = background_value
    limited_mask[cutoff:, :] = 0
    return limited_image, limited_mask


def _pad_and_resize(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int,
    background_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    side = max(height, width)
    canvas = np.full((side, side), background_value, dtype=np.uint8)
    mask_canvas = np.zeros((side, side), dtype=np.uint8)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = image
    mask_canvas[top : top + height, left : left + width] = mask
    interpolation = cv2.INTER_AREA if side > target_size else cv2.INTER_LANCZOS4
    output = cv2.resize(canvas, (target_size, target_size), interpolation=interpolation)
    output_mask = cv2.resize(
        mask_canvas,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return output, (output_mask > 0).astype(np.uint8)


def standardize_hand(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int = 768,
    margin_fraction: float = 0.06,
    background_mode: str = "black",
    intensity_mode: str = "normalized_clahe",
    flip_orientation: bool = False,
    rotation_override_degrees: float | None = None,
    forearm_extension_fraction: float = 0.24,
    skip_forearm_limit: bool = False,
) -> StandardizedHand:
    """Remove background artifacts, align the hand, and standardize its scale.

    PCA determines the hand axis but cannot distinguish fingertips from the wrist.
    ``flip_orientation`` selects the opposite, equivalent axis direction for audited
    exceptions. ``rotation_override_degrees`` applies an exact audited rotation and skips
    automatic residual deskewing. Adaptive background mode removes markers without painting a black
    silhouette onto naturally light or dark radiographs. Preserve intensity mode keeps
    source radiograph tones and skips edge feathering; only geometric interpolation from
    rotation and resizing can then alter retained hand pixels.
    """
    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError("Image and mask must both be 2D grayscale arrays")
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have identical dimensions")
    if target_size < 64:
        raise ValueError("target_size must be at least 64")
    if not 0 <= margin_fraction <= 0.25:
        raise ValueError("margin_fraction must be between 0 and 0.25")
    if background_mode not in VALID_BACKGROUND_MODES:
        raise ValueError(
            f"background_mode must be one of {sorted(VALID_BACKGROUND_MODES)}"
        )
    if intensity_mode not in VALID_INTENSITY_MODES:
        raise ValueError(f"intensity_mode must be one of {sorted(VALID_INTENSITY_MODES)}")
    if not 0.05 <= forearm_extension_fraction <= 0.5:
        raise ValueError("forearm_extension_fraction must be between 0.05 and 0.5")
    if rotation_override_degrees is not None and not math.isfinite(
        rotation_override_degrees
    ):
        raise ValueError("rotation_override_degrees must be finite")
    if rotation_override_degrees is not None and flip_orientation:
        raise ValueError(
            "rotation_override_degrees and flip_orientation cannot both be enabled"
        )

    binary_mask = keep_largest_component(mask)
    angle = hand_orientation_degrees(binary_mask)
    processed = (
        image.copy()
        if intensity_mode == "preserve"
        else normalize_hand_intensity(image, binary_mask)
    )
    background_value = (
        0 if background_mode == "black" else estimate_background_tone(processed, binary_mask)
    )
    applied_rotation = (
        float(rotation_override_degrees)
        if rotation_override_degrees is not None
        else -angle + (180.0 if flip_orientation else 0.0)
    )
    rotated_image = rotate_bound(
        processed,
        applied_rotation,
        cv2.INTER_CUBIC,
        border_value=background_value,
    )
    rotated_mask = rotate_bound(binary_mask, applied_rotation, cv2.INTER_NEAREST)
    rotated_mask = (rotated_mask > 0).astype(np.uint8)
    if not skip_forearm_limit:
        rotated_image, rotated_mask = limit_forearm_length(
            rotated_image,
            rotated_mask,
            extension_fraction=forearm_extension_fraction,
            background_value=background_value,
        )
    rotated_mask = keep_largest_component(rotated_mask)
    rotated_image[rotated_mask == 0] = background_value
    residual_angle = hand_orientation_degrees(rotated_mask)
    if rotation_override_degrees is None and abs(residual_angle) >= 0.25:
        correction = -residual_angle
        rotated_image = rotate_bound(
            rotated_image,
            correction,
            cv2.INTER_CUBIC,
            border_value=background_value,
        )
        rotated_mask = rotate_bound(rotated_mask, -residual_angle, cv2.INTER_NEAREST)
        rotated_mask = keep_largest_component(rotated_mask)
        rotated_image[rotated_mask == 0] = background_value
        applied_rotation += correction
    cropped_image, cropped_mask = _crop_to_mask(
        rotated_image, rotated_mask, margin_fraction=margin_fraction
    )

    if intensity_mode == "preserve":
        composited = cropped_image.copy()
        composited[cropped_mask == 0] = background_value
    else:
        # Feather only the silhouette edge into the audited background tone.
        feathered = cv2.GaussianBlur(cropped_mask.astype(np.float32), (0, 0), sigmaX=0.8)
        feathered = np.clip(feathered, 0.0, 1.0)
        composited = np.rint(
            cropped_image.astype(np.float32) * feathered
            + float(background_value) * (1.0 - feathered)
        ).astype(np.uint8)
    output, output_mask = _pad_and_resize(
        composited,
        cropped_mask,
        target_size,
        background_value=background_value,
    )
    output[output_mask == 0] = background_value
    return StandardizedHand(
        image=output,
        mask=output_mask,
        rotation_degrees=applied_rotation,
    )
