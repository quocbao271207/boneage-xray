"""Image-quality EDA and visual review artifacts for hand X-rays.

The detector is deliberately conservative: it ranks likely quality problems and
writes a review/exclusion list, but never deletes source images.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_REVIEW_IDS = (1381, 1489, 1514, 1737, 1729)
FLAG_LABELS_VI = {
    "low_contrast": "tương phản thấp",
    "light_background": "nền sáng",
    "off_center": "tay lệch tâm",
    "rotated": "tay xoay",
    "small_hand": "tay nhỏ trong ảnh",
    "marker": "marker/vật thể sáng",
    "crop_risk": "nguy cơ bị cắt",
    "blur_review": "cần duyệt độ mờ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="Optional directory of same-name binary hand masks; preferred over heuristics.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analysis-size", type=int, default=640)
    parser.add_argument("--review-ids", type=int, nargs="*", default=DEFAULT_REVIEW_IDS)
    parser.add_argument(
        "--only-review-ids",
        action="store_true",
        help="Analyze only --review-ids while tuning the detector.",
    )
    return parser.parse_args()


def robust_normalize(image: np.ndarray) -> np.ndarray:
    pixels = image.astype(np.float32)
    low, high = np.percentile(pixels, [0.5, 99.5])
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = np.clip((pixels - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255).astype(np.uint8)


def resize_for_analysis(image: np.ndarray, longest_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape
    scale = min(1.0, longest_side / max(height, width))
    if scale == 1.0:
        return image.copy(), scale
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _component_candidates(binary: np.ndarray) -> list[dict[str, Any]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    candidates = []
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        area_fraction = area / binary.size
        width_fraction = box_width / width
        height_fraction = box_height / height
        box_fraction = box_width * box_height / binary.size
        fill_fraction = area / (box_width * box_height)
        if area_fraction < 0.0015:
            continue
        # Film/cassette rims span almost the whole image and are not anatomy.
        if width_fraction > 0.86 and height_fraction > 0.86:
            continue
        if box_fraction > 0.70 or (box_fraction > 0.52 and fill_fraction < 0.34):
            continue
        center_x, center_y = centroids[label]
        distance = math.hypot(center_x / width - 0.5, center_y / height - 0.5)
        centrality = math.exp(-2.5 * distance**2)
        span_bonus = min(1.0, (box_height + box_width) / (height + width) * 2.0)
        score = area * (0.55 + centrality) * (0.75 + span_bonus)
        candidates.append(
            {
                "label": label,
                "bbox": (int(x), int(y), int(box_width), int(box_height)),
                "area": int(area),
                "score": float(score),
                "labels": labels,
            }
        )
    return candidates


def suppress_film_rims(evidence: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    """Suppress long straight cassette edges near the image periphery."""
    height, width = normalized.shape
    minimum_side = min(height, width)
    edges = cv2.Canny(normalized, 45, 135)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=max(35, round(minimum_side * 0.09)),
        minLineLength=round(minimum_side * 0.32),
        maxLineGap=round(minimum_side * 0.035),
    )
    if lines is None:
        return evidence
    rim_mask = np.zeros_like(normalized, dtype=np.uint8)
    thickness = max(5, round(minimum_side * 0.018))
    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)[:4]
        midpoint_x = (x1 + x2) / (2 * width)
        midpoint_y = (y1 + y2) / (2 * height)
        is_peripheral = max(abs(midpoint_x - 0.5), abs(midpoint_y - 0.5)) >= 0.23
        if is_peripheral:
            cv2.line(rim_mask, (x1, y1), (x2, y2), 1, thickness)
    output = evidence.copy()
    output[rim_mask > 0] = 0
    return output


def estimate_hand_mask(normalized: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Estimate anatomy from spatially dense local detail, ignoring sparse film rims."""
    height, width = normalized.shape
    minimum_side = min(height, width)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(normalized)

    # First connect the brightest anatomical structures. Bone pixels form a
    # dense articulated cluster; the long cassette edges are removed by Hough.
    bright_threshold = float(np.percentile(clahe, 82))
    bright = (clahe >= bright_threshold).astype(np.float32)
    bright = suppress_film_rims(bright, normalized)
    border = max(2, round(minimum_side * 0.015))
    bright[:border, :] = 0
    bright[-border:, :] = 0
    bright[:, :border] = 0
    bright[:, -border:] = 0
    join = max(7, round(minimum_side * 0.035))
    if join % 2 == 0:
        join += 1
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join, join))
    bright_binary = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_CLOSE, structure)
    bright_binary = cv2.dilate(bright_binary, structure, iterations=1)
    candidates = _component_candidates(bright_binary)

    # Low-exposure images may not form a connected bright cluster. Fall back
    # to a spatial density map of local contrast and gradients in that case.
    if not candidates:
        candidates = _local_detail_candidates(clahe, normalized, minimum_side)
    if not candidates:
        return np.ones_like(normalized, dtype=np.uint8), (0, 0, width, height)
    selected = max(candidates, key=lambda item: item["score"])
    mask = (selected["labels"] == selected["label"]).astype(np.uint8)
    points = cv2.findNonZero(mask)
    if points is None:
        return np.ones_like(normalized, dtype=np.uint8), (0, 0, width, height)
    x, y, box_width, box_height = cv2.boundingRect(points)
    margin = round(max(box_width, box_height) * 0.11)
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(width, x + box_width + margin)
    bottom = min(height, y + box_height + margin)
    return mask, (left, top, right - left, bottom - top)


def _local_detail_candidates(
    clahe: np.ndarray, normalized: np.ndarray, minimum_side: int
) -> list[dict[str, Any]]:
    """Fallback candidate extraction for unusually dark/flat radiographs."""
    sigma = max(9.0, minimum_side / 18.0)
    background = cv2.GaussianBlur(clahe, (0, 0), sigmaX=sigma, sigmaY=sigma)
    positive_detail = cv2.subtract(clahe, background).astype(np.float32)

    grad_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    detail_scale = max(float(np.percentile(positive_detail, 99)), 1.0)
    gradient_scale = max(float(np.percentile(gradient, 99)), 1.0)
    evidence = np.clip(positive_detail / detail_scale, 0, 1)
    evidence += 0.55 * np.clip(gradient / gradient_scale, 0, 1)
    evidence = suppress_film_rims(evidence, normalized)
    threshold = max(0.18, float(np.percentile(evidence, 76)))
    detail_pixels = (evidence >= threshold).astype(np.float32)
    density_sigma = max(7.0, minimum_side / 28.0)
    density = cv2.GaussianBlur(
        detail_pixels, (0, 0), sigmaX=density_sigma, sigmaY=density_sigma
    )
    # A hand contains many nearby bone and soft-tissue edges. A cassette rim is
    # long but locally sparse, so smoothing an edge map suppresses it naturally.
    density_peak = float(np.percentile(density, 99.7))
    if density_peak <= 0:
        return []
    binary = (density >= density_peak * 0.30).astype(np.uint8)
    join = max(5, round(minimum_side * 0.02))
    if join % 2 == 0:
        join += 1
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join, join)),
    )

    return _component_candidates(binary)


def orientation_degrees(mask: np.ndarray) -> float:
    y_coordinates, x_coordinates = np.nonzero(mask)
    if len(x_coordinates) < 10:
        return 0.0
    points = np.column_stack([x_coordinates, y_coordinates]).astype(np.float32)
    _, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
    axis_x, axis_y = eigenvectors[0]
    angle = math.degrees(math.atan2(float(axis_x), float(axis_y)))
    angle = ((angle + 90) % 180) - 90
    return float(angle)


def bbox_from_mask(
    mask: np.ndarray, margin_fraction: float = 0.07
) -> tuple[int, int, int, int]:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    height, width = mask.shape
    if points is None:
        return (0, 0, width, height)
    x, y, box_width, box_height = cv2.boundingRect(points)
    margin = round(max(box_width, box_height) * margin_fraction)
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(width, x + box_width + margin)
    bottom = min(height, y + box_height + margin)
    return (left, top, right - left, bottom - top)


def load_hand_mask(
    mask_path: Path | None,
    original_shape: tuple[int, int],
    analysis_shape: tuple[int, int],
) -> np.ndarray | None:
    if mask_path is None or not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != original_shape:
        return None
    resized = cv2.resize(
        mask,
        (analysis_shape[1], analysis_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    binary = (resized >= 128).astype(np.uint8)
    return binary if binary.any() else None


def marker_candidates(
    normalized: np.ndarray,
    hand_bbox: tuple[int, int, int, int],
    hand_mask: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """Find compact high-contrast objects outside the estimated anatomy box."""
    height, width = normalized.shape
    x, y, box_width, box_height = hand_bbox
    if hand_mask is None:
        outside = np.ones_like(normalized, dtype=np.uint8)
        outside[y : y + box_height, x : x + box_width] = 0
    else:
        clearance = max(3, round(min(height, width) * 0.012))
        anatomy = cv2.dilate(
            (hand_mask > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clearance, clearance)),
        )
        outside = (anatomy == 0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)
    threshold = max(210, int(np.percentile(clahe, 98.8)))
    bright = ((clahe >= threshold) & (outside > 0)).astype(np.uint8)
    join = max(3, round(min(height, width) * 0.01))
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (join, join)),
    )
    bright = cv2.dilate(bright, np.ones((3, 3), dtype=np.uint8), iterations=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    boxes = []
    for label in range(1, count):
        candidate_x, candidate_y, candidate_width, candidate_height, area = stats[label]
        bbox_area_fraction = candidate_width * candidate_height / bright.size
        aspect = max(
            candidate_width / max(candidate_height, 1),
            candidate_height / max(candidate_width, 1),
        )
        if 0.00025 <= bbox_area_fraction <= 0.10 and area >= 8 and aspect <= 6.0:
            boxes.append(
                (
                    int(candidate_x),
                    int(candidate_y),
                    int(candidate_width),
                    int(candidate_height),
                )
            )
    return boxes


def analyze_image(
    path: Path, analysis_size: int = 640, mask_path: Path | None = None
) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"id": path.stem, "path": str(path), "error": "OpenCV could not decode image"}
    original_height, original_width = image.shape
    analysis_image, scale = resize_for_analysis(image, analysis_size)
    normalized = robust_normalize(analysis_image)
    hand_mask = load_hand_mask(mask_path, image.shape, analysis_image.shape)
    mask_source = "external" if hand_mask is not None else "heuristic"
    if hand_mask is None:
        hand_mask, bbox = estimate_hand_mask(normalized)
    else:
        bbox = bbox_from_mask(hand_mask)
    x, y, box_width, box_height = bbox
    height, width = normalized.shape

    hand_pixels = analysis_image[hand_mask > 0]
    background_pixels = analysis_image[hand_mask == 0]
    if not len(background_pixels):
        background_pixels = analysis_image.reshape(-1)
    contrast = abs(float(np.median(hand_pixels)) - float(np.median(background_pixels)))
    bbox_center_x = x + box_width / 2
    bbox_center_y = y + box_height / 2
    center_offset = math.hypot(bbox_center_x / width - 0.5, bbox_center_y / height - 0.5)

    crop = normalized[y : y + box_height, x : x + box_width]
    crop_512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    enhanced_crop = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(crop_512)
    laplacian = cv2.Laplacian(enhanced_crop, cv2.CV_32F)
    blur_score = float(laplacian.var())
    gradient_x = cv2.Sobel(enhanced_crop, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(enhanced_crop, cv2.CV_32F, 0, 1, ksize=3)
    edge_score = float(np.mean(cv2.magnitude(gradient_x, gradient_y)))
    markers = marker_candidates(normalized, bbox, hand_mask)

    border = max(4, round(min(height, width) * 0.05))
    raw_border_pixels = np.concatenate(
        [
            analysis_image[:border, :].reshape(-1),
            analysis_image[-border:, :].reshape(-1),
            analysis_image[:, :border].reshape(-1),
            analysis_image[:, -border:].reshape(-1),
        ]
    )
    background_median = float(np.median(raw_border_pixels))
    # The wrist normally enters at the bottom; contact there is not a crop failure.
    edge_band = max(1, round(min(height, width) * 0.004))
    upper_limit = round(height * 0.78)
    touches_edge = bool(
        hand_mask[:edge_band, :].any()
        or hand_mask[:upper_limit, :edge_band].any()
        or hand_mask[:upper_limit, -edge_band:].any()
    )

    return {
        "id": int(path.stem) if path.stem.isdigit() else path.stem,
        "path": str(path.resolve()),
        "error": None,
        "width": original_width,
        "height": original_height,
        "analysis_scale": scale,
        "mask_source": mask_source,
        "background_median": background_median,
        "background_tone": "light" if background_median >= 40 else "dark",
        "foreground_background_contrast": contrast,
        "hand_bbox_x_fraction": x / width,
        "hand_bbox_y_fraction": y / height,
        "hand_bbox_width_fraction": box_width / width,
        "hand_bbox_height_fraction": box_height / height,
        "hand_bbox_area_fraction": box_width * box_height / normalized.size,
        "hand_center_offset": center_offset,
        "orientation_degrees": orientation_degrees(hand_mask),
        "touches_image_edge": touches_edge,
        "marker_candidate_count": len(markers),
        "marker_candidates": markers,
        "blur_score": blur_score,
        "edge_score": edge_score,
    }


def add_dataset_flags(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics[metrics["error"].isna()].copy()
    if valid.empty:
        return metrics
    small_threshold = float(valid["hand_bbox_area_fraction"].quantile(0.08))
    contrast_threshold = float(valid["foreground_background_contrast"].quantile(0.08))
    blur_threshold = float(valid["blur_score"].quantile(0.01))
    edge_threshold = float(valid["edge_score"].quantile(0.01))

    output = metrics.copy()
    output["flag_low_contrast"] = output["foreground_background_contrast"] <= contrast_threshold
    output["flag_light_background"] = output["background_tone"] == "light"
    output["flag_off_center"] = output["hand_center_offset"] >= 0.14
    output["flag_rotated"] = output["orientation_degrees"].abs() >= 15.0
    output["flag_small_hand"] = output["hand_bbox_area_fraction"] <= small_threshold
    output["flag_marker"] = output["marker_candidate_count"] > 0
    output["flag_crop_risk"] = output["touches_image_edge"]
    output["flag_blur_review"] = (output["blur_score"] <= blur_threshold) | (
        output["edge_score"] <= edge_threshold
    )
    output["candidate_blur_exclusion"] = (output["blur_score"] <= blur_threshold) & (
        output["edge_score"] <= edge_threshold
    )
    flag_columns = [column for column in output.columns if column.startswith("flag_")]
    output["quality_flags"] = output[flag_columns].apply(
        lambda row: ";".join(
            column.removeprefix("flag_") for column, value in row.items() if value
        ),
        axis=1,
    )
    output.attrs["thresholds"] = {
        "small_hand_area_fraction_p08": small_threshold,
        "low_contrast_p08": contrast_threshold,
        "blur_score_p01": blur_threshold,
        "edge_score_p01": edge_threshold,
    }
    return output


def vietnamese_flag_labels(flags: str) -> str:
    if not flags:
        return "không có cờ"
    return "; ".join(FLAG_LABELS_VI.get(flag, flag) for flag in flags.split(";") if flag)


def _diagnostic_view(
    path: Path, row: pd.Series, analysis_size: int, mask_path: Path | None
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    resized, _ = resize_for_analysis(image, analysis_size)
    normalized = robust_normalize(resized)
    hand_mask = load_hand_mask(mask_path, image.shape, resized.shape)
    if hand_mask is None:
        hand_mask, bbox = estimate_hand_mask(normalized)
    else:
        bbox = bbox_from_mask(hand_mask)
    markers = marker_candidates(normalized, bbox, hand_mask)
    x, y, width, height = bbox
    overlay = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (40, 220, 70), 3)
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (40, 220, 70), 2)
    center = (round(x + width / 2), round(y + height / 2))
    angle = math.radians(float(row["orientation_degrees"]))
    axis_length = round(max(width, height) * 0.35)
    delta_x = round(math.sin(angle) * axis_length)
    delta_y = round(math.cos(angle) * axis_length)
    cv2.line(
        overlay,
        (center[0] - delta_x, center[1] - delta_y),
        (center[0] + delta_x, center[1] + delta_y),
        (255, 170, 20),
        3,
    )
    for marker_x, marker_y, marker_width, marker_height in markers:
        cv2.rectangle(
            overlay,
            (marker_x, marker_y),
            (marker_x + marker_width, marker_y + marker_height),
            (240, 40, 40),
            2,
        )
    crop = normalized[y : y + height, x : x + width]
    crop = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(crop)
    return overlay, crop


def save_review_montage(
    metrics: pd.DataFrame,
    review_ids: list[int],
    output_path: Path,
    analysis_size: int,
    mask_dir: Path | None,
) -> None:
    selected = metrics[metrics["id"].isin(review_ids)].copy()
    order = {image_id: index for index, image_id in enumerate(review_ids)}
    selected["review_order"] = selected["id"].map(order)
    selected = selected.sort_values("review_order")
    if selected.empty:
        return

    fig, axes = plt.subplots(len(selected), 3, figsize=(14, 4.3 * len(selected)), squeeze=False)
    for row_index, (_, row) in enumerate(selected.iterrows()):
        path = Path(row["path"])
        original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask_path = mask_dir / path.name if mask_dir is not None else None
        overlay, crop = _diagnostic_view(path, row, analysis_size, mask_path)
        axes[row_index, 0].imshow(original, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 0].set_title(f"ID {row['id']} — ảnh gốc")
        axes[row_index, 1].imshow(overlay)
        axes[row_index, 1].set_title(
            f"xanh=mask/ROI ({row['mask_source']}), cam=trục tay, đỏ=marker"
        )
        axes[row_index, 2].imshow(crop, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 2].set_title(
            f"crop chuẩn hóa | {vietnamese_flag_labels(row['quality_flags'])}\n"
            f"góc={row['orientation_degrees']:.1f}°, "
            f"diện tích={row['hand_bbox_area_fraction']:.2f}, "
            f"blur={row['blur_score']:.1f}"
        )
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle(
        "EDA X-quang bàn tay: ảnh gốc, chẩn đoán và crop đề xuất", fontsize=16
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


def save_blur_review_montage(
    metrics: pd.DataFrame,
    output_path: Path,
    analysis_size: int,
    mask_dir: Path | None,
    limit: int = 25,
) -> None:
    selected = metrics[metrics["candidate_blur_exclusion"]].copy()
    if selected.empty:
        return
    selected["severity"] = selected["blur_score"].rank(pct=True) + selected["edge_score"].rank(
        pct=True
    )
    selected = selected.nsmallest(limit, "severity")
    columns = 5
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.8 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (_, row) in zip(axes.flat, selected.iterrows(), strict=False):
        path = Path(row["path"])
        mask_path = mask_dir / path.name if mask_dir is not None else None
        _, crop = _diagnostic_view(path, row, analysis_size, mask_path)
        axis.imshow(crop, cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"ID {row['id']} | blur={row['blur_score']:.1f}\nedge={row['edge_score']:.1f}"
        )
        axis.axis("off")
    fig.suptitle(
        "Ảnh ít chi tiết nhất — cần duyệt thủ công, không tự động xóa", fontsize=16
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    mask_dir = args.mask_dir.resolve() if args.mask_dir is not None else None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.only_review_ids:
        paths = [image_dir / f"{image_id}.png" for image_id in args.review_ids]
        paths = [path for path in paths if path.exists()]
    else:
        paths = sorted(image_dir.glob("*.png"), key=lambda path: int(path.stem))
    work = [
        (path, args.analysis_size, mask_dir / path.name if mask_dir is not None else None)
        for path in paths
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(lambda item: analyze_image(*item), work))

    metrics = add_dataset_flags(pd.DataFrame(rows))
    metrics.to_csv(output_dir / "quality_metrics.csv", index=False)
    review = metrics[metrics["id"].isin(args.review_ids)].copy()
    review.to_csv(output_dir / "review_cases.csv", index=False)
    blur_candidates = metrics[
        metrics.get("candidate_blur_exclusion", False) == True  # noqa: E712
    ]
    blur_candidates.to_csv(output_dir / "blur_review_candidates.csv", index=False)
    save_review_montage(
        metrics,
        list(args.review_ids),
        output_dir / "review_montage.png",
        args.analysis_size,
        mask_dir,
    )
    save_blur_review_montage(
        metrics,
        output_dir / "blur_review_candidates.png",
        args.analysis_size,
        mask_dir,
    )

    flag_columns = [column for column in metrics.columns if column.startswith("flag_")]
    summary = {
        "images_analyzed": len(metrics),
        "decode_errors": int(metrics["error"].notna().sum()),
        "background_tone_counts": metrics["background_tone"].value_counts().to_dict(),
        "flag_counts": {
            column.removeprefix("flag_"): int(metrics[column].sum()) for column in flag_columns
        },
        "joint_low_blur_edge_candidates": int(len(blur_candidates)),
        "thresholds": metrics.attrs.get("thresholds", {}),
        "review_ids": list(args.review_ids),
    }
    (output_dir / "quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
