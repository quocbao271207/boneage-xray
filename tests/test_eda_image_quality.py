import cv2
import numpy as np

from analysis.eda_image_quality import (
    add_dataset_flags,
    bbox_from_mask,
    estimate_hand_mask,
    marker_candidates,
    orientation_degrees,
    robust_normalize,
    suppress_film_rims,
)


def test_robust_normalize_uses_full_uint8_range() -> None:
    image = np.tile(np.arange(40, 180, dtype=np.uint8), (64, 1))
    normalized = robust_normalize(image)
    assert normalized.dtype == np.uint8
    assert normalized.min() == 0
    assert normalized.max() == 255


def test_hand_mask_ignores_full_frame_and_finds_center_object() -> None:
    image = np.full((360, 280), 25, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (271, 351), 180, 4)
    cv2.ellipse(image, (140, 195), (55, 125), 8, 0, 360, 145, -1)
    for x_coordinate in (105, 125, 145, 165, 185):
        cv2.line(image, (140, 170), (x_coordinate, 55), 220, 9)
    mask, (x, y, width, height) = estimate_hand_mask(robust_normalize(image))
    assert mask.sum() > 0
    assert width < image.shape[1] * 0.9
    assert height < image.shape[0] * 0.9
    assert x < image.shape[1] / 2 < x + width
    assert y < image.shape[0] / 2 < y + height


def test_orientation_reports_rotation_away_from_vertical() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(mask, (100, 100), (20, 75), 28, 0, 360, 1, -1)
    assert 20 <= abs(orientation_degrees(mask)) <= 35


def test_bbox_from_mask_adds_margin_without_leaving_image() -> None:
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[20:90, 15:65] = 1
    x, y, width, height = bbox_from_mask(mask, margin_fraction=0.1)
    assert (x, y) == (8, 13)
    assert (width, height) == (64, 84)


def test_long_peripheral_line_is_suppressed_without_erasing_center() -> None:
    image = np.zeros((200, 160), dtype=np.uint8)
    cv2.line(image, (5, 15), (150, 15), 255, 4)
    cv2.ellipse(image, (80, 105), (25, 65), 0, 0, 360, 180, -1)
    evidence = np.ones_like(image, dtype=np.float32)
    suppressed = suppress_film_rims(evidence, image)
    assert suppressed[15, 80] == 0
    assert suppressed[105, 80] == 1


def test_marker_candidates_are_outside_hand_bbox() -> None:
    image = np.full((240, 200), 30, dtype=np.uint8)
    cv2.ellipse(image, (105, 130), (40, 85), 0, 0, 360, 150, -1)
    cv2.putText(image, "L", (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 4)
    boxes = marker_candidates(image, (55, 35, 100, 195))
    assert boxes
    assert all(x + width <= 55 or x >= 155 for x, _, width, _ in boxes)


def test_dataset_flags_require_both_blur_signals_for_exclusion() -> None:
    import pandas as pd

    rows = []
    for index in range(100):
        rows.append(
            {
                "id": index,
                "error": None,
                "foreground_background_contrast": 30.0,
                "background_tone": "dark",
                "hand_center_offset": 0.02,
                "orientation_degrees": 0.0,
                "hand_bbox_area_fraction": 0.4,
                "marker_candidate_count": 0,
                "touches_image_edge": False,
                "blur_score": 100.0,
                "edge_score": 20.0,
            }
        )
    rows[0]["blur_score"] = 1.0
    rows[0]["edge_score"] = 1.0
    flagged = add_dataset_flags(pd.DataFrame(rows))
    assert bool(flagged.loc[0, "candidate_blur_exclusion"])
    assert int(flagged["candidate_blur_exclusion"].sum()) == 1
