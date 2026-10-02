"""Build deterministic, group-aware folds with conservative duplicate confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: int, second: int) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


def hamming_distance(first: str, second: str) -> int:
    try:
        return (int(first, 16) ^ int(second, 16)).bit_count()
    except (TypeError, ValueError):
        return 64


class BKNode:
    """Node in an exact metric tree for 64-bit perceptual hashes."""

    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, BKNode] = {}

    def query(self, value: int, radius: int) -> list[int]:
        distance = (self.value ^ value).bit_count()
        matches = list(self.indices) if distance <= radius else []
        for edge, child in self.children.items():
            if distance - radius <= edge <= distance + radius:
                matches.extend(child.query(value, radius))
        return matches

    def insert(self, value: int, index: int) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.indices.append(index)
            return
        child = self.children.get(distance)
        if child is None:
            self.children[distance] = BKNode(value, index)
        else:
            child.insert(value, index)


def _candidate_pairs(hashes: list[str], max_distance: int) -> list[tuple[int, int]]:
    """Return every prior pair within the exact Hamming-distance threshold."""
    if not hashes:
        return []
    values = [int(hash_value, 16) for hash_value in hashes]
    root = BKNode(values[0], 0)
    pairs: list[tuple[int, int]] = []
    for index, value in enumerate(values[1:], start=1):
        pairs.extend((previous, index) for previous in root.query(value, max_distance))
        root.insert(value, index)
    return pairs


@lru_cache(maxsize=None)
def _canonical_preview(path_value: str) -> np.ndarray:
    image = cv2.imread(path_value, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unable to decode duplicate candidate: {path_value}")
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        raise ValueError(f"Duplicate candidate has no intensity range: {path_value}")
    image = np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)
    height, width = image.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.float32)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = image
    return cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)


def _global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    mean_first, mean_second = float(first.mean()), float(second.mean())
    variance_first, variance_second = float(first.var()), float(second.var())
    covariance = float(((first - mean_first) * (second - mean_second)).mean())
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    denominator = (mean_first**2 + mean_second**2 + c1) * (variance_first + variance_second + c2)
    return numerator / denominator if denominator else 0.0


def confirm_image_pair(first_path: str, second_path: str) -> bool:
    first, second = _canonical_preview(first_path), _canonical_preview(second_path)
    return _global_ssim(first, second) >= 0.985 and float(np.abs(first - second).mean()) <= 0.035


def find_duplicate_groups(
    df: pd.DataFrame,
    *,
    max_distance: int = 4,
    confirm_pair: Callable[[int, int], bool] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Group exact bytes and only perceptual candidates confirmed by image comparison."""
    frame = df.reset_index(drop=True)
    union_find = UnionFind(len(frame))
    exact_groups = 0
    if "sha256" in frame:
        for _, indices in frame.groupby("sha256", sort=False).groups.items():
            positions = list(indices)
            if len(positions) > 1:
                exact_groups += 1
                for position in positions[1:]:
                    union_find.union(positions[0], position)

    candidate_pairs = _candidate_pairs(frame["phash"].astype(str).tolist(), max_distance)
    print(
        f"Found {len(candidate_pairs)} pHash candidate pairs at distance <= {max_distance}",
        flush=True,
    )
    confirmed_pairs = 0
    for first, second in candidate_pairs:
        if confirm_pair is not None and confirm_pair(first, second):
            union_find.union(first, second)
            confirmed_pairs += 1

    group_ids = [
        f"group_{int(frame.at[union_find.find(index), 'image_id'])}" for index in range(len(frame))
    ]
    return group_ids, {
        "exact_sha256_groups": exact_groups,
        "phash_candidate_pairs": len(candidate_pairs),
        "confirmed_near_duplicate_pairs": confirmed_pairs,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/data/manifest.parquet")
    parser.add_argument("--output", default="artifacts/data/splits_v2.csv")
    parser.add_argument("--audit-output", default="artifacts/data/splits_v2_audit.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    train = manifest[manifest["split"] == "train"].copy().reset_index(drop=True)
    image_paths = train["image_path"].astype(str).tolist()
    workers = min(8, os.cpu_count() or 1)
    print(f"Preloading {len(image_paths)} canonical previews with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        previews = list(executor.map(_canonical_preview, image_paths))

    def confirmation(first: int, second: int) -> bool:
        first_preview, second_preview = previews[first], previews[second]
        return (
            _global_ssim(first_preview, second_preview) >= 0.985
            and float(np.abs(first_preview - second_preview).mean()) <= 0.035
        )

    train["group_id"], duplicate_audit = find_duplicate_groups(train, confirm_pair=confirmation)
    train["age_bin"] = pd.cut(
        train["bone_age_months"],
        bins=np.arange(0, 241, 24),
        labels=False,
        include_lowest=True,
        right=False,
    ).astype(int)
    train["stratum"] = train["sex"].astype(str) + "_" + train["age_bin"].astype(str)
    counts = train["stratum"].value_counts()
    for stratum in counts[counts < args.folds].index:
        sex = stratum.split("_", maxsplit=1)[0]
        train.loc[train["stratum"] == stratum, "stratum"] = f"{sex}_rare"

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    train["fold"] = -1
    for fold, (_, validation_indices) in enumerate(
        splitter.split(train, train["stratum"], groups=train["group_id"])
    ):
        train.loc[validation_indices, "fold"] = fold

    output = train[["image_id", "fold", "stratum", "group_id"]].copy()
    output["split_version"] = "v2"
    output["seed"] = args.seed
    if not (output["fold"] >= 0).all() or not output["image_id"].is_unique:
        raise RuntimeError("Split assignment is incomplete or contains duplicate IDs")
    if (output.groupby("group_id")["fold"].nunique() > 1).any():
        raise RuntimeError("Duplicate group leakage across folds")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(output_path)
    audit = {
        **duplicate_audit,
        "rows": len(output),
        "groups": int(output["group_id"].nunique()),
        "multi_image_groups": int((output["group_id"].value_counts() > 1).sum()),
        "fold_counts": {
            str(key): int(value)
            for key, value in output["fold"].value_counts().sort_index().items()
        },
        "groups_crossing_folds": int((output.groupby("group_id")["fold"].nunique() > 1).sum()),
        "split_sha256": _sha256(output_path),
        "seed": args.seed,
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
