"""PyTorch dataset boundary for bone-age images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class BoneAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        if transform is None:
            raise ValueError("BoneAgeDataset requires an explicit transform")
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.df.iloc[idx]
        image_id = int(row["image_id"])
        image_path = Path(str(row["image_path"]))
        try:
            with Image.open(image_path) as source:
                image = source.convert("L")
                tensor = self.transform(image)
        except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Unable to decode image_id={image_id} at {image_path}") from exc

        if tensor.ndim != 3 or tensor.shape[0] != 3 or not torch.isfinite(tensor).all():
            raise ValueError(
                f"Invalid transformed tensor for image_id={image_id}: {tuple(tensor.shape)}"
            )
        sex = torch.tensor([float(row["sex"])], dtype=torch.float32)
        age = row.get("bone_age_months")
        target = float(age) if pd.notna(age) else float("nan")
        metadata = {"image_id": image_id, "image_path": str(image_path)}
        return tensor, sex, torch.tensor(target, dtype=torch.float32), metadata
