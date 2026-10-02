from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from boneage.dataset import BoneAgeDataset


def frame(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        {"image_id": [7], "image_path": [str(path)], "sex": [1], "bone_age_months": [120.0]}
    )


def transform(_: Image.Image) -> torch.Tensor:
    return torch.zeros((3, 16, 16))


def test_dataset_reads_image_and_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("L", (20, 30), color=100).save(image_path)
    tensor, sex, target, metadata = BoneAgeDataset(frame(image_path), transform=transform)[0]
    assert tensor.shape == (3, 16, 16)
    assert sex.item() == 1.0
    assert target.item() == 120.0
    assert metadata["image_id"] == 7


def test_dataset_fails_on_missing_image(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="image_id=7"):
        BoneAgeDataset(frame(tmp_path / "missing.png"), transform=transform)[0]


def test_dataset_requires_transform(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit transform"):
        BoneAgeDataset(frame(tmp_path / "sample.png"), transform=None)
