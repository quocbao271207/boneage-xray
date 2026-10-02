import numpy as np
import pytest
import torch
from PIL import Image

from boneage.preprocessing import BoneAgePreprocess, percentile_normalization


def textured_image() -> Image.Image:
    row = np.linspace(10, 240, 320, dtype=np.uint8)
    return Image.fromarray(np.tile(row, (240, 1)))


@pytest.mark.parametrize("mode", ["P0", "P1", "P2"])
def test_preprocessing_shapes_and_finite_values(mode: str) -> None:
    tensor = BoneAgePreprocess(mode=mode, target_size=128, is_train=False)(textured_image())
    assert tensor.shape == (3, 128, 128)
    assert torch.isfinite(tensor).all()
    assert float(tensor.min()) >= -3.0
    assert float(tensor.max()) <= 3.0


def test_evaluation_preprocessing_is_deterministic() -> None:
    preprocess = BoneAgePreprocess(mode="P1", target_size=128, is_train=False)
    assert torch.equal(preprocess(textured_image()), preprocess(textured_image()))


def test_p2_crops_obvious_empty_border() -> None:
    pixels = np.zeros((300, 300), dtype=np.uint8)
    pixels[80:280, 110:220] = np.tile(np.linspace(30, 240, 110, dtype=np.uint8), (200, 1))
    image = Image.fromarray(pixels)
    p1 = BoneAgePreprocess(mode="P1", target_size=128, is_train=False)(image)
    p2 = BoneAgePreprocess(mode="P2", target_size=128, is_train=False)(image)
    assert not torch.equal(p1, p2)


def test_constant_image_fails_fast() -> None:
    with pytest.raises(ValueError, match="no usable intensity range"):
        percentile_normalization(np.zeros((32, 32), dtype=np.uint8))
