import numpy as np
import torch
from torch import nn

from boneage.xai import (
    explanation_metrics,
    occlusion_sensitivity,
    resolve_target_layer,
    signed_grad_cam,
)


class ToyRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        nn.init.ones_(self.conv.weight)

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        return self.conv(image).mean(dim=(1, 2, 3)) + sex[:, 0] * 0.1


class ToyPooledBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(3, 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.after_pool = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.after_pool(self.pool(self.spatial(image))).flatten(1)


class ToyPooledModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ToyPooledBackbone()

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).mean(dim=1) + sex[:, 0]


def test_signed_grad_cam_and_occlusion_have_input_resolution() -> None:
    model = ToyRegression().eval()
    image = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8).repeat(1, 3, 1, 1)
    sex = torch.zeros((1, 1))

    output, cam = signed_grad_cam(model, model.conv, image, sex)
    occlusion = occlusion_sensitivity(model, image, sex, patch_size=4, stride=2)

    assert np.isfinite(output)
    assert cam.shape == (8, 8)
    assert occlusion.shape == (8, 8)
    assert np.isfinite(cam).all()
    assert np.isfinite(occlusion).all()
    assert np.max(np.abs(cam)) <= 1
    assert np.max(np.abs(occlusion)) <= 1


def test_explanation_metrics_report_fold_stability_and_border_attention() -> None:
    first = np.ones((10, 10), dtype=np.float32)
    first[2:8, 2:8] = 2
    second = first * 2

    metrics = explanation_metrics([first, second], first)

    assert metrics["fold_gradcam_correlation"] == 1.0
    assert metrics["gradcam_occlusion_correlation"] == 1.0
    assert 0 < metrics["border_attention_fraction"] < 1


def test_target_layer_selection_ignores_convolution_after_global_pooling() -> None:
    model = ToyPooledModel().eval()

    name, layer = resolve_target_layer(
        model,  # type: ignore[arg-type]
        resolution=16,
        device=torch.device("cpu"),
    )

    assert name == "spatial"
    assert layer is model.backbone.spatial
