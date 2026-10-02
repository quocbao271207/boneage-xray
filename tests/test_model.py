import torch

from boneage.models import BoneAgeModel


def test_model_forward_and_backbone_freeze() -> None:
    model = BoneAgeModel("mobilenetv3_small_050", pretrained=False)
    output = model(torch.zeros((2, 3, 64, 64)), torch.tensor([[0.0], [1.0]]))
    assert output.shape == (2,)
    assert torch.isfinite(output).all()

    model.freeze_backbone(True)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
