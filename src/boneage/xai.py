"""Grad-CAM and occlusion explanations for bone-age regression."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import nn

from .dicom_io import DicomCase, iter_zip_dicom_cases, load_dicom_file
from .models import BoneAgeModel
from .preprocessing import BoneAgePreprocess
from .train import select_device


@dataclass
class ExplainableModel:
    model: BoneAgeModel
    target_layer: nn.Module
    target_layer_name: str
    target_mean: float
    target_std: float
    fold: int


@dataclass(frozen=True)
class EnsembleContract:
    backbone: str
    preprocess_mode: str
    resolution: int
    dataset_version: str


def _module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise ValueError(f"Unknown target layer: {name}")
    return modules[name]


def resolve_target_layer(
    model: BoneAgeModel,
    *,
    resolution: int,
    device: torch.device,
    requested: str | None = None,
) -> tuple[str, nn.Module]:
    candidates = [
        (name, module)
        for name, module in model.backbone.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if not candidates:
        raise ValueError("Backbone has no Conv2d layer for Grad-CAM")
    if requested:
        layer = _module_by_name(model.backbone, requested)
        if not isinstance(layer, nn.Conv2d):
            raise ValueError("Grad-CAM target layer must be a Conv2d module")
        candidates = [(requested, layer)]

    spatial_shapes: dict[str, tuple[int, int]] = {}
    handles = []
    for name, module in candidates:
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, layer_name=name: spatial_shapes.__setitem__(
                    layer_name, tuple(output.shape[-2:])
                )
            )
        )
    try:
        with torch.inference_mode():
            model(
                torch.zeros((1, 3, resolution, resolution), device=device),
                torch.zeros((1, 1), device=device),
            )
    finally:
        for handle in handles:
            handle.remove()

    spatial_candidates = [
        (name, module)
        for name, module in candidates
        if name in spatial_shapes and min(spatial_shapes[name]) > 1
    ]
    if not spatial_candidates:
        target = requested or "automatic selection"
        raise ValueError(f"Grad-CAM target {target} has no spatial feature map")
    return spatial_candidates[-1]


def load_ensemble(
    run_dir: Path,
    *,
    device: torch.device,
    target_layer: str | None = None,
) -> tuple[list[ExplainableModel], EnsembleContract]:
    checkpoint_paths = sorted(run_dir.glob("fold_*/best.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold checkpoints found under {run_dir}")

    loaded: list[ExplainableModel] = []
    contract: EnsembleContract | None = None
    seen_folds: set[int] = set()
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {
            "model_state",
            "backbone",
            "preprocess_mode",
            "resolution",
            "target_mean",
            "target_std",
            "fold",
        }
        missing = required - checkpoint.keys()
        if missing:
            raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
        current = EnsembleContract(
            backbone=str(checkpoint["backbone"]),
            preprocess_mode=str(checkpoint["preprocess_mode"]),
            resolution=int(checkpoint["resolution"]),
            dataset_version=str(checkpoint.get("dataset_version", "unknown")),
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError("Ensemble checkpoints have incompatible preprocessing/model contracts")
        fold = int(checkpoint["fold"])
        if fold in seen_folds:
            raise ValueError(f"Ensemble contains duplicate fold={fold}")
        seen_folds.add(fold)

        model = BoneAgeModel(current.backbone, pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        layer_name, layer = resolve_target_layer(
            model,
            resolution=current.resolution,
            device=device,
            requested=target_layer,
        )
        loaded.append(
            ExplainableModel(
                model=model,
                target_layer=layer,
                target_layer_name=layer_name,
                target_mean=float(checkpoint["target_mean"]),
                target_std=float(checkpoint["target_std"]),
                fold=fold,
            )
        )
    assert contract is not None
    return loaded, contract


def signed_grad_cam(
    model: nn.Module,
    target_layer: nn.Module,
    image: torch.Tensor,
    sex: torch.Tensor,
) -> tuple[float, np.ndarray]:
    """Return the normalized regression output and a signed Grad-CAM map."""
    activation: torch.Tensor | None = None

    def capture(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal activation
        activation = output

    handle = target_layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        output = model(image, sex)
        if output.numel() != 1:
            raise ValueError("Grad-CAM requires a single input case")
        if activation is None or activation.ndim != 4:
            raise RuntimeError("Target layer did not produce a spatial activation map")
        gradient = torch.autograd.grad(output.sum(), activation, retain_graph=False)[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = functional.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        # Centering removes a spatially uniform regression offset and makes the map describe
        # which regions contribute more or less than the image-wide average.
        cam = cam - cam.mean()
        scale = cam.abs().amax().clamp_min(torch.finfo(cam.dtype).eps)
        normalized = (cam / scale).detach().cpu().numpy().astype(np.float32)
        return float(output.detach().cpu().item()), normalized
    finally:
        handle.remove()


def occlusion_sensitivity(
    model: nn.Module,
    image: torch.Tensor,
    sex: torch.Tensor,
    *,
    patch_size: int,
    stride: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Signed prediction change when local regions are replaced by the image mean."""
    if image.shape[0] != 1 or image.ndim != 4:
        raise ValueError("Occlusion sensitivity requires a single BCHW image")
    height, width = image.shape[-2:]
    if patch_size < 1 or stride < 1 or patch_size > min(height, width):
        raise ValueError("Invalid occlusion patch_size/stride")

    y_starts = sorted(set(range(0, height - patch_size + 1, stride)) | {height - patch_size})
    x_starts = sorted(set(range(0, width - patch_size + 1, stride)) | {width - patch_size})
    windows = [(y, x) for y in y_starts for x in x_starts]
    aggregate = torch.zeros((height, width), device=image.device)
    coverage = torch.zeros((height, width), device=image.device)
    fill = image.mean(dim=(2, 3), keepdim=True)

    with torch.inference_mode():
        baseline = model(image, sex)[0]
        for offset in range(0, len(windows), batch_size):
            batch_windows = windows[offset : offset + batch_size]
            variants = image.repeat(len(batch_windows), 1, 1, 1)
            for index, (y, x) in enumerate(batch_windows):
                variants[index, :, y : y + patch_size, x : x + patch_size] = fill[0]
            batch_sex = sex.repeat(len(batch_windows), 1)
            changes = baseline - model(variants, batch_sex)
            for change, (y, x) in zip(changes, batch_windows, strict=True):
                aggregate[y : y + patch_size, x : x + patch_size] += change
                coverage[y : y + patch_size, x : x + patch_size] += 1
    sensitivity = aggregate / coverage.clamp_min(1)
    scale = sensitivity.abs().amax().clamp_min(torch.finfo(sensitivity.dtype).eps)
    return (sensitivity / scale).cpu().numpy().astype(np.float32)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = left.astype(np.float64).ravel()
    y = right.astype(np.float64).ravel()
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def explanation_metrics(cams: list[np.ndarray], occlusion: np.ndarray) -> dict[str, float | None]:
    if not cams:
        raise ValueError("At least one Grad-CAM map is required")
    mean_cam = np.mean(np.stack(cams), axis=0)
    border = max(1, round(min(mean_cam.shape) * 0.12))
    border_mask = np.zeros(mean_cam.shape, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    magnitude = np.abs(mean_cam)
    corner_fraction = float(magnitude[border_mask].sum() / max(magnitude.sum(), 1e-12))
    pairwise = [
        correlation
        for index, left in enumerate(cams)
        for right in cams[index + 1 :]
        if (correlation := _safe_correlation(left, right)) is not None
    ]
    return {
        "gradcam_occlusion_correlation": _safe_correlation(mean_cam, occlusion),
        "fold_gradcam_correlation": float(np.mean(pairwise)) if pairwise else None,
        "border_attention_fraction": corner_fraction,
        "positive_gradcam_fraction": float((mean_cam > 0).mean()),
    }


def explain_case(
    case: DicomCase,
    models: list[ExplainableModel],
    contract: EnsembleContract,
    *,
    device: torch.device,
    occlusion_fraction: float,
    occlusion_stride_fraction: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    preprocess = BoneAgePreprocess(
        mode=contract.preprocess_mode,
        target_size=contract.resolution,
        is_train=False,
    )
    image = preprocess(case.image).unsqueeze(0).to(device)
    sex = torch.tensor([[case.sex]], dtype=torch.float32, device=device)
    patch_size = max(8, round(contract.resolution * occlusion_fraction))
    stride = max(4, round(contract.resolution * occlusion_stride_fraction))

    predictions: list[float] = []
    cams: list[np.ndarray] = []
    occlusion_maps: list[np.ndarray] = []
    for loaded in models:
        normalized, cam = signed_grad_cam(loaded.model, loaded.target_layer, image, sex)
        predictions.append(normalized * loaded.target_std + loaded.target_mean)
        cams.append(cam)
        occlusion_maps.append(
            occlusion_sensitivity(
                loaded.model,
                image,
                sex,
                patch_size=patch_size,
                stride=stride,
            )
        )

    mean_cam = np.mean(np.stack(cams), axis=0)
    mean_occlusion = np.mean(np.stack(occlusion_maps), axis=0)
    base_image = image[0, 0].detach().cpu().numpy() * 0.229 + 0.485
    base_image = np.clip(base_image, 0.0, 1.0)
    payload: dict[str, Any] = {
        "case_key": case.case_key,
        "sex": case.sex_label,
        "prediction_months": float(np.mean(predictions)),
        "ensemble_std_months": float(np.std(predictions, ddof=0)),
        "fold_predictions_months": {
            str(model.fold): float(prediction)
            for model, prediction in zip(models, predictions, strict=True)
        },
        "checkpoint_count": len(models),
        "model_contract": {
            "backbone": contract.backbone,
            "dataset_version": contract.dataset_version,
            "preprocess_mode": contract.preprocess_mode,
            "resolution": contract.resolution,
        },
        "target_layer": models[0].target_layer_name,
        "xai_metrics": explanation_metrics(cams, mean_occlusion),
        "interpretation": {
            "gradcam_positive": "above-average local support for an older prediction",
            "gradcam_negative": "below-average local support for an older prediction",
            "occlusion_positive": "occluding the region lowered the predicted age",
            "occlusion_negative": "occluding the region raised the predicted age",
            "scope": "model-behavior visualization, not a causal or clinical explanation",
        },
    }
    return payload, base_image, mean_cam, mean_occlusion


def render_explanation(
    destination: Path,
    payload: dict[str, Any],
    base_image: np.ndarray,
    gradcam: np.ndarray,
    occlusion: np.ndarray,
) -> None:
    figure = Figure(figsize=(12, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    axes[0].imshow(base_image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Model input")
    for axis, heatmap, title in (
        (axes[1], gradcam, "Centered Grad-CAM"),
        (axes[2], occlusion, "Occlusion sensitivity"),
    ):
        axis.imshow(base_image, cmap="gray", vmin=0, vmax=1)
        axis.imshow(heatmap, cmap="coolwarm", vmin=-1, vmax=1, alpha=np.abs(heatmap) * 0.75)
        axis.set_title(title)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"{payload['case_key']} | {payload['prediction_months']:.1f} months | "
        f"ensemble SD {payload['ensemble_std_months']:.1f}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    figure.savefig(temporary, dpi=160, format="png")
    figure.clear()
    os.replace(temporary, destination)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _source_cases(args: argparse.Namespace) -> Iterable[DicomCase]:
    if args.dicom:
        return [load_dicom_file(Path(args.dicom), sex_override=args.sex_override)]
    return iter_zip_dicom_cases(
        Path(args.zip),
        max_cases=args.max_cases,
        sex_override=args.sex_override,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate privacy-safe Grad-CAM and occlusion explanations for DICOM images."
    )
    parser.add_argument("--run-dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dicom")
    source.add_argument("--zip")
    parser.add_argument("--output-dir", default="artifacts/xai")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--sex-override", choices=["M", "F"])
    parser.add_argument("--target-layer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--occlusion-fraction", type=float, default=0.20)
    parser.add_argument("--occlusion-stride-fraction", type=float, default=0.12)
    args = parser.parse_args()
    if not 0.05 <= args.occlusion_fraction <= 0.5:
        parser.error("--occlusion-fraction must be between 0.05 and 0.5")
    if not 0.02 <= args.occlusion_stride_fraction <= args.occlusion_fraction:
        parser.error("--occlusion-stride-fraction must be positive and no larger than the patch")

    device = select_device(args.device)
    models, contract = load_ensemble(
        Path(args.run_dir),
        device=device,
        target_layer=args.target_layer,
    )
    destination = Path(args.output_dir)
    results: list[dict[str, Any]] = []
    for case in _source_cases(args):
        payload, base_image, gradcam, occlusion = explain_case(
            case,
            models,
            contract,
            device=device,
            occlusion_fraction=args.occlusion_fraction,
            occlusion_stride_fraction=args.occlusion_stride_fraction,
        )
        render_explanation(
            destination / f"{case.case_key}.png",
            payload,
            base_image,
            gradcam,
            occlusion,
        )
        _atomic_json(destination / f"{case.case_key}.json", payload)
        results.append(payload)
        print(
            f"Explained {case.case_key}: {payload['prediction_months']:.1f} months "
            f"(ensemble SD {payload['ensemble_std_months']:.1f})",
            flush=True,
        )
    if not results:
        raise RuntimeError("No DICOM cases were explained")

    summary = pd.DataFrame(
        {
            "case_key": [item["case_key"] for item in results],
            "sex": [item["sex"] for item in results],
            "prediction_months": [item["prediction_months"] for item in results],
            "ensemble_std_months": [item["ensemble_std_months"] for item in results],
            "gradcam_occlusion_correlation": [
                item["xai_metrics"]["gradcam_occlusion_correlation"] for item in results
            ],
            "fold_gradcam_correlation": [
                item["xai_metrics"]["fold_gradcam_correlation"] for item in results
            ],
            "border_attention_fraction": [
                item["xai_metrics"]["border_attention_fraction"] for item in results
            ],
        }
    )
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "summary.csv.tmp"
    summary.to_csv(temporary, index=False)
    os.replace(temporary, destination / "summary.csv")
    _atomic_json(
        destination / "run.json",
        {
            "case_count": len(results),
            "run_dir": str(Path(args.run_dir)),
            "device": str(device),
            "privacy": "source names and DICOM identifiers are intentionally omitted",
            "clinical_use": False,
        },
    )


if __name__ == "__main__":
    main()
