"""Leakage-safe local adaptation of an RSNA bone-age ensemble."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from boneage.metrics import bootstrap_mae_ci, compute_metrics
from boneage.models import BoneAgeModel
from boneage.preprocessing import BoneAgePreprocess
from boneage.train import select_device, set_seed
from boneage.uncertainty import finite_sample_quantile, prediction_bounds

ADAPTATION_IMAGE_COLUMNS = {"raw_image_path", "standardized_image_path"}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def build_patient_split(
    manifest: pd.DataFrame,
    *,
    seed: int,
    image_path_column: str = "standardized_image_path",
    require_standardization_passed: bool | None = None,
    reference_split: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assign deterministic 60/20/20 train/validation/test partitions by patient."""
    if image_path_column not in ADAPTATION_IMAGE_COLUMNS:
        raise ValueError(
            f"image_path_column must be one of {sorted(ADAPTATION_IMAGE_COLUMNS)}"
        )
    if require_standardization_passed is None:
        require_standardization_passed = image_path_column == "standardized_image_path"
    required = {
        "case_key",
        "patient_key",
        "study_key",
        "sex",
        "sex_valid",
        "report_bone_age_months",
        image_path_column,
    }
    if require_standardization_passed:
        required.add("standardization_status")
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Hospital manifest is missing columns: {missing}")
    path_is_valid = manifest[image_path_column].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    usable_mask = (
        manifest["sex_valid"].astype(bool)
        & manifest["report_bone_age_months"].notna()
        & path_is_valid
    )
    if require_standardization_passed:
        usable_mask &= manifest["standardization_status"].eq("passed")
    usable = manifest[usable_mask].copy()
    if len(usable) < 25:
        raise ValueError("At least 25 usable hospital images are required for adaptation")
    usable["adaptation_image_path"] = usable[image_path_column].astype(str)
    usable["input_variant"] = (
        "raw" if image_path_column == "raw_image_path" else "standardized"
    )
    usable["partition"] = ""
    if reference_split is not None:
        reference_required = {"patient_key", "partition"}
        reference_missing = sorted(reference_required - set(reference_split.columns))
        if reference_missing:
            raise ValueError(f"Reference split is missing columns: {reference_missing}")
        reference = reference_split[list(reference_required)].drop_duplicates()
        if not set(reference["partition"]).issubset({"train", "validation", "test"}):
            raise ValueError("Reference split contains an invalid partition")
        if (
            not reference.empty
            and reference.groupby("patient_key")["partition"].nunique().max() != 1
        ):
            raise ValueError("A patient crosses partitions in the reference split")
        partition_by_patient = reference.drop_duplicates("patient_key").set_index(
            "patient_key"
        )["partition"]
        usable["partition"] = usable["patient_key"].map(partition_by_patient).fillna("")

    unassigned = usable[usable["partition"].eq("")]
    if not unassigned.empty:
        ages = unassigned["report_bone_age_months"].astype(float)
        age_band = pd.cut(
            ages,
            bins=[0, 48, 96, 144, 192, np.inf],
            labels=False,
            include_lowest=True,
        ).astype(int)
        strata = unassigned["sex"].astype(int).astype(str) + "_" + age_band.astype(str)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (_, held_indices) in enumerate(
            splitter.split(unassigned, strata, groups=unassigned["patient_key"])
        ):
            partition = "test" if fold == 0 else "validation" if fold == 1 else "train"
            usable.loc[unassigned.iloc[held_indices].index, "partition"] = partition
    if (usable["partition"] == "").any():
        raise RuntimeError("Hospital split assignment is incomplete")
    patient_partitions = usable.groupby("patient_key")["partition"].nunique()
    if int(patient_partitions.max()) != 1:
        raise RuntimeError("A patient crosses adaptation partitions")
    return usable.reset_index(drop=True)


class HospitalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: BoneAgePreprocess) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(str(row["adaptation_image_path"])) as source:
            image = self.transform(source.convert("L"))
        return (
            image,
            torch.tensor([float(row["sex"])], dtype=torch.float32),
            torch.tensor(float(row["report_bone_age_months"]), dtype=torch.float32),
            index,
        )


@dataclass(frozen=True)
class SourceCheckpoint:
    path: Path
    fold: int
    backbone: str
    resolution: int
    preprocess_mode: str
    dataset_version: str
    target_mean: float
    target_std: float
    state: dict[str, torch.Tensor]


def _load_sources(run_dir: Path, expected_count: int) -> list[SourceCheckpoint]:
    paths = sorted(run_dir.glob("fold_*/best.pt"))
    if len(paths) != expected_count:
        raise RuntimeError(f"Expected {expected_count} source checkpoints, found {len(paths)}")
    sources = []
    contract: tuple[str, int, str] | None = None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        current = (
            str(checkpoint["backbone"]),
            int(checkpoint["resolution"]),
            str(checkpoint["preprocess_mode"]),
        )
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError("Source checkpoints have incompatible contracts")
        sources.append(
            SourceCheckpoint(
                path=path,
                fold=int(checkpoint["fold"]),
                backbone=current[0],
                resolution=current[1],
                preprocess_mode=current[2],
                dataset_version=str(checkpoint.get("dataset_version", "unknown")),
                target_mean=float(checkpoint["target_mean"]),
                target_std=float(checkpoint["target_std"]),
                state=checkpoint["model_state"],
            )
        )
    return sources


def _loader(
    frame: pd.DataFrame,
    *,
    source: SourceCheckpoint,
    training: bool,
    batch_size: int,
    seed: int,
) -> DataLoader:
    dataset = HospitalDataset(
        frame,
        BoneAgePreprocess(source.preprocess_mode, source.resolution, training),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _predict(
    model: BoneAgeModel,
    loader: DataLoader,
    source: SourceCheckpoint,
    device: torch.device,
) -> np.ndarray:
    result = np.full(len(loader.dataset), np.nan, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for images, sex, _target, indices in loader:
            normalized = model(images.to(device), sex.to(device)).cpu().numpy()
            result[indices.numpy()] = normalized * source.target_std + source.target_mean
    if not np.isfinite(result).all():
        raise RuntimeError("Adaptation inference produced incomplete predictions")
    return result


def _study_frame(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    working = frame.copy()
    working["prediction_months"] = predictions
    return (
        working.groupby("study_key", as_index=False)
        .agg(
            patient_key=("patient_key", "first"),
            sex=("sex", "first"),
            target_months=("report_bone_age_months", "first"),
            prediction_months=("prediction_months", "mean"),
            image_count=("case_key", "count"),
        )
        .sort_values("study_key")
    )


def _metrics(studies: pd.DataFrame) -> dict[str, Any]:
    target = studies["target_months"].to_numpy(dtype=float)
    prediction = studies["prediction_months"].to_numpy(dtype=float)
    output = compute_metrics(target, prediction)
    output["mae_ci_95"] = list(bootstrap_mae_ci(target, prediction, seed=20261001))
    output["study_count"] = int(len(studies))
    output["by_sex"] = {}
    for sex, group in studies.groupby("sex"):
        output["by_sex"]["M" if float(sex) == 1 else "F"] = {
            "study_count": int(len(group)),
            "mae": float(
                np.mean(np.abs(group["target_months"] - group["prediction_months"]))
            ),
        }
    return output


def _paired_comparison(
    baseline_path: Path,
    candidate: pd.DataFrame,
    *,
    baseline_name: str,
    seed: int,
) -> dict[str, Any]:
    baseline = pd.read_csv(baseline_path)
    required = {"study_key", "target_months", "prediction_months"}
    missing = sorted(required - set(baseline.columns))
    if missing:
        raise ValueError(f"Baseline predictions are missing columns: {missing}")
    paired = baseline[list(required)].merge(
        candidate[list(required)],
        on="study_key",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("No common held-out studies for paired adaptation comparison")
    if not np.allclose(
        paired["target_months_baseline"], paired["target_months_candidate"], atol=1e-6
    ):
        raise ValueError("Paired adaptation targets disagree")
    target = paired["target_months_candidate"].to_numpy(dtype=float)
    baseline_prediction = paired["prediction_months_baseline"].to_numpy(dtype=float)
    candidate_prediction = paired["prediction_months_candidate"].to_numpy(dtype=float)
    baseline_metrics = compute_metrics(target, baseline_prediction)
    candidate_metrics = compute_metrics(target, candidate_prediction)
    paired_delta = np.abs(candidate_prediction - target) - np.abs(baseline_prediction - target)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired_delta), size=(5000, len(paired_delta)))
    bootstrap_delta = paired_delta[indices].mean(axis=1)
    delta_ci = np.quantile(bootstrap_delta, [0.025, 0.975])
    return {
        "baseline_name": baseline_name,
        "study_count": int(len(paired)),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "mae_delta_candidate_minus_baseline": float(paired_delta.mean()),
        "mae_delta_ci_95": [float(delta_ci[0]), float(delta_ci[1])],
        "candidate_better": bool(
            delta_ci[1] < 0
            and candidate_metrics["p95_ae"] <= baseline_metrics["p95_ae"]
            and candidate_metrics["over_24_rate_upper_95"]
            <= baseline_metrics["over_24_rate_upper_95"]
        ),
    }


def _train_source(
    source: SourceCheckpoint,
    split: pd.DataFrame,
    config: dict[str, Any],
    *,
    mode: str,
    output_dir: Path,
    device: torch.device,
) -> Path:
    seed = int(config["seed"]) + source.fold
    set_seed(seed)
    train_frame = split[split["partition"] == "train"].reset_index(drop=True)
    validation_frame = split[split["partition"] == "validation"].reset_index(drop=True)
    input_variant = str(split["input_variant"].iloc[0])
    train_loader = _loader(
        train_frame,
        source=source,
        training=True,
        batch_size=int(config["batch_size"]),
        seed=seed,
    )
    validation_loader = _loader(
        validation_frame,
        source=source,
        training=False,
        batch_size=int(config["batch_size"]),
        seed=seed,
    )
    model = BoneAgeModel(source.backbone, pretrained=False).to(device)
    model.load_state_dict(source.state, strict=True)
    model.freeze_backbone(True)
    learning_rate = float(
        config["head_learning_rate"] if mode == "head" else config["full_learning_rate"]
    )
    optimizer = optim.AdamW(
        model.parameters(), learning_rate, weight_decay=float(config["weight_decay"])
    )
    criterion = nn.SmoothL1Loss(beta=float(config.get("smooth_l1_beta", 0.25)))
    epochs = int(config["epochs"])
    unfreeze_epoch = int(config.get("gradual_unfreeze_epoch", 2))
    best_mae = float("inf")
    best_path = output_dir / f"fold_{source.fold}" / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        if mode == "gradual" and epoch == unfreeze_epoch:
            model.freeze_backbone(False)
        model.train()
        if mode == "head" or epoch < unfreeze_epoch:
            model.backbone.eval()
        for images, sex, target, _indices in train_loader:
            optimizer.zero_grad(set_to_none=True)
            normalized_target = (target.to(device) - source.target_mean) / source.target_std
            prediction = model(images.to(device), sex.to(device))
            loss = criterion(prediction, normalized_target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite local-adaptation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        validation_prediction = _predict(model, validation_loader, source, device)
        validation_mae = float(
            np.mean(
                np.abs(
                    validation_prediction
                    - validation_frame["report_bone_age_months"].to_numpy(dtype=float)
                )
            )
        )
        print(
            f"[{mode}] fold={source.fold} epoch={epoch}/{epochs} "
            f"validation_mae={validation_mae:.4f}",
            flush=True,
        )
        if validation_mae < best_mae:
            best_mae = validation_mae
            checkpoint = {
                "model_state": model.state_dict(),
                "backbone": source.backbone,
                "dataset_version": (
                    f"{source.dataset_version}+hospital-{input_variant}-{mode}"
                ),
                "hospital_input_variant": input_variant,
                "preprocess_mode": source.preprocess_mode,
                "resolution": source.resolution,
                "target_mean": source.target_mean,
                "target_std": source.target_std,
                "fold": source.fold,
                "epoch": epoch,
                "best_val_mae": best_mae,
                "adaptation_mode": mode,
                "source_checkpoint": str(source.path),
                "seed": seed,
            }
            temporary = best_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, best_path)
    return best_path


def run_adaptation(config: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"head", "gradual"}:
        raise ValueError("mode must be 'head' or 'gradual'")
    source_run = Path(str(config["source_run_dir"]))
    artifact_dir = Path(str(config["artifact_dir"]))
    output_dir = artifact_dir / mode
    manifest = pd.read_parquet(Path(str(config["hospital_manifest_path"])))
    image_path_column = str(config.get("image_path_column", "standardized_image_path"))
    reference_split_path = config.get("reference_split_path")
    reference_split = (
        pd.read_csv(Path(str(reference_split_path))) if reference_split_path else None
    )
    split = build_patient_split(
        manifest,
        seed=int(config["seed"]),
        image_path_column=image_path_column,
        require_standardization_passed=config.get("require_standardization_passed"),
        reference_split=reference_split,
    )
    missing_files = [
        path for path in split["adaptation_image_path"] if not Path(str(path)).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} adaptation image files")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_columns = [
        "case_key",
        "patient_key",
        "study_key",
        "sex",
        "report_bone_age_months",
        "adaptation_image_path",
        "input_variant",
        "partition",
    ]
    split[split_columns].to_csv(artifact_dir / "split.csv", index=False)
    sources = _load_sources(source_run, int(config.get("expected_checkpoint_count", 5)))
    device = select_device(str(config.get("device", "auto")))
    for source in sources:
        _train_source(
            source,
            split,
            config,
            mode=mode,
            output_dir=output_dir,
            device=device,
        )

    validation = split[split["partition"] == "validation"].reset_index(drop=True)
    test = split[split["partition"] == "test"].reset_index(drop=True)
    validation_fold_predictions = []
    test_fold_predictions = []
    base_test_predictions = []
    for source in sources:
        validation_loader = _loader(
            validation,
            source=source,
            training=False,
            batch_size=int(config["batch_size"]),
            seed=int(config["seed"]),
        )
        test_loader = _loader(
            test,
            source=source,
            training=False,
            batch_size=int(config["batch_size"]),
            seed=int(config["seed"]),
        )
        adapted = torch.load(
            output_dir / f"fold_{source.fold}" / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = BoneAgeModel(source.backbone, pretrained=False).to(device)
        model.load_state_dict(adapted["model_state"], strict=True)
        validation_fold_predictions.append(_predict(model, validation_loader, source, device))
        test_fold_predictions.append(_predict(model, test_loader, source, device))
        model.load_state_dict(source.state, strict=True)
        base_test_predictions.append(_predict(model, test_loader, source, device))

    validation_prediction = np.mean(np.vstack(validation_fold_predictions), axis=0)
    test_prediction = np.mean(np.vstack(test_fold_predictions), axis=0)
    base_prediction = np.mean(np.vstack(base_test_predictions), axis=0)
    validation_studies = _study_frame(validation, validation_prediction)
    test_studies = _study_frame(test, test_prediction)
    base_studies = _study_frame(test, base_prediction)
    radius = finite_sample_quantile(
        np.abs(
            validation_studies["target_months"].to_numpy(dtype=float)
            - validation_studies["prediction_months"].to_numpy(dtype=float)
        ),
        float(config.get("prediction_interval_coverage", 0.90)),
    )
    target = test_studies["target_months"].to_numpy(dtype=float)
    lower, upper = prediction_bounds(
        test_studies["prediction_months"].to_numpy(dtype=float), radius
    )
    test_studies["prediction_lower_months"] = lower
    test_studies["prediction_upper_months"] = upper
    test_studies.to_csv(output_dir / "test_predictions.csv", index=False)
    base_metrics = _metrics(base_studies)
    adapted_metrics = _metrics(test_studies)
    adapted_metrics["prediction_interval"] = {
        "coverage_target": float(config.get("prediction_interval_coverage", 0.90)),
        "radius_months": radius,
        "empirical_coverage": float(np.mean((target >= lower) & (target <= upper))),
    }
    decision = {
        "promote": bool(
            adapted_metrics["mae"] < base_metrics["mae"]
            and adapted_metrics["p95_ae"] <= base_metrics["p95_ae"]
            and adapted_metrics["over_24_rate_upper_95"]
            <= base_metrics["over_24_rate_upper_95"]
        ),
        "mode": mode,
        "input_variant": str(split["input_variant"].iloc[0]),
        "image_path_column": image_path_column,
        "base": base_metrics,
        "adapted": adapted_metrics,
        "train_patient_count": int(
            split.loc[split["partition"].eq("train"), "patient_key"].nunique()
        ),
        "validation_patient_count": int(
            split.loc[split["partition"].eq("validation"), "patient_key"].nunique()
        ),
        "test_patient_count": int(test_studies["patient_key"].nunique()),
        "test_study_count": int(len(test_studies)),
    }
    comparison = config.get("paired_comparison")
    if comparison:
        decision["paired_comparison"] = _paired_comparison(
            Path(str(comparison["baseline_predictions_path"])),
            test_studies,
            baseline_name=str(comparison["baseline_name"]),
            seed=int(config["seed"]),
        )
    _atomic_json(output_dir / "evaluation.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hospital_finetune.yaml")
    parser.add_argument("--mode", choices=["head", "gradual"], required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = run_adaptation(config, args.mode)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
