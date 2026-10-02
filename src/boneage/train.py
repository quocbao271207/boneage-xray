"""Leakage-safe baselines and resumable fold training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader

from .data_contracts import validate_manifest
from .dataset import BoneAgeDataset
from .metrics import bootstrap_mae_ci, compute_metrics
from .models import BoneAgeModel
from .preprocessing import BoneAgePreprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def evaluate_naive_baselines(
    manifest: pd.DataFrame, splits: pd.DataFrame, run_dir: Path
) -> dict[str, Any]:
    data = manifest[manifest["split"] == "train"].merge(
        splits, on="image_id", validate="one_to_one"
    )
    predictions = {name: np.full(len(data), np.nan) for name in ("global", "sex", "metadata")}
    features = ["width", "height", "file_bytes", "sex"]
    for fold in sorted(data["fold"].unique()):
        training = data[data["fold"] != fold]
        validation = data[data["fold"] == fold]
        predictions["global"][validation.index] = training["bone_age_months"].median()
        medians = training.groupby("sex")["bone_age_months"].median()
        predictions["sex"][validation.index] = validation["sex"].map(medians)
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(training[features], training["bone_age_months"])
        predictions["metadata"][validation.index] = model.predict(validation[features])

    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("Baseline OOF predictions are incomplete")
    target = data["bone_age_months"].to_numpy()
    metrics = {name: compute_metrics(target, values) for name, values in predictions.items()}
    for name, values in predictions.items():
        metrics[name]["mae_ci_95"] = list(bootstrap_mae_ci(target, values))
        data[f"prediction_{name}"] = values
    run_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(run_dir / "oof_predictions.csv", index=False)
    atomic_json(run_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def make_loader(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    training: bool,
    seed: int,
) -> DataLoader:
    transform = BoneAgePreprocess(
        mode=str(config["preprocess_mode"]),
        target_size=int(config["resolution"]),
        is_train=training,
    )
    dataset = BoneAgeDataset(frame, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=training,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=int(config["num_workers"]) > 0,
    )


def predict_loader(
    model: BoneAgeModel,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    image_ids: list[int] = []
    with torch.inference_mode():
        for images, sex, target, metadata in loader:
            output = model(images.to(device), sex.to(device))
            prediction = output.cpu().numpy() * target_std + target_mean
            predictions.extend(prediction.tolist())
            targets.extend(target.numpy().tolist())
            image_ids.extend(metadata["image_id"].numpy().tolist())
    return np.asarray(image_ids), np.asarray(targets), np.asarray(predictions)


def train_fold(
    manifest: pd.DataFrame,
    splits: pd.DataFrame,
    config: dict[str, Any],
    *,
    fold: int,
    pilot: bool,
) -> dict[str, Any]:
    seed = int(config["seed"]) + fold
    set_seed(seed)
    device = select_device(str(config["device"]))
    data = manifest[manifest["split"] == "train"].merge(
        splits, on="image_id", validate="one_to_one"
    )
    training = data[data["fold"] != fold].copy()
    validation = data[data["fold"] == fold].copy()
    epochs = int(config["epochs"])
    run_dir = Path(str(config["run_dir"]))
    if pilot:
        training = training.sample(min(64, len(training)), random_state=seed)
        validation = validation.sample(min(32, len(validation)), random_state=seed)
        epochs = min(2, epochs)
        run_dir = Path(str(config.get("pilot_run_dir", "artifacts/runs/pilot_v2")))

    target_mean = float(training["bone_age_months"].mean())
    target_std = float(training["bone_age_months"].std())
    if target_std <= 0:
        raise ValueError("Training target standard deviation must be positive")
    training_loader = make_loader(training, config=config, training=True, seed=seed)
    validation_loader = make_loader(validation, config=config, training=False, seed=seed)

    model = BoneAgeModel(str(config["backbone"]), pretrained=bool(config["pretrained"])).to(device)
    model.freeze_backbone(int(config["freeze_backbone_epochs"]) > 0)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.SmoothL1Loss(beta=0.25)
    fold_dir = run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    best_mae = float("inf")
    stale_epochs = 0
    checkpoint_path = fold_dir / "best.pt"
    last_checkpoint_path = fold_dir / "last.pt"
    manifest_sha256 = sha256_file(Path(str(config["manifest_path"])))
    split_sha256 = sha256_file(Path(str(config["splits_path"])))
    start_epoch = 1

    resume_path = last_checkpoint_path if last_checkpoint_path.is_file() else checkpoint_path
    if not pilot and resume_path.is_file():
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        resumable = (
            int(resume_checkpoint.get("fold", -1)) == fold
            and resume_checkpoint.get("manifest_sha256") == manifest_sha256
            and resume_checkpoint.get("split_sha256") == split_sha256
        )
        if resumable:
            model.load_state_dict(resume_checkpoint["model_state"])
            if "optimizer_state" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            if "scheduler_state" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            start_epoch = int(resume_checkpoint["epoch"]) + 1
            best_mae = float(resume_checkpoint["best_val_mae"])
            stale_epochs = int(resume_checkpoint.get("stale_epochs", 0))
            print(f"Resuming fold {fold} at epoch {start_epoch}", flush=True)

    status_path = run_dir / "status.json"
    freeze_epochs = int(config["freeze_backbone_epochs"])
    if start_epoch > freeze_epochs:
        model.freeze_backbone(False)
    atomic_json(
        status_path,
        {
            "state": "running",
            "fold": fold,
            "epoch": start_epoch - 1,
            "device": str(device),
            "pilot": pilot,
            "process_id": os.getpid(),
        },
    )
    for epoch in range(start_epoch, epochs + 1):
        if epoch == freeze_epochs + 1:
            model.freeze_backbone(False)
        model.train()
        if epoch <= freeze_epochs:
            model.backbone.eval()
        losses: list[float] = []
        started = time.time()
        for images, sex, target, _ in training_loader:
            optimizer.zero_grad(set_to_none=True)
            normalized_target = (target.to(device) - target_mean) / target_std
            output = model(images.to(device), sex.to(device))
            loss = criterion(output, normalized_target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at fold={fold}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        image_ids, targets, predictions = predict_loader(
            model, validation_loader, device, target_mean, target_std
        )
        metrics = compute_metrics(targets, predictions)
        record = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "seconds": round(time.time() - started, 2),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        atomic_json(
            status_path,
            {
                "state": "running",
                **record,
                "device": str(device),
                "pilot": pilot,
                "process_id": os.getpid(),
            },
        )

        improved = metrics["mae"] < best_mae
        if improved:
            best_mae = metrics["mae"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "backbone": str(config["backbone"]),
            "dataset_version": str(config.get("dataset_version", "raw")),
            "manifest_sha256": manifest_sha256,
            "preprocess_mode": str(config["preprocess_mode"]),
            "resolution": int(config["resolution"]),
            "target_mean": target_mean,
            "target_std": target_std,
            "fold": fold,
            "epoch": epoch,
            "best_val_mae": best_mae,
            "stale_epochs": stale_epochs,
            "split_sha256": split_sha256,
            "seed": seed,
        }
        last_temporary = last_checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, last_temporary)
        last_temporary.replace(last_checkpoint_path)
        if improved:
            best_temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, best_temporary)
            best_temporary.replace(checkpoint_path)
        if stale_epochs >= int(config["early_stopping_patience"]):
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    image_ids, targets, predictions = predict_loader(
        model, validation_loader, device, target_mean, target_std
    )
    fold_predictions = pd.DataFrame(
        {
            "image_id": image_ids,
            "target_months": targets,
            "prediction_months": predictions,
            "fold": fold,
        }
    )
    fold_predictions.to_csv(fold_dir / "oof.csv", index=False)
    final_metrics = compute_metrics(targets, predictions)
    final_metrics["mae_ci_95"] = list(bootstrap_mae_ci(targets, predictions, seed=seed))
    atomic_json(fold_dir / "metrics.json", final_metrics)
    atomic_json(
        status_path,
        {
            "state": "fold_complete",
            "fold": fold,
            "metrics": final_metrics,
            "device": str(device),
            "pilot": pilot,
        },
    )
    return final_metrics


def combine_oof(run_dir: Path, expected_rows: int) -> dict[str, Any] | None:
    paths = sorted(run_dir.glob("fold_*/oof.csv"))
    if not paths:
        return None
    combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if combined["image_id"].duplicated().any():
        raise RuntimeError("OOF predictions contain duplicate image IDs")
    combined.sort_values("image_id").to_csv(run_dir / "oof_predictions.csv", index=False)
    metrics = compute_metrics(combined["target_months"], combined["prediction_months"])
    metrics["mae_ci_95"] = list(
        bootstrap_mae_ci(combined["target_months"], combined["prediction_months"])
    )
    metrics["rows"] = len(combined)
    metrics["complete"] = len(combined) == expected_rows
    atomic_json(run_dir / "oof_metrics.json", metrics)
    return metrics


def fold_is_complete(run_dir: Path, fold: int) -> bool:
    """Validate the minimal artifact set before skipping a completed fold."""
    fold_dir = run_dir / f"fold_{fold}"
    checkpoint_path = fold_dir / "best.pt"
    predictions_path = fold_dir / "oof.csv"
    metrics_path = fold_dir / "metrics.json"
    if not all(path.is_file() for path in (checkpoint_path, predictions_path, metrics_path)):
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        predictions = pd.read_csv(predictions_path)
        json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
        return False
    required_checkpoint_keys = {"model_state", "target_mean", "target_std", "fold"}
    return (
        required_checkpoint_keys.issubset(checkpoint)
        and int(checkpoint["fold"]) == fold
        and len(predictions) > 0
        and "image_id" in predictions
        and predictions["image_id"].is_unique
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strong.yaml")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--stage", choices=["full", "pilot"], default="full")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    manifest = pd.read_parquet(config["manifest_path"])
    validate_manifest(manifest)
    splits = pd.read_csv(config["splits_path"])
    train_ids = set(manifest.loc[manifest["split"] == "train", "image_id"])
    if set(splits["image_id"]) != train_ids or splits["image_id"].duplicated().any():
        raise ValueError("Split file does not match the train manifest")

    if config["model_type"] == "metadata":
        evaluate_naive_baselines(manifest, splits, Path(str(config["run_dir"])))
        return
    if config["model_type"] != "image":
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    folds = sorted(splits["fold"].unique()) if args.all_folds else [args.fold]
    if folds == [None]:
        raise ValueError("Specify --fold N or --all-folds")
    pilot = args.stage == "pilot"
    run_dir = (
        Path(str(config.get("pilot_run_dir", "artifacts/runs/pilot_v2")))
        if pilot
        else Path(str(config["run_dir"]))
    )
    try:
        for fold in folds:
            fold = int(fold)
            if not pilot and fold_is_complete(run_dir, fold):
                print(f"Skipping verified complete fold {fold}", flush=True)
                continue
            train_fold(manifest, splits, config, fold=fold, pilot=pilot)
            if pilot:
                break
        metrics = combine_oof(run_dir, expected_rows=int((manifest["split"] == "train").sum()))
        if metrics is not None:
            atomic_json(
                run_dir / "status.json",
                {"state": "complete" if metrics["complete"] else "partial", "oof_metrics": metrics},
            )
    except Exception as error:
        atomic_json(
            run_dir / "status.json",
            {"state": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
