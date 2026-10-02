"""Fail-closed deterministic ensemble inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import BoneAgeDataset
from .models import BoneAgeModel
from .preprocessing import BoneAgePreprocess


def validate_predictions(
    image_ids: np.ndarray,
    predictions: np.ndarray,
    *,
    expected_rows: int,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    if len(image_ids) != expected_rows or len(predictions) != expected_rows:
        raise ValueError("Inference output row count mismatch")
    if not np.isfinite(predictions).all():
        raise ValueError("Inference produced non-finite predictions")
    if ((predictions < lower) | (predictions > upper)).any():
        bad = int(((predictions < lower) | (predictions > upper)).sum())
        raise ValueError(f"Inference produced {bad} predictions outside {lower}..{upper} months")
    output = pd.DataFrame({"Case ID": image_ids.astype(int), "prediction_months": predictions})
    if not output["Case ID"].is_unique:
        raise ValueError("Inference output contains duplicate Case IDs")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strong.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output", default="artifacts/predictions/test_v2.csv")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    run_dir = Path(args.run_dir or config["run_dir"])
    checkpoints = sorted(run_dir.glob("fold_*/best.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No trained fold checkpoints found under {run_dir}")
    manifest = pd.read_parquet(config["manifest_path"])
    frame = manifest[manifest["split"] == args.split].copy()
    if frame.empty:
        raise ValueError(f"No samples found for split={args.split}")

    ensemble: list[np.ndarray] = []
    image_ids: np.ndarray | None = None
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        preprocess = BoneAgePreprocess(
            mode=str(checkpoint["preprocess_mode"]),
            target_size=int(checkpoint["resolution"]),
            is_train=False,
        )
        loader = DataLoader(
            BoneAgeDataset(frame, transform=preprocess),
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
        )
        model = BoneAgeModel(str(checkpoint["backbone"]), pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        fold_predictions: list[float] = []
        fold_ids: list[int] = []
        with torch.inference_mode():
            for images, sex, _, metadata in loader:
                normalized = model(images, sex).numpy()
                predictions = normalized * float(checkpoint["target_std"]) + float(
                    checkpoint["target_mean"]
                )
                fold_predictions.extend(predictions.tolist())
                fold_ids.extend(metadata["image_id"].numpy().tolist())
        current_ids = np.asarray(fold_ids)
        if image_ids is None:
            image_ids = current_ids
        elif not np.array_equal(image_ids, current_ids):
            raise RuntimeError("Checkpoint inference order mismatch")
        ensemble.append(np.asarray(fold_predictions))

    predictions = np.mean(np.vstack(ensemble), axis=0)
    lower, upper = float(config["target_min_months"]), float(config["target_max_months"])
    assert image_ids is not None
    output = validate_predictions(
        image_ids,
        predictions,
        expected_rows=len(frame),
        lower=lower,
        upper=upper,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(f"Wrote {len(output)} predictions from {len(checkpoints)} checkpoints to {destination}")


if __name__ == "__main__":
    main()
