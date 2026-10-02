"""Preprocessing and OOF evaluation entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import yaml
from tqdm import tqdm

from .dataset import BoneAgeDataset
from .metrics import bootstrap_mae_ci, compute_metrics
from .preprocessing import BoneAgePreprocess


def run_preprocessing_audit(config: dict[str, object]) -> None:
    manifest = pd.read_parquet(str(config["manifest_out"]))
    sample = manifest.groupby(["split", "sex"], group_keys=False).sample(
        n=25, random_state=20260924, replace=False
    )
    dataset = BoneAgeDataset(
        sample,
        transform=BoneAgePreprocess(
            mode=str(config.get("preprocess_mode", "P1")),
            target_size=384,
            is_train=False,
        ),
    )
    tensors = [dataset[index][0] for index in tqdm(range(len(dataset)), desc="Preprocessing audit")]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    grid = torchvision.utils.make_grid([tensor * std + mean for tensor in tensors], nrow=10)
    output = Path(
        str(
            config.get(
                "preprocessing_audit_output",
                "artifacts/reports/preprocessing/montage_P1_v2.png",
            )
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(20, 20))
    plt.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=140)
    plt.close()
    print(f"Saved {output}")


def evaluate_oof(run_dir: Path, manifest_path: Path) -> None:
    predictions = pd.read_csv(run_dir / "oof_predictions.csv")
    manifest = pd.read_parquet(manifest_path)[["image_id", "sex", "bone_age_months"]]
    data = predictions.merge(manifest, on="image_id", validate="one_to_one")
    if not (data["target_months"] == data["bone_age_months"]).all():
        raise ValueError("OOF target does not match manifest")
    report: dict[str, object] = {
        "overall": compute_metrics(data["target_months"], data["prediction_months"]),
        "mae_ci_95": list(bootstrap_mae_ci(data["target_months"], data["prediction_months"])),
        "by_sex": {},
        "by_age_band": {},
    }
    for sex, group in data.groupby("sex"):
        report["by_sex"][str(int(sex))] = compute_metrics(
            group["target_months"], group["prediction_months"]
        )
    data["age_band"] = (data["target_months"] // 24).astype(int)
    for age_band, group in data.groupby("age_band"):
        report["by_age_band"][str(int(age_band))] = compute_metrics(
            group["target_months"], group["prediction_months"]
        )
    output = run_dir / "evaluation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--mode", choices=["preprocessing-audit", "oof"], required=True)
    parser.add_argument("--run-dir", default="artifacts/runs/codex_cpu")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest_path = Path(str(config.get("manifest_path", config.get("manifest_out", ""))))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if args.mode == "preprocessing-audit":
        audit_config = {**config, "manifest_out": str(manifest_path)}
        run_preprocessing_audit(audit_config)
    else:
        evaluate_oof(Path(args.run_dir), manifest_path)


if __name__ == "__main__":
    main()
