from pathlib import Path

import yaml


def load_config(path: str) -> dict[str, object]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_selected_rsna_model_uses_frozen_data_contract() -> None:
    model = load_config("configs/standardized_v3_efficientnet_b0_512.yaml")

    assert model["dataset_version"] == "standardized-v3-preserve"
    assert model["backbone"] == "efficientnet_b0.ra_in1k"
    assert model["resolution"] == 512
    assert model["splits_path"] == "artifacts/data/splits_standardized_v3_preserve.csv"
    assert model["seed"] == 20260924


def test_hospital_directions_share_source_model_and_seed() -> None:
    original = load_config("configs/hospital_finetune_raw.yaml")
    standardized = load_config("configs/hospital_finetune_standardized.yaml")

    assert original["source_run_dir"] == standardized["source_run_dir"]
    assert original["seed"] == standardized["seed"] == 20261001
    assert original["epochs"] == standardized["epochs"]
    assert original["gradual_unfreeze_epoch"] == standardized["gradual_unfreeze_epoch"]
    assert original["image_path_column"] == "raw_image_path"
    assert standardized["image_path_column"] == "standardized_image_path"
    assert original["require_standardization_passed"] is False
    assert standardized["require_standardization_passed"] is True


def test_external_validation_builds_both_hospital_directions() -> None:
    config = load_config("configs/hospital_external_validation_both.yaml")

    assert config["hospital_input_variant"] == "both"
    assert config["standardization_enabled"] is True
    assert config["expected_checkpoint_count"] == 5
