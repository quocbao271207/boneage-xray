"""Build and evaluate a de-identified hospital bone-age dataset from a ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import yaml
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

from analysis.audit_hand_orientation import enclosed_hole_fraction, finger_likeness
from analysis.eda_image_quality import estimate_hand_mask, robust_normalize
from analysis.hospital_reports import (
    BoneAgeLabel,
    numeric_identifier_tokens,
    parse_acquisition_date,
    parse_bone_age_label,
    parse_patient_age_months,
)
from boneage.dicom_io import DicomCase, dicom_dataset_to_uint8, normalize_sex
from boneage.hand_standardization import standardize_hand
from boneage.metrics import bootstrap_mae_ci, compute_metrics
from boneage.preprocessing import BoneAgePreprocess
from boneage.train import select_device
from boneage.uncertainty import calibrate_oof_interval, prediction_bounds
from boneage.xai import (
    EnsembleContract,
    ExplainableModel,
    explain_case,
    load_ensemble,
    render_explanation,
)

SAFE_MANIFEST_COLUMNS = {
    "case_key",
    "patient_key",
    "study_key",
    "sex",
    "sex_valid",
    "chronological_age_months",
    "raw_image_path",
    "standardized_image_path",
    "standardization_status",
    "mask_quality_flags",
    "mask_area_fraction",
    "mask_bbox_width_fraction",
    "mask_bbox_height_fraction",
    "rotation_degrees",
    "report_bone_age_months",
    "report_lower_months",
    "report_upper_months",
    "report_precision",
    "report_match_method",
}


def _safe_key(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(image).save(temporary, format="PNG")
    os.replace(temporary, path)


def _clean_identifier(value: object) -> str:
    return re.sub(r"\W+", "", str(value)).upper()


def _pdf_text(payload: bytes, max_bytes: int) -> str:
    if not 100 <= len(payload) <= max_bytes:
        raise ValueError("PDF size is outside the configured safety range")
    marker = payload.find(b"%PDF-")
    if marker < 0:
        raise ValueError("PDF header is missing")
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(io.BytesIO(payload[marker:]), strict=False)
    if len(reader.pages) > 10:
        raise ValueError("PDF has more than 10 pages")
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def assess_mask_quality(mask: np.ndarray) -> dict[str, Any]:
    height, width = mask.shape
    binary = mask > 0
    points = cv2.findNonZero(binary.astype(np.uint8))
    if points is None:
        return {"flags": ["empty_mask"]}
    x, y, box_width, box_height = cv2.boundingRect(points)
    area_fraction = float(binary.mean())
    width_fraction = box_width / width
    height_fraction = box_height / height
    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height
    center_offset = float(np.hypot(center_x - 0.5, center_y - 0.56))
    edge = max(1, round(min(mask.shape) * 0.008))
    unsafe_edge_contact = bool(
        binary[:edge, :].any()
        or binary[: round(height * 0.9), :edge].any()
        or binary[: round(height * 0.9), -edge:].any()
    )
    flags: list[str] = []
    if not 0.06 <= area_fraction <= 0.60:
        flags.append("implausible_area")
    if not 0.20 <= width_fraction <= 0.92:
        flags.append("implausible_width")
    if height_fraction < 0.45:
        flags.append("implausible_height")
    if center_offset > 0.38:
        flags.append("off_center")
    if unsafe_edge_contact:
        flags.append("unsafe_edge_contact")
    if enclosed_hole_fraction(binary.astype(np.uint8)) > 0.01:
        flags.append("mask_holes")
    return {
        "flags": flags,
        "area_fraction": area_fraction,
        "bbox_width_fraction": width_fraction,
        "bbox_height_fraction": height_fraction,
        "center_offset": center_offset,
    }


def standardize_hospital_record(task: dict[str, Any]) -> dict[str, Any]:
    case_key = str(task["case_key"])
    source_path = Path(str(task["source_path"]))
    output_path = Path(str(task["output_path"]))
    mask_path = Path(str(task["mask_path"]))
    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"case_key": case_key, "standardization_status": "decode_failed"}
    normalized = robust_normalize(image)
    mask, _ = estimate_hand_mask(normalized)
    quality = assess_mask_quality(mask)
    _atomic_png(mask_path, (mask > 0).astype(np.uint8) * 255)
    base = {
        "case_key": case_key,
        "mask_quality_flags": ";".join(quality["flags"]),
        "mask_area_fraction": quality.get("area_fraction"),
        "mask_bbox_width_fraction": quality.get("bbox_width_fraction"),
        "mask_bbox_height_fraction": quality.get("bbox_height_fraction"),
        "rotation_degrees": None,
        "standardized_image_path": None,
    }
    if quality["flags"]:
        return {**base, "standardization_status": "needs_mask_review"}
    try:
        result = standardize_hand(
            image,
            mask,
            target_size=int(task["target_size"]),
            background_mode="adaptive_uniform",
            intensity_mode="preserve",
        )
        top_score = finger_likeness(result.mask, 0.04, 0.35)
        bottom_score = finger_likeness(result.mask, 0.65, 0.96)
        if bottom_score - top_score >= 2.5:
            result = standardize_hand(
                image,
                mask,
                target_size=int(task["target_size"]),
                background_mode="adaptive_uniform",
                intensity_mode="preserve",
                flip_orientation=True,
            )
        if abs(result.rotation_degrees) > 60:
            return {
                **base,
                "standardization_status": "needs_orientation_review",
                "rotation_degrees": result.rotation_degrees,
            }
        _atomic_png(output_path, result.image)
        return {
            **base,
            "standardization_status": "passed",
            "standardized_image_path": str(output_path.resolve()),
            "rotation_degrees": result.rotation_degrees,
        }
    except (RuntimeError, ValueError, cv2.error):
        return {**base, "standardization_status": "failed"}


def _collect_dicom_records(
    archive: zipfile.ZipFile,
    raw_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    entries = sorted(
        (entry for entry in archive.infolist() if entry.filename.lower().endswith(".dcm")),
        key=lambda entry: entry.filename,
    )
    records: list[dict[str, Any]] = []
    internal: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for entry in tqdm(entries, desc="Decoding hospital DICOMs"):
        payload = archive.read(entry)
        dataset = pydicom.dcmread(io.BytesIO(payload), force=False)
        pixels = dicom_dataset_to_uint8(dataset)
        sop_instance_uid = str(dataset.get("SOPInstanceUID", ""))
        source_identity = f"{sop_instance_uid}|{entry.filename}"
        case_key = _safe_key("case", source_identity)
        if case_key in seen_keys:
            raise ValueError("Duplicate de-identified DICOM case key")
        seen_keys.add(case_key)
        patient_id = _clean_identifier(dataset.get("PatientID", ""))
        study_date = re.sub(r"\D+", "", str(dataset.get("StudyDate", "")))
        study_identity = f"{patient_id}|{study_date}" if patient_id else source_identity
        patient_identity = patient_id if patient_id else source_identity
        patient_key = _safe_key("patient", patient_identity)
        study_key = _safe_key("study", study_identity)
        try:
            sex, sex_label = normalize_sex(dataset.get("PatientSex", ""))
            sex_valid = True
        except ValueError:
            sex, sex_label, sex_valid = float("nan"), "unknown", False
        raw_path = raw_dir / f"{case_key}.png"
        _atomic_png(raw_path, pixels)
        records.append(
            {
                "case_key": case_key,
                "patient_key": patient_key,
                "study_key": study_key,
                "sex": sex,
                "sex_label": sex_label,
                "sex_valid": sex_valid,
                "chronological_age_months": parse_patient_age_months(
                    dataset.get("PatientAge", "")
                ),
                "raw_image_path": str(raw_path.resolve()),
                "entry_stem_hash": _safe_key("stem", Path(entry.filename).stem.casefold()),
                "report_bone_age_months": None,
                "report_lower_months": None,
                "report_upper_months": None,
                "report_precision": None,
                "report_match_method": None,
            }
        )
        internal.append(
            {
                "case_key": case_key,
                "patient_id": patient_id,
                "study_date": study_date,
                "entry_stem": Path(entry.filename).stem.casefold(),
            }
        )
    return records, internal


def _attach_report_labels(
    archive: zipfile.ZipFile,
    records: list[dict[str, Any]],
    internal: list[dict[str, str]],
    *,
    max_pdf_bytes: int,
) -> dict[str, int]:
    known_ids = {item["patient_id"] for item in internal if item["patient_id"]}
    by_patient: dict[str, list[dict[str, str]]] = {}
    by_stem: dict[str, list[dict[str, str]]] = {}
    for item in internal:
        by_patient.setdefault(item["patient_id"], []).append(item)
        by_stem.setdefault(item["entry_stem"], []).append(item)
    labels_by_case: dict[str, list[tuple[BoneAgeLabel, str]]] = {}
    audit = {
        "pdf_total": 0,
        "pdf_size_rejected": 0,
        "pdf_parse_failed": 0,
        "pdf_label_missing_or_ambiguous": 0,
        "pdf_pair_missing_or_ambiguous": 0,
        "pdf_paired": 0,
    }
    pdf_entries = [entry for entry in archive.infolist() if entry.filename.lower().endswith(".pdf")]
    for entry in tqdm(pdf_entries, desc="Parsing hospital reports"):
        audit["pdf_total"] += 1
        if not 100 <= entry.file_size <= max_pdf_bytes:
            audit["pdf_size_rejected"] += 1
            continue
        try:
            text = _pdf_text(archive.read(entry), max_pdf_bytes)
        except Exception:  # noqa: BLE001 - malformed vendor PDFs must not stop the audit
            audit["pdf_parse_failed"] += 1
            continue
        label = parse_bone_age_label(text)
        if label is None:
            audit["pdf_label_missing_or_ambiguous"] += 1
            continue
        patient_candidates = numeric_identifier_tokens(text) & known_ids
        candidates: list[dict[str, str]] = []
        method = ""
        if len(patient_candidates) == 1:
            patient_id = next(iter(patient_candidates))
            patient_records = by_patient.get(patient_id, [])
            report_date = parse_acquisition_date(text)
            dated = [item for item in patient_records if item["study_date"] == report_date]
            if report_date and dated:
                candidates, method = dated, "patient_id_and_study_date"
            elif len({item["study_date"] for item in patient_records}) == 1:
                candidates, method = patient_records, "patient_id_single_study"
        if not candidates:
            stem_candidates = by_stem.get(Path(entry.filename).stem.casefold(), [])
            if stem_candidates:
                candidates, method = stem_candidates, "exact_filename_stem"
        if not candidates:
            audit["pdf_pair_missing_or_ambiguous"] += 1
            continue
        audit["pdf_paired"] += 1
        for candidate in candidates:
            labels_by_case.setdefault(candidate["case_key"], []).append((label, method))

    indexed = {record["case_key"]: record for record in records}
    for case_key, labels in labels_by_case.items():
        unique = {
            (label.midpoint_months, label.lower_months, label.upper_months): (label, method)
            for label, method in labels
        }
        if len(unique) != 1:
            continue
        label, method = next(iter(unique.values()))
        indexed[case_key].update(
            {
                "report_bone_age_months": label.midpoint_months,
                "report_lower_months": label.lower_months,
                "report_upper_months": label.upper_months,
                "report_precision": label.precision,
                "report_match_method": method,
            }
        )
    return audit


def build_hospital_manifest(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    archive_path = Path(str(config["archive_path"]))
    output_root = Path(str(config["output_root"]))
    artifact_dir = Path(str(config["artifact_dir"]))
    raw_dir = output_root / "raw"
    standardized_dir = output_root / "standardized_v3_compatible"
    mask_dir = output_root / "masks_heuristic"
    with zipfile.ZipFile(archive_path) as archive:
        records, internal = _collect_dicom_records(archive, raw_dir)
        pdf_audit = _attach_report_labels(
            archive,
            records,
            internal,
            max_pdf_bytes=int(config.get("max_pdf_bytes", 20 * 1024 * 1024)),
        )

    standardization_enabled = bool(config.get("standardization_enabled", True))
    if standardization_enabled:
        tasks = [
            {
                "case_key": record["case_key"],
                "source_path": record["raw_image_path"],
                "output_path": standardized_dir / f"{record['case_key']}.png",
                "mask_path": mask_dir / f"{record['case_key']}.png",
                "target_size": int(config.get("standardization_target_size", 1536)),
            }
            for record in records
        ]
        workers = int(config.get("workers", 4))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            standardization = list(
                tqdm(
                    executor.map(standardize_hospital_record, tasks, chunksize=4),
                    total=len(tasks),
                    desc="Standardizing hospital radiographs",
                )
            )
    else:
        standardization = [
            {
                "case_key": record["case_key"],
                "mask_quality_flags": "",
                "mask_area_fraction": None,
                "mask_bbox_width_fraction": None,
                "mask_bbox_height_fraction": None,
                "rotation_degrees": None,
                "standardized_image_path": None,
                "standardization_status": "not_run_raw_only",
            }
            for record in records
        ]
    standardization_by_key = {item["case_key"]: item for item in standardization}
    for record in records:
        record.update(standardization_by_key[record["case_key"]])
        record.pop("entry_stem_hash", None)
        record.pop("sex_label", None)
    manifest = pd.DataFrame(records)
    if not set(manifest.columns).issubset(SAFE_MANIFEST_COLUMNS):
        unsafe = sorted(set(manifest.columns) - SAFE_MANIFEST_COLUMNS)
        raise RuntimeError(f"Unsafe columns would be written to hospital manifest: {unsafe}")
    if manifest["case_key"].duplicated().any():
        raise RuntimeError("Hospital manifest contains duplicate case keys")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "manifest.parquet"
    temporary = manifest_path.with_suffix(".parquet.tmp")
    manifest.to_parquet(temporary, index=False)
    os.replace(temporary, manifest_path)
    audit: dict[str, Any] = {
        "dicom_total": len(manifest),
        "sex_valid": int(manifest["sex_valid"].sum()),
        "report_label_cases": int(manifest["report_bone_age_months"].notna().sum()),
        "report_label_studies": int(
            manifest.loc[manifest["report_bone_age_months"].notna(), "study_key"].nunique()
        ),
        "standardization_status": manifest["standardization_status"].value_counts().to_dict(),
        "pdf": pdf_audit,
        "privacy": "manifest contains hashes and clinical aggregates only; no direct identifiers",
    }
    _atomic_json(artifact_dir / "build_audit.json", audit)
    return manifest, audit


def _predict_variant(
    frame: pd.DataFrame,
    path_column: str,
    models: list[ExplainableModel],
    contract: EnsembleContract,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    transform = BoneAgePreprocess(contract.preprocess_mode, contract.resolution, False)
    fold_outputs = [np.full(len(frame), np.nan, dtype=np.float64) for _ in models]
    valid_indices = [
        index
        for index, row in frame.iterrows()
        if bool(row["sex_valid"]) and isinstance(row[path_column], str) and row[path_column]
    ]
    for offset in tqdm(range(0, len(valid_indices), batch_size), desc=f"Predicting {path_column}"):
        indices = valid_indices[offset : offset + batch_size]
        tensors = []
        sexes = []
        for index in indices:
            with Image.open(str(frame.at[index, path_column])) as source:
                tensors.append(transform(source.convert("L")))
            sexes.append([float(frame.at[index, "sex"])])
        images = torch.stack(tensors).to(device)
        sex = torch.tensor(sexes, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for model_index, loaded in enumerate(models):
                normalized = loaded.model(images, sex).cpu().numpy()
                fold_outputs[model_index][indices] = (
                    normalized * loaded.target_std + loaded.target_mean
                )
    mean = np.full(len(frame), np.nan, dtype=np.float64)
    standard_deviation = np.full(len(frame), np.nan, dtype=np.float64)
    if valid_indices:
        stacked = np.vstack(fold_outputs)[:, valid_indices]
        mean[valid_indices] = np.mean(stacked, axis=0)
        standard_deviation[valid_indices] = np.std(stacked, axis=0)
    return mean, standard_deviation


def _study_predictions(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    selected = frame.dropna(subset=[prediction_column, "report_bone_age_months"]).copy()
    return (
        selected.groupby("study_key", as_index=False)
        .agg(
            patient_key=("patient_key", "first"),
            sex=("sex", "first"),
            target_months=("report_bone_age_months", "first"),
            lower_months=("report_lower_months", "first"),
            upper_months=("report_upper_months", "first"),
            prediction_months=(prediction_column, "mean"),
            image_count=("case_key", "count"),
        )
        .sort_values("study_key")
    )


def _external_metrics(
    studies: pd.DataFrame,
    *,
    interval_radius_months: float | None = None,
) -> dict[str, Any]:
    if studies.empty:
        return {"study_count": 0}
    target = studies["target_months"].to_numpy(dtype=float)
    prediction = studies["prediction_months"].to_numpy(dtype=float)
    metrics = compute_metrics(target, prediction)
    metrics["mae_ci_95"] = list(bootstrap_mae_ci(target, prediction, seed=20260927))
    below = np.maximum(studies["lower_months"].to_numpy(dtype=float) - prediction, 0)
    above = np.maximum(prediction - studies["upper_months"].to_numpy(dtype=float), 0)
    metrics.update(
        {
            "study_count": int(len(studies)),
            "image_count": int(studies["image_count"].sum()),
            "within_report_range": float(np.mean((below == 0) & (above == 0))),
            "range_distance_mae": float(np.mean(below + above)),
            "by_sex": {},
        }
    )
    if interval_radius_months is not None:
        interval_lower, interval_upper = prediction_bounds(
            prediction,
            interval_radius_months,
        )
        metrics["prediction_interval"] = {
            "radius_months": float(interval_radius_months),
            "empirical_coverage": float(
                np.mean((target >= interval_lower) & (target <= interval_upper))
            ),
            "mean_width_months": float(np.mean(interval_upper - interval_lower)),
        }
    for sex, group in studies.groupby("sex"):
        errors = np.abs(group["prediction_months"] - group["target_months"])
        metrics["by_sex"]["M" if float(sex) == 1 else "F"] = {
            "study_count": int(len(group)),
            "mae": float(errors.mean()),
        }
    return metrics


def _run_xai_sample(
    frame: pd.DataFrame,
    models: list[ExplainableModel],
    contract: EnsembleContract,
    *,
    device: torch.device,
    destination: Path,
    count: int,
    image_path_column: str,
    prediction_column: str,
) -> list[dict[str, Any]]:
    usable = frame.dropna(
        subset=[
            image_path_column,
            prediction_column,
            "report_bone_age_months",
        ]
    ).copy()
    if usable.empty or count < 1:
        return []
    usable["absolute_error"] = (
        usable[prediction_column] - usable["report_bone_age_months"]
    ).abs()
    worst = usable.nlargest(min(count // 2, len(usable)), "absolute_error")
    remainder = usable.drop(index=worst.index).sort_values("report_bone_age_months")
    needed = min(count - len(worst), len(remainder))
    if needed:
        positions = np.linspace(0, len(remainder) - 1, needed).round().astype(int)
        selected = pd.concat([worst, remainder.iloc[positions]]).drop_duplicates("case_key")
    else:
        selected = worst
    results = []
    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="Generating XAI"):
        with Image.open(getattr(row, image_path_column)) as source:
            image = source.convert("L").copy()
        case = DicomCase(
            case_key=row.case_key,
            image=image,
            sex=float(row.sex),
            sex_label="M" if float(row.sex) == 1 else "F",
        )
        payload, base, gradcam, occlusion = explain_case(
            case,
            models,
            contract,
            device=device,
            occlusion_fraction=0.20,
            occlusion_stride_fraction=0.12,
        )
        payload["reference_bone_age_months"] = float(row.report_bone_age_months)
        payload["absolute_error_months"] = abs(
            payload["prediction_months"] - payload["reference_bone_age_months"]
        )
        render_explanation(destination / f"{row.case_key}.png", payload, base, gradcam, occlusion)
        _atomic_json(destination / f"{row.case_key}.json", payload)
        results.append(payload)
    return results


def evaluate_hospital_manifest(
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    artifact_dir = Path(str(config["artifact_dir"]))
    device = select_device(str(config.get("device", "auto")))
    models, contract = load_ensemble(
        Path(str(config["run_dir"])),
        device=device,
    )
    expected_checkpoint_count = int(config.get("expected_checkpoint_count", 5))
    if len(models) != expected_checkpoint_count:
        raise RuntimeError(
            f"Expected {expected_checkpoint_count} checkpoints, found {len(models)}"
        )
    interval = calibrate_oof_interval(
        Path(str(config["run_dir"])),
        coverage=float(config.get("prediction_interval_coverage", 0.90)),
    )
    input_variant = str(config.get("hospital_input_variant", "both"))
    if input_variant not in {"raw", "standardized", "both"}:
        raise ValueError("hospital_input_variant must be raw, standardized, or both")
    empty = np.full(len(manifest), np.nan, dtype=np.float64)
    raw_prediction, raw_std = (empty.copy(), empty.copy())
    standardized_prediction, standardized_std = (empty.copy(), empty.copy())
    if input_variant in {"raw", "both"}:
        raw_prediction, raw_std = _predict_variant(
            manifest,
            "raw_image_path",
            models,
            contract,
            device=device,
            batch_size=int(config.get("batch_size", 16)),
        )
    if input_variant in {"standardized", "both"}:
        standardized_prediction, standardized_std = _predict_variant(
            manifest,
            "standardized_image_path",
            models,
            contract,
            device=device,
            batch_size=int(config.get("batch_size", 16)),
        )
    manifest = manifest.copy()
    manifest["prediction_raw_months"] = raw_prediction
    manifest["ensemble_std_raw_months"] = raw_std
    manifest["prediction_standardized_months"] = standardized_prediction
    manifest["ensemble_std_standardized_months"] = standardized_std
    raw_lower, raw_upper = prediction_bounds(raw_prediction, interval["radius_months"])
    standardized_lower, standardized_upper = prediction_bounds(
        standardized_prediction,
        interval["radius_months"],
    )
    manifest["prediction_raw_lower_months"] = raw_lower
    manifest["prediction_raw_upper_months"] = raw_upper
    manifest["prediction_standardized_lower_months"] = standardized_lower
    manifest["prediction_standardized_upper_months"] = standardized_upper
    prediction_path = artifact_dir / "predictions.csv"
    manifest.to_csv(prediction_path, index=False)

    raw_studies = _study_predictions(manifest, "prediction_raw_months")
    standardized_studies = _study_predictions(manifest, "prediction_standardized_months")
    paired_keys = set(raw_studies["study_key"]) & set(standardized_studies["study_key"])
    raw_paired = raw_studies[raw_studies["study_key"].isin(paired_keys)]
    standardized_paired = standardized_studies[
        standardized_studies["study_key"].isin(paired_keys)
    ]
    metrics: dict[str, Any] = {
        "model_contract": {
            "backbone": contract.backbone,
            "dataset_version": contract.dataset_version,
            "preprocess_mode": contract.preprocess_mode,
            "resolution": contract.resolution,
            "checkpoint_count": len(models),
            "hospital_input_variant": input_variant,
        },
        "prediction_interval": interval,
        "raw": _external_metrics(
            raw_studies, interval_radius_months=interval["radius_months"]
        ),
        "standardized": _external_metrics(
            standardized_studies, interval_radius_months=interval["radius_months"]
        ),
        "paired_raw": _external_metrics(
            raw_paired, interval_radius_months=interval["radius_months"]
        ),
        "paired_standardized": _external_metrics(
            standardized_paired, interval_radius_months=interval["radius_months"]
        ),
    }
    if paired_keys:
        metrics["paired_mae_delta_standardized_minus_raw"] = (
            metrics["paired_standardized"]["mae"] - metrics["paired_raw"]["mae"]
        )
    xai_path_column = (
        "raw_image_path" if input_variant == "raw" else "standardized_image_path"
    )
    xai_prediction_column = (
        "prediction_raw_months"
        if input_variant == "raw"
        else "prediction_standardized_months"
    )
    xai_results = _run_xai_sample(
        manifest,
        models,
        contract,
        device=device,
        destination=artifact_dir / "xai",
        count=int(config.get("xai_cases", 8)),
        image_path_column=xai_path_column,
        prediction_column=xai_prediction_column,
    )
    metrics["xai"] = {
        "case_count": len(xai_results),
        "mean_gradcam_occlusion_correlation": float(
            np.mean(
                [
                    item["xai_metrics"]["gradcam_occlusion_correlation"]
                    for item in xai_results
                    if item["xai_metrics"]["gradcam_occlusion_correlation"] is not None
                ]
            )
        )
        if xai_results
        else None,
    }
    _atomic_json(artifact_dir / "metrics.json", metrics)
    return metrics


def evaluate_external_gate(
    build: dict[str, Any],
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate predeclared external-validation gates without tuning on the result."""
    gates = dict(config.get("external_gates", {}))
    if not gates:
        return {"passed": False, "reason": "external_gates_missing", "checks": {}}
    input_variant = str(config.get("hospital_input_variant", "standardized"))
    metric_key = "raw" if input_variant == "raw" else "standardized"
    evaluated = metrics[metric_key]
    run_dir = Path(str(config["run_dir"]))
    internal = json.loads((run_dir / "oof_metrics.json").read_text(encoding="utf-8"))
    subgroup_maes = [
        float(group["mae"]) for group in evaluated.get("by_sex", {}).values()
    ]
    worst_subgroup_gap = (
        max(subgroup_maes) - float(evaluated["mae"]) if subgroup_maes else float("inf")
    )
    observed = {
        "input_variant": metric_key,
        "internal_external_mae_gap_months": float(evaluated["mae"] - internal["mae"]),
        "p95_months": float(evaluated["p95_ae"]),
        "over_24_rate_upper_95": float(evaluated["over_24_rate_upper_95"]),
        "worst_sex_mae_gap_months": float(worst_subgroup_gap),
    }
    checks = {
        "internal_external_mae_gap": observed["internal_external_mae_gap_months"]
        <= float(gates["max_internal_external_mae_gap_months"]),
        "p95": observed["p95_months"] <= float(gates["max_p95_months"]),
        "over_24_rate_upper_95": observed["over_24_rate_upper_95"]
        <= float(gates["max_over_24_rate_upper_95"]),
        "worst_sex_mae_gap": observed["worst_sex_mae_gap_months"]
        <= float(gates["max_subgroup_mae_gap_months"]),
    }
    if metric_key == "standardized" and "min_standardization_pass_rate" in gates:
        total = int(build["dicom_total"])
        passed_count = int(build["standardization_status"].get("passed", 0))
        pass_rate = passed_count / total if total else 0.0
        observed["standardization_pass_rate"] = float(pass_rate)
        checks["standardization_pass_rate"] = pass_rate >= float(
            gates["min_standardization_pass_rate"]
        )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": observed,
        "thresholds": gates,
        "action": "package_research_ui" if all(checks.values()) else "run_local_adaptation",
    }


def _write_report(
    path: Path,
    build: dict[str, Any],
    metrics: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    raw = metrics["raw"]
    input_variant = metrics["model_contract"].get("hospital_input_variant", "both")
    standardized = metrics["standardized"]
    delta = metrics.get("paired_mae_delta_standardized_minus_raw")
    lines = [
        "# Hospital external validation - bone age",
        "",
        "Research-only evaluation. No direct patient identifiers are stored in this report.",
        "",
        "## Coverage",
        "",
        f"- DICOM images: {build['dicom_total']}",
        f"- Cases with supported sex metadata: {build['sex_valid']}",
        f"- Cases paired to a report label: {build['report_label_cases']}",
        f"- Unique labeled studies: {build['report_label_studies']}",
        f"- Standardization status: {build['standardization_status']}",
        "",
        "## External metrics",
        "",
        f"- Hospital input variant selected for the gate: {input_variant}.",
        (
            f"- Raw-P0 studies: {raw.get('study_count', 0)}; "
            f"MAE: {raw.get('mae', float('nan')):.2f} months."
        ),
        *(
            [
                (
                    f"- Standardized-compatible studies: {standardized.get('study_count', 0)}; "
                    f"MAE: {standardized.get('mae', float('nan')):.2f} months."
                )
            ]
            if standardized.get("study_count", 0)
            else []
        ),
        f"- Paired MAE delta (standardized - raw): {delta:.2f} months."
        if delta is not None
        else "- Paired MAE delta unavailable.",
        (
            f"- {metrics['prediction_interval']['coverage']:.0%} OOF-calibrated prediction "
            f"interval radius: ±{metrics['prediction_interval']['radius_months']:.2f} months."
        ),
        f"- Predeclared external gate passed: {decision['passed']}.",
        "",
        "## Interpretation limits",
        "",
        "- PDF report ranges are evaluated at their midpoint and also with a range-aware error.",
        (
            "- Raw hospital images are not mask-cropped, rotated, background-filled, or "
            "geometrically standardized."
            if input_variant == "raw"
            else (
                "- Heuristic masks that fail geometry gates are excluded rather than silently "
                "standardized."
            )
        ),
        (
            f"- Model: {metrics['model_contract']['backbone']} at "
            f"{metrics['model_contract']['resolution']} px with "
            f"{metrics['model_contract']['checkpoint_count']} fold checkpoints."
        ),
        "- Grad-CAM and occlusion describe model behavior; they are not causal clinical evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hospital_external_validation.yaml")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-xai", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.skip_xai:
        config["xai_cases"] = 0
    artifact_dir = Path(str(config["artifact_dir"]))
    if args.skip_build:
        source_dir = Path(str(config.get("manifest_source_artifact_dir", artifact_dir)))
        manifest = pd.read_parquet(source_dir / "manifest.parquet")
        build_audit = json.loads((source_dir / "build_audit.json").read_text(encoding="utf-8"))
    else:
        manifest, build_audit = build_hospital_manifest(config)
    metrics = evaluate_hospital_manifest(manifest, config)
    decision = evaluate_external_gate(build_audit, metrics, config)
    _atomic_json(artifact_dir / "decision.json", decision)
    _write_report(artifact_dir / "report.md", build_audit, metrics, decision)
    print(
        json.dumps(
            {"build": build_audit, "metrics": metrics, "decision": decision},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
