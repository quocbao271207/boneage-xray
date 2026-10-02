"""Research-only Streamlit UI for bone-age decision support."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pydicom
import streamlit as st
import torch
import yaml
from PIL import Image

from boneage.dicom_io import DicomCase, dicom_dataset_to_uint8, normalize_sex
from boneage.train import select_device
from boneage.uncertainty import calibrate_oof_interval, prediction_bounds
from boneage.xai import EnsembleContract, ExplainableModel, explain_case, load_ensemble


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@st.cache_resource(show_spinner="Đang nạp ensemble...")
def load_runtime(
    run_dir_text: str,
    device_text: str,
    coverage: float,
    calibrated_radius_months: float | None,
) -> tuple[list[ExplainableModel], EnsembleContract, torch.device, dict[str, Any]]:
    run_dir = Path(run_dir_text)
    device = select_device(device_text)
    models, contract = load_ensemble(run_dir, device=device)
    interval = (
        calibrate_oof_interval(run_dir, coverage=coverage)
        if calibrated_radius_months is None
        else {
            "method": "held_out_local_validation",
            "coverage": coverage,
            "radius_months": calibrated_radius_months,
        }
    )
    return models, contract, device, interval


def decode_upload(payload: bytes) -> tuple[np.ndarray, float, str]:
    dataset = pydicom.dcmread(io.BytesIO(payload), force=False)
    image = dicom_dataset_to_uint8(dataset)
    sex, sex_label = normalize_sex(dataset.get("PatientSex", ""))
    return image, sex, sex_label


def xai_overlay(base: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    grayscale = np.clip(base, 0.0, 1.0)
    rgb = np.repeat(grayscale[..., None], 3, axis=2)
    color = matplotlib.colormaps["coolwarm"]((heatmap + 1.0) / 2.0)[..., :3]
    alpha = (np.abs(heatmap) * 0.72)[..., None]
    return np.clip(rgb * (1.0 - alpha) + color * alpha, 0.0, 1.0)


def format_age(months: float) -> str:
    rounded = int(round(months))
    return f"{rounded} tháng ({rounded // 12} tuổi {rounded % 12} tháng)"


def review_reasons(payload: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    metrics = payload["xai_metrics"]
    reasons = []
    if payload["ensemble_std_months"] > thresholds["max_ensemble_std_months"]:
        reasons.append("các fold chưa đồng thuận")
    if metrics["border_attention_fraction"] > thresholds["max_border_attention_fraction"]:
        reasons.append("attention vùng biên cao")
    correlation = metrics["gradcam_occlusion_correlation"]
    if correlation is None or correlation < thresholds["min_gradcam_occlusion_correlation"]:
        reasons.append("Grad-CAM và occlusion chưa đồng thuận")
    return reasons


def main() -> None:
    st.set_page_config(page_title="Bone Age Research Support", layout="wide")
    config = load_config(Path("configs/research_ui.yaml"))
    deployment_path = Path("artifacts/deployment/research_model.json")
    deployment = (
        json.loads(deployment_path.read_text(encoding="utf-8"))
        if deployment_path.is_file()
        else {}
    )
    input_variant = str(deployment.get("hospital_input_variant", "raw"))
    if input_variant != "raw":
        st.error("UI bệnh viện chỉ chấp nhận model có contract raw-only.")
        return
    run_dir = str(deployment.get("run_dir", config["run_dir"]))
    calibrated_radius = deployment.get("prediction_interval_radius_months")
    st.title("Hỗ trợ đánh giá tuổi xương")
    st.caption(
        "Bản nghiên cứu — không dùng để tự động chẩn đoán hoặc thay thế bác sĩ chẩn đoán hình ảnh."
    )
    uploaded = st.file_uploader("Chọn một ảnh DICOM bàn tay–cổ tay", type=["dcm"])
    if uploaded is None:
        st.info(
            "Dữ liệu chỉ được xử lý trong phiên chạy cục bộ; "
            "UI không hiển thị định danh DICOM."
        )
        return
    payload_bytes = uploaded.getvalue()
    if len(payload_bytes) > int(config["max_upload_bytes"]):
        st.error("Tệp vượt giới hạn kích thước cho phép.")
        return
    try:
        raw, sex, sex_label = decode_upload(payload_bytes)
    except ValueError:
        selected = st.radio("DICOM thiếu hoặc sai PatientSex", ["Nữ", "Nam"], horizontal=True)
        sex_label = "F" if selected == "Nữ" else "M"
        sex = 0.0 if sex_label == "F" else 1.0
        try:
            dataset = pydicom.dcmread(io.BytesIO(payload_bytes), force=False)
            raw = dicom_dataset_to_uint8(dataset)
        except Exception as error:  # noqa: BLE001 - show a safe boundary error only
            st.error(f"Không thể đọc DICOM: {type(error).__name__}")
            return
    except Exception as error:  # noqa: BLE001 - show a safe boundary error only
        st.error(f"Không thể đọc DICOM: {type(error).__name__}")
        return

    model_image = raw

    models, contract, device, interval = load_runtime(
        run_dir,
        str(config.get("device", "auto")),
        float(config["prediction_interval_coverage"]),
        float(calibrated_radius) if calibrated_radius is not None else None,
    )
    if len(models) != 5:
        st.error(f"Ensemble chưa đủ 5 folds (hiện có {len(models)}).")
        return
    case = DicomCase(
        case_key="session_case",
        image=Image.fromarray(model_image),
        sex=sex,
        sex_label=sex_label,
    )
    with st.spinner("Đang dự đoán và sinh XAI..."):
        explanation, model_input, gradcam, occlusion = explain_case(
            case,
            models,
            contract,
            device=device,
            occlusion_fraction=0.20,
            occlusion_stride_fraction=0.12,
        )
    lower, upper = prediction_bounds(
        np.array([explanation["prediction_months"]]), interval["radius_months"]
    )
    st.subheader("Kết quả hỗ trợ")
    result_columns = st.columns(3)
    result_columns[0].metric("Tuổi xương dự đoán", format_age(explanation["prediction_months"]))
    result_columns[1].metric(
        "Khoảng dự đoán 90%", f"{lower[0]:.0f}–{upper[0]:.0f} tháng"
    )
    result_columns[2].metric(
        "Độ lệch giữa các fold", f"{explanation['ensemble_std_months']:.1f} tháng"
    )

    panels = st.columns(3)
    panels[0].image(raw, caption="1. Raw", clamp=True, use_container_width=True)
    panels[1].image(
        model_input,
        caption="2. Model input P0 từ ảnh raw",
        clamp=True,
        use_container_width=True,
    )
    panels[2].image(
        xai_overlay(model_input, gradcam),
        caption="3. XAI — Centered Grad-CAM",
        clamp=True,
        use_container_width=True,
    )
    with st.expander("Đối chiếu occlusion sensitivity"):
        st.image(
            xai_overlay(model_input, occlusion),
            caption="Occlusion sensitivity",
            clamp=True,
            width=520,
        )
        st.json(
            {
                "model": explanation["model_contract"],
                "xai_metrics": explanation["xai_metrics"],
                "hospital_input_variant": input_variant,
            }
        )
    reasons = review_reasons(explanation, config["review_thresholds"])
    if reasons:
        st.warning("Cần bác sĩ review: " + "; ".join(reasons) + ".")
    else:
        st.info("Mô hình vượt các gate kỹ thuật; kết quả vẫn cần bác sĩ xác nhận.")


if __name__ == "__main__":
    main()
