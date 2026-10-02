# boneage-xray

Quy trình nghiên cứu dự đoán tuổi xương từ X-quang bàn tay, gồm huấn luyện trên RSNA,
đánh giá ngoài trên dữ liệu bệnh viện, tinh chỉnh theo bệnh nhân và giao diện kiểm tra.
Dự án ưu tiên chống rò rỉ dữ liệu, so sánh cùng ca và theo dõi cả sai số trung bình lẫn
đuôi sai số lớn.

> Chỉ dùng cho nghiên cứu. Không dùng để chẩn đoán, điều trị hoặc thay thế bác sĩ.

## Các bản mạnh nhất được giữ lại

| Nhánh | Mô hình | Tập đánh giá | Độ lệch tuyệt đối trung bình |
|---|---|---:|---:|
| RSNA | EfficientNet-B0, ảnh chuẩn hóa v3, 512 px, 5 fold | 12.609 ảnh | 6,642 tháng |
| Bệnh viện — ảnh chuẩn hóa | EfficientNet-B0, mở dần toàn bộ mô hình | 147 ca | 9,032 tháng |
| Bệnh viện — ảnh gốc | EfficientNet-B0, mở dần toàn bộ mô hình | 209 ca | 9,731 tháng |

Trên đúng 147 ca bệnh viện dùng chung, ảnh gốc đạt 9,228 tháng và ảnh chuẩn hóa đạt
9,032 tháng; chênh lệch 0,196 tháng có khoảng tin cậy 95% từ -0,604 đến +0,973 tháng.
Vì vậy kho giữ cả hai hướng bệnh viện thay vì suy luận từ hai tập kiểm tra khác kích thước.

Chi tiết nằm trong [docs/models.md](docs/models.md) và các JSON dưới `results/`.
Checkpoint không được đưa lên Git vì dung lượng lớn; tái tạo bằng pipeline bên dưới.

## Dữ liệu

Ảnh, DICOM và nhãn không nằm trong Git. Bốn nguồn dữ liệu và cấu trúc thư mục đích được
ghi tại [data/README.md](data/README.md). Sau khi tải, thư mục cục bộ có dạng:

```text
data/
  raw/                              # Kaggle RSNA gốc
  external/rsna_hand_masks/         # mask đi kèm nhánh chuẩn hóa Kaggle
  processed/standardized_v3_preserve/
  hospital/original/                # gói DICOM bệnh viện gốc
  hospital/standardized/            # ảnh bệnh viện chuẩn hóa để đối chiếu
```

## Cài đặt

Yêu cầu Python 3.11 và `uv`:

```powershell
uv sync --extra dev
```

## Quy trình tuần tự

```powershell
# 1. Kiểm tra dữ liệu Kaggle và tạo manifest
uv run python -m boneage.build_manifest --config configs/data.yaml

# 2. Tạo 5 fold chống rò rỉ theo nhóm trùng và giới/tuổi
uv run python -m boneage.build_splits `
  --manifest artifacts/data/manifest.parquet `
  --output artifacts/data/splits_v2.csv `
  --audit-output artifacts/data/splits_v2_audit.json

# 3. Tạo ảnh chuẩn hóa v3 giữ sắc độ gốc
uv run python -m boneage.standardize_dataset --config configs/standardization_v3.yaml

# 4. Huấn luyện và đánh giá mô hình RSNA mạnh nhất
uv run python -m boneage.train `
  --config configs/standardized_v3_efficientnet_b0_512.yaml --all-folds
uv run python -m boneage.evaluate `
  --config configs/standardized_v3_efficientnet_b0_512.yaml `
  --mode oof `
  --run-dir artifacts/runs/standardized_v3_preserve_efficientnet_b0_512_screen

# 5. Đánh giá bệnh viện đồng thời trên ảnh gốc và ảnh chuẩn hóa
uv run python -m analysis.hospital_external_validation `
  --config configs/hospital_external_validation_both.yaml

# 6. Tinh chỉnh hai hướng trên split bệnh nhân cố định
uv run python -m analysis.hospital_finetune `
  --config configs/hospital_finetune_standardized.yaml --mode gradual
uv run python -m analysis.hospital_finetune `
  --config configs/hospital_finetune_raw.yaml --mode gradual

# 7. Chạy kiểm tra mã nguồn
uv run ruff check .
uv run pytest -q

# 8. Mở giao diện nghiên cứu
uv run streamlit run analysis/research_ui.py
```

Có thể chạy các bước chính bằng `scripts/run_research_pipeline.ps1`. Script dừng ngay khi
một bước thất bại và không tự động bỏ qua kiểm tra chất lượng.

## Quy tắc nghiên cứu

- Mọi phép so sánh trực tiếp phải dùng đúng cùng mã ảnh hoặc mã lần chụp.
- Chia dữ liệu bệnh viện theo mã bệnh nhân ẩn danh; một bệnh nhân không được xuất hiện ở
  nhiều tập huấn luyện, hiệu chỉnh và kiểm tra.
- Tập kiểm tra được giữ kín; không dùng để chọn epoch, siêu tham số hoặc ngưỡng.
- Báo cáo đồng thời độ lệch trung bình, trung vị, phân vị 95, độ lệch lớn nhất và số ca lệch
  trên 12/24 tháng.
- Mô hình chỉ được giữ khi cải thiện trung bình mà không làm xấu đáng kể đuôi sai số.
- Không đưa ảnh, DICOM, bảng theo từng ca, thông tin định danh hoặc checkpoint lên Git.
- Mọi kết quả bệnh viện phải được mô tả là nghiên cứu và cần bác sĩ xác nhận.

Xem quy trình quyết định đầy đủ tại [docs/research-protocol.md](docs/research-protocol.md).

## Cấu trúc kho

```text
analysis/   đánh giá ngoài, tinh chỉnh, so sánh và giao diện nghiên cứu
configs/    cấu hình dữ liệu, chuẩn hóa và các mô hình được tái lập
docs/       quy tắc nghiên cứu, mô hình và nguồn dữ liệu
results/    chỉ số tổng hợp của các bản mạnh nhất, không chứa dữ liệu từng ca
scripts/    trình chạy tuần tự
src/        package `boneage`
tests/      kiểm thử tự động
```
