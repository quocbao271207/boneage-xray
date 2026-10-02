# Các mô hình được giữ lại

## 1. RSNA — EfficientNet-B0 512

- Dữ liệu: ảnh chuẩn hóa v3 giữ sắc độ gốc.
- Huấn luyện: 5 fold cố định, kết hợp ảnh và giới tính.
- Độ lệch tuyệt đối trung bình: 6,642 tháng trên 12.609 dự đoán ngoài fold.
- Phân vị 95 của độ lệch: 17,584 tháng.
- Số ảnh lệch trên 24 tháng: 194 (1,54%).
- Kết quả: `results/rsna_efficientnet_b0_512/evaluation.json`.

Đây là mô hình nguồn cho cả hai nhánh đánh giá bệnh viện.

## 2. Bệnh viện — ảnh chuẩn hóa, mở dần toàn bộ mô hình

- Tập kiểm tra: 147 ca có ảnh chuẩn hóa.
- Độ lệch tuyệt đối trung bình: 9,032 tháng.
- Phân vị 95: 22,886 tháng.
- Số ca lệch trên 24 tháng: 6/147.
- Kết quả: `results/hospital_standardized_gradual/evaluation.json`.

Đây là con số thấp nhất quan sát được trên dữ liệu bệnh viện, nhưng mẫu số nhỏ hơn do chỉ
gồm các ca có bản chuẩn hóa.

## 3. Bệnh viện — ảnh gốc, mở dần toàn bộ mô hình

- Tập kiểm tra: 209 ca.
- Độ lệch tuyệt đối trung bình: 9,731 tháng.
- Phân vị 95: 23,427 tháng.
- Số ca lệch trên 24 tháng: 10/209.
- Kết quả: `results/hospital_original_gradual/evaluation.json`.

Nhánh này được chọn cho giao diện nghiên cứu vì bao phủ tập rộng hơn và không phụ thuộc vào
phép tách bàn tay, xoay hoặc cắt ảnh. Trên đúng 147 ca chung, kết quả là 9,228 tháng so với
9,032 tháng của ảnh chuẩn hóa; khoảng tin cậy của chênh lệch đi qua 0.

## Quy tắc lưu mô hình

Checkpoint không nằm trong Git. Mỗi lần tái tạo phải giữ cùng cấu hình, split, seed và ghi
đầy đủ hash của checkpoint. Không thay thế mô hình được chọn chỉ dựa trên một fold hoặc một
tập kiểm tra có mẫu số khác.
