# Quy trình nghiên cứu và tiêu chí quyết định

## Trình tự bắt buộc

1. Kiểm tra số lượng, khả năng giải mã và hash của dữ liệu nguồn.
2. Gom ảnh trùng hoặc gần trùng trước khi chia fold.
3. Khóa split theo bệnh nhân hoặc nhóm trùng; không sửa split sau khi xem kết quả.
4. Kiểm tra trực quan ảnh chuẩn hóa trước khi huấn luyện.
5. Huấn luyện đủ các fold đã định trước và chỉ báo cáo dự đoán ngoài fold.
6. Đánh giá RSNA trước, sau đó mới đánh giá ngoài trên bệnh viện.
7. Nếu hiệu năng ngoài miền không đạt, tinh chỉnh trên split bệnh nhân 60/20/20 cố định.
8. So sánh ảnh gốc và ảnh chuẩn hóa trên đúng cùng ca trước khi kết luận.
9. Chọn mô hình bằng tập hiệu chỉnh; chỉ mở tập kiểm tra một lần cho báo cáo cuối.
10. Đưa mô hình vào giao diện nghiên cứu kèm khoảng dự đoán và tín hiệu cần xem lại.

## Chỉ số phải báo cáo

- Độ lệch tuyệt đối trung bình và khoảng tin cậy 95%.
- Trung vị, phân vị 90/95 và độ lệch lớn nhất.
- Số và tỷ lệ ca lệch trên 12, 24, 36 và 48 tháng.
- Độ chệch có dấu, kết quả theo giới và dải tuổi.
- Số ca, số ảnh và tỷ lệ bao phủ của mỗi nhánh.

## Điều kiện giữ mô hình

- So sánh theo cặp trên cùng ca khi hai phương án dùng dữ liệu đầu vào khác nhau.
- Không chấp nhận cải thiện trung bình nếu phân vị 95 hoặc tỷ lệ lệch trên 24 tháng xấu đi
  vượt nhiễu lấy mẫu.
- Không coi một fold, một thử nghiệm thăm dò hoặc một tập có mẫu số khác là kết quả cuối.
- Không chọn ngưỡng từ chính tập dùng để báo cáo hiệu năng.

## Quản trị dữ liệu

- Ảnh và DICOM chỉ lưu ngoài Git qua các liên kết trong `data/README.md`.
- Tên, mã bệnh nhân gốc, UID và đường dẫn nguồn không được ghi vào artifact chia sẻ.
- Bảng theo từng ca và ảnh X-quang không được đưa vào kho mã nguồn.
- Kết quả chỉ phục vụ nghiên cứu và luôn cần người có chuyên môn xác nhận.
