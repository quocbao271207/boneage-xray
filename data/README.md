# Dữ liệu ngoài Git

Không đưa ảnh, DICOM, nhãn bệnh viện hoặc bảng dự đoán từng ca vào Git. Tải dữ liệu từ
bốn nguồn dưới đây và đặt vào đúng thư mục đích.

| Nhóm dữ liệu | Liên kết | Thư mục cục bộ |
|---|---|---|
| Kaggle RSNA — ảnh gốc | [Google Drive](https://drive.google.com/drive/folders/1q9Y8BG5Q0hhfoOG9CCdj9SvgStSjV04Y) | `data/raw/` |
| Kaggle RSNA — ảnh chuẩn hóa v3 | [Google Drive](https://drive.google.com/drive/folders/1Z9qC_Ha_qzf61kMs8zxbVh4Hfj5sju0n) | `data/processed/standardized_v3_preserve/` |
| Bệnh viện — ảnh DICOM gốc | [Google Drive](https://drive.google.com/drive/folders/1oH4xdToS0kp7kyBkk1ZYjDQr9vhv6DWY) | `data/hospital/original/` |
| Bệnh viện — ảnh chuẩn hóa | [Google Drive](https://drive.google.com/drive/folders/1xzkmKRkSoOl-F1s0ntuCl-5qfvxMEKaQ) | `data/hospital/standardized/` |

Nguồn công khai gốc của RSNA: [Kaggle — RSNA Bone Age](https://www.kaggle.com/datasets/kmader/rsna-bone-age).

## Kiểm tra tối thiểu

- Kaggle: 12.611 ảnh huấn luyện, 200 ảnh kiểm tra và hai CSV nguồn.
- Ảnh chuẩn hóa v3: 12.609 ảnh huấn luyện và 200 ảnh kiểm tra.
- Bệnh viện gốc: 1.144 ảnh DICOM trong gói nguồn.
- Bệnh viện chuẩn hóa: 789 ảnh có bản chuẩn hóa để đối chiếu.

Không đổi tên tệp nguồn trước khi chạy manifest. Mọi báo cáo xuất ra phải dùng mã ẩn danh.
