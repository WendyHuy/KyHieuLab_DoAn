# Đánh giá bổ sung Ký Hiệu Lab v5

Ngày 04/11/2026

Bản v5 mở rộng phép đánh giá và kiểm tra nguồn dữ liệu, giữ mô hình và ngưỡng 0,65 của v4. Đợt này không thu thêm video thực, không huấn luyện thêm và không lựa chọn cấu hình theo kết quả ngoài bộ từ.

## Phạm vi và kiểm tra tính đúng của thực nghiệm

Bộ học có 8 video, 5 nhãn, một mã người ký. Phép giữ từng clip ngoài phần học chỉ bao phủ 5 video nguồn thuộc hai nhãn ai cho và không cần. Ba nhãn bực mình, ghen tị và xin lỗi chưa có mẫu kiểm tra riêng. Ba seed dùng lại cùng năm clip, không tạo thêm mẫu độc lập.

Đã đối chiếu 45 checkpoint: đủ 45 tổ hợp phương pháp, seed và clip; mỗi phần học đúng bảy video còn lại; mã băm video giữ riêng không nằm trong phần học. Đã tính lại 45 cặp trung bình và độ lệch chuẩn từ chính video học để so với các mảng lưu; sai khác lớn nhất 0. Không phát hiện chuẩn hóa dùng video giữ riêng trong phép kiểm tra này.

## Kết quả theo seed và theo video nguồn

| Mô hình | Đúng ở seed chính | Chấp nhận đúng ở seed chính | Clip đúng ở cả 3 seed | Clip có ít nhất một seed chấp nhận sai |
| --- | --- | --- | --- | --- |
| MLP | 4/5 | 2/5 | 3/5 | 2/5 |
| TCN gốc | 4/5 | 3/5 | 3/5 | 1/5 |
| TCN tăng cường | 3/5 | 2/5 | 2/5 | 3/5 |

Bảng độ ổn định đếm trên năm video nguồn. Cột có seed chấp nhận sai không phải tỷ lệ lỗi trên người dùng mới. Các dự đoán từng seed và từng clip nằm trong evaluation_v5_clip_stability.csv.

## Thăm dò video ngoài bộ từ

Chọn trước toàn bộ 29 clip đủ hồ sơ học thử trong bộ v3 có nhãn ngoài năm nhãn hiện tại: 17 clip giahuy và 12 clip tp, 29 nhãn khác nhau. Những video này không thuộc tám nguồn học của mô hình v4. Hai clip thiếu hồ sơ ở trạng thái pending không được dùng. Video được chuẩn hóa theo cùng quy tắc 12 fps và kích thước trước khi trích điểm lại.

Cả 29 clip thăm dò vẫn có reviewStatus pending và trainingEligible false trong hồ sơ gốc; nhãn được hiểu theo thông tin người thu cung cấp và giả định người thu đã đối chiếu động tác. Không thay đổi các trạng thái này. Đây là tập tiện có từ lần thu cũ, chưa phải test mù hoặc mẫu đại diện cho mọi hoạt động ngoài bộ từ.

| Mô hình | Đã xử lý | Chấp nhận nhầm | Từ chối | Tỷ lệ chấp nhận nhầm trong nhóm này |
| --- | --- | --- | --- | --- |
| MLP | 29 | 12 | 17 | 41.4% |
| TCN gốc | 29 | 14 | 15 | 48.3% |
| TCN tăng cường | 29 | 10 | 19 | 34.5% |

Số clip lỗi hoặc bỏ qua: 0. Mẫu lỗi được lưu riêng trong JSON, không tính thành từ chối đúng. Các tỷ lệ trên dùng số clip thực sự xử lý được, không được suy rộng thành FAR đã được hiệu chỉnh của sản phẩm.

Hình trong gói: reports/figures/evaluation_v5_outside_vocabulary.png. Bảng dự đoán từng clip nằm trong reports/evaluation_v5_outside_predictions.csv; ma trận một hàng ngoài bộ từ nằm trong reports/evaluation_v5_outside_confusion.csv.

## Quyết định sử dụng và việc còn thiếu

Bản hiện tại dùng cho trình bày và phiên thử có giám sát. Điểm softmax cao chưa đủ để xác nhận ký hiệu thật sự thuộc bộ năm nhãn. Cần luôn có cách ký lại, nhập tay hoặc xác nhận bởi người kiểm tra; kết quả nên được đọc là nhãn gợi ý trong bộ đang học.

Chưa đáp ứng kiểm chứng độc lập đủ năm nhãn hoặc người ký mới, hiệu chỉnh ngưỡng ngoài bộ từ và điều kiện dịch công khai. Ưu tiên tiếp theo là video thực của nhiều người và phiên, có validation và test tách trước; dữ liệu ngoài bộ từ dùng chọn ngưỡng phải tách khỏi phần dùng kiểm tra cuối. Không tăng ngưỡng chỉ để làm đẹp kết quả của 29 clip vừa xem.

## Tái lập

Chạy `python research/audit_v5.py` từ thư mục gốc bản v5 để tính lại từ gói đặc trưng đã trích, có mã băm đối chiếu. Gói reports/evaluation_v5_features chỉ chứa đặc trưng dẫn xuất của 29 video cũ, không phải 29 video mới và không được nhập vào phần học.

Để trích điểm lại từ video thật, giải nén gói v3 đầy đủ và chạy `python research/audit_v5.py --source-v3 /duong/dan/KHL_NhanDien_v3 --work-dir /thu/muc/tam`. Cần requirements-research.txt và FFmpeg cho chế độ này. Protocol được ghi trước dự đoán trong research/evaluation_v5_protocol.json; mã băm trọng số và cấu hình giữ nguyên. MP4 chuẩn hóa nằm ở work-dir và không được đóng gói vào bản v5.
