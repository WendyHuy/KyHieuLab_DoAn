# Tiến độ Ký Hiệu Lab ngày 07 11 2026

## Đã bàn giao bản v6 cho phạm vi đồ án

Ứng dụng nhận diện được ở mức demo có giám sát trong bộ năm ký hiệu. TCN gốc vẫn mặc định. Nguồn giữ riêng ở seed chính cho gợi ý đúng 4/5, chấp nhận đúng 3/5; chỉ có hai nhãn và cùng một người ký được kiểm tra. Chưa có đánh giá độc lập ba nhãn đơn mẫu hoặc người ký mới.

Đã hoàn tất trong phiên sáng 07/11:

- Học sâu thêm TCN sáu đầu ra, gồm nhóm ngoài bộ năm nhãn, trên 37 nguồn cũ.
- Chạy 30 lượt học ghép cặp với ba seed, dự đoán 204 lượt nguồn giữ riêng. Kiểm tra không trùng nguồn học, chuẩn hóa train-only và sai khác JAX/NumPy.
- Ở seed chính, đối chứng nhận nhầm ngoài bộ từ 10/29, ứng viên 3/29; nhưng chấp nhận đúng clip hợp lệ giảm từ 3/5 xuống 1/5. Trung bình ba seed cũng giảm nhận đúng. Quy tắc chọn đã chốt KHÔNG ĐẠT: giữ TCN gốc mặc định, ứng viên chỉ thử nghiệm. Không dò lại ngưỡng 0,65 theo kết quả này.
- Khôi phục 29 video nguồn cũ, xác minh MP4 v3 với protocol; bản mã hóa lại khác byte v5. Đặc trưng nghiên cứu vẫn giữ nguyên bản v5 đã khóa, không thay bằng đặc trưng mới. Hai ví dụ ngoài bộ nhãn trong giao diện là nguồn ứng viên đã học.
- Hoàn thiện hiển thị từ chối, ghi quyết định gốc, nhận diện phát lại theo từng mô hình, hai ví dụ ngoài bộ nhãn và thống kê phiếu.
- Đạt 40 kiểm tra dữ liệu/API và 22 tình huống trình duyệt thật, không bỏ qua. Camera tự động dùng nguồn tổng hợp, chưa phải webcam vật lý.
- Hoàn thiện báo cáo MD, Word và PDF 11 trang; đã xem đủ 11 trang sau lần xuất cuối. Có ma trận nền và ma trận nghiên cứu mới với mẫu số 5 và 29 tách riêng.

## Checkpoint và hồ sơ sử dụng lại

`research/rejection_v6.py`, `research/rejection_v6_protocol.json`, `research/rejection_v6_checkpoints/`, `models/tcn_reject.*`, `reports/rejection_v6.json`, `reports/rejection_v6_verification.json`. Không đổi protocol sau khi xem kết quả. Lệnh nghiên cứu dùng lại checkpoint có cùng dấu vân tay; chỉ dùng --rerun khi có lý do thực nghiệm cụ thể.

Báo cáo chính: `reports/Bao_cao_DoAn_KyHieuLab_v6.md`, `.docx`, `.pdf`. Kiểm tra phần mềm: `reports/QA_V6.md`, `test_results_v6.json`, `browser_checks_v6.json`. Phần v4/v5 là hồ sơ lịch sử, không tự thay số.

## Công việc cần dữ liệu hoặc thiết bị thực

1. Quay thêm đủ năm nhãn ở người và phiên mới, đặc biệt ba nhãn chỉ có một mẫu. Kế hoạch 250 vị trí chưa quay nằm trong data/collection_plan_250.csv.
2. Thu hoạt động ngoài bộ từ mới, chia validation/test trước khi tối ưu. Không gọi 29 nguồn đã dùng phát triển là test mù mới.
3. Chạy webcam vật lý, Windows và CSV trong Excel trên máy báo cáo; đo độ trễ tại máy đó.
4. Điền thông tin môn, lớp và giảng viên theo biểu mẫu thật nếu người dùng cung cấp.

Không có video thực mới hoặc phần cứng tương ứng trong phiên này. Nếu chưa có nguồn mới, không thể hoàn thành các mục trên chỉ bằng học lại hoặc lặp seed. Không tự tăng số video, người ký, trạng thái duyệt hay chất lượng ngoài phạm vi.

## Lịch tiếp tục trong hôm nay

Mỗi giờ từ 10:45 đến 23:45 ngày 07/11/2026, giờ Việt Nam; lịch kết thúc trong ngày. Đây là các lượt trợ lý tiếp tục công việc, không phải GPU huấn luyện không nghỉ. Mỗi lượt khôi phục bản ZIP hiện hành trước khi xử lý, đọc tiến độ và không ghi đè phiên đang hoạt động. Dùng version guard hiện hành khi lưu lại cùng tệp.

Ưu tiên dữ liệu mới thực sự liên quan nếu người dùng cung cấp, hoặc lỗi có thể tái hiện. Nếu không có đầu vào mới và không có lỗi chưa xử lý, báo ngắn rằng phần mềm/báo cáo đã hoàn tất ở phạm vi demo, các mục còn lại cần video hoặc thiết bị thực. Không chạy lại bộ kiểm tra/học đã đạt để tạo tiến độ giả, không thêm giao diện hoặc mô hình chỉ để kéo dài công việc. Không triển khai công khai.
