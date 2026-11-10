# Công việc phối hợp cho bản v5

Ngày 04/11/2026, bản cập nhật được thực hiện bằng ba phiên chuyên trách cùng một phiên tích hợp. Đây là phân công các tác vụ trợ lý trong phiên làm việc, không phải danh sách thành viên sinh viên hoặc cơ chế tự học của mô hình.

| Phiên | Công việc | Đầu ra có thể kiểm tra |
| --- | --- | --- |
| Đánh giá | Rà soát nguồn học và chuẩn hóa; thăm dò 29 clip ngoài bộ từ; tái lập số học | research/audit_v5.py, reports/evaluation_v5.json, CSV và gói đặc trưng thăm dò |
| Thiết kế | Thiết kế lại bốn mục giao diện; xử lý camera, chọn video, kết quả và ghi phiếu | web/index.html, web/style.css, web/app.js |
| Kiểm thử | Kiểm tra dữ liệu/API và trình duyệt thật; dùng đầu ra tạm riêng | tests/run_checks_v5.py, tests/test_browser_v5.py, reports/QA_V5.md |
| Tích hợp | Kết nối số liệu, tổng hợp phiếu, chuẩn bị lần chạy đầu, báo cáo và đóng gói | app.py, tools/summarize_trials.py, README và báo cáo v5 |

Các phần được đối chiếu chung trước bàn giao. Kết quả mô hình, kiểm tra phần mềm và camera giả lập được ghi riêng. Các phiên phối hợp không tạo thêm người ký hoặc video thực; trọng số và ngưỡng của nghiên cứu v4 được khóa trong đợt đánh giá bổ sung.
