# Thử nghiệm lớp ngoài bộ từ Ký Hiệu Lab v6

Ngày 07/11/2026

Đợt này bổ sung học có giám sát với một lớp ngoài bộ từ, sau khi đã thấy kết quả thăm dò v5. Đây là nghiên cứu phát triển hồi cứu, không phải test mù mới. Không có video thực mới được quay.

## Thiết kế ghép cặp trước khi học

Protocol được đóng băng trước học: cùng năm lượt giữ clip nhãn hỗ trợ, cùng ba seed, 500 bước, batch 64, learning rate 0,001, ngưỡng 0,65. Mỗi lượt giữ riêng một trong năm clip thuộc ai cho và không cần; 29 nguồn ngoài bộ từ được xếp theo SHA256 nguồn rồi chia vòng vào năm nhóm. Toàn bộ biến thể của nguồn giữ riêng đứng ngoài phần học.

TCN gốc học trên bảy clip hỗ trợ còn lại. Ứng viên sáu lớp học trên cùng bảy clip đó cộng 23 hoặc 24 nguồn ngoài bộ từ thuộc phần học, với lấy mẫu cân bằng sáu lớp. Mọi tăng cường và chuẩn hóa chỉ tạo từ phần học. Ứng viên thay đổi đồng thời lớp ngoài bộ từ và lấy mẫu cân bằng; phép so sánh chưa tách riêng tác dụng từng thay đổi.

## Kết quả theo seed

| Phương pháp | Seed | Hỗ trợ gợi ý đầu đúng /5 | Hỗ trợ chấp nhận đúng /5 | Hỗ trợ chấp nhận sai | Hỗ trợ bị từ chối | Ngoài bộ từ nhận nhầm /29 |
| --- | --- | --- | --- | --- | --- | --- |
| TCN gốc 5 lớp | 20261103 | 4 | 3 | 0 | 2 | 10 |
| TCN gốc 5 lớp | 20261104 | 5 | 3 | 0 | 2 | 9 |
| TCN gốc 5 lớp | 20261105 | 4 | 3 | 1 | 1 | 13 |
| TCN có lớp ngoài bộ từ | 20261103 | 4 | 1 | 0 | 4 | 3 |
| TCN có lớp ngoài bộ từ | 20261104 | 1 | 1 | 1 | 3 | 1 |
| TCN có lớp ngoài bộ từ | 20261105 | 2 | 2 | 0 | 3 | 3 |

Các hàng seed dùng lại cùng 5 nguồn nhãn hỗ trợ và cùng 29 nguồn ngoài bộ từ; không cộng thành 15 hoặc 87 video độc lập. Số bị từ chối ở nhóm hỗ trợ là false rejection trong nhóm này; chấp nhận nhưng sai nhãn được ghi riêng.

Không ghép 5 clip nhãn hỗ trợ và 29 clip ngoài bộ từ thành một accuracy chung. Cột ngoài bộ từ là kết quả giữ nguồn riêng bằng mô hình học trong từng lượt, khác phép thăm dò v5 dùng một mô hình học đủ tám clip hỗ trợ.

## Quyết định mô hình

Quy tắc thay mặc định đã chốt trước học: ở seed chính và trung bình ba seed, ứng viên phải không giảm gợi ý đầu đúng, không giảm chấp nhận đúng, không tăng chấp nhận sai của nhóm hỗ trợ, đồng thời giảm số nhận nhầm ngoài bộ từ. Không quét hoặc hạ ngưỡng theo tập giữ riêng.

Kết quả quy tắc: chưa đạt; giữ TCN gốc làm mặc định và để ứng viên ở chế độ thực nghiệm.

| Điều kiện | Đạt |
| --- | --- |
| primary_known_top1_nondecreasing | Có |
| primary_known_correct_accepts_nondecreasing | Chưa |
| primary_known_wrong_accepts_not_increasing | Có |
| primary_outside_false_accepts_reduced | Có |
| mean_known_top1_nondecreasing | Chưa |
| mean_known_correct_accepts_nondecreasing | Chưa |
| mean_known_wrong_accepts_not_increasing | Có |
| mean_outside_false_accepts_reduced | Có |

## Trọng số đủ dữ liệu và giới hạn

Trọng số ứng viên bàn giao được học lại trên đủ 37 nguồn cũ: 8 nguồn của năm nhãn hỗ trợ và 29 nguồn được gán lớp __OTHER__. Đây là trọng số demo sau nghiên cứu, khác các trọng số dùng giữ nguồn riêng. Các JSON nguồn gốc vẫn giữ nguyên pending/trainingEligible false; dùng học thử không biến thành nhãn đã được chuyên gia duyệt.

Xác suất softmax tính trên đủ sáu đầu ra. Khi lớp cao nhất là __OTHER__, hệ thống từ chối dù điểm cao; nếu lớp cao nhất thuộc bộ hỗ trợ, cần qua ngưỡng 0,65 và đủ tay. Không bỏ lớp ngoài rồi chuẩn hóa lại điểm năm lớp hỗ trợ.

Chưa có kiểm tra độc lập cho ba nhãn đơn mẫu hoặc người ký mới. Ngoài bộ từ là một nhóm dị thể của các nhãn đã biết từ hồ sơ; độ bao phủ ngoài đời còn hạn chế. Kể cả đạt quy tắc kỹ thuật nội bộ, kết quả không bảo đảm dùng dịch công khai.

## Tái lập

Từ thư mục gốc chạy `python research/rejection_v6.py`. Checkpoint có cùng dấu vân tay mã, protocol và đặc trưng sẽ được dùng lại. `--rerun` học lại có chủ đích. Không cần video thô ngoài bộ từ; gói đặc trưng dẫn xuất v5 đã kèm mã băm. Cần requirements-research.txt.
