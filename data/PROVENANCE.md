# Nguồn dữ liệu của bản v4

Bản v4 lấy 8 video của 5 nhãn từ `KyHieuLab_DanhGia_CaiTien_v3_2026-11-01.zip`. SHA256 gói nguồn: `c5f1c0173cdf38ce4d75aacc7f5e824231afc3300cefc56be7133efb58793ad3`.

`source_video_sha256` trong manifest là mã của video gốc theo hồ sơ v3, không phải mã của MP4 đã chuẩn hóa. `normalized_video_sha256` dùng kiểm tra MP4 đi kèm v4. `v3_video` ghi đường dẫn clip trong gói nguồn; `original_start` và `original_end` là mốc cắt theo hồ sơ nguồn.

Danh sách năm nhãn được chốt trong `research/protocol.json` trước khi học. Dữ liệu gốc không được tăng số lượng khi chạy ba seed hoặc sinh tăng cường. Hồ sơ duyệt và JSON nguồn được giữ nguyên. Hai video pending thiếu thông tin và 29 nhãn ngoài phạm vi không được nhập vào mô hình năm nhãn. Gói đầy đủ v3 vẫn lưu riêng.

## Bổ sung ở v6 ngày 07 11 2026

Mô hình TCN từ chối là nghiên cứu phát triển mới, có sáu đầu ra: năm nhãn hỗ trợ và một nhóm `__OTHER__`. Mô hình này học thử từ tám nguồn hỗ trợ và đặc trưng đã khóa của 29 nguồn ngoài bộ từ, sau khi đã biết chẩn đoán v5. Ba mô hình năm đầu ra và nghiên cứu nền vẫn giữ nguyên. Dữ liệu mới được sử dụng cho học thử không được gọi là test mù mới.

`outside_manifest_v6.json` lưu các bản video phát lại của 29 nguồn cũ, mã băm nguồn v3, mã băm video chuẩn hóa hiện tại và tham chiếu mã băm chuẩn hóa v5. Đã đối chiếu byte đầu vào của đủ 29 MP4 v3 với protocol v5. Các bản mã hóa hiện tại có mã băm khác v5; chúng phục vụ phát lại trải nghiệm và QA. Đặc trưng phục vụ học v6 vẫn là các NPZ đã khóa trong `reports/evaluation_v5_features`, không được thay bằng đặc trưng mới chỉ vì khôi phục video.

Số nguồn vẫn là 37 lần thực hiện cũ, không có video thực mới. Hai ví dụ ngoài bộ từ trong giao diện là nguồn ứng viên đã học. Hệ thống đánh dấu phát lại các bản mã hóa này theo quan hệ nguồn học, không dùng một mã băm byte mới để khẳng định mẫu độc lập. Hồ sơ pending và trainingEligible false của nguồn vẫn giữ nguyên.
