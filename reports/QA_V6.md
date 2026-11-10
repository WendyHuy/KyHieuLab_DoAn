# Kiểm tra phần mềm Ký Hiệu Lab v6

Ngày kiểm tra 07/11/2026. Phạm vi kiểm tra gồm API, giao diện, quyết định từ chối và tính đúng của phiếu thực nghiệm. Kết quả này đo hoạt động phần mềm; số đo chất lượng nhận diện nằm trong báo cáo thực nghiệm riêng.

## Kiểm tra API và quyết định từ chối

`tests/run_checks_v6.py --require-candidate` đã chạy đạt **40 kiểm tra**, không bỏ qua trường hợp nào. Bộ này giữ 28 kiểm tra hồi quy và thêm 12 kiểm tra v6. Kết quả máy đọc được nằm trong `reports/test_results_v6.json`.

Các kiểm tra mới xác nhận:

- Mô hình có lớp nội bộ ngoài bộ từ vẫn chỉ hiển thị năm nhãn hỗ trợ. Quyết định lớp ngoài bộ từ luôn dẫn đến từ chối; điểm các gợi ý còn lại không được chuẩn hóa lại để tạo độ chắc chắn giả.
- Quyết định và điểm gốc được giữ trong phiếu. Nếu lớp gốc là ngoài bộ từ nhưng gợi ý đứng đầu sau lọc trùng nhãn thực tế, lượt đó vẫn được tính sai ở quyết định gốc, thay vì đổi thành đúng theo gợi ý.
- Tệp trọng số ứng viên là tùy chọn. Ứng viên chỉ thành mặc định khi báo cáo cho phép thăng cấp, đề xuất đúng tên và mã băm khớp trọng số đang nạp. Báo cáo cũ hoặc mã băm sai giữ mô hình mặc định gốc.
- Các phiếu cũ vẫn đọc được; các cột lý do từ chối và trạng thái ngoài bộ từ là phần bổ sung.
- Nhận diện phát lại được xác định theo dữ liệu đã học của từng mô hình. Một video ngoài bộ từ có thể là mẫu học của ứng viên nhưng không phải mẫu học của mô hình gốc.
- Khi video cũ được mã hóa lại, chỉ ghép mã băm mới vào hồ sơ phát lại nếu đồng thời khớp mã video nguồn và mã băm video gốc. Không tự coi mỗi tệp khác byte là một lần thực hiện mới.
- Hai video ngoài bộ từ phục vụ demo được đọc bằng danh sách cho phép; đường dẫn lạ hoặc vượt thư mục bị từ chối.

Một số kiểm tra dùng cấu hình sáu đầu ra được tạo có chủ đích để ép nhánh ngoài bộ từ. Đây là fixture kiểm tra logic phần mềm, không phải dữ liệu đo chất lượng mô hình. Mã nguồn và trọng số huấn luyện thực không bị thay đổi bởi các fixture.

## Kiểm tra trình duyệt

`tests/test_browser_v6.py` dùng Playwright với Chromium thật và máy chủ Python thật. Đã chạy đạt **22 kiểm tra** trên Chromium 153.0.8010.0 headless. Kết quả đầy đủ được ghi trong `reports/browser_checks_v6.json`. Đã kiểm tra desktop 1440 × 1100 và mobile 390 × 844; bốn tab không gây tràn ngang trang. Không có lỗi JavaScript chưa xử lý, lỗi CSP hay lỗi pattern HTML trong phiên cuối. Các phép thử có nguồn video thật, chọn/tải clip, ghi phiếu, CSV, mô hình ứng viên, bốn tab và bố cục màn hình nhỏ. Danh sách bốn mô hình được lấy từ cấu hình máy chủ và mô hình mặc định khớp quyết định thăng cấp đã kiểm tra mã băm. Ứng viên thật đã nhận lại video c3b59369 thành lớp ngoài bộ từ, hiển thị thông báo tiếng Việt và giữ quyết định gốc trong CSV. Phần so sánh CV cùng cách giữ nguồn được hiển thị riêng với kết quả thăm dò v5.

Nguồn video ngoài bộ từ dùng trong demo là lần thực hiện cũ đã có trong dữ liệu phát triển của ứng viên, được khôi phục và mã hóa lại. Kiểm tra xem/nhận diện đoạn này chỉ xác minh luồng demo và ghi đúng hồ sơ phát lại; không được dùng như một video kiểm tra độc lập mới.

Kiểm tra camera tự động dùng nguồn tổng hợp; bản Chromium headless này không hỗ trợ thiết bị camera giả qua getUserMedia nên phiên cuối dùng MediaStream canvas được đưa vào có chủ đích. Bài kiểm tra vẫn dùng MediaRecorder thật để tạo WebM và gửi vào bộ xử lý Python thật. Phép thử từ chối quyền camera cũng được tạo chủ động. Chưa có kiểm tra webcam vật lý, Windows hoặc phần cứng điện thoại.

Mọi nhật ký kiểm tra nằm trong thư mục tạm riêng. Phiếu giả lập và video do QA tạo không nằm trong thư mục đầu ra của người dùng hay số liệu đánh giá mô hình. Ảnh chụp màn hình chỉ dùng rà soát nội bộ, không nằm trong gói bàn giao.

## Chạy lại

Kiểm tra API cần môi trường có `requirements-research.txt` và FFmpeg:

```text
python tests/run_checks_v6.py --require-candidate
```

Phần kiểm tra trình duyệt là tùy chọn, không cần để sử dụng demo:

```text
python -m pip install playwright
python -m playwright install chromium
python tests/test_browser_v6.py --require-candidate
```

Có thể dùng bản Chromium tin cậy đã có với `--executable`. Cờ `--screenshots-dir` chỉ dành cho ảnh QA nội bộ. Máy chủ và trình duyệt chạy trong cùng phiên kiểm tra để truy cập localhost; bộ kiểm tra không đọc hoặc xóa phiếu thật của người dùng.
