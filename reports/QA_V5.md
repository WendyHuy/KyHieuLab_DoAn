# Kiểm tra phần mềm Ký Hiệu Lab v5

Ngày kiểm tra 04/11/2026. Phạm vi là độ đúng của luồng phần mềm, dữ liệu, nhật ký thực nghiệm và tương tác giao diện. Kiểm tra này không thay thế thực nghiệm nhận diện trên người ký mới.

## Kiểm tra API và dữ liệu

`tests/run_checks_v5.py` đã chạy đạt 28 kiểm tra trong môi trường Linux Python 3.12. Bộ này gồm 23 kiểm tra hồi quy và 5 kiểm tra mới cho v5. Kết quả máy đọc được nằm trong `reports/test_results_v5.json`.

Các nội dung được kiểm tra gồm:

- Nguồn video không trùng giữa phần học và clip giữ riêng; 45 kết quả dùng đúng năm clip nguồn qua ba seed.
- Kích thước đặc trưng, mặt nạ điểm thiếu, timestamp và mã băm MP4 chuẩn hóa.
- Nhận diện bằng video thật qua HTTP cho cả ba mô hình, áp dụng đoạn cắt, từ chối dữ liệu không hợp lệ và đoạn không có người.
- Ghi phiếu theo kết quả gốc trên máy chủ, ngăn ghi trùng, ngăn ghi bằng mã kết quả đã hết hạn và xử lý hai yêu cầu ghi cùng một lượt đồng thời.
- Tổng hợp riêng số lượt, số video khác mã băm, lượt phát lại video đã học, phiên bản trọng số, dữ liệu trong và ngoài bộ từ.
- Ma trận quyết định có cột từ chối, mẫu số đúng khi tính tỷ lệ chấp nhận và nhận nhầm ngoài bộ từ.
- Khi nhật ký hỏng hoặc thiếu trường, API trả lỗi 400 có số dòng; không âm thầm bỏ dòng hoặc sửa tệp gốc.

Toàn bộ phiếu dùng kiểm tra được ghi trong thư mục tạm riêng và được dọn sau phiên. Các số liệu từ phiếu giả lập dùng kiểm tra phần mềm không được tính thành dữ liệu thực nghiệm mô hình.

## Kiểm tra trình duyệt

Trình chạy `tests/test_browser_v5.py` dùng Playwright với Chromium thật, nhận diện video bằng máy chủ Python thật và thư mục đầu ra tạm riêng. Đã chạy đạt **18 kiểm tra** trên Chromium 153.0.8010.0 headless, qua Playwright. Tệp `reports/browser_checks_v5.json` lưu kết quả cụ thể. Đã kiểm tra desktop 1440 × 1100 và bố cục 390 × 844; cả bốn màn hình di động không tràn ngang trang.

Bản Chromium headless dùng trong phiên không hỗ trợ thiết bị camera giả qua getUserMedia, nên phép quay dùng **MediaStream từ canvas được đưa vào có chủ đích**. MediaRecorder thực của trình duyệt vẫn tạo WebM; video được gửi vào bộ xử lý Python thật, nhận kết quả từ chối thiếu tay, ghi phiếu và tổng hợp mẫu số ngoài bộ từ. Các track đã được xác nhận dừng sau khi quay. Tình huống từ chối quyền camera được tạo chủ động trong bài kiểm tra. Không có xác nhận về camera vật lý, Windows hoặc phần cứng điện thoại từ các phép kiểm tra này.

Các tương tác đã đạt gồm bốn tab và bàn phím, xem MP4 thật, chọn/tải clip rồi nhận diện, mốc cắt sai, hủy kết quả cũ khi đổi mô hình/đoạn ký, bỏ qua phản hồi mạng đến muộn, ghi phiếu một lần và tải CSV. Không còn lỗi JavaScript chưa xử lý, lỗi CSP hoặc lỗi pattern HTML trong lần chạy cuối.

Trình duyệt thật phát hiện thuộc tính pattern của mã người/phiên chưa tương thích quy tắc UnicodeSets mới. Hai pattern đã được sửa để thoát dấu gạch ngang, và việc từ chối mã không hợp lệ đã được kiểm tra lại. Trên điện thoại, bảng so sánh được chuyển thành thẻ để số lượt nhận nhầm và tỷ lệ ngoài bộ từ luôn nhìn thấy.

Ảnh chụp màn hình chỉ được dùng để rà soát bố cục trong phiên kiểm tra, không nằm trong gói bàn giao.

## Chạy lại

Tại thư mục gốc, dùng môi trường có `requirements-research.txt` và FFmpeg:

```text
python tests/run_checks_v5.py
```

Phần trình duyệt là kiểm tra tùy chọn, không cần để dùng demo:

```text
python -m pip install playwright
python -m playwright install chromium
python tests/test_browser_v5.py
```

Có thể dùng một bản Chromium tin cậy đã cài với `--executable`. Tham số `--screenshots-dir` chỉ dành cho ảnh QA nội bộ. Trình duyệt và máy chủ được khởi động trong cùng tiến trình kiểm tra để truy cập localhost; các phiếu thật của người dùng không bị đọc hoặc xóa.
