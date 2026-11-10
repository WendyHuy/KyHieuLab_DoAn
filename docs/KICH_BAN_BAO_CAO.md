# Kịch bản trình bày đồ án Ký Hiệu Lab v6

## Trình bày trong khoảng 8 đến 10 phút

1. **Bài toán và phạm vi** khoảng 1 phút: nhận diện một clip ký hiệu riêng lẻ trong 5 nhãn; nêu đây là bản minh chứng kỹ thuật.
2. **Dữ liệu** khoảng 1 phút: 8 video thực, một người ký; hai nhãn có nhiều lượt, ba nhãn đơn mẫu. Không gọi các bản tăng cường là video thực mới.
3. **Phương pháp** khoảng 2 phút: trích điểm tay/thân bằng MediaPipe, tạo chuỗi 32 × 159, so sánh MLP, TCN gốc, TCN tăng cường và TCN học thêm nhóm ngoài bộ nhãn.
4. **Thực nghiệm** khoảng 2 phút: giải thích giữ nguồn riêng và ba seed. Nêu kết quả nền TCN gợi ý đúng 4/5 clip; khảo sát v5 nhận nhầm 14/29 clip ngoài bộ từ. Nghiên cứu phát triển v6 ghép cặp mới cho kết quả nhận nhầm 10/29 ở đối chứng và 3/29 ở ứng viên, nhưng ứng viên chỉ chấp nhận đúng 1/5 clip hợp lệ thay vì 3/5. TCN gốc tiếp tục mặc định. Hai số 14/29 và 10/29 dùng trọng số khác nhau, không so trực tiếp để tính mức cải thiện. Không cộng seed thành số video độc lập.
5. **Demo** khoảng 2 phút: mở một clip mẫu ở mục Nhận diện, xem nhãn gợi ý; đổi đoạn cắt hoặc mô hình rồi nhận diện lại. Qua mục Thực nghiệm ghi phiếu có mã người/phiên, xem thống kê và tải CSV. Mở mục Đánh giá để giải thích mẫu số của các kết quả. Ghi rõ clip mẫu đã được học. Nếu thử video mới, giữ nguyên cả trường hợp sai hoặc chưa chắc chắn.
6. **Kết luận** khoảng 1 phút: hệ thống chạy được ở phạm vi thăm dò; ưu tiên quay đủ nhiều người và phiên để đo khả năng áp dụng cho người mới.

## Chuẩn bị trước buổi báo cáo

- Giải nén gói và chạy `CHAY_DEMO.bat` khi có mạng để cài thư viện và chạy nhận diện ít nhất một clip trước; MediaPipe có thể tải mô hình ở lần trích điểm đầu.
- Chạy `tools/doctor.py`, thử MP4 mẫu, kiểm tra camera và quyền truy cập trình duyệt trên máy trình bày.
- Lưu báo cáo PDF và video mẫu trên máy. Mạng nơ-ron đã có trọng số, không cần huấn luyện trước lớp.
- Dùng thư mục đầu ra riêng cho tập chạy thử; không trộn phiếu luyện demo với phiếu dùng đánh giá mô hình.
- Điền tên môn, lớp, giảng viên, thành viên theo biểu mẫu thật của môn; không tự thêm thông tin chưa xác nhận.

## Câu hỏi thường gặp khi bảo vệ

**Đây có phải thuật toán mới không?** Phần đóng góp là bộ dữ liệu tự thu, cách ghép đặc trưng, ứng dụng và thực nghiệm đối chiếu. MediaPipe, MLP, tích chập thời gian và tăng cường dữ liệu là phương pháp đã có.

**Tại sao chỉ 5 nhãn?** Chốt phạm vi nhỏ để kiểm soát demo và trình bày đủ quy trình. Bộ 34 nhãn v3 thiếu mẫu lặp; bản v4 cũng còn ba nhãn chưa có mẫu kiểm tra độc lập.

**Có phải đạt 80 phần trăm cho cả hệ thống không?** Con số 4/5 của TCN ở seed chính là trên 5 clip thuộc hai nhãn, cùng người ký, giữ từng clip ngoài phần học. Không suy rộng cho cả 5 nhãn hoặc người mới. Trọng số demo học đủ 8 mẫu khác với các trọng số dùng từng lượt kiểm tra.

**Tại sao tăng cường dữ liệu không tốt hơn?** Đợt thử này chưa chứng minh lợi ích ổn định. Bộ gốc quá nhỏ; biến thể không tạo thêm người ký hoặc động tác thực mới. Giữ TCN gốc làm mặc định và báo cáo cả kết quả ứng viên thấp hơn.

**Máy có được học sâu thêm không?** Có. Bản v6 học TCN sáu đầu ra từ 37 nguồn cũ, gồm nhóm ngoài năm nhãn. Nghiên cứu phát triển có 30 lượt học theo nguồn giữ riêng và ba seed. Ứng viên giảm nhận nhầm ngoài bộ từ nhưng từ chối mẫu hợp lệ nhiều hơn nên chưa được chọn mặc định. Đây là thay đổi kết hợp lớp ngoài bộ từ và lấy mẫu cân bằng, chưa tách tác dụng từng phần.

**Tại sao ứng viên nhận lại đúng tất cả mẫu trong demo mà vẫn không chọn?** Nhận lại nguồn đã học chỉ là phát lại phần học. Quyết định dựa trên nguồn giữ ngoài phần học; kết quả ở đó cho thấy ứng viên giảm nhận đúng. Hai ví dụ ngoài bộ nhãn trong giao diện cũng là nguồn ứng viên đã học.

**Tại sao không học tăng cường?** Bài toán hiện có nhãn ký hiệu và được giải bằng học có giám sát. Phản hồi giao diện được lưu để xem lại; chưa có môi trường và hàm thưởng được kiểm chứng để dùng học tăng cường.

**Có dùng mặt không?** Holistic có thể trích điểm mặt nhưng vector của dự án chỉ dùng tay và 13 điểm thân trên. Đây là giới hạn đáng chú ý với các ký hiệu hoặc sắc thái cần nét mặt.

**Bước tiếp theo là gì?** Quay bộ 250 clip theo kế hoạch; chia riêng người/phiên cho học, chọn cấu hình và kiểm tra; bổ sung hoạt động ngoài bộ từ, hiệu chỉnh ngưỡng trên validation rồi kiểm tra cuối.

**Hệ thống đã hoàn thiện chưa?** Đã có phần mềm và hồ sơ nghiên cứu ở mức minh chứng đồ án. Nhận diện chưa được kiểm chứng đủ năm nhãn hoặc người mới; khả năng từ chối còn đánh đổi với nhận đúng. Bản v6 kiểm tra trình duyệt thật và luồng ghi phiếu, không dùng các kiểm tra phần mềm để thay phép đo mô hình trên dữ liệu thực mới.
