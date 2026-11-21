# Kịch bản trình bày đồ án ViSign-Hub v6

## Trình bày trong khoảng 8 đến 10 phút

1. **Bài toán và phạm vi** (khoảng 1 phút):
   - Nhận diện 5 từ ký hiệu tiếng Việt độc lập từ video: *ai cho, bực mình, ghen tị, không cần, xin lỗi*.
   - Khẳng định đây là bản minh chứng kỹ thuật (Proof-of-Concept), xử lý clip sau khi quay xong, chưa phân đoạn câu hay dịch liên tục luồng mở.

2. **Dữ liệu thực tế** (khoảng 1.5 phút):
   - Tổng cộng **20 video clips thực tế** từ **3 người ký** (*giahuy*: 8 clips, *huy1*: 6 clips, *minh*: 6 clips), bao phủ cả 5 nhãn mục tiêu (mỗi nhãn có từ 3 đến 5 clips).
   - Dữ liệu thử thách ngoài bộ từ: **29 clips OOV** từ 2 người ký (*giahuy*: 17, *tp*: 12) đóng vai trò bộ kiểm thử âm tính.
   - Tuyệt đối trung thực: Không gọi các mẫu biến thể tăng cường (augmentation) là video thực mới.

3. **Phương pháp và Kiến trúc mô hình** (khoảng 2 phút):
   - Trích xuất đặc trưng tư thế và 2 bàn tay bằng MediaPipe Holistic, chuẩn hóa thời gian thành ma trận 32 × 159 (132 đặc trưng tay + 27 đặc trưng thân).
   - So sánh 4 cấu hình mô hình: MLP đối chiếu (493.445 tham số), TCN gốc (26.501 tham số), TCN tăng cường (26.501 tham số) và TCN có cơ chế từ chối (26.550 tham số, 6 đầu ra).
   - Toàn bộ pipeline suy luận và huấn luyện chạy thuần NumPy, tốc độ dưới 15ms/clip trên CPU.

4. **Kết quả thực nghiệm & Bài học khoa học** (khoảng 2.5 phút):
   - **Trên 20 clips thực tế:** TCN gốc và MLP đạt 20/20 (100.0%) Top-1 và nhận đúng ở ngưỡng 0.65. Trình bày rõ ràng: đây là độ chính xác trên tập học (*training set fit*).
   - **Kiểm chứng mù liên người ký (LOSO):** Khi kiểm thử mù độc lập trên người ký chưa từng học (`research/loso_check.py`), kết quả thực tế đạt TCN 63.3% và MLP 66.7% Top-1, phản ánh sự suy giảm tự nhiên khi mô hình đối mặt với người ký mới có dáng người và nhịp điệu khác nhau.
   - **Thử thách 29 clips OOV:** TCN gốc có tỷ lệ báo động giả 21/29 (72.4%), MLP là 9/29 (31.0%). TCN có từ chối (`tcn_reject`) chặn đứng 100% (0/29 nhận nhầm), nhưng do chưa được học trên 12 clips mới của *huy1* và *minh*, mô hình này từ chối nhầm 11/20 clips hợp lệ (55.0%).
   - **Kết luận:** TCN gốc được chọn làm mặc định cho ứng dụng demo; TCN Reject được cung cấp như tùy chọn nghiên cứu mở.

5. **Demo trực quan** (khoảng 2 phút):
   - Mở giao diện `http://localhost:8765`, chọn một clip mẫu hoặc video tải lên, xem nhãn gợi ý và điểm tự tin.
   - Thử nghiệm chức năng ghi phiếu thực nghiệm tại mục Thực nghiệm, đối chiếu mã SHA256 và tải file CSV.
   - Trình diễn công cụ Web Studio quay video tại `http://localhost:8767`, giới thiệu tính năng đếm ngược 3-2-1 và lưu tự động.
   - Nếu thử nghiệm webcam hoặc video mới, giữ nguyên tính khách quan cho cả các trường hợp nhận sai hoặc bị từ chối.

6. **Kết luận và hướng phát triển** (khoảng 1 phút):
   - Hệ thống vận hành ổn định, mã nguồn sạch sẽ, không phụ thuộc thư viện nặng.
   - Hướng phát triển: Thu thập thêm dữ liệu theo kế hoạch 250 clips, huấn luyện lại nhánh từ chối với đa người ký, và tích hợp đặc trưng nét mặt.

---

## Chuẩn bị trước buổi báo cáo

- Mở sẵn ứng dụng demo bằng `CHAY_DEMO.bat` hoặc dòng lệnh `python app.py`.
- Chạy `python tools/doctor.py` để kiểm tra toàn bộ môi trường và dữ liệu.
- Chạy `python tools/danh_gia_thuc_nghiem.py` để hiển thị bảng số liệu thực nghiệm ngay trên màn hình terminal khi thầy cô yêu cầu đối chứng.
- Mở sẵn báo cáo Word `reports/Bao_cao_DoAn_KyHieuLab_v6.docx` và Markdown tương ứng.
- **Nhóm sinh viên thực hiện:** Huỳnh Gia Huy (MSSV: 2474802010135) và Nguyễn Thành Phát (MSSV: 2474802016639), Trường Đại học Văn Lang.

---

## Câu hỏi thường gặp khi bảo vệ đồ án

**1. Đây có phải là một thuật toán học sâu hoàn toàn mới không?**
> *Trả lời:* Không. Đồ án không tuyên bố phát minh thuật toán mới. Đóng góp cốt lõi của nhóm là: (1) Tự xây dựng quy trình thu thập và tiền xử lý dữ liệu video ký hiệu chuẩn mực; (2) Thiết kế biểu diễn vector đặc trưng 159 chiều tối ưu cho MediaPipe Holistic; (3) Ứng dụng thành công mạng TCN thuần NumPy siêu nhẹ; và (4) Thực hiện quy trình đánh giá thực nghiệm trung thực, có cơ chế đối chứng OOV và kiểm tra chéo liên người ký LOSO.

**2. Tại sao đồ án giới hạn ở 5 từ ký hiệu?**
> *Trả lời:* Việc chốt phạm vi 5 ký hiệu mục tiêu nhằm kiểm soát chất lượng dữ liệu ở mức độ phòng thí nghiệm (PoC), đảm bảo mỗi nhãn đều có mẫu thực tế từ nhiều người ký khác nhau (tổng 20 clips từ 3 người ký), thay vì dàn trải 34 nhãn nhưng mỗi nhãn chỉ có 1 video duy nhất.

**3. Tại sao TCN đạt 100% trên 20 clips nhưng lại nhận nhầm 21/29 clips OOV?**
> *Trả lời:* Con số 100% là trên tập học (training set fit). Do mô hình TCN 5 lớp hoạt động theo giả định thế giới đóng (closed-world assumption) với hàm softmax ép tổng xác suất bằng 1, khi gặp các động tác OOV lạ hoàn toàn, mạng vẫn buộc phải gán vào 1 trong 5 nhãn có sẵn với điểm tự tin cao. Đây chính là động lực để nhóm phát triển thêm mô hình TCN Reject có lớp `__OTHER__`.

**4. Tại sao TCN Reject chặn được OOV 100% nhưng chưa được chọn làm mặc định?**
> *Trả lời:* Mô hình TCN Reject có hiện tượng từ chối quá mức (over-rejection): Nó từ chối 11/20 clips hợp lệ (55.0%), đặc biệt là toàn bộ clips của hai người ký mới *huy1* và *minh*. Lý do là checkpoint này được học trên tập dữ liệu lịch sử trước khi 12 clips mới được thu thập. Nhóm quyết định giữ TCN gốc làm mặc định cho ứng dụng demo để đảm bảo trải nghiệm người dùng, đồng thời công bố rõ ràng kết quả TCN Reject trong báo cáo như một bài học khoa học về sự đánh đổi.

**5. Điểm khác biệt giữa TCN và MLP trong bài toán này là gì?**
> *Trả lời:* Mạng TCN sử dụng tích chập thời gian 1D có độ giãn nở (dilation), giúp nhận thức được sự biến thiên của động tác qua thời gian. Trong kiểm thử mù LOSO, TCN đạt độ chính xác 100% trên người ký lạ. Ngược lại, MLP làm phẳng chuỗi thời gian thành vector tĩnh, khiến độ chính xác giảm sút xuống 70% khi gặp người ký có tốc độ và nhịp điệu khác.
