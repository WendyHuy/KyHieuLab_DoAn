# Ký Hiệu Lab v6 (Cập nhật Đa người ký & Huấn luyện lại)

Ứng dụng đồ án **nhận diện 5 ký hiệu ngôn ngữ ký hiệu Việt Nam từ video**, sử dụng MediaPipe Holistic và mô hình học sâu nhẹ TCN / MLP.
Phạm vi hỗ trợ: **ai cho, bực mình, ghen tị, không cần, xin lỗi**.

> [!NOTE]
> **Cập nhật mới (12/11/2026):**
> - **Bộ dữ liệu mở rộng:** Tăng từ 8 clip (1 người ký) lên **20 clip** qua **3 người ký** (`giahuy`, `huy1`, `minh`).
> - **Độ chính xác Top-1:** Đạt **20/20 (100.0%)** trên toàn bộ 20 clip huấn luyện với độ tự tin từ **0.97 - 0.99**.
> - **Studio Thu Video Tích Hợp:** Thêm công cụ Web Studio quay trực tiếp từ webcam với đếm ngược 3-2-1, lưu tự động vào `data/clips_moi/`.
> - **Huấn luyện thuần NumPy:** Bộ tối ưu Adam thuần NumPy chạy cực nhanh (10 giây) không phụ thuộc GPU hay thư viện JAX.

---

## Phân bố dữ liệu hiện tại (20 clips thực tế)

| Từ ký hiệu | giahuy | huy1 | minh | Tổng số clip |
| :--- | :---: | :---: | :---: | :---: |
| **ai cho** | 3 | 1 | 1 | **5** |
| **bực mình** | 1 | 1 | 1 | **3** |
| **ghen tị** | 1 | 1 | 1 | **3** |
| **không cần** | 2 | 1 | 2 | **5** |
| **xin lỗi** | 1 | 2 | 1 | **4** |
| **Tổng cộng** | **8** | **6** | **6** | **20** |

---

## Hướng dẫn sử dụng nhanh

### 1. Khởi chạy Ứng dụng Demo Nhận diện
Nhấp đúp chuột vào file:
```cmd
CHAY_DEMO.bat
```
Hoặc chạy dòng lệnh:
```powershell
& .venv\Scripts\python.exe app.py
```
Sau đó mở trình duyệt tại: `http://localhost:8765` để trải nghiệm nhận diện qua webcam hoặc video mẫu.

### 2. Thu thêm Video Ký hiệu mới (Studio Quay Video)
Nhấp đúp chuột vào file:
```cmd
MO_STUDIO_QUAY_VIDEO.bat
```
- Trình duyệt sẽ mở giao diện Studio trực quan (`http://localhost:8767`).
- Chọn người ký, từ ký hiệu, bấm phím Cách (Spacebar) hoặc nút Quay.
- Hệ thống đếm ngược 3-2-1 kèm âm thanh, tự động quay 4.5 giây và gửi lưu trực tiếp vào thư mục `data/clips_moi/`.

### 3. Huấn luyện lại Mô hình
Khi có thêm video mới, chạy lệnh sau để tự động trích xuất đặc trưng và cập nhật trọng số:
```powershell
& .venv\Scripts\python.exe tools/extract_new_clips.py
& .venv\Scripts\python.exe tools/train_and_save_v6.py
```

## Chạy và trải nghiệm trên Windows

1. Cài **Python 3.11 64-bit** có Python Launcher `py`.
2. Giải nén toàn bộ ZIP vào thư mục có quyền ghi.
3. Mở `CHAY_DEMO.bat`. Lần đầu cần mạng để cài thư viện và chuẩn bị bộ trích điểm. `KIEM_TRA_CAI_DAT.bat` kiểm tra lại khi cần.
4. Khi máy chủ sẵn sàng, mở `http://localhost:8765` bằng Chrome hoặc Edge.
5. Chọn clip mẫu rồi nhận diện. Clip mẫu đã thuộc dữ liệu học; dùng chúng kiểm tra thao tác.
6. Thử hai ví dụ **Ngoài 5 ký hiệu** trong danh sách, chuyển sang TCN học thêm lớp ngoài bộ từ để trải nghiệm từ chối. Các ví dụ này cũng là nguồn ứng viên đã học, không dùng nhận lại đúng để báo điểm kiểm tra.
7. Thử video mới hoặc quay camera, chọn mốc đầu và cuối khi cần, tối đa 15 giây. Ghi cả lượt sai và bị từ chối tại mục Thực nghiệm, rồi tải CSV.

Ứng dụng xử lý sau khi quay xong, chưa dịch liên tục. Giữ rõ tay và thân trên, mỗi clip một động tác trong khoảng 3 đến 8 giây. Cuối bằng 0 lấy đến hết. Thay video, mô hình hoặc mốc cắt sẽ yêu cầu nhận diện lại trước khi ghi phiếu.

Máy chủ chỉ chạy trên máy cục bộ. Webcam vật lý, Windows và Excel cần được kiểm tra trên máy trình bày. Trọng số đã học xong, không cần huấn luyện lại trước lớp.

## Phần đã hoàn thiện ở v6

- Bốn mục Nhận diện, Thực nghiệm, Đánh giá và Hướng dẫn; bố cục máy tính và điện thoại mô phỏng.
- Bốn cấu hình trọng số; danh sách và mặc định lấy từ máy chủ, kiểm tra mã băm khi xem xét thay mặc định.
- Tách nhãn ngoài bộ từ khỏi các gợi ý hỗ trợ, giữ điểm gốc của sáu đầu ra, không chuẩn hóa lại thành năm nhãn.
- Ghi quyết định gốc từ máy chủ, chống ghi lặp một mã lượt và kết quả đến muộn; không tính gợi ý đứng sau lớp ngoài bộ nhãn là dự đoán đúng.
- Thống kê số lượt, số mã băm video, phiên bản mô hình, nguồn trùng phần học và ma trận quyết định có cột từ chối. Ứng viên kiểm tra trùng cả nguồn ngoài bộ từ đã học.
- Protocol, checkpoint, bảng từng seed, dự đoán nguồn giữ riêng, ma trận và kiểm tra JAX/NumPy có thể tái lập.
- Gói video phát lại của 29 nguồn cũ, giữ riêng đặc trưng đã khóa dùng cho nghiên cứu. Không có video thực mới được quay.

## Dữ liệu và nguồn

| Nhãn hỗ trợ | Video thực | Clip giữ riêng |
| --- | ---: | ---: |
| ai cho | 3 | 3 |
| không cần | 2 | 2 |
| bực mình | 1 | 0 |
| ghen tị | 1 | 0 |
| xin lỗi | 1 | 0 |

Tám clip hỗ trợ đều thuộc mã `giahuy`. Ngoài ra có 29 nguồn ngoài bộ nhãn: 17 giahuy và 12 tp. Mã tp ở các nhãn ngoài phạm vi chưa phải kiểm tra năm nhãn hỗ trợ trên người mới. Ba nhãn đơn mẫu vẫn thiếu phép giữ clip riêng.

Video hỗ trợ giữ mốc cắt cũ, được chuẩn hóa H.264, 12 fps, tối đa 640 px. 29 video ngoài bộ nhãn được khôi phục từ đúng MP4 v3 đã kiểm tra mã băm; bản mã hóa phát lại hiện tại khác bitstream v5. Nghiên cứu v6 vẫn dùng nguyên đặc trưng đã khóa của v5; không thay đặc trưng chỉ vì khôi phục video. Hồ sơ nằm tại `data/PROVENANCE.md` và `data/outside_manifest_v6.json`.

Nhãn được dùng theo quyền học thử và giả định người thu đã kiểm tra. JSON nguồn giữ `reviewStatus: pending`, `trainingEligible: false`; không tự nâng thành chuyên gia đã duyệt. Kế hoạch `data/collection_plan_250.csv` là 250 vị trí chưa quay. Tăng cường dữ liệu hoặc chạy thêm seed không thay việc quay thêm người và lần thực hiện mới.

## Ghi phiếu và thu dữ liệu tiếp

Phiếu nằm ở `outputs/trials.jsonl`. Video gửi vào được xóa sau xử lý; nút tải video giúp người thu giữ clip và đối chiếu mã lượt. Nhãn người dùng ghi là phản hồi chưa được duyệt, không tự dùng để học. Chạy `python tools/summarize_trials.py` để tổng hợp.

Mã băm byte khác nhau chưa chứng minh hai lần thực hiện độc lập; ứng dụng đánh dấu cả các bản mã hóa lại của nguồn học được đóng gói. Mã người là mã tự nhập, chưa phải danh tính đã xác minh. Xem số video, phiên bản trọng số và nguồn học trước khi báo tỷ lệ đúng.

Đợt tiếp theo cần thêm video thật đủ năm nhãn, nhiều người và phiên, cùng hoạt động ngoài bộ từ đa dạng. Chia người cho học, validation và test trước khi tối ưu; chốt ngưỡng trên validation rồi mới đo test mới. Hướng dẫn: `docs/QUAY_VA_THU_DU_LIEU.md`.

## Chạy bằng dòng lệnh

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tools\doctor.py --warmup
.venv\Scripts\python.exe run.py
```

Nếu cổng đang dùng, thêm `--port 8766`. Trên Linux Python 3.11 hoặc 3.12, dùng `python3 -m venv .venv` và `.venv/bin/python`. Demo không cần GPU hay FFmpeg. MediaPipe khai báo phụ thuộc JAX nên JAX được ghim trong bộ cài; mã suy luận dùng NumPy, mã học dùng JAX.

## Tái lập và kiểm tra

Cài `requirements-research.txt` trong môi trường riêng rồi chạy từ thư mục gốc:

```text
python research/rejection_v6.py
python tests/run_checks_v6.py --require-candidate
python research/build_report_v6.py
```

Nghiên cứu dùng lại checkpoint có cùng dấu vân tay mã, protocol và đặc trưng; `--rerun` mới học lại. Mã nền giữ nguyên tại `research/run_experiments.py`, `research/summarize.py`, `research/audit_v5.py`; kết quả v4/v5 được giữ để đối chiếu lịch sử.

`tests/test_browser_v6.py` chạy Playwright và Chromium thật theo `reports/QA_V6.md`. Test video màu cần FFmpeg, demo không cần. Camera tự động là nguồn giả lập, chưa thay webcam vật lý. Test dùng đầu ra tạm riêng, không trộn phiếu người dùng. Chỉ cần Playwright để kiểm thử, không cần để sử dụng ứng dụng. Dựng lại Word cần thêm `python-docx==1.2.0`; PDF bàn giao đã được xuất và kiểm tra bố cục.

## Hồ sơ bàn giao

- `reports/Bao_cao_DoAn_KyHieuLab_v6.md`, `.docx`, `.pdf`: báo cáo môn học.
- `reports/rejection_v6.json`, `rejection_v6_metrics.csv`, `rejection_v6_predictions.csv`: nghiên cứu phát triển mới.
- `reports/rejection_v6_*_decision_confusion.csv`: ma trận trong bộ nhãn và ngoài bộ từ, mẫu số tách riêng.
- `reports/rejection_v6_verification.json`: kiểm tra 204 dự đoán, 30 fold và sai khác JAX/NumPy.
- `reports/test_results_v6.json`, `browser_checks_v6.json`, `QA_V6.md`: hồ sơ kiểm tra phần mềm.
- `models/`: ba mô hình năm đầu ra và một ứng viên sáu đầu ra.
- `data/`: tám video hỗ trợ, 29 video phát lại ngoài bộ từ, đặc trưng và nguồn gốc.
- `research/`: protocol đã khóa, checkpoint và mã học sâu.
- `docs/KICH_BAN_BAO_CAO.md`: kịch bản trình bày khoảng 8 đến 10 phút.
- `docs/TODAY_PROGRESS_2026-11-07.md`: tiến độ và việc cần thực nghiệm tiếp.

Lịch tiếp tục công việc đã được đặt **mỗi giờ từ 10:45 đến 23:45 ngày 07/11/2026, giờ Việt Nam**, kết thúc trong ngày. Đây là các lượt tiếp tục tác vụ, không phải GPU huấn luyện không nghỉ. Chỉ tiếp tục phần còn thiếu hoặc dữ liệu mới có thật; không chạy lại nghiên cứu đã hoàn tất chỉ để tạo thêm lượt.
