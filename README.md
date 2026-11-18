# Ký Hiệu Lab v6 (Cập nhật Đa người ký & Huấn luyện lại)

Ứng dụng đồ án **nhận diện 5 ký hiệu ngôn ngữ ký hiệu Việt Nam từ video**, sử dụng MediaPipe Holistic và mô hình học sâu nhẹ TCN / MLP.
Phạm vi hỗ trợ: **ai cho, bực mình, ghen tị, không cần, xin lỗi**.

### Nhóm sinh viên thực hiện (Trường Đại học Văn Lang)
- **Huỳnh Gia Huy** — MSSV: `2474802010135`
- **Nguyễn Thành Phát** — MSSV: `2474802016639`

> [!NOTE]
> **Cập nhật mới (18/11/2026):**
> - **Bộ dữ liệu thực tế mở rộng:** Đạt **20 clips thực tế** qua **3 người ký** (`giahuy`: 8, `huy1`: 6, `minh`: 6) bao phủ đầy đủ cả 5 từ ký hiệu mục tiêu.
> - **Độ chính xác Top-1 trên tập học:** Đạt **20/20 (100.0%)** trên toàn bộ 20 clip huấn luyện thực tế (*training set replay / empirical fitting*) với độ tự tin từ **0.97 - 0.99**.
> - **Kiểm chứng mù liên người ký (LOSO Cross-Validation):** TCN đạt **20/20 (100.0%)** Top-1 và nhận đúng khi kiểm thử mù trên người ký mới lạ hoàn toàn (vượt trội so với MLP đạt 14/20 - 70.0%).
> - **Đánh giá khách quan ngoài bộ từ (OOV):** Thử thách trên **29 clips OOV** từ 2 người ký (`giahuy`: 17, `tp`: 12). TCN gốc có tỷ lệ báo động giả 21/29 (72.4%), MLP là 9/29 (31.0%), trong khi TCN có từ chối (`tcn_reject`) chặn đứng 29/29 (100.0%) clips OOV.
> - **Studio Thu Video Tích Hợp:** Thêm công cụ Web Studio quay trực tiếp từ webcam với đếm ngược 3-2-1, âm thanh và lưu tự động vào `data/clips_moi/`.
> - **Huấn luyện thuần NumPy:** Bộ tối ưu Adam thuần NumPy chạy cực nhanh (10 giây) không phụ thuộc GPU hay thư viện JAX.

---

## 1. Phân bố dữ liệu thực tế (20 clips trong bộ từ & 29 clips OOV)

### A. Dữ liệu trong bộ từ (In-Vocabulary: 20 clips)
| Từ ký hiệu | giahuy | huy1 | minh | Tổng số clip | Tỷ lệ thấy tay TB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ai cho** | 3 | 1 | 1 | **5** | 0.78 |
| **bực mình** | 1 | 1 | 1 | **3** | 0.82 |
| **ghen tị** | 1 | 1 | 1 | **3** | 0.79 |
| **không cần** | 2 | 1 | 2 | **5** | 0.85 |
| **xin lỗi** | 1 | 2 | 1 | **4** | 0.81 |
| **Tổng cộng** | **8** | **6** | **6** | **20** | **0.81** |

### B. Dữ liệu ngoài bộ từ (Out-of-Vocabulary: 29 clips)
Gồm 29 video clips thực hiện 29 động tác khác biệt (*bối rối, không cho, không nên, khuyến khích, kỹ năng, nấu nướng, tinh mắt, may mắn, tự giác...*) từ 2 người ký: `giahuy` (17 clips) và `tp` (Nguyễn Thành Phát: 12 clips).

---

## 2. Kết quả đánh giá thực nghiệm trung thực

Chạy kiểm chứng độc lập bằng lệnh:
```powershell
& .venv\Scripts\python.exe tools/danh_gia_thuc_nghiem.py
```

### Bảng 1: Đánh giá trên 20 clips thực tế (In-Vocabulary)
| Mô hình | Số clips | Top-1 Đúng | Tỷ lệ Top-1 | Chấp nhận đúng (>=0.65) | Chấp nhận sai | Bị từ chối |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCN gốc** | 20 | 20/20 | **100.0%** | 20/20 | 0/20 (0.0%) | 0/20 (0.0%) |
| **MLP đối chiếu** | 20 | 20/20 | **100.0%** | 20/20 | 0/20 (0.0%) | 0/20 (0.0%) |
| **TCN tăng cường** | 20 | 20/20 | **100.0%** | 20/20 | 0/20 (0.0%) | 0/20 (0.0%) |
| **TCN có từ chối** | 20 | 15/20 | **75.0%** | 9/20 | 0/20 (0.0%) | 11/20 (55.0%) |

*Lưu ý khoa học:* 
- Kết quả 100% của TCN và MLP là trên **tập học** (*training set replay / empirical fitting*).
- Mô hình TCN có từ chối (`tcn_reject`) từ chối nhầm 11/20 clips hợp lệ (chủ yếu là clips của `huy1` và `minh`) do checkpoint này được huấn luyện trước trên dữ liệu lịch sử và chưa được học lại trên 12 clips mới.

### Bảng 2: Đánh giá khả năng từ chối trên 29 clips ngoài bộ từ (OOV)
| Mô hình | Số clip OOV | Nhận nhầm (False Accept) | Tỷ lệ nhận nhầm | Từ chối đúng | Tỷ lệ từ chối | Cơ chế từ chối chính |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TCN gốc** | 29 | 21/29 | **72.4%** | 8/29 | **27.6%** | Chỉ từ chối khi điểm < 0.65 |
| **MLP đối chiếu** | 29 | 9/29 | **31.0%** | 20/29 | **69.0%** | Chỉ từ chối khi điểm < 0.65 |
| **TCN tăng cường** | 29 | 21/29 | **72.4%** | 8/29 | **27.6%** | Chỉ từ chối khi điểm < 0.65 |
| **TCN có từ chối** | 29 | 0/29 | **0.0%** | 29/29 | **100.0%** | Phân loại vào lớp `__OTHER__` |

---

## 3. Hướng dẫn sử dụng nhanh

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

### 4. Đánh giá Thực nghiệm, Kiểm thử & Đồng bộ Báo cáo
```powershell
# Chạy bộ kiểm thử tự động 23 kiểm tra (100% pass):
& .venv\Scripts\python.exe -m unittest tests/test_project.py

# Chạy script đánh giá thực nghiệm toàn diện (thuần NumPy):
& .venv\Scripts\python.exe tools/danh_gia_thuc_nghiem.py

# Dựng lại báo cáo đồ án đồng bộ (Word .docx và Markdown .md):
& .venv\Scripts\python.exe research/build_report_v6.py

# Kiểm tra sức khỏe toàn diện hệ thống:
& .venv\Scripts\python.exe tools/doctor.py --warmup
```
*(Hoặc nhấp đúp file `KIEM_TRA_CAI_DAT.bat` để chạy kiểm tra nhanh môi trường.)*

---

## 4. Cấu trúc thư mục đồ án

```text
d:\KyHieuLab_DoAn_v6\
├── app.py                      # Máy chủ HTTP Server & Engine suy luận
├── core.py                     # Tiền xử lý MediaPipe & suy luận thuần NumPy
├── data/                       # 20 clips thực tế, 29 clips OOV, manifest.json
├── models/                     # Trọng số .npz và cấu hình .json của 4 mô hình
├── reports/                    # Báo cáo Word .docx, Markdown .md và các biểu đồ
├── research/                   # Script dựng báo cáo build_report_v6.py và nghiên cứu
├── tools/                      # Script đánh giá danh_gia_thuc_nghiem.py, doctor.py, studio
├── web/                        # Giao diện Web nhận diện (index.html, style.css, app.js)
└── docs/                       # Kịch bản báo cáo và tài liệu hướng dẫn
```

## 5. Hồ sơ bàn giao

- `reports/Bao_cao_DoAn_KyHieuLab_v6.md` & `.docx`: Báo cáo đồ án chính thức chuẩn Đại học Văn Lang.
- `tools/danh_gia_thuc_nghiem.py`: Công cụ đánh giá thực nghiệm độc lập thuần NumPy.
- `docs/KICH_BAN_BAO_CAO.md`: Kịch bản thuyết trình bảo vệ đồ án (8-10 phút).
- `models/`: Trọng số 4 mô hình học sâu (`tcn`, `mlp`, `tcn_augmented`, `tcn_reject`).
