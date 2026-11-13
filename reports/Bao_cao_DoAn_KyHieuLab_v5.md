# Nhận diện ký hiệu tiếng Việt từ video

Báo cáo đồ án thử nghiệm Ký Hiệu Lab phiên bản 5

Nhóm sinh viên thực hiện:
- Huỳnh Gia Huy | MSSV: 2474802010135 | Trường Đại học Văn Lang
- Nguyễn Thành Phát | MSSV: 2474802016639 | Trường Đại học Văn Lang

Ngày cập nhật và kiểm tra 04/11/2026

### Tóm tắt dự án

Dự án xây dựng ứng dụng gợi ý một ký hiệu riêng lẻ từ đoạn video đã quay. Bản v5 kế thừa ba cấu hình học trên tám video của v4, thiết kế lại trải nghiệm nhận diện và ghi phiếu, chạy kiểm tra trình duyệt thật, đồng thời bổ sung khảo sát 29 video có nhãn ngoài bộ năm ký hiệu. Mã nguồn, trọng số, ma trận và hồ sơ kiểm tra được bàn giao cùng quy trình tái lập.

Bản này có thể dùng trình bày quy trình kỹ thuật và thực nghiệm có giám sát. Dữ liệu học vẫn chỉ có một người ký và ba nhãn đơn mẫu; phép giữ riêng chỉ bao phủ năm clip của hai nhãn. Khảo sát ngoài bộ từ phát hiện TCN gốc vẫn chấp nhận nhầm 14/29 clip. Vì vậy kết quả cần được đọc là gợi ý trong năm nhãn, chưa đủ để xác nhận đầu vào thật sự thuộc bộ từ hoặc mở nhận diện công khai.

| Phương pháp | Gợi ý đầu đúng | Chấp nhận đúng | Chấp nhận sai |
| --- | --- | --- | --- |
| MLP | 4/5 | 2/5 | 0 |
| TCN gốc | 4/5 | 3/5 | 0 |
| TCN tăng cường | 3/5 | 2/5 | 1 |

Bảng dùng seed chính 20261103, với ngưỡng 0,65 giữ cố định trước thực nghiệm. TCN gốc là mặc định; ứng viên tăng cường được giữ để so sánh vì chưa cải thiện ổn định. Mỗi mô hình còn được chạy thêm hai seed trên cùng năm clip; số lượt lặp không làm tăng số video kiểm tra độc lập.

Đóng góp của đồ án gồm quy trình xử lý video, cách ghép đặc trưng tay và thân, ứng dụng nhận diện có cơ chế chưa chắc chắn, cùng thực nghiệm so sánh tái lập. MediaPipe, MLP và mạng tích chập thời gian là các phương pháp đã có; đồ án không tuyên bố phát minh thuật toán mới.

## 1 Bài toán và dữ liệu thực hiện

Đầu vào là video một người thực hiện một động tác, hoặc một đoạn được chọn trong video. Đầu ra gồm nhãn gợi ý, điểm của ba nhãn đứng đầu và trạng thái chấp nhận hay chưa chắc chắn. Người kiểm tra có thể ghi nhãn thực tế, kể cả ngoài bộ từ, để tổng hợp các lượt sử dụng. Hệ thống xử lý sau khi kết thúc clip, chưa giải quyết phân đoạn câu và dịch liên tục.

| Nhãn | Video thực | Clip có thể giữ riêng | Trạng thái đánh giá |
| --- | --- | --- | --- |
| ai cho | 3 | 3 | Hồi cứu cùng người |
| không cần | 2 | 2 | Hồi cứu cùng người |
| bực mình | 1 | 0 | Chưa có mẫu kiểm tra |
| ghen tị | 1 | 0 | Chưa có mẫu kiểm tra |
| xin lỗi | 1 | 0 | Chưa có mẫu kiểm tra |

### Nguồn và lựa chọn phạm vi

Bộ v3 có 39 video, trong đó 37 video được dùng học thử cho 34 nhãn và hai video thiếu thông tin tiếp tục ở trạng thái chờ. Bản v4 chọn hai nhãn có mẫu lặp và ba nhãn minh họa có tỷ lệ thấy tay cao theo hồ sơ. Danh sách được ghi trước khi huấn luyện v4. Tám video đều thuộc mã giahuy; số video thực mới trong đợt này là 0.

Gói v4 chứa video và đặc trưng của nhóm năm nhãn. Hồ sơ 39 video và JSON gốc được giữ để đối chiếu; gói đầy đủ v3 được lưu riêng. Việc giảm từ 34 xuống năm nhãn làm thay đổi bài toán, nên chênh lệch điểm v3 và v4 không được dùng như mức cải thiện trên cùng bộ kiểm tra.

### Quy ước nhãn và thông tin duyệt

Động tác được dùng theo giả định người thu đã đối chiếu nhãn. Các trường reviewStatus và trainingEligible trong hồ sơ gốc được giữ nguyên; học thử theo yêu cầu không biến dữ liệu thành đã được chuyên gia duyệt. Các biến thể cách 1 và cách 2 của những nhãn ngoài phạm vi không được tự gộp vào năm nhãn đang học.

## 2 Đặc trưng và mô hình nhận diện

MediaPipe Holistic được dùng để ước lượng bàn tay và tư thế từ hình ảnh [1]. Bộ đặc trưng của dự án lấy 21 điểm mỗi bàn tay và 13 điểm thân trên. Điểm mặt không đi vào mô hình; đây là giới hạn với các ký hiệu hoặc sắc thái cần thông tin nét mặt.

| Thành phần | Số đặc trưng | Cách biểu diễn |
| --- | --- | --- |
| Một bàn tay | 66 | 63 tọa độ hình dạng, 2 tọa độ vị trí và 1 mặt nạ |
| Hai bàn tay | 132 | Giữ riêng tay trái và tay phải |
| Thân trên | 27 | 13 điểm XY và 1 mặt nạ |
| Một bước thời gian | 159 | 132 cộng 27 |
| Một đoạn ký | 32 × 159 | Lấy mẫu lại theo thời gian, có xét mặt nạ |

Hình dạng tay được đưa về gốc cổ tay và chia theo kích thước bàn tay. Vị trí cổ tay và thân được tính trong mặt phẳng XY theo tâm hai vai và khoảng cách vai. Tỷ lệ khung hình được tính vào trục ngang. Không trộn tọa độ sâu của bàn tay với hệ tọa độ sâu của tư thế. Điểm thiếu được ghi bằng mặt nạ; nội suy không tự biến đoạn mất tay dài thành tay hợp lệ.

| Mạng | Cấu trúc chính | Tham số |
| --- | --- | --- |
| MLP | Làm phẳng 5.088 → 96 → 48 → 5 | 493445 |
| TCN gốc | 159 → 48; 2 khối tích chập có nối tắt; pooling; 48 → 5 | 26501 |
| TCN tăng cường | Cùng kiến trúc TCN, thay cách tạo và lấy mẫu phần học | 26501 |

Mạng tích chập thời gian tham khảo hướng mô hình hóa chuỗi bằng tích chập [2]. Bản triển khai dùng kernel 3, dilation 1 và 2, padding SAME, nối tắt và gộp trung bình cùng cực đại ở các bước có tay. Mạng nhìn cả đoạn đã quay; không phải bản TCN nhân quả để suy luận luồng trực tiếp. Hàm kích hoạt ReLU, đầu ra softmax năm nhãn.

## 3 Xử lý dữ liệu và cấu trúc ứng dụng

### Chuẩn hóa video và trích điểm

Tám clip giữ các mốc nội dung đã cắt ở v3, được chuyển sang H.264, yuv420p, 12 khung hình mỗi giây, cạnh dài tối đa 640 pixel và bỏ âm thanh. Không cắt thêm phần động tác theo dự đoán của mô hình. Một clip bực mình có tốc độ khai báo 1.000 fps và 4.576 khung; bản chuẩn hóa còn 55 khung, giữ thời lượng xấp xỉ đoạn gốc. Nhật ký chuyển đổi lưu trong video_normalization_audit.json.

Đặc trưng được trích lại từ chính MP4 đã chuẩn hóa, thay vì gắn đặc trưng cũ cho video mới. Chuỗi được đưa về 32 bước; thống kê chuẩn hóa được học riêng từ phần học ở mỗi lượt thực nghiệm. Khi nhận diện, nếu ít hơn tám bước có tay thì hệ thống từ chối. Nếu có đủ tay nhưng điểm cao nhất dưới 0,65, kết quả là chưa chắc chắn.

### Luồng xử lý ứng dụng

| Bước | Xử lý |
| --- | --- |
| 1 | Trình duyệt chọn video, quay 5 giây hoặc mở clip mẫu |
| 2 | Người dùng chọn mốc đầu/cuối và mô hình |
| 3 | Máy chủ kiểm tra dữ liệu, trích điểm đoạn đã chọn và chạy trọng số |
| 4 | Trả nhãn, điểm, tỷ lệ thấy tay và thời gian xử lý |
| 5 | Người kiểm tra ghi nhãn thực tế theo mã lượt; tải phiếu CSV |

Máy chủ chỉ lắng nghe tại 127.0.0.1. Phép phân loại dùng NumPy; demo không cần GPU hay FFmpeg. JAX được ghim trong bộ cài vì MediaPipe khai báo phụ thuộc này; mã huấn luyện dùng JAX trực tiếp. Video gửi vào được đặt trong thư mục tạm và xóa sau xử lý. Có nút tải video đang chọn để người thu chủ động giữ clip cho lần bổ sung dữ liệu.

Phiếu lấy kết quả gốc từ máy chủ bằng inference_id. Trình duyệt không tự cung cấp điểm hoặc nhãn dự đoán để ghi đè. Đổi video, mốc cắt hay mô hình sẽ hủy kết quả đang chờ ghi; một mã lượt chỉ ghi một lần. Phiếu phản hồi chưa được duyệt và không tự cập nhật trọng số.

Giao diện v5 có bốn mục Nhận diện, Thực nghiệm, Đánh giá và Hướng dẫn. Tổng hợp phiếu tách số lượt với số mã băm video, hiển thị lượt trùng video học và các phiên bản trọng số; có lựa chọn ghi ngoài bộ ký hiệu. Ma trận quyết định dùng cột từ chối để phản ánh kết quả thực trả về. Mã người do người dùng nhập chưa phải danh tính đã được xác minh.

## 4 Thiết kế thực nghiệm

Thực nghiệm cố định ba phương pháp và ba seed trước khi học. Mỗi trong năm clip có mẫu lặp được giữ riêng; mô hình học trên bảy video còn lại rồi dự đoán clip giữ riêng. Mẫu tăng cường và trung bình, độ lệch chuẩn chỉ được tạo từ bảy video này. Không chia khung của cùng video qua hai phần [3].

| Thông số | Giá trị |
| --- | --- |
| Seed chính và hai seed lặp | 20261103, 20261104, 20261105 |
| Số bước và batch | 500 bước, batch 64 |
| Bộ tối ưu | Adam, learning rate 0,001 |
| Điều chuẩn | L2 hệ số 0,00001; label smoothing 0,02 |
| Tăng cường cơ sở | 20 biến thể thời gian và nhiễu nhẹ cho mỗi video học |
| Biến thể TCN tăng cường | Lấy mẫu cân bằng nhãn, biến thiên mốc cắt và mất tay ngắn |
| Ngưỡng chấp nhận | 0,65; chưa hiệu chỉnh từ validation |
| Đơn vị giữ riêng | Video nguồn, đối chiếu mã và SHA256 |

Có 3 phương pháp × 3 seed × 5 clip = 45 lượt học giữ clip riêng. Thêm ba lượt học trên đủ tám video để tạo trọng số demo. Checkpoint lưu mã video học, seed, danh sách video học, thống kê chuẩn hóa và lịch sử loss. Dấu vân tay dữ liệu và mã giúp tránh dùng lại kết quả cũ khi đầu vào thay đổi.

### Chỉ số và phạm vi suy luận

Báo cáo số gợi ý đầu đúng, Top 3, Precision, Recall, F1 từng nhãn, ma trận số đếm và số lượt chấp nhận đúng hoặc sai. Macro F1 chỉ lấy trung bình trên ai cho và không cần, là hai nhãn có mẫu kiểm tra [4]. Ba hàng không có mẫu trong CSV đầy đủ không phải bằng chứng mô hình không mắc lỗi ở ba nhãn đó.

Ba seed dùng lại cùng năm clip, chỉ đo sự thay đổi khi khởi tạo và học; không tạo thành 15 clip độc lập. Đây là kiểm tra hồi cứu với dữ liệu đã biết, cùng một người ký. Chưa có tập validation riêng để chọn ngưỡng, test mù mới, hoặc test người ký mới. Khi có nhiều người, cần chia theo người và phiên [5].

## 5 Kết quả ở seed chính

*Hình 1 Ma trận gợi ý đầu trên năm clip giữ ngoài phần học*. Hình trong gói: `reports/figures/confusion_primary.png`.

| Mô hình | Đúng | Top 3 đúng | Macro F1 | Nhận đúng | Nhận sai |
| --- | --- | --- | --- | --- | --- |
| MLP | 4/5 | 5/5 | 0.800 | 2 | 0 |
| TCN gốc | 4/5 | 5/5 | 0.900 | 3 | 0 |
| TCN tăng cường | 3/5 | 5/5 | 0.429 | 2 | 1 |

*Hình 2 Ma trận có cơ chế từ chối ở ngưỡng 0,65*. Hình trong gói: `reports/figures/confusion_reject_primary.png`.

Hàng là nhãn người thu cung cấp; cột là dự đoán. Cột khác gộp ba nhãn không có mẫu giữ riêng. TCN gốc gợi ý đầu đúng 4/5 clip, chấp nhận ba lượt đúng và từ chối hai lượt. Đây là số đo nhỏ ở seed chính; không suy ra độ chính xác tổng quát bằng 80% hoặc độ chính xác chấp nhận bằng 100%.

## 6 Độ ổn định và phân tích lỗi

| Mô hình | Seed 03 | Seed 04 | Seed 05 | Trung bình đúng |
| --- | --- | --- | --- | --- |
| MLP | 4/5 | 5/5 | 3/5 | 80.0% |
| TCN gốc | 4/5 | 5/5 | 4/5 | 86.7% |
| TCN tăng cường | 3/5 | 5/5 | 3/5 | 73.3% |

TCN gốc cho kết quả 4/5, 5/5 và 4/5; mức trung bình 86,7% chỉ là trung bình ba lần học trên cùng năm clip. Ở seed thứ ba, TCN gốc có một lượt chấp nhận sai. TCN tăng cường đạt 3/5, 5/5 và 3/5, chưa cho thấy lợi ích ổn định. Mô hình mặc định tiếp tục là TCN gốc đã chốt trước khi đo.

| Mô hình | Nhãn | Precision | Recall | F1 | Mẫu |
| --- | --- | --- | --- | --- | --- |
| MLP | ai cho | 1.000 | 0.667 | 0.800 | 3 |
| MLP | không cần | 0.667 | 1.000 | 0.800 | 2 |
| TCN gốc | ai cho | 1.000 | 0.667 | 0.800 | 3 |
| TCN gốc | không cần | 1.000 | 1.000 | 1.000 | 2 |
| TCN tăng cường | ai cho | 0.750 | 1.000 | 0.857 | 3 |
| TCN tăng cường | không cần | 0.000 | 0.000 | 0.000 | 2 |

### Trường hợp sai và cách đọc chỉ số

MLP: 48f69b84 ai cho → không cần.

TCN gốc: 95907815 ai cho → bực mình.

TCN tăng cường: 3d328846 không cần → xin lỗi; 4c5f6aa4 không cần → ai cho.

MLP và TCN gốc cùng đúng bốn clip nhưng Macro F1 hai nhãn khác nhau vì nhãn nhận nhầm khác nhau. Bộ kiểm tra quá nhỏ để khẳng định ưu thế có ý nghĩa thống kê. Tăng cường dữ liệu tạo biến thể học, chưa thay thế được người ký và lần thực hiện mới. Loss thấp hoặc nhận lại đúng mẫu học không đủ chứng minh khả năng áp dụng rộng.

## 7 Khảo sát động tác ngoài bộ từ

Đợt bổ sung chọn trước toàn bộ 29 video đủ hồ sơ học thử từ v3 có nhãn nằm ngoài năm nhãn của v4: 17 clip giahuy và 12 clip tp. Chúng không thuộc tám nguồn học của trọng số demo. Video được chuẩn hóa theo cùng quy tắc trước khi trích điểm; không có video thực mới được quay trong đợt này.

| Mô hình | Chấp nhận nhầm | Từ chối | Tỷ lệ trong 29 clip |
| --- | --- | --- | --- |
| MLP | 12 | 17 | 41.4% |
| TCN gốc | 14 | 15 | 48.3% |
| TCN tăng cường | 10 | 19 | 34.5% |

*Hình 3 Khảo sát hồi cứu 29 clip ngoài bộ năm nhãn*. Hình trong gói: `reports/figures/evaluation_v5_outside_vocabulary.png`.

Có 29/29 video được xử lý, không có clip lỗi hoặc bị bỏ qua. Trọng số học đủ tám video và ngưỡng 0,65 được khóa trước các dự đoán này. Mẫu thuộc hồ sơ cũ, trạng thái duyệt vẫn pending; đây là khảo sát tiện có, chưa phải test mù hoặc tỷ lệ lỗi đại diện cho mọi hoạt động ngoài bộ từ. Mã tp ở các nhãn ngoài phạm vi không tạo thành phép kiểm tra nhận diện nhãn được hỗ trợ trên người mới.

Phép giữ clip riêng sử dụng trọng số học trên bảy video, còn khảo sát ngoài bộ từ dùng trọng số demo học đủ tám video. Hai phần không được gộp thành một accuracy hay một ma trận tổng thể. Không tăng ngưỡng hoặc đổi mô hình theo kết quả vừa xem; cần bộ validation và test ngoài bộ từ tách trước để chọn và kiểm tra cơ chế từ chối.

Rà soát lại 45 checkpoint xác nhận mã video nguồn giữ riêng không nằm trong phần học. Đã tính lại 45 cặp thống kê chuẩn hóa từ phần học; sai khác lớn nhất 0. Hồ sơ dự đoán, ma trận riêng và đặc trưng thăm dò được lưu để chạy lại số học mà không nhập chúng vào phần học.

## 8 Trải nghiệm và kiểm tra phần mềm

| Phép kiểm tra | Kết quả và giới hạn |
| --- | --- |
| Dữ liệu và API | 28/28 đạt; gồm chia nguồn, thống kê phiếu, ma trận quyết định, phiếu trùng và ghi đồng thời |
| Trình duyệt thật | 18 kiểm tra đạt với Chromium 153.0.8010.0 qua Playwright; có khung desktop và mobile mô phỏng |
| Video đầu vào thật | Luồng chọn clip, nhận diện qua máy chủ và xuất phiếu được chạy bằng MP4 trong bộ mẫu; đây là mẫu học |
| Camera | Kiểm tra quyền và quay bằng camera giả lập; chưa đo webcam vật lý hoặc Windows thật |

Bốn mục chức năng giúp tách thao tác nhận diện, ghi phiếu và đọc kết quả. Các trạng thái chọn clip, đang xử lý, chưa chắc chắn và đã ghi phiếu được hiển thị riêng. Đổi nguồn video, mô hình hoặc mốc cắt yêu cầu nhận diện lại; phản hồi mạng đến muộn không được dùng để ghi nhãn cho cấu hình mới. Camera được tắt khi kết thúc quay hoặc rời luồng sử dụng.

Phiếu lấy dự đoán và điểm từ máy chủ, chống ghi hai lần cùng mã lượt và xử lý hai yêu cầu ghi đồng thời. Sổ bị hỏng trả lỗi thay vì âm thầm bỏ dòng. Tổng hợp có mẫu số riêng cho clip trong bộ từ và ngoài bộ từ, cảnh báo lượt trùng video học và liệt kê phiên bản trọng số. Mọi phiếu QA dùng thư mục tạm, không nằm trong đầu ra người dùng.

Số đo lịch sử ngày 03/11 trên tám video học có trung vị 3.74 giây và p95 5.51 giây trên CPU Linux chia sẻ. Đó là thời gian trích điểm và phân loại trong phiên cũ, chưa phải tốc độ trên máy trình bày hoặc bằng chứng nhận diện thời gian thực.

### Mức sẵn sàng

Đã có bản đồ án kỹ thuật chạy được và được kiểm tra trên trình duyệt. Có thể tổ chức phiên demo có giám sát trong bộ năm ký hiệu, kèm kết quả và giới hạn dữ liệu. Tỷ lệ nhận nhầm ngoài bộ từ còn cao và chưa kiểm tra độc lập ba nhãn hoặc người mới; hệ thống chưa đủ cơ sở để sử dụng như bộ nhận diện hay dịch công khai.

Trước buổi báo cáo, cần chạy một video mẫu khi còn mạng để hoàn tất bộ trích điểm, rồi thử camera vật lý và mở CSV trên máy trình bày. Tệp CHAY_DEMO.bat có bước chuẩn bị lần đầu; KIEM_TRA_CAI_DAT.bat giúp kiểm tra lại khi cần.

## 9 Chạy thử và tái lập kết quả

### Chạy bản đã học

Giải nén toàn bộ gói, cài Python 3.11 64-bit rồi mở CHAY_DEMO.bat. Khi hiện dòng sẵn sàng, vào http://localhost:8765. Lần cài và trích điểm đầu có thể cần mạng để tải thư viện, mô hình MediaPipe. Chọn một clip mẫu rồi nhận diện; clip này đã nằm trong phần học. Dùng camera hoặc video mới để kiểm tra thêm, ghi rõ đây là lượt thăm dò.

Nếu chạy bằng PowerShell, tạo .venv bằng py -3.11 -m venv .venv, cài requirements.txt bằng .venv\Scripts\python.exe -m pip install -r requirements.txt, rồi chạy .venv\Scripts\python.exe run.py. Lệnh tools/doctor.py --warmup kiểm tra thư viện, trọng số và một video thật; có thể đổi cổng bằng --port 8766.

### Chạy lại thí nghiệm

Cài requirements-research.txt, chạy research/prepare_features.py nếu cần trích lại điểm, research/run_experiments.py để học, rồi research/summarize.py để tạo chỉ số và matrix. Tệp checkpoint cùng dấu vân tay được dùng lại; --rerun sẽ học lại và thay kết quả tương ứng. Trọng số đủ tám mẫu dùng demo được lưu riêng với kết quả giữ mẫu ngoài phần học.

Chạy tests/run_checks_v5.py để kiểm tra dữ liệu và API. tests/test_browser_v5.py chạy luồng trình duyệt thật với Playwright và Chromium; tham số đường dẫn trình duyệt được mô tả trong tài liệu QA. Các thư mục đầu ra đều tách khỏi phiếu người dùng. Kiểm tra camera giả lập không thay thế việc đo webcam vật lý. Báo cáo được tạo bằng research/build_report.py.

### Cách trình bày kết quả

Mở đầu bằng mục tiêu năm ký hiệu và giới hạn một người ký; giải thích vector 159 chiều và cách giữ video riêng. Trình bày ma trận trong bộ từ cùng kết quả ba seed, sau đó nêu rõ 14/29 clip ngoài bộ từ bị TCN gốc chấp nhận nhầm. Demo chọn video, nhận diện, đổi cấu hình, ghi phiếu và đọc thống kê. Kịch bản cùng câu hỏi bảo vệ nằm trong docs/KICH_BAN_BAO_CAO.md.

Chạy research/audit_v5.py để kiểm tra checkpoint và phát lại dự đoán ngoài bộ từ từ đặc trưng đã đóng gói. Khi cần trích lại từ video, cung cấp --source-v3 trỏ tới bộ v3 đầy đủ và thư mục tạm theo phần trợ giúp của lệnh. Tập 29 clip này dùng đánh giá riêng, không được tự thêm vào dữ liệu học hoặc dùng lại như validation mới.

Phiếu mới nằm tại outputs/trials.jsonl và không được đưa vào tập học tự động. Chạy tools/summarize_trials.py để tổng hợp số lượt, số mã băm video, tỷ lệ chấp nhận và nhận nhầm ngoài bộ từ. Các nhãn này vẫn là phản hồi người dùng chưa được chuyên gia xác nhận.

## 10 Kết luận và hướng phát triển

Dự án đã có bản minh chứng kỹ thuật với năm ký hiệu, thiết kế lại trải nghiệm, sổ thực nghiệm có kiểm tra nguồn và hồ sơ chạy trình duyệt thật. Đợt đánh giá bổ sung xác nhận 45 phần học được tách đúng, đồng thời phát hiện rủi ro nhận nhầm ngoài bộ từ. Hoàn thiện phần mềm chưa giải quyết được giới hạn tám video của một người; các bước mở rộng cần tập trung vào dữ liệu thực và đánh giá người mới.

### Phát triển tiếp theo trong phạm vi môn học

Kế hoạch thu tiếp đề xuất 5 nhãn × 5 người × 10 lượt = 250 video, chia hai phiên mỗi người. Tệp collection_plan_250.csv chỉ chứa vị trí dự kiến chưa quay. Có thể chia ba người cho học, một cho validation và một cho test; giữ mọi biến thể của cùng người và nguồn trong cùng phần. Một người test vẫn còn hạn chế, cần ghi rõ khi công bố kết quả.

Ưu tiên duyệt nhãn, thu hoạt động ngoài bộ từ và chọn ngưỡng trên validation trước khi đo test. Khi dữ liệu đủ hơn mới mở rộng nhãn hoặc bổ sung đặc trưng mặt. Mục tiêu đợt tiếp theo là có kết quả theo từng nhãn, từng người và phiên; không đặt độ chính xác tùy ý rồi sửa cách chia để đạt con số mong muốn.

### Tài liệu tham khảo

[1] Google MediaPipe. Holistic. Mô tả pipeline tay, mặt và tư thế. https://chuoling.github.io/mediapipe/solutions/holistic.html

[2] Bai, Kolter và Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. 2018. https://arxiv.org/abs/1803.01271

[3] scikit-learn. Common pitfalls and recommended practices. Tránh rò rỉ dữ liệu và học tiền xử lý chỉ từ phần học. https://scikit-learn.org/stable/common_pitfalls.html

[4] scikit-learn. classification_report. Quy ước Precision, Recall, F1 và support. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

[5] scikit-learn. Cross-validation. Chia theo nhóm và người thực hiện. https://scikit-learn.org/stable/modules/cross_validation.html

## Phụ lục ma trận số đếm ở seed chính

### MLP gợi ý đầu

| true_label | ai cho | bực mình | ghen tị | không cần | xin lỗi |
| --- | --- | --- | --- | --- | --- |
| ai cho | 2 | 0 | 0 | 1 | 0 |
| bực mình | 0 | 0 | 0 | 0 | 0 |
| ghen tị | 0 | 0 | 0 | 0 | 0 |
| không cần | 0 | 0 | 0 | 2 | 0 |
| xin lỗi | 0 | 0 | 0 | 0 | 0 |

### MLP có từ chối

| true_label | ai cho | bực mình | ghen tị | không cần | xin lỗi | CHUA_CHAC |
| --- | --- | --- | --- | --- | --- | --- |
| ai cho | 2 | 0 | 0 | 0 | 0 | 1 |
| bực mình | 0 | 0 | 0 | 0 | 0 | 0 |
| ghen tị | 0 | 0 | 0 | 0 | 0 | 0 |
| không cần | 0 | 0 | 0 | 0 | 0 | 2 |
| xin lỗi | 0 | 0 | 0 | 0 | 0 | 0 |

### TCN gốc gợi ý đầu

| true_label | ai cho | bực mình | ghen tị | không cần | xin lỗi |
| --- | --- | --- | --- | --- | --- |
| ai cho | 2 | 1 | 0 | 0 | 0 |
| bực mình | 0 | 0 | 0 | 0 | 0 |
| ghen tị | 0 | 0 | 0 | 0 | 0 |
| không cần | 0 | 0 | 0 | 2 | 0 |
| xin lỗi | 0 | 0 | 0 | 0 | 0 |

### TCN gốc có từ chối

| true_label | ai cho | bực mình | ghen tị | không cần | xin lỗi | CHUA_CHAC |
| --- | --- | --- | --- | --- | --- | --- |
| ai cho | 2 | 0 | 0 | 0 | 0 | 1 |
| bực mình | 0 | 0 | 0 | 0 | 0 | 0 |
| ghen tị | 0 | 0 | 0 | 0 | 0 | 0 |
| không cần | 0 | 0 | 0 | 1 | 0 | 1 |
| xin lỗi | 0 | 0 | 0 | 0 | 0 | 0 |

### TCN tăng cường gợi ý đầu

| true_label | ai cho | bực mình | ghen tị | không cần | xin lỗi |
| --- | --- | --- | --- | --- | --- |
| ai cho | 3 | 0 | 0 | 0 | 0 |
| bực mình | 0 | 0 | 0 | 0 | 0 |
| ghen tị | 0 | 0 | 0 | 0 | 0 |
| không cần | 1 | 0 | 0 | 0 | 1 |
| xin lỗi | 0 | 0 | 0 | 0 | 0 |

### TCN tăng cường có từ chối

| true_label | ai cho | bực mình | ghen tị | không cần | xin lỗi | CHUA_CHAC |
| --- | --- | --- | --- | --- | --- | --- |
| ai cho | 2 | 0 | 0 | 0 | 0 | 1 |
| bực mình | 0 | 0 | 0 | 0 | 0 | 0 |
| ghen tị | 0 | 0 | 0 | 0 | 0 | 0 |
| không cần | 0 | 0 | 0 | 0 | 1 | 1 |
| xin lỗi | 0 | 0 | 0 | 0 | 0 | 0 |

Hàng có support bằng 0 là nhãn chưa có mẫu kiểm tra; không diễn giải là không có lỗi. Ba seed lặp cùng năm clip, không phải mười lăm mẫu độc lập.
