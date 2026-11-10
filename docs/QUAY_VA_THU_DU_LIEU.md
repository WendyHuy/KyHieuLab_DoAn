# Hướng dẫn thu video cho đợt thực nghiệm tiếp theo

Kế hoạch `data/collection_plan_250.csv` có 250 vị trí chưa quay: 5 nhãn × 5 người × 10 lượt. Đây là quy mô đề xuất cho đợt thăm dò, không bảo đảm chất lượng nhận diện hay đủ kiểm chứng diện rộng.

## Cách quay

- Người tham gia biết mục đích dùng video cho đồ án và đồng ý trước khi lưu hoặc chia sẻ.
- Dùng mã S01 đến S05 thay tên thật. Cả năm người phải quay đủ cả năm nhãn, sau khi được hướng dẫn và đối chiếu động tác.
- Mỗi người quay 2 phiên, mỗi nhãn 5 lượt mỗi phiên. Có thể dùng những ngày hoặc điều kiện sáng khác nhau, vẫn giữ nhìn rõ tay và thân trên.
- Máy quay cố định, ngang tầm thân trên, tránh ngược sáng; giữ đủ bàn tay trong khung. Quay 3–8 giây cho một động tác, chừa một ít khoảng nghỉ ở hai đầu; không đếm ngón tay trong đoạn ký.
- Giữ video gốc. Khi cắt, ghi mốc bắt đầu/kết thúc; không cắt mất phần vào hoặc kết thúc có ý nghĩa của động tác. Không lật trái/phải chỉ để tạo thêm dữ liệu.

## Ghi hồ sơ

Tên gợi ý: `S01_B01_ai_cho_R01.mp4`. Lưu mã người, phiên, nhãn, lượt, đường dẫn, mốc cắt, người duyệt và trạng thái duyệt. Nếu tải video từ demo, dùng mã lượt trong tên file để đối chiếu `inference_id` trong CSV; `video_sha256` hỗ trợ phát hiện trùng byte.

Nhãn người dùng chọn mới là phản hồi chưa duyệt. Kiểm tra lại video và động tác trước khi đưa vào tập học chính thức. Không tự đổi `reviewStatus` hoặc `trainingEligible` của hồ sơ gốc chỉ vì mô hình dự đoán đúng.

## Chia dữ liệu trước khi tối ưu

Đề xuất S01–S03 cho học, S04 cho validation và S05 cho test. Giữ toàn bộ clip của cùng người và mọi bản cắt/tăng cường của chúng trong cùng phần. Các video gốc đang có của giahuy cần được nhận diện là cùng một người nếu gán mã mới; không gán mã khác để tạo người kiểm tra giả.

Chọn mô hình và ngưỡng trên validation. Chỉ dùng test sau khi khóa cấu hình. Một người test vẫn là mẫu nhỏ; báo rõ kết quả theo người và phạm vi dữ liệu, không coi đây là kiểm chứng cho mọi người dùng.

Thu thêm các clip nghỉ, chỉnh tóc, cầm đồ và ký hiệu không nằm trong 5 nhãn. Tách chúng giữa validation và test để đo tỷ lệ nhận nhầm ngoài bộ từ. Không dùng chỉ vài video nền màu để suy ra tỷ lệ nhận nhầm thực tế.
