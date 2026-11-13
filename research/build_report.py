"""Build matching Markdown and Times New Roman Word course reports."""
from pathlib import Path
import csv
import json
from docx import Document
from docx.shared import Inches, Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
oof = json.loads((ROOT / "research/oof_results.json").read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
tests = json.loads((OUT / "test_results_v5.json").read_text(encoding="utf-8"))
ui = json.loads((OUT / "browser_checks_v5.json").read_text(encoding="utf-8"))
pipeline = json.loads((OUT / "video_pipeline_replay.json").read_text(encoding="utf-8"))
models = metrics["models"]
blocks = []


def page(text): blocks.append(("page", text))
def h(text): blocks.append(("h", text))
def p(text): blocks.append(("p", text))
def table(headers, rows, widths): blocks.append(("table", (headers, rows, widths)))
def fig(name, caption): blocks.append(("fig", (name, caption)))


page("Nhận diện ký hiệu tiếng Việt từ video")
p("Báo cáo đồ án thử nghiệm Ký Hiệu Lab phiên bản 4")
p("Nhóm sinh viên: Huỳnh Gia Huy (MSSV: 2474802010135) & Nguyễn Thành Phát (MSSV: 2474802016639) | Trường Đại học Văn Lang")
p("Ngày hoàn thiện bản kỹ thuật 03/11/2026")
h("Tóm tắt dự án")
p("Dự án xây dựng ứng dụng nhận diện một ký hiệu riêng lẻ từ đoạn video đã quay. Bản v4 giới hạn ở năm nhãn để tổ chức lại dữ liệu, huấn luyện ba cấu hình, đối chiếu kết quả và cung cấp demo có ghi phiếu thực nghiệm. Hệ thống đã có mã nguồn, trọng số học xong, video mẫu, ma trận nhầm lẫn và quy trình chạy lại.")
p("Kết quả cho phép dùng bản này làm minh chứng kỹ thuật trong đồ án, với giới hạn dữ liệu được công bố. Tám video thực chỉ thuộc một người ký; ba trong năm nhãn mới có một mẫu. Đánh giá giữ từng clip ngoài phần học chỉ bao phủ năm video của hai nhãn, chưa xác nhận khả năng nhận diện người mới hoặc đủ cả năm nhãn.")
table(["Phương pháp", "Gợi ý đầu đúng", "Chấp nhận đúng", "Chấp nhận sai"], [(m["name"], f"{m['primary']['top1_correct']}/5", f"{m['primary']['accepted_correct']}/5", m["primary"]["accepted_wrong"]) for m in models], [4.3, 4, 4, 4])
p("Bảng dùng seed chính 20261103, với ngưỡng 0,65 giữ cố định trước thực nghiệm. TCN gốc là mặc định; ứng viên tăng cường được giữ để so sánh vì chưa cải thiện ổn định. Mỗi mô hình còn được chạy thêm hai seed trên cùng năm clip; số lượt lặp không làm tăng số video kiểm tra độc lập.")
p("Đóng góp của đồ án gồm quy trình xử lý video, cách ghép đặc trưng tay và thân, ứng dụng nhận diện có cơ chế chưa chắc chắn, cùng thực nghiệm so sánh tái lập. MediaPipe, MLP và mạng tích chập thời gian là các phương pháp đã có; đồ án không tuyên bố phát minh thuật toán mới.")

page("1 Bài toán và dữ liệu thực hiện")
p("Đầu vào là video một người thực hiện một động tác, hoặc một đoạn được chọn trong video. Đầu ra gồm nhãn gợi ý, điểm của ba nhãn đứng đầu và trạng thái chấp nhận hay chưa chắc chắn. Người kiểm tra có thể ghi nhãn thực tế, kể cả ngoài bộ từ, để tổng hợp các lượt sử dụng. Hệ thống xử lý sau khi kết thúc clip, chưa giải quyết phân đoạn câu và dịch liên tục.")
table(["Nhãn", "Video thực", "Clip có thể giữ riêng", "Trạng thái đánh giá"], [("ai cho",3,3,"Hồi cứu cùng người"),("không cần",2,2,"Hồi cứu cùng người"),("bực mình",1,0,"Chưa có mẫu kiểm tra"),("ghen tị",1,0,"Chưa có mẫu kiểm tra"),("xin lỗi",1,0,"Chưa có mẫu kiểm tra")], [3.4,2.7,4.5,5.7])
h("Nguồn và lựa chọn phạm vi")
p("Bộ v3 có 39 video, trong đó 37 video được dùng học thử cho 34 nhãn và hai video thiếu thông tin tiếp tục ở trạng thái chờ. Bản v4 chọn hai nhãn có mẫu lặp và ba nhãn minh họa có tỷ lệ thấy tay cao theo hồ sơ. Danh sách được ghi trước khi huấn luyện v4. Tám video đều thuộc mã giahuy; số video thực mới trong đợt này là 0.")
p("Gói v4 chứa video và đặc trưng của nhóm năm nhãn. Hồ sơ 39 video và JSON gốc được giữ để đối chiếu; gói đầy đủ v3 được lưu riêng. Việc giảm từ 34 xuống năm nhãn làm thay đổi bài toán, nên chênh lệch điểm v3 và v4 không được dùng như mức cải thiện trên cùng bộ kiểm tra.")
h("Quy ước nhãn và thông tin duyệt")
p("Động tác được dùng theo giả định người thu đã đối chiếu nhãn. Các trường reviewStatus và trainingEligible trong hồ sơ gốc được giữ nguyên; học thử theo yêu cầu không biến dữ liệu thành đã được chuyên gia duyệt. Các biến thể cách 1 và cách 2 của những nhãn ngoài phạm vi không được tự gộp vào năm nhãn đang học.")

page("2 Đặc trưng và mô hình nhận diện")
p("MediaPipe Holistic được dùng để ước lượng bàn tay và tư thế từ hình ảnh [1]. Bộ đặc trưng của dự án lấy 21 điểm mỗi bàn tay và 13 điểm thân trên. Điểm mặt không đi vào mô hình; đây là giới hạn với các ký hiệu hoặc sắc thái cần thông tin nét mặt.")
table(["Thành phần", "Số đặc trưng", "Cách biểu diễn"], [("Một bàn tay",66,"63 tọa độ hình dạng, 2 tọa độ vị trí và 1 mặt nạ"),("Hai bàn tay",132,"Giữ riêng tay trái và tay phải"),("Thân trên",27,"13 điểm XY và 1 mặt nạ"),("Một bước thời gian",159,"132 cộng 27"),("Một đoạn ký","32 × 159","Lấy mẫu lại theo thời gian, có xét mặt nạ")], [4.1,3.2,9])
p("Hình dạng tay được đưa về gốc cổ tay và chia theo kích thước bàn tay. Vị trí cổ tay và thân được tính trong mặt phẳng XY theo tâm hai vai và khoảng cách vai. Tỷ lệ khung hình được tính vào trục ngang. Không trộn tọa độ sâu của bàn tay với hệ tọa độ sâu của tư thế. Điểm thiếu được ghi bằng mặt nạ; nội suy không tự biến đoạn mất tay dài thành tay hợp lệ.")
table(["Mạng", "Cấu trúc chính", "Tham số"], [("MLP","Làm phẳng 5.088 → 96 → 48 → 5",493445),("TCN gốc","159 → 48; 2 khối tích chập có nối tắt; pooling; 48 → 5",26501),("TCN tăng cường","Cùng kiến trúc TCN, thay cách tạo và lấy mẫu phần học",26501)], [3.6,9.7,3])
p("Mạng tích chập thời gian tham khảo hướng mô hình hóa chuỗi bằng tích chập [2]. Bản triển khai dùng kernel 3, dilation 1 và 2, padding SAME, nối tắt và gộp trung bình cùng cực đại ở các bước có tay. Mạng nhìn cả đoạn đã quay; không phải bản TCN nhân quả để suy luận luồng trực tiếp. Hàm kích hoạt ReLU, đầu ra softmax năm nhãn.")

page("3 Xử lý dữ liệu và cấu trúc ứng dụng")
h("Chuẩn hóa video và trích điểm")
p("Tám clip giữ các mốc nội dung đã cắt ở v3, được chuyển sang H.264, yuv420p, 12 khung hình mỗi giây, cạnh dài tối đa 640 pixel và bỏ âm thanh. Không cắt thêm phần động tác theo dự đoán của mô hình. Một clip bực mình có tốc độ khai báo 1.000 fps và 4.576 khung; bản chuẩn hóa còn 55 khung, giữ thời lượng xấp xỉ đoạn gốc. Nhật ký chuyển đổi lưu trong video_normalization_audit.json.")
p("Đặc trưng được trích lại từ chính MP4 đã chuẩn hóa, thay vì gắn đặc trưng cũ cho video mới. Chuỗi được đưa về 32 bước; thống kê chuẩn hóa được học riêng từ phần học ở mỗi lượt thực nghiệm. Khi nhận diện, nếu ít hơn tám bước có tay thì hệ thống từ chối. Nếu có đủ tay nhưng điểm cao nhất dưới 0,65, kết quả là chưa chắc chắn.")
h("Luồng xử lý ứng dụng")
table(["Bước", "Xử lý"], [("1","Trình duyệt chọn video, quay 5 giây hoặc mở clip mẫu"),("2","Người dùng chọn mốc đầu/cuối và mô hình"),("3","Máy chủ kiểm tra dữ liệu, trích điểm đoạn đã chọn và chạy trọng số"),("4","Trả nhãn, điểm, tỷ lệ thấy tay và thời gian xử lý"),("5","Người kiểm tra ghi nhãn thực tế theo mã lượt; tải phiếu CSV")], [1.5,14.8])
p("Máy chủ chỉ lắng nghe tại 127.0.0.1. Phép phân loại dùng NumPy; demo không cần GPU hay FFmpeg. JAX được ghim trong bộ cài vì MediaPipe khai báo phụ thuộc này; mã huấn luyện dùng JAX trực tiếp. Video gửi vào được đặt trong thư mục tạm và xóa sau xử lý. Có nút tải video đang chọn để người thu chủ động giữ clip cho lần bổ sung dữ liệu.")
p("Phiếu lấy kết quả gốc từ máy chủ bằng inference_id. Trình duyệt không tự cung cấp điểm hoặc nhãn dự đoán để ghi đè. Đổi video, mốc cắt hay mô hình sẽ hủy kết quả đang chờ ghi; một mã lượt chỉ ghi một lần. Phiếu phản hồi chưa được duyệt và không tự cập nhật trọng số.")

page("4 Thiết kế thực nghiệm")
p("Thực nghiệm cố định ba phương pháp và ba seed trước khi học. Mỗi trong năm clip có mẫu lặp được giữ riêng; mô hình học trên bảy video còn lại rồi dự đoán clip giữ riêng. Mẫu tăng cường và trung bình, độ lệch chuẩn chỉ được tạo từ bảy video này. Không chia khung của cùng video qua hai phần [3].")
table(["Thông số", "Giá trị"], [("Seed chính và hai seed lặp","20261103, 20261104, 20261105"),("Số bước và batch","500 bước, batch 64"),("Bộ tối ưu","Adam, learning rate 0,001"),("Điều chuẩn","L2 hệ số 0,00001; label smoothing 0,02"),("Tăng cường cơ sở","20 biến thể thời gian và nhiễu nhẹ cho mỗi video học"),("Biến thể TCN tăng cường","Lấy mẫu cân bằng nhãn, biến thiên mốc cắt và mất tay ngắn"),("Ngưỡng chấp nhận","0,65; chưa hiệu chỉnh từ validation"),("Đơn vị giữ riêng","Video nguồn, đối chiếu mã và SHA256")], [5.3,11])
p("Có 3 phương pháp × 3 seed × 5 clip = 45 lượt học giữ clip riêng. Thêm ba lượt học trên đủ tám video để tạo trọng số demo. Checkpoint lưu mã video học, seed, danh sách video học, thống kê chuẩn hóa và lịch sử loss. Dấu vân tay dữ liệu và mã giúp tránh dùng lại kết quả cũ khi đầu vào thay đổi.")
h("Chỉ số và phạm vi suy luận")
p("Báo cáo số gợi ý đầu đúng, Top 3, Precision, Recall, F1 từng nhãn, ma trận số đếm và số lượt chấp nhận đúng hoặc sai. Macro F1 chỉ lấy trung bình trên ai cho và không cần, là hai nhãn có mẫu kiểm tra [4]. Ba hàng không có mẫu trong CSV đầy đủ không phải bằng chứng mô hình không mắc lỗi ở ba nhãn đó.")
p("Ba seed dùng lại cùng năm clip, chỉ đo sự thay đổi khi khởi tạo và học; không tạo thành 15 clip độc lập. Đây là kiểm tra hồi cứu với dữ liệu đã biết, cùng một người ký. Chưa có tập validation riêng để chọn ngưỡng, test mù mới, hoặc test người ký mới. Khi có nhiều người, cần chia theo người và phiên [5].")

page("5 Kết quả ở seed chính")
fig("confusion_primary.png", "Hình 1 Ma trận gợi ý đầu trên năm clip giữ ngoài phần học")
table(["Mô hình", "Đúng", "Top 3 đúng", "Macro F1", "Nhận đúng", "Nhận sai"], [(m["name"],f"{m['primary']['top1_correct']}/5",f"{m['primary']['top3_correct']}/5",f"{m['primary']['macro_f1_two_tested_labels']:.3f}",m["primary"]["accepted_correct"],m["primary"]["accepted_wrong"]) for m in models], [4.1,2,2.7,2.7,2.4,2.4])
fig("confusion_reject_primary.png", "Hình 2 Ma trận có cơ chế từ chối ở ngưỡng 0,65")
p("Hàng là nhãn người thu cung cấp; cột là dự đoán. Cột khác gộp ba nhãn không có mẫu giữ riêng. TCN gốc gợi ý đầu đúng 4/5 clip, chấp nhận ba lượt đúng và từ chối hai lượt. Đây là số đo nhỏ ở seed chính; không suy ra độ chính xác tổng quát bằng 80% hoặc độ chính xác chấp nhận bằng 100%.")

page("6 Độ ổn định và phân tích lỗi")
table(["Mô hình", "Seed 03", "Seed 04", "Seed 05", "Trung bình đúng"], [(m["name"],*[f"{r['top1_correct']}/5" for r in m["per_seed"]],f"{m['mean_top1_accuracy_over_seeds']*100:.1f}%") for m in models], [4.5,2.6,2.6,2.6,4])
p("TCN gốc cho kết quả 4/5, 5/5 và 4/5; mức trung bình 86,7% chỉ là trung bình ba lần học trên cùng năm clip. Ở seed thứ ba, TCN gốc có một lượt chấp nhận sai. TCN tăng cường đạt 3/5, 5/5 và 3/5, chưa cho thấy lợi ích ổn định. Mô hình mặc định tiếp tục là TCN gốc đã chốt trước khi đo.")
detail=[]
for m in models:
    with (OUT/(m['id']+'_per_label_primary.csv')).open(encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            detail.append((m['name'],row['label'],f"{float(row['precision']):.3f}",f"{float(row['recall']):.3f}",f"{float(row['f1']):.3f}",int(float(row['support']))))
table(["Mô hình", "Nhãn", "Precision", "Recall", "F1", "Mẫu"],detail,[3.9,3.0,2.9,2.4,2.4,1.7])
h("Trường hợp sai và cách đọc chỉ số")
primary=[r for r in oof if r['seed']==metrics['primary_seed']]
for m in models:
    errors=[r for r in primary if r['method']==m['id'] and not r['top1_correct']]
    text='; '.join(r['id']+' '+r['provided']+' → '+str(r['predicted']) for r in errors)
    p(m['name']+': '+text+'.')
p("MLP và TCN gốc cùng đúng bốn clip nhưng Macro F1 hai nhãn khác nhau vì nhãn nhận nhầm khác nhau. Bộ kiểm tra quá nhỏ để khẳng định ưu thế có ý nghĩa thống kê. Tăng cường dữ liệu tạo biến thể học, chưa thay thế được người ký và lần thực hiện mới. Loss thấp hoặc nhận lại đúng mẫu học không đủ chứng minh khả năng áp dụng rộng.")

page("7 Kiểm tra ứng dụng và mức sẵn sàng")
table(["Phép kiểm tra", "Kết quả"], [("Dữ liệu và đường API",f"{tests['tests_run']} kiểm tra đạt; bao gồm tách nguồn, mặt nạ điểm thiếu, cắt đoạn, lỗi video, ghi phiếu và CSV"),("Giao diện mô phỏng DOM",f"{ui['checks_passed']} kiểm tra đạt; dùng fetch và camera mô phỏng, chưa phải kiểm thử trình duyệt thật"),("Nhận diện lại tám video học",f"TCN gốc đúng {pipeline['top1_correct']}/8 và chấp nhận {pipeline['accepted']}/8 qua đường video thực; chỉ là phát lại tập học"),("Thiết bị vật lý","Chưa đo camera thật, Windows, điện thoại hoặc tải nhiều người")], [5.2,11.1])
p(f"Qua tám video học được gửi lại vào đường xử lý, thời gian trung vị là {pipeline['median_seconds']:.2f} giây, p95 {pipeline['p95_seconds']:.2f} giây trên Linux Python 3.12 và CPU chia sẻ. Phép đo gồm trích điểm và phân loại, chưa gồm thời gian người dùng ký. Không so sánh trực tiếp với tốc độ v3 vì khác video chuẩn hóa và điều kiện CPU; chưa dùng để tuyên bố thời gian thực.")
h("Quyết định sử dụng")
p("Bản v4 đã hoàn tất phạm vi kỹ thuật để trình bày quy trình và demo có giám sát. Có thể dùng làm hồ sơ đồ án thử nghiệm nếu phù hợp yêu cầu môn học, kèm số mẫu và hạn chế. Chưa đủ bằng chứng cho nhận diện công khai, kiểm chứng đủ năm nhãn, người ký mới hay dịch câu. Kết quả môn học còn phụ thuộc rubric và buổi chạy trên máy trình bày.")
p("Mã kiểm thử dùng thư mục tạm riêng; không xóa hoặc trộn phiếu của người sử dụng. Mẫu nền đen bị từ chối chỉ là kiểm tra chức năng thiếu tay, chưa đo được tỷ lệ nhận nhầm trong sinh hoạt. Nhãn ngoài bộ từ được ghi riêng trong phiếu để hỗ trợ thí nghiệm mới.")
h("Phần cần chạy trên máy trình bày")
p("Cài thư viện trước buổi báo cáo, thử MP4 mẫu, bật camera và quay một clip. Kiểm tra quyền camera, thời gian chờ, mốc cắt, trạng thái chưa chắc chắn và mở CSV bằng Excel. Lưu lại cả trường hợp sai; không dùng các lượt luyện demo để báo như một tập test mù mới.")

page("8 Hướng dẫn tái lập và trình bày")
h("Chạy bản đã học")
p("Giải nén toàn bộ gói, cài Python 3.11 64-bit rồi mở CHAY_DEMO.bat. Khi hiện dòng sẵn sàng, vào http://localhost:8765. Lần cài và trích điểm đầu có thể cần mạng để tải thư viện, mô hình MediaPipe. Chọn một clip mẫu rồi nhận diện; clip này đã nằm trong phần học. Dùng camera hoặc video mới để kiểm tra thêm, ghi rõ đây là lượt thăm dò.")
p("Nếu chạy bằng PowerShell, tạo .venv bằng py -3.11 -m venv .venv, cài requirements.txt bằng .venv\\Scripts\\python.exe -m pip install -r requirements.txt, rồi chạy .venv\\Scripts\\python.exe run.py. tools/doctor.py kiểm tra thư viện và trọng số; có thể đổi cổng bằng --port 8766.")
h("Chạy lại thí nghiệm")
p("Cài requirements-research.txt, chạy research/prepare_features.py nếu cần trích lại điểm, research/run_experiments.py để học, rồi research/summarize.py để tạo chỉ số và matrix. Tệp checkpoint cùng dấu vân tay được dùng lại; --rerun sẽ học lại và thay kết quả tương ứng. Trọng số đủ tám mẫu dùng demo được lưu riêng với kết quả giữ mẫu ngoài phần học.")
p("tests/test_project.py cần thêm FFmpeg để tạo video kiểm tra nền màu. tests/test_ui.cjs cần Node.js và jsdom, chỉ kiểm tra DOM mô phỏng. Báo cáo Word được tạo bằng research/build_report.py với python-docx; bản PDF trong gói đã được xuất và kiểm tra bố cục.")
h("Cách trình bày kết quả")
p("Mở đầu bằng mục tiêu năm ký hiệu và giới hạn một người ký; tiếp theo giải thích vector 159 chiều và cách chia video. Trình bày bảng ba mô hình, ma trận có từ chối và kết quả ba seed. Demo nhận diện một clip, đổi cấu hình để nhận diện lại và ghi phiếu CSV. Kết thúc bằng phần dữ liệu còn thiếu và kế hoạch kiểm tra người mới. Kịch bản 8–10 phút và câu hỏi bảo vệ nằm trong docs/KICH_BAN_BAO_CAO.md.")
p("Phiếu mới nằm tại outputs/trials.jsonl và không được đưa vào tập học tự động. Chạy tools/summarize_trials.py để tổng hợp số lượt, số mã băm video, tỷ lệ chấp nhận và nhận nhầm ngoài bộ từ. Các nhãn này vẫn là phản hồi người dùng chưa được chuyên gia xác nhận.")

page("9 Kết luận và hướng phát triển")
p("Dự án đã hoàn thành một bản minh chứng kỹ thuật nhận diện năm ký hiệu từ video: chuẩn hóa dữ liệu, trích điểm, học ba cấu hình, đánh giá giữ mẫu riêng với ba seed, ứng dụng cục bộ và công cụ ghi phiếu. TCN gốc được giữ làm mặc định; phép tăng cường đang thử chưa cải thiện ổn định. Số liệu hiện tại thể hiện kết quả hồi cứu trên hai nhãn, chưa bao phủ độc lập cả năm nhãn.")
h("Phát triển tiếp theo trong phạm vi môn học")
p("Kế hoạch thu tiếp đề xuất 5 nhãn × 5 người × 10 lượt = 250 video, chia hai phiên mỗi người. Tệp collection_plan_250.csv chỉ chứa vị trí dự kiến chưa quay. Có thể chia ba người cho học, một cho validation và một cho test; giữ mọi biến thể của cùng người và nguồn trong cùng phần. Một người test vẫn còn hạn chế, cần ghi rõ khi công bố kết quả.")
p("Ưu tiên duyệt nhãn, thu hoạt động ngoài bộ từ và chọn ngưỡng trên validation trước khi đo test. Khi dữ liệu đủ hơn mới mở rộng nhãn hoặc bổ sung đặc trưng mặt. Mục tiêu đợt tiếp theo là có kết quả theo từng nhãn, từng người và phiên; không đặt độ chính xác tùy ý rồi sửa cách chia để đạt con số mong muốn.")
h("Tài liệu tham khảo")
p("[1] Google MediaPipe. Holistic. Mô tả pipeline tay, mặt và tư thế. https://chuoling.github.io/mediapipe/solutions/holistic.html")
p("[2] Bai, Kolter và Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. 2018. https://arxiv.org/abs/1803.01271")
p("[3] scikit-learn. Common pitfalls and recommended practices. Tránh rò rỉ dữ liệu và học tiền xử lý chỉ từ phần học. https://scikit-learn.org/stable/common_pitfalls.html")
p("[4] scikit-learn. classification_report. Quy ước Precision, Recall, F1 và support. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html")
p("[5] scikit-learn. Cross-validation. Chia theo nhóm và người thực hiện. https://scikit-learn.org/stable/modules/cross_validation.html")


def upgrade_v5():
    global blocks
    evaluation = json.loads((OUT / "evaluation_v5.json").read_text(encoding="utf-8"))
    groups = []
    for item in blocks:
        if item[0] == "page":
            groups.append([item])
        else:
            groups[-1].append(item)

    def replace_paragraph(group, prefix, text):
        for index, (kind, value) in enumerate(group):
            if kind == "p" and value.startswith(prefix):
                group[index] = ("p", text)
                return
        raise ValueError("Không tìm thấy đoạn báo cáo: " + prefix)

    cover = groups[0]
    replace_paragraph(cover, "Báo cáo đồ án thử nghiệm", "Báo cáo đồ án thử nghiệm Ký Hiệu Lab phiên bản 5")
    replace_paragraph(cover, "Ngày hoàn thiện", "Ngày cập nhật và kiểm tra 04/11/2026")
    replace_paragraph(cover, "Dự án xây dựng", "Dự án xây dựng ứng dụng gợi ý một ký hiệu riêng lẻ từ đoạn video đã quay. Bản v5 kế thừa ba cấu hình học trên tám video của v4, thiết kế lại trải nghiệm nhận diện và ghi phiếu, chạy kiểm tra trình duyệt thật, đồng thời bổ sung khảo sát 29 video có nhãn ngoài bộ năm ký hiệu. Mã nguồn, trọng số, ma trận và hồ sơ kiểm tra được bàn giao cùng quy trình tái lập.")
    replace_paragraph(cover, "Kết quả cho phép", "Bản này có thể dùng trình bày quy trình kỹ thuật và thực nghiệm có giám sát. Dữ liệu học vẫn chỉ có một người ký và ba nhãn đơn mẫu; phép giữ riêng chỉ bao phủ năm clip của hai nhãn. Khảo sát ngoài bộ từ phát hiện TCN gốc vẫn chấp nhận nhầm 14/29 clip. Vì vậy kết quả cần được đọc là gợi ý trong năm nhãn, chưa đủ để xác nhận đầu vào thật sự thuộc bộ từ hoặc mở nhận diện công khai.")
    flow = groups[3]
    flow.append(("p", "Giao diện v5 có bốn mục Nhận diện, Thực nghiệm, Đánh giá và Hướng dẫn. Tổng hợp phiếu tách số lượt với số mã băm video, hiển thị lượt trùng video học và các phiên bản trọng số; có lựa chọn ghi ngoài bộ ký hiệu. Ma trận quyết định dùng cột từ chối để phản ánh kết quả thực trả về. Mã người do người dùng nhập chưa phải danh tính đã được xác minh."))

    oov = evaluation["outside_vocabulary"]
    audit = evaluation["provenance"]
    outside_page = [
        ("page", "7 Khảo sát động tác ngoài bộ từ"),
        ("p", "Đợt bổ sung chọn trước toàn bộ 29 video đủ hồ sơ học thử từ v3 có nhãn nằm ngoài năm nhãn của v4: 17 clip giahuy và 12 clip tp. Chúng không thuộc tám nguồn học của trọng số demo. Video được chuẩn hóa theo cùng quy tắc trước khi trích điểm; không có video thực mới được quay trong đợt này."),
        ("table", (["Mô hình", "Chấp nhận nhầm", "Từ chối", "Tỷ lệ trong 29 clip"], [(m["name"], m["false_accepts"], m["rejected"], f"{100*m['false_accept_rate']:.1f}%") for m in oov["models"]], [4.3, 4, 3.4, 4.6])),
        ("fig", ("evaluation_v5_outside_vocabulary.png", "Hình 3 Khảo sát hồi cứu 29 clip ngoài bộ năm nhãn")),
        ("p", "Có 29/29 video được xử lý, không có clip lỗi hoặc bị bỏ qua. Trọng số học đủ tám video và ngưỡng 0,65 được khóa trước các dự đoán này. Mẫu thuộc hồ sơ cũ, trạng thái duyệt vẫn pending; đây là khảo sát tiện có, chưa phải test mù hoặc tỷ lệ lỗi đại diện cho mọi hoạt động ngoài bộ từ. Mã tp ở các nhãn ngoài phạm vi không tạo thành phép kiểm tra nhận diện nhãn được hỗ trợ trên người mới."),
        ("p", "Phép giữ clip riêng sử dụng trọng số học trên bảy video, còn khảo sát ngoài bộ từ dùng trọng số demo học đủ tám video. Hai phần không được gộp thành một accuracy hay một ma trận tổng thể. Không tăng ngưỡng hoặc đổi mô hình theo kết quả vừa xem; cần bộ validation và test ngoài bộ từ tách trước để chọn và kiểm tra cơ chế từ chối."),
        ("p", f"Rà soát lại {audit['checkpoints_checked']} checkpoint xác nhận mã video nguồn giữ riêng không nằm trong phần học. Đã tính lại {audit['normalizer_arrays_checked']} cặp thống kê chuẩn hóa từ phần học; sai khác lớn nhất {audit['normalizer_max_abs_error']:g}. Hồ sơ dự đoán, ma trận riêng và đặc trưng thăm dò được lưu để chạy lại số học mà không nhập chúng vào phần học."),
    ]
    software_page = [
        ("page", "8 Trải nghiệm và kiểm tra phần mềm"),
        ("table", (["Phép kiểm tra", "Kết quả và giới hạn"], [
            ("Dữ liệu và API", f"{tests['tests_run']}/{tests['tests_run']} đạt; gồm chia nguồn, thống kê phiếu, ma trận quyết định, phiếu trùng và ghi đồng thời"),
            ("Trình duyệt thật", f"{ui['checks_passed']} kiểm tra đạt với Chromium {ui['browser']['version']} qua Playwright; có khung desktop và mobile mô phỏng"),
            ("Video đầu vào thật", "Luồng chọn clip, nhận diện qua máy chủ và xuất phiếu được chạy bằng MP4 trong bộ mẫu; đây là mẫu học"),
            ("Camera", "Kiểm tra quyền và quay bằng camera giả lập; chưa đo webcam vật lý hoặc Windows thật"),
        ], [4.8, 11.5])),
        ("p", "Bốn mục chức năng giúp tách thao tác nhận diện, ghi phiếu và đọc kết quả. Các trạng thái chọn clip, đang xử lý, chưa chắc chắn và đã ghi phiếu được hiển thị riêng. Đổi nguồn video, mô hình hoặc mốc cắt yêu cầu nhận diện lại; phản hồi mạng đến muộn không được dùng để ghi nhãn cho cấu hình mới. Camera được tắt khi kết thúc quay hoặc rời luồng sử dụng."),
        ("p", "Phiếu lấy dự đoán và điểm từ máy chủ, chống ghi hai lần cùng mã lượt và xử lý hai yêu cầu ghi đồng thời. Sổ bị hỏng trả lỗi thay vì âm thầm bỏ dòng. Tổng hợp có mẫu số riêng cho clip trong bộ từ và ngoài bộ từ, cảnh báo lượt trùng video học và liệt kê phiên bản trọng số. Mọi phiếu QA dùng thư mục tạm, không nằm trong đầu ra người dùng."),
        ("p", f"Số đo lịch sử ngày 03/11 trên tám video học có trung vị {pipeline['median_seconds']:.2f} giây và p95 {pipeline['p95_seconds']:.2f} giây trên CPU Linux chia sẻ. Đó là thời gian trích điểm và phân loại trong phiên cũ, chưa phải tốc độ trên máy trình bày hoặc bằng chứng nhận diện thời gian thực."),
        ("h", "Mức sẵn sàng"),
        ("p", "Đã có bản đồ án kỹ thuật chạy được và được kiểm tra trên trình duyệt. Có thể tổ chức phiên demo có giám sát trong bộ năm ký hiệu, kèm kết quả và giới hạn dữ liệu. Tỷ lệ nhận nhầm ngoài bộ từ còn cao và chưa kiểm tra độc lập ba nhãn hoặc người mới; hệ thống chưa đủ cơ sở để sử dụng như bộ nhận diện hay dịch công khai."),
        ("p", "Trước buổi báo cáo, cần chạy một video mẫu khi còn mạng để hoàn tất bộ trích điểm, rồi thử camera vật lý và mở CSV trên máy trình bày. Tệp CHAY_DEMO.bat có bước chuẩn bị lần đầu; KIEM_TRA_CAI_DAT.bat giúp kiểm tra lại khi cần."),
    ]
    usage = groups[8]
    usage[0] = ("page", "9 Chạy thử và tái lập kết quả")
    replace_paragraph(usage, "Nếu chạy bằng PowerShell", "Nếu chạy bằng PowerShell, tạo .venv bằng py -3.11 -m venv .venv, cài requirements.txt bằng .venv\\Scripts\\python.exe -m pip install -r requirements.txt, rồi chạy .venv\\Scripts\\python.exe run.py. Lệnh tools/doctor.py --warmup kiểm tra thư viện, trọng số và một video thật; có thể đổi cổng bằng --port 8766.")
    replace_paragraph(usage, "tests/test_project.py", "Chạy tests/run_checks_v5.py để kiểm tra dữ liệu và API. tests/test_browser_v5.py chạy luồng trình duyệt thật với Playwright và Chromium; tham số đường dẫn trình duyệt được mô tả trong tài liệu QA. Các thư mục đầu ra đều tách khỏi phiếu người dùng. Kiểm tra camera giả lập không thay thế việc đo webcam vật lý. Báo cáo được tạo bằng research/build_report.py.")
    usage.insert(-1, ("p", "Chạy research/audit_v5.py để kiểm tra checkpoint và phát lại dự đoán ngoài bộ từ từ đặc trưng đã đóng gói. Khi cần trích lại từ video, cung cấp --source-v3 trỏ tới bộ v3 đầy đủ và thư mục tạm theo phần trợ giúp của lệnh. Tập 29 clip này dùng đánh giá riêng, không được tự thêm vào dữ liệu học hoặc dùng lại như validation mới."))
    replace_paragraph(usage, "Mở đầu bằng mục tiêu", "Mở đầu bằng mục tiêu năm ký hiệu và giới hạn một người ký; giải thích vector 159 chiều và cách giữ video riêng. Trình bày ma trận trong bộ từ cùng kết quả ba seed, sau đó nêu rõ 14/29 clip ngoài bộ từ bị TCN gốc chấp nhận nhầm. Demo chọn video, nhận diện, đổi cấu hình, ghi phiếu và đọc thống kê. Kịch bản cùng câu hỏi bảo vệ nằm trong docs/KICH_BAN_BAO_CAO.md.")
    conclusion = groups[9]
    conclusion[0] = ("page", "10 Kết luận và hướng phát triển")
    replace_paragraph(conclusion, "Dự án đã hoàn thành", "Dự án đã có bản minh chứng kỹ thuật với năm ký hiệu, thiết kế lại trải nghiệm, sổ thực nghiệm có kiểm tra nguồn và hồ sơ chạy trình duyệt thật. Đợt đánh giá bổ sung xác nhận 45 phần học được tách đúng, đồng thời phát hiện rủi ro nhận nhầm ngoài bộ từ. Hoàn thiện phần mềm chưa giải quyết được giới hạn tám video của một người; các bước mở rộng cần tập trung vào dữ liệu thực và đánh giá người mới.")
    blocks = [item for group in groups[:7] + [outside_page, software_page, usage, conclusion] for item in group]


upgrade_v5()


def build(output_name="Bao_cao_DoAn_KyHieuLab_v5", footer_date="04/11/2026"):
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.6)
    section.right_margin = Cm(2.3)
    section.footer_distance = Cm(0.9)
    for style in doc.styles:
        if style.type not in (1, 2):
            continue
        style.font.name = "Times New Roman"
        style.font.color.rgb = RGBColor(0, 0, 0)
        fonts = style.element.get_or_add_rPr().find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            style.element.get_or_add_rPr().append(fonts)
        for key in list(fonts.attrib):
            if "theme" in key.lower():
                fonts.attrib.pop(key)
        for key in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(qn("w:" + key), "Times New Roman")
        borders = style.element.find(".//" + qn("w:pBdr"))
        if borders is not None:
            borders.getparent().remove(borders)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(12.5)
    normal.paragraph_format.line_spacing = 1.06
    normal.paragraph_format.space_after = Pt(6)
    for name, size in [("Title", 20), ("Heading 1", 16), ("Heading 2", 13)]:
        style = doc.styles[name]
        style.font.size = Pt(size)
        style.font.bold = True
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(7)
        style.paragraph_format.space_after = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Ký Hiệu Lab | " + footer_date + " | Trang ").font.size = Pt(9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    doc.core_properties.author = "Huỳnh Gia Huy, Nguyễn Thành Phát"
    doc.core_properties.title = "Nhận diện ký hiệu tiếng Việt từ video"
    md = []
    first = True
    for kind, value in blocks:
        if kind == "page":
            para = doc.add_paragraph(value, style="Title" if first else "Heading 1")
            para.paragraph_format.page_break_before = not first
            md.append(("# " if first else "## ") + value)
            first = False
        elif kind == "h":
            doc.add_paragraph(value, style="Heading 2")
            md.append("### " + value)
        elif kind == "p":
            para = doc.add_paragraph(value)
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT if any(token in value for token in ("http", ".py", "requirements", "collection_plan")) else WD_ALIGN_PARAGRAPH.JUSTIFY
            md.append(value)
        elif kind == "fig":
            name, caption = value
            para = doc.add_paragraph()
            para.paragraph_format.keep_with_next = True
            para.paragraph_format.space_after = Pt(2)
            para.add_run().add_picture(str(OUT / "figures" / name), width=Cm(16.3))
            para = doc.add_paragraph(caption)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.italic = True
                run.font.size = Pt(10.5)
            md.append(f"*{caption}*. Hình trong gói: `reports/figures/{name}`.")
        elif kind == "table":
            headers, rows, widths = value
            tab = doc.add_table(rows=1, cols=len(headers))
            tab.autofit = False
            tab.alignment = WD_TABLE_ALIGNMENT.CENTER
            for index, width in enumerate(widths):
                tab.columns[index].width = Cm(width)
                tab.rows[0].cells[index].text = headers[index]
            for row in rows:
                cells = tab.add_row().cells
                for index, text in enumerate(row):
                    cells[index].text = str(text)
            for ri, row in enumerate(tab.rows):
                row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
                if ri == 0:
                    row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
                for ci, cell in enumerate(row.cells):
                    cell.width = Cm(widths[ci])
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    props = cell._tc.get_or_add_tcPr()
                    if ri == 0:
                        shading = OxmlElement("w:shd")
                        shading.set(qn("w:fill"), "E8EDF2")
                        props.append(shading)
                    margins = OxmlElement("w:tcMar")
                    for edge in ("top", "bottom", "left", "right"):
                        item = OxmlElement("w:" + edge)
                        item.set(qn("w:w"), "55")
                        item.set(qn("w:type"), "dxa")
                        margins.append(item)
                    props.append(margins)
                    for para in cell.paragraphs:
                        para.paragraph_format.space_after = Pt(1)
                        para.paragraph_format.space_before = Pt(1)
                        para.paragraph_format.line_spacing = 1.01
                        for run in para.runs:
                            run.font.size = Pt(11)
                            run.bold = ri == 0
            borders = OxmlElement("w:tblBorders")
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                item = OxmlElement("w:" + edge)
                for key, content in [("val", "single"), ("sz", "4"), ("color", "D9D9D9")]:
                    item.set(qn("w:" + key), content)
                borders.append(item)
            tab._tbl.tblPr.append(borders)
            spacer = doc.add_paragraph()
            spacer.paragraph_format.line_spacing = Pt(3)
            spacer.paragraph_format.space_after = Pt(0)
            md.append("\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows]))
    md.append("## Phụ lục ma trận số đếm ở seed chính")
    for model in models:
        for reject in [False, True]:
            path = OUT / (model["id"] + "_confusion" + ("_reject" if reject else "") + ".csv")
            with path.open(encoding="utf-8-sig") as file:
                lines = list(csv.reader(file))
            md.append("### " + model["name"] + (" có từ chối" if reject else " gợi ý đầu"))
            md.append("\n".join(["| " + " | ".join(lines[0]) + " |", "| " + " | ".join(["---"] * len(lines[0])) + " |"] + ["| " + " | ".join(row) + " |" for row in lines[1:]]))
    md.append("Hàng có support bằng 0 là nhãn chưa có mẫu kiểm tra; không diễn giải là không có lỗi. Ba seed lặp cùng năm clip, không phải mười lăm mẫu độc lập.")
    doc.save(OUT / (output_name + ".docx"))
    (OUT / (output_name + ".md")).write_text("\n\n".join(md) + "\n", encoding="utf-8")
    print(OUT / (output_name + ".docx"))


if __name__ == "__main__":
    build()
