"""Build the course report from frozen historical results and the v6 study."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as base


def load(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def main():
    study = load("rejection_v6.json")
    checks = load("test_results_v6.json")
    browser = load("browser_checks_v6.json")
    if not checks["passed"] or not browser["passed"]:
        raise RuntimeError("Chỉ dựng bản bàn giao sau khi các kiểm tra phần mềm đã đạt.")
    old = load("metrics.json")
    methods = study["methods"]
    default = study["recommended_default"]
    selected = "TCN học thêm lớp ngoài bộ từ" if default == "tcn_reject" else "TCN gốc"
    primary = {m["id"]: m["primary"] for m in methods}
    baseline = primary["tcn"]
    candidate = primary["tcn_reject"]
    promoted = study["promotion_passed"]
    decision = ("Ứng viên đạt toàn bộ quy tắc đã chốt và được chọn làm mặc định trong bản demo."
                if promoted else "Ứng viên chưa đạt toàn bộ quy tắc đã chốt; TCN gốc tiếp tục làm mặc định, ứng viên được giữ như một lựa chọn thử nghiệm.")
    base.blocks = []
    page, h, p, table, fig = base.page, base.h, base.p, base.table, base.fig

    page("Nhận diện ký hiệu tiếng Việt từ video")
    p("Báo cáo đồ án Ký Hiệu Lab phiên bản 6")
    p("Sinh viên Huỳnh Gia Huy | MSSV 2474802010135 | Trường Đại học Văn Lang")
    p("Ngày cập nhật 07/11/2026")
    h("Tóm tắt kết quả")
    p("Dự án đã có ứng dụng cục bộ để nhận diện thử năm ký hiệu riêng lẻ từ video, chọn đoạn cần xử lý, quay clip bằng camera và ghi phiếu thực nghiệm. Bản v6 bổ sung một mô hình học nhóm ngoài bộ nhãn, so sánh với TCN gốc trên các nguồn giữ riêng, đồng thời hoàn thiện phần hiển thị và ghi nhận quyết định từ chối.")
    p(f"Trong so sánh v6 ở seed chính, TCN gốc chấp nhận đúng {baseline['known']['accepted_correct']}/5 clip hợp lệ và nhận nhầm {baseline['outside']['false_accepts']}/29 clip ngoài bộ từ; ứng viên tương ứng là {candidate['known']['accepted_correct']}/5 và {candidate['outside']['false_accepts']}/29. {decision}")
    table(["Phạm vi", "Số lượng và ý nghĩa"], [
        ("Nhãn hỗ trợ", "ai cho, bực mình, ghen tị, không cần, xin lỗi"),
        ("Nguồn ký hiệu hỗ trợ", "8 video của một người ký; 5 clip thuộc 2 nhãn có thể giữ riêng"),
        ("Nguồn ngoài bộ từ", "29 video cũ được đưa vào nghiên cứu phát triển cơ chế từ chối"),
        ("Video thực mới trong v6", "0; tăng cường và đổi seed không tạo thêm lần thực hiện thực"),
        ("Mặc định của ứng dụng", selected),
    ], [5.1, 11.2])
    p("Bản hiện tại đủ để trình bày quy trình kỹ thuật và tổ chức demo có giám sát trong phạm vi đồ án. Chưa có kiểm chứng độc lập cho ba nhãn đơn mẫu, người ký mới hoặc webcam vật lý trên máy báo cáo. Thực nghiệm mới là nghiên cứu phát triển hồi cứu sau khi đã xem kết quả v5, vì vậy chưa được xem như test mù cho chất lượng sử dụng rộng.")
    p("Đóng góp của đồ án là tích hợp xử lý video, đặc trưng tay và thân, thực nghiệm tái lập và ứng dụng có sổ ghi kết quả. MediaPipe, MLP và TCN là các phương pháp đã có; dự án không tuyên bố phát minh thuật toán học sâu mới.")

    page("1 Bài toán và dữ liệu")
    p("Đầu vào là một video chứa một người thực hiện một động tác, hoặc một đoạn được chọn trong video. Đầu ra gồm nhãn gợi ý, điểm chưa hiệu chuẩn và trạng thái chấp nhận hoặc từ chối. Ứng dụng xử lý cả clip sau khi quay xong; chưa phân đoạn câu hay dịch ngôn ngữ ký hiệu liên tục.")
    table(["Nhãn", "Video thực", "Clip giữ riêng", "Giới hạn"], [
        ("ai cho", 3, 3, "Cùng người ký"), ("không cần", 2, 2, "Cùng người ký"),
        ("bực mình", 1, 0, "Chưa kiểm tra riêng"), ("ghen tị", 1, 0, "Chưa kiểm tra riêng"),
        ("xin lỗi", 1, 0, "Chưa kiểm tra riêng"),
    ], [4.1, 2.6, 3.4, 6.2])
    p("Tám video thuộc mã giahuy trong bộ v3. Người thu cho phép dùng các động tác và nhãn đã kiểm tra để học thử. Hồ sơ nguồn vẫn giữ reviewStatus pending và trainingEligible false; quyền học thử không được chuyển thành xác nhận chuyên gia đã duyệt.")
    p("Ngoài ra có 29 video có nhãn khác năm nhãn hỗ trợ, gồm 17 clip giahuy và 12 clip tp. Trong v5 chúng chỉ dùng khảo sát với trọng số đã khóa; sang v6, sau khi đã thấy vấn đề nhận nhầm, chúng được sử dụng để phát triển mô hình từ chối. Việc tái sử dụng này được công bố rõ, không giữ cách gọi chúng là tập kiểm tra mới độc lập.")
    h("Giữ nguồn và phiên bản dữ liệu")
    p("Mã video, mã người ký, nhãn gốc và SHA256 được giữ để đối chiếu. Các biến thể của cùng nguồn luôn đi cùng nhau trong phần học hoặc phần giữ riêng. Mã tp chỉ có ở các nhãn ngoài phạm vi, nên chưa tạo thành phép kiểm tra nhận diện năm nhãn trên người mới.")
    p("Kế hoạch collection_plan_250.csv chứa 250 vị trí dự kiến thu tiếp, chưa phải 250 video đã có. Gói bàn giao giữ tám MP4 hỗ trợ, đặc trưng đã khóa và bản video phát lại của 29 clip ngoài bộ từ. Các video phát lại được mã hóa từ đúng nguồn v3; mã băm bitstream khác bản chuẩn hóa v5 nên không gắn chúng với đặc trưng mới. Nghiên cứu v6 dùng nguyên đặc trưng đã khóa của v5.")

    page("2 Xử lý video và mô hình")
    p("Tám clip hỗ trợ giữ mốc nội dung đã cắt ở v3 và được chuẩn hóa thành H.264, yuv420p, 12 fps, cạnh dài tối đa 640 pixel, bỏ âm thanh. Một nguồn có tốc độ khai báo 1.000 fps đã được chuyển về 12 fps để tránh giải mã dư thừa. Đặc trưng được trích lại từ chính video chuẩn hóa. Giao diện cho chọn mốc đầu và cuối, tối đa 15 giây, để bỏ khoảng chờ khi cần.")
    table(["Thành phần", "Kích thước", "Cách biểu diễn"], [
        ("Hai bàn tay", 132, "Mỗi tay có 63 tọa độ hình dạng, 2 vị trí theo vai và 1 mặt nạ"),
        ("Thân trên", 27, "13 điểm XY và 1 mặt nạ"),
        ("Một chuỗi", "32 × 159", "Lấy mẫu theo thời gian, giữ riêng mặt nạ điểm thiếu"),
    ], [3.9, 3.2, 9.2])
    p("MediaPipe Holistic trích điểm tay và tư thế [1]. Hình dạng tay được tính theo gốc cổ tay và kích thước bàn tay; vị trí cổ tay và thân tính theo tâm vai, khoảng cách vai và tỷ lệ khung hình. Không dùng đặc trưng mặt. Thống kê chuẩn hóa chỉ được tính trên phần học của mỗi lượt; nội suy không tự điền tay hợp lệ qua khoảng mất điểm dài.")
    table(["Mạng", "Kiến trúc và phần học", "Tham số"], [
        ("MLP", "5.088 → 96 → 48 → 5", "493.445"),
        ("TCN gốc", "159 → 48; hai khối tích chập nối tắt; pooling; 5 đầu ra", "26.501"),
        ("TCN tăng cường", "Cùng TCN gốc; cân bằng nhãn và thay biến thể phần học", "26.501"),
        ("TCN từ chối", "Cùng TCN; 6 đầu ra, gồm một nhóm ngoài bộ nhãn", "26.550"),
    ], [3.6, 9.8, 2.9])
    p("TCN dùng kernel 3, dilation 1 và 2, nối tắt, gộp trung bình và cực đại ở các bước có tay [2]. Tất cả mô hình dùng toàn đoạn đã quay. Nếu ít hơn tám bước có tay thì ứng dụng từ chối; nếu điểm đầu dưới 0,65 thì báo chưa chắc chắn. Riêng ứng viên luôn từ chối khi nhóm ngoài bộ nhãn đứng đầu, dù điểm của nhóm đó là bao nhiêu. Điểm không được đổi thành xác suất đúng đã hiệu chuẩn.")

    page("3 Thực nghiệm nền trong bộ năm nhãn")
    p("Thực nghiệm v4 và v5 được giữ nguyên để đối chiếu lịch sử. Với mỗi clip có mẫu lặp, giữ clip đó ngoài phần học và học trên bảy video còn lại. Có năm clip đủ điều kiện thuộc ai cho và không cần. Ba nhãn đơn mẫu chưa có mẫu kiểm tra. Mọi tăng cường và thống kê chuẩn hóa được tạo sau khi chia nguồn [3].")
    table(["Thông số", "Giá trị"], [
        ("Seed", "20261103, 20261104, 20261105"),
        ("Tối ưu", "500 bước; batch 64; Adam; learning rate 0,001"),
        ("Điều chuẩn", "L2 0,00001; label smoothing 0,02"),
        ("Ngưỡng", "0,65, khóa trước đánh giá và chưa hiệu chỉnh"),
        ("Số lượt lịch sử", "3 phương pháp × 3 seed × 5 clip = 45 lượt"),
    ], [5, 11.3])
    table(["Phương pháp", "Gợi ý đúng", "Macro F1", "Nhận đúng", "Nhận sai"], [
        (m["name"], f"{m['primary']['top1_correct']}/5", f"{m['primary']['macro_f1_two_tested_labels']:.3f}", m["primary"]["accepted_correct"], m["primary"]["accepted_wrong"]) for m in old["models"]
    ], [4.1, 3.1, 3.0, 3.0, 3.1])
    p("Bảng dùng seed chính 20261103. Macro F1 chỉ lấy trung bình hai nhãn có mẫu giữ riêng. TCN gốc đúng bốn gợi ý đầu, chấp nhận ba gợi ý đúng và không chấp nhận gợi ý sai ở seed này. Số lượng năm clip quá nhỏ để kết luận ưu thế có ý nghĩa thống kê.")
    p("Qua ba seed, số gợi ý đầu đúng của TCN gốc lần lượt là 4, 5 và 4 trên cùng năm clip; MLP là 4, 5 và 3; TCN tăng cường là 3, 5 và 3. Ba lần khởi tạo đo biến động của quá trình học, không tạo thành 15 clip độc lập. Nhận lại đúng video đã học cũng không chứng minh khả năng nhận diện người mới.")

    page("4 Ma trận kết quả nền")
    fig("confusion_primary.png", "Hình 1 Gợi ý đầu trên năm clip giữ riêng của thực nghiệm nền")
    fig("confusion_reject_primary.png", "Hình 2 Quyết định có từ chối ở ngưỡng 0,65 của thực nghiệm nền")
    p("Hàng là nhãn người thu cung cấp, cột là kết quả mô hình. Ba nhãn không có clip giữ riêng được thể hiện trong CSV đầy đủ với support bằng 0; số 0 ở đó không phải bằng chứng không mắc lỗi. Ma trận nguồn riêng của nghiên cứu v6 nằm trong hồ sơ rejection_v6, không cộng vào ma trận nền này.")

    page("5 Phát triển cơ chế từ chối")
    p("Khảo sát v5 dùng trọng số học đủ tám video với ngưỡng đã khóa trên 29 clip ngoài bộ từ: MLP chấp nhận nhầm 12/29, TCN gốc 14/29 và TCN tăng cường 10/29. Kết quả này cho thấy điểm softmax cao trong năm nhãn chưa đủ xác định đầu vào thuộc bộ nhãn. Đó là khảo sát tiện có, chưa phải tỷ lệ lỗi đại diện cho mọi động tác thực tế.")
    p("Bản v6 thêm đầu ra thứ sáu cho nhóm ngoài bộ nhãn và lấy batch cân bằng sáu lớp. Năm nhãn hỗ trợ không đổi. Cấu hình được học sâu bằng JAX; mô hình demo chạy suy luận bằng NumPy. Phép so sánh đo cả việc học nhóm ngoài bộ nhãn và lấy mẫu cân bằng, chưa cô lập đóng góp của từng thay đổi.")
    h("Cách giữ nguồn trong nghiên cứu mới")
    p("Chốt protocol, danh sách nguồn và quy tắc chọn mô hình trước khi học v6. Năm fold lần lượt giữ một trong năm clip hợp lệ. Hai mươi chín nguồn ngoài bộ nhãn được sắp theo mã băm rồi chia cố định thành năm nhóm; ở từng fold, ứng viên học trên bảy clip hợp lệ và 23 hoặc 24 clip ngoài bộ nhãn, dự đoán clip hợp lệ cùng nhóm ngoài bộ nhãn giữ riêng. TCN gốc học trên cùng bảy clip hợp lệ để làm đối chứng.")
    p("Mọi biến thể và thống kê chuẩn hóa chỉ lấy từ phần học. Mã nguồn và SHA256 của phần giữ riêng được kiểm tra không nằm trong phần học. Có ba seed trên cùng cách chia nguồn, cùng ngưỡng 0,65 và số bước 500. Các clip ngoài bộ nhãn không xuất hiện trong phần học của fold dự đoán chính chúng; tuy nhiên toàn bộ nghiên cứu vẫn là phát triển hồi cứu sau khi đã biết chẩn đoán v5.")
    h("Quy tắc chọn mặc định")
    p("Ứng viên chỉ được chọn nếu số gợi ý đầu đúng và số chấp nhận đúng trên năm clip hợp lệ không giảm, số chấp nhận sai không tăng, đồng thời số nhận nhầm ngoài bộ từ giảm. Các điều kiện phải đúng ở seed chính và trung bình ba seed. Không dò lại ngưỡng theo kết quả giữ riêng. Sau so sánh, trọng số ứng viên cho demo được học trên đủ 37 nguồn, tách với checkpoint giữ riêng.")

    page("6 Kết quả nghiên cứu phát triển")
    table(["Mô hình", "Gợi ý đúng", "Nhận đúng", "Từ chối hợp lệ", "Nhận sai", "Nhầm ngoài bộ"], [
        (m["name"], f"{m['primary']['known']['top1_correct']}/5", m["primary"]["known"]["accepted_correct"], m["primary"]["known"].get("false_rejects", 5-m["primary"]["known"]["accepted"]), m["primary"]["known"]["accepted_wrong"], f"{m['primary']['outside']['false_accepts']}/29") for m in methods
    ], [3.5, 2.5, 2.4, 2.7, 2.2, 3.0])
    p("Bảng dùng seed chính 20261103. Từ chối mẫu hợp lệ là số clip thuộc bộ năm nhãn bị từ chối, dù lý do là thiếu tay, điểm thấp hay nhóm ngoài bộ nhãn đứng đầu. Nhận sai là chấp nhận một nhãn không khớp clip hợp lệ. Nhầm ngoài bộ là chấp nhận một trong năm nhãn cho clip ngoài phạm vi. Mẫu số 5 và 29 được báo riêng.")
    rows = []
    for m in methods:
        for item in m["by_seed"]:
            rows.append((m["name"], item["seed"], f"{item['known']['top1_correct']}/5", f"{item['known']['accepted_correct']}/5", item["known"]["accepted_wrong"], f"{item['outside']['false_accepts']}/29"))
    table(["Mô hình", "Seed", "Gợi ý đúng", "Nhận đúng", "Nhận sai", "Nhầm ngoài bộ"], rows, [3.5, 2.6, 2.5, 2.5, 2.2, 3.0])
    p(decision + " Các điều kiện được lưu từng mục trong reports/rejection_v6.json để kiểm tra lại quyết định.")
    p("Hai tập clip có mục đích khác nhau nên không cộng thành một accuracy tổng hợp. Ba seed dùng lại cùng 34 nguồn giữ riêng, không phải 102 video độc lập. Ba nhãn đơn mẫu và người ký mới vẫn chưa được kiểm chứng. Không so trực tiếp số nhận nhầm trong bảng này với 14/29 của v5 để suy ra mức cải thiện: v5 dùng trọng số học đủ tám clip, còn v6 dùng trọng số theo fold học bảy clip.")

    page("7 Ma trận của nghiên cứu phát triển")
    supported = study["scope"]["supported_labels"]
    method_names = {m["id"]: m["name"] for m in methods}
    with (ROOT / "reports/rejection_v6_known_decision_confusion.csv").open(encoding="utf-8-sig") as file:
        known_rows = list(csv.DictReader(file))
    repeated = study["scope"]["known_tested_labels"]
    table(["Mô hình", "Nhãn thật", "ai cho", "không cần", "Nhãn khác", "Từ chối"], [
        (method_names[r["method"]], r["true_label"], r["ai cho"], r["không cần"], sum(int(r[label]) for label in supported if label not in repeated), r["__REJECT__"])
        for r in known_rows if r["true_label"] in repeated
    ], [3.6, 3.3, 2.0, 2.4, 2.5, 2.5])
    p("Ma trận trên chỉ gồm năm clip hợp lệ thuộc hai nhãn có mẫu lặp. Nhãn khác gộp các dự đoán bực mình, ghen tị và xin lỗi; CSV đi kèm giữ đủ từng cột. Ba nhãn đó không có hàng kiểm tra với support dương. Mỗi phương pháp có tổng số đếm bằng năm.")
    with (ROOT / "reports/rejection_v6_outside_decision_confusion.csv").open(encoding="utf-8-sig") as file:
        outside_rows = list(csv.DictReader(file))
    table(["Mô hình", "ai cho", "bực mình", "ghen tị", "không cần", "xin lỗi", "Từ chối"], [
        (method_names[r["method"]], *[r[label] for label in supported], r["__REJECT__"]) for r in outside_rows
    ], [3.5, 1.8, 2.1, 2.0, 2.4, 2.0, 2.5])
    p("Ma trận thứ hai chỉ gồm 29 clip ngoài bộ từ, tất cả có nhãn thật là ngoài phạm vi. Mỗi phương pháp có tổng bằng 29. Mọi dự đoán được chấp nhận vào một trong năm nhãn đều là nhận nhầm; cột từ chối ghi đúng quyết định của hệ thống, gồm điểm thấp hoặc nhóm ngoài bộ nhãn đứng đầu.")
    fig("rejection_v6_matched_cv.png", "Hình 3 Kết quả ghép cặp ở seed chính với mẫu số tách riêng")
    p("Không cộng hai ma trận để báo một accuracy tổng hợp. Ma trận này thuộc nghiên cứu phát triển hồi cứu; các nguồn đã được biết trước, dù từng nguồn được giữ ngoài phần học khi tạo dự đoán cho chính nó.")

    page("8 Ứng dụng và kiểm tra phần mềm")
    table(["Hạng mục", "Kết quả kiểm tra v6"], [
        ("Dữ liệu và máy chủ", f"{checks['tests_run']} kiểm tra đạt; có nguồn dữ liệu, mask, API và phiếu thực nghiệm"),
        ("Trình duyệt thật", f"{browser['checks_passed']} tình huống đạt qua Chromium và Playwright"),
        ("Đầu vào", "Có đường MP4 thực qua bộ trích điểm; clip mẫu là dữ liệu học"),
        ("Camera tự động", "Dùng nguồn giả lập để thử luồng quay; chưa kiểm tra webcam vật lý"),
        ("Máy trình bày", "Windows, camera thật và CSV trong Excel cần chạy trên thiết bị thực"),
    ], [4.6, 11.7])
    p("Giao diện có bốn mục Nhận diện, Thực nghiệm, Đánh giá và Hướng dẫn, với bố cục trên máy tính và màn hình điện thoại mô phỏng. Danh sách mô hình lấy từ máy chủ. Đổi video, mô hình hoặc mốc cắt yêu cầu nhận diện lại; kết quả mạng đến muộn không được dùng để ghi phiếu cho cấu hình khác. Camera được dừng khi rời thao tác sử dụng.")
    p("Khi nhóm ngoài bộ nhãn đứng đầu, giao diện hiển thị có thể nằm ngoài bộ năm ký hiệu và chỉ trình bày các gợi ý thuộc năm nhãn hỗ trợ. Điểm gợi ý không được chuẩn hóa lại. Sổ phiếu vẫn giữ lớp và điểm quyết định gốc, tránh tính nhầm một gợi ý đứng sau thành kết quả nhận diện đúng.")
    p("Máy chủ lấy dự đoán từ kết quả đã xử lý qua mã inference_id, không nhận điểm tự khai từ trình duyệt. Mỗi mã chỉ được ghi một lần; sổ lỗi trả thông báo thay vì bỏ dòng. Tổng hợp phân biệt số lượt, video có mã băm khác nhau, mã người do người dùng nhập, video trùng nguồn học và phiên bản trọng số. Với ứng viên, nguồn học gồm cả nhóm ngoài bộ từ. Nhãn phản hồi chưa được duyệt và không tự thay trọng số.")
    p("Máy chủ chỉ lắng nghe tại 127.0.0.1. Video đầu vào được xóa khi xử lý xong; người dùng có thể tải giữ clip để lập bộ thu mới. Kết quả kiểm tra tự động được ghi trong thư mục tạm riêng. Tất cả kiểm tra phần mềm xác nhận đường xử lý chạy được, chưa thay cho phép đo độ chính xác bằng dữ liệu thực mới.")

    page("9 Chạy thử và tái lập")
    h("Trải nghiệm bản demo")
    p("Giải nén toàn bộ gói, cài Python 3.11 64-bit rồi mở CHAY_DEMO.bat. Khi máy chủ sẵn sàng, vào http://localhost:8765 bằng Chrome hoặc Edge. Lần đầu cần mạng để cài thư viện và chuẩn bị MediaPipe. KIEM_TRA_CAI_DAT.bat chạy lại kiểm tra khi cần. Chọn clip mẫu, nhận diện, chuyển qua mục Thực nghiệm, ghi nhãn và tải CSV.")
    p("Tiếp theo thử một clip mới hoặc quay bằng camera trong khoảng 3 đến 8 giây, nhìn rõ tay và thân trên. Dùng mốc đầu và cuối để bỏ khoảng chờ, tối đa 15 giây; cuối bằng 0 lấy đến hết. Ghi cả lượt sai và bị từ chối. Một video được nhận diện nhiều lần vẫn chỉ là một lần thực hiện động tác, không tăng số mẫu độc lập.")
    h("Các lệnh tái lập")
    p("Cài requirements-research.txt trong môi trường riêng. research/run_experiments.py tái lập nghiên cứu nền; research/summarize.py dựng bảng và ma trận nền. research/audit_v5.py phát lại khảo sát v5 đã khóa. research/rejection_v6.py tái lập nghiên cứu phát triển, dùng checkpoint và protocol ghép cặp của v6; đọc --help trước khi yêu cầu học lại.")
    p("Chạy tests/run_checks_v6.py để kiểm tra dữ liệu và API. tests/test_browser_v6.py chạy trình duyệt thật; đường dẫn Chromium và thư mục đầu ra được mô tả trong reports/QA_V6.md. Kiểm tra tạo video màu cần FFmpeg, còn demo chỉ cần bộ thư viện Python. Báo cáo Word và Markdown được dựng bằng research/build_report_v6.py; cần python-docx để tạo lại Word.")
    p("Phiếu của người dùng nằm ở outputs/trials.jsonl. Lệnh tools/summarize_trials.py tạo thống kê mới mà không nhập phản hồi vào phần học. Các checkpoint và báo cáo đã bàn giao có thể đọc ngay, không cần học lại. Hai ví dụ ngoài bộ nhãn trong giao diện là nguồn ứng viên đã học, dùng trải nghiệm thao tác từ chối; không dùng nhận lại đúng các mẫu này làm điểm kiểm tra. Video và mã băm nguồn được lưu trong data/outside_manifest_v6.json; đặc trưng giữ nguyên bản đã khóa để tái lập số học.")
    h("Lịch tiếp tục ngày 07 11 2026")
    p("Lịch trợ lý tiếp tục công việc được đặt mỗi giờ từ 10:45 đến 23:45 theo giờ Việt Nam, kết thúc trong ngày. Mỗi lượt đọc tiến độ và dùng lại checkpoint, chỉ xử lý việc còn thiếu hoặc dữ liệu mới có thật. Lịch này không thay thế tiến trình GPU chạy liên tục và không tạo thêm video thực. Phần cần người quay và máy vật lý được giữ thành công việc còn thiếu.")

    page("10 Kết luận và kế hoạch thực nghiệm")
    p(f"Dự án đã có bản đồ án kỹ thuật chạy được, mã học sâu, bốn cấu hình trọng số, ma trận và kiểm tra trình duyệt thật. Bản v6 bổ sung cơ chế từ chối với nguồn giữ riêng và ghi nhận đầy đủ ảnh hưởng tới clip hợp lệ. Mặc định hiện tại là {selected}. Có thể dùng để trình bày và trải nghiệm có giám sát trong phạm vi năm ký hiệu, với giới hạn dữ liệu được nêu cùng kết quả.")
    p("Để kết luận chất lượng sử dụng thực tế, cần thu thêm lần thực hiện mới cho cả năm nhãn, đặc biệt ba nhãn đang có một mẫu. Kế hoạch tiếp theo là 5 nhãn × 5 người × 10 lượt = 250 clip, chia hai phiên mỗi người. Có thể dành ba người cho học, một người cho validation và một người cho test; mọi biến thể và phiên liên quan phải được quản lý theo nguồn và người [5]. Một người test vẫn còn ít, nên cần báo kết quả theo từng người và từng nhãn.")
    p("Thu thêm hoạt động ngoài bộ từ đa dạng, duyệt nhãn, chốt ngưỡng trên validation trước khi chạy test mới. Giữ phần test đó kín trong quá trình phát triển. Trước buổi báo cáo, chạy camera vật lý, cắt đoạn và xuất CSV trên chính máy trình bày, ghi thời gian xử lý và giữ cả trường hợp thất bại. Các bước này chưa thể xác nhận chỉ bằng việc chạy lại video đang có.")
    h("Tài liệu tham khảo")
    for text in [
        "[1] Google MediaPipe. Holistic. https://chuoling.github.io/mediapipe/solutions/holistic.html",
        "[2] Bai, Kolter và Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. 2018. https://arxiv.org/abs/1803.01271",
        "[3] scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html",
        "[4] scikit-learn. classification_report. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html",
        "[5] scikit-learn. Cross-validation. https://scikit-learn.org/stable/modules/cross_validation.html",
    ]:
        p(text)
    base.build("Bao_cao_DoAn_KyHieuLab_v6", "07/11/2026")


if __name__ == "__main__":
    main()
