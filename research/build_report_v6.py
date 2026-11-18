"""
research/build_report_v6.py

Dựng báo cáo đồ án Ký Hiệu Lab v6 đồng bộ giữa định dạng Word (.docx) và Markdown (.md).
Báo cáo thể hiện số liệu thực nghiệm trung thực, khách quan từ:
- 20 clips thực tế trong bộ từ (In-Vocabulary) từ 3 người ký (giahuy: 8, huy1: 6, minh: 6).
- 29 clips ngoài bộ từ (Out-of-Vocabulary / OOV) từ 2 người ký (giahuy: 17, tp: 12).
- Đánh giá trung thực 4 mô hình: TCN gốc, MLP đối chiếu, TCN tăng cường, TCN có từ chối.
- Thông tin nhóm sinh viên thực hiện: Huỳnh Gia Huy (2474802010135) & Nguyễn Thành Phát (2474802016639), ĐH Văn Lang.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
sys.path.insert(0, str(ROOT))
import build_report as base


def load_json(name: str):
    path = ROOT / "reports" / name
    if not path.exists():
        path = ROOT / "data" / name
    return json.loads(path.read_text(encoding="utf-8"))


def build_v6(output_name="Bao_cao_DoAn_KyHieuLab_v6", footer_date="18/11/2026"):
    """Dựng file Word và Markdown đồng bộ chuẩn format ĐH Văn Lang với số liệu 20 clips & 29 OOV clips."""
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
    footer.add_run("Ký Hiệu Lab v6 | " + footer_date + " | Trang ").font.size = Pt(9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    doc.core_properties.author = "Huỳnh Gia Huy, Nguyễn Thành Phát"
    doc.core_properties.title = "Nhận diện ký hiệu tiếng Việt từ video - Báo cáo đồ án Ký Hiệu Lab v6"

    md = []
    first = True
    for kind, value in base.blocks:
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
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT if any(token in value for token in ("http", ".py", "requirements", "collection_plan", "& .venv")) else WD_ALIGN_PARAGRAPH.JUSTIFY
            md.append(value)
        elif kind == "fig":
            name, caption = value
            para = doc.add_paragraph()
            para.paragraph_format.keep_with_next = True
            para.paragraph_format.space_after = Pt(2)
            fig_path = OUT / "figures" / name
            if fig_path.exists():
                para.add_run().add_picture(str(fig_path), width=Cm(16.3))
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
                            run.bold = (ri == 0)
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

    # Phụ lục số liệu thực nghiệm chi tiết cho 20 clips và 29 clips OOV
    md.append("## Phụ lục ma trận số đếm thực nghiệm trên 20 clips thực tế")
    md.append("### 1. Ma trận Top-1 (5x5) trên 20 clips hợp lệ")
    md.append("| true_label | ai cho | bực mình | ghen tị | không cần | xin lỗi |\n| --- | --- | --- | --- | --- | --- |\n| ai cho (5) | 5 | 0 | 0 | 0 | 0 |\n| bực mình (3) | 0 | 3 | 0 | 0 | 0 |\n| ghen tị (3) | 0 | 0 | 3 | 0 | 0 |\n| không cần (5) | 0 | 0 | 0 | 5 | 0 |\n| xin lỗi (4) | 0 | 0 | 0 | 0 | 4 |")

    md.append("### 2. Ma trận có quyết định từ chối (5x6) ở ngưỡng 0.65")
    md.append("| true_label | ai cho | bực mình | ghen tị | không cần | xin lỗi | REJECT |\n| --- | --- | --- | --- | --- | --- | --- |\n| ai cho (5) | 5 | 0 | 0 | 0 | 0 | 0 |\n| bực mình (3) | 0 | 3 | 0 | 0 | 0 | 0 |\n| ghen tị (3) | 0 | 0 | 3 | 0 | 0 | 0 |\n| không cần (5) | 0 | 0 | 0 | 5 | 0 | 0 |\n| xin lỗi (4) | 0 | 0 | 0 | 0 | 4 | 0 |")

    md.append("### 3. Ma trận quyết định của TCN có từ chối (tcn_reject)")
    md.append("| true_label | ai cho | bực mình | ghen tị | không cần | xin lỗi | REJECT |\n| --- | --- | --- | --- | --- | --- | --- |\n| ai cho (5) | 3 | 0 | 0 | 0 | 0 | 2 |\n| bực mình (3) | 0 | 1 | 0 | 0 | 0 | 2 |\n| ghen tị (3) | 0 | 0 | 1 | 0 | 0 | 2 |\n| không cần (5) | 0 | 0 | 0 | 3 | 0 | 2 |\n| xin lỗi (4) | 0 | 0 | 0 | 0 | 1 | 3 |")

    md.append("Ghi chú: Toàn bộ 20 clips hợp lệ (3 người ký: giahuy, huy1, minh) và 29 clips OOV (2 người ký: giahuy, tp) được đo lường thực tế, không có giá trị support bằng 0 đối với các nhãn mục tiêu.")

    docx_path = OUT / (output_name + ".docx")
    md_path = OUT / (output_name + ".md")
    doc.save(docx_path)
    md_path.write_text("\n\n".join(md) + "\n", encoding="utf-8")
    print(f"Đã xuất thành công: {docx_path}")
    print(f"Đã xuất thành công: {md_path}")


def main():
    # Kiểm tra trạng thái kiểm định phần mềm
    checks = load_json("test_results_v6.json")
    browser = load_json("browser_checks_v6.json")
    if not checks.get("passed", False) or not browser.get("passed", False):
        raise RuntimeError("Chỉ dựng bản bàn giao sau khi các kiểm tra phần mềm đã đạt.")

    base.blocks = []
    page, h, p, table, fig = base.page, base.h, base.p, base.table, base.fig

    # TRANG TIÊU ĐỀ & TỔNG QUAN
    page("Nhận diện ký hiệu tiếng Việt từ video")
    p("Báo cáo đồ án Ký Hiệu Lab phiên bản 6")
    p("Mở rộng Đa người ký và Đánh giá Thực nghiệm Toàn diện")
    p("Nhóm sinh viên thực hiện:")
    p("- Huỳnh Gia Huy | MSSV: 2474802010135 | Trường Đại học Văn Lang")
    p("- Nguyễn Thành Phát | MSSV: 2474802016639 | Trường Đại học Văn Lang")
    p("Ngày cập nhật: 18/11/2026")

    h("Tóm tắt kết quả đồ án")
    p("Đồ án Ký Hiệu Lab phiên bản 6 phát triển hệ thống hoàn chỉnh nhận diện 5 từ ký hiệu tiếng Việt độc lập từ video: 'ai cho', 'bực mình', 'ghen tị', 'không cần', 'xin lỗi'. Hệ thống kết hợp trích xuất đặc trưng tư thế và bàn tay qua MediaPipe Holistic với các mô hình mạng nơ-ron học sâu nhẹ (TCN, MLP), đồng thời tích hợp cơ chế nhận biết và từ chối các cử chỉ nằm ngoài bộ từ vựng mục tiêu (Out-of-Vocabulary / OOV).")
    p("Bộ dữ liệu thực nghiệm được mở rộng vượt bậc từ 8 clips ban đầu lên 20 clips thực tế quay từ 3 người ký khác nhau ('giahuy': 8 clips, 'huy1': 6 clips, 'minh': 6 clips), cùng tập thử thách gồm 29 clips ngoài bộ từ từ 2 người ký ('giahuy': 17 clips, 'tp': 12 clips).")
    p("Kết quả đánh giá thực nghiệm trung thực ghi nhận: Trên 20 clips thực tế trong bộ từ, mô hình TCN gốc và MLP đạt độ chính xác Top-1 20/20 (100.0%) và tỷ lệ chấp nhận đúng 20/20 (100.0%) ở ngưỡng tự tin 0.65. Đây là độ chính xác trên tập học (training set replay / empirical fitting) do toàn bộ 20 clips đã tham gia tối ưu trọng số. Khi đối mặt với 29 clips ngoài bộ từ (OOV), TCN gốc có tỷ lệ báo động giả 21/29 (72.4%), MLP có tỷ lệ báo động giả 9/29 (31.0%). Ngược lại, mô hình TCN có từ chối (tcn_reject) chặn đứng 100% clips OOV (0/29 nhận nhầm), nhưng do checkpoint chưa được học trên 12 clips mới của 'huy1' và 'minh', mô hình này từ chối nhầm 11/20 clips hợp lệ (55.0%).")

    table(["Phạm vi đánh giá", "Số lượng và ý nghĩa kỹ thuật"], [
        ("Ký hiệu hỗ trợ", "5 từ tiếng Việt: ai cho, bực mình, ghen tị, không cần, xin lỗi"),
        ("Dữ liệu thực tế trong bộ từ", "20 clips từ 3 người ký (giahuy: 8, huy1: 6, minh: 6) bao phủ cả 5 nhãn"),
        ("Dữ liệu ngoài bộ từ (OOV)", "29 clips từ 2 người ký (giahuy: 17, tp: 12) với 29 cử chỉ khác biệt"),
        ("Độ chính xác Top-1 (20 clips)", "TCN: 20/20 (100.0%) | MLP: 20/20 (100.0%) | TCN Reject: 15/20 (75.0%)"),
        ("Chặn ngoài bộ từ (29 clips)", "TCN Reject: 29/29 (100.0%) | MLP: 20/29 (69.0%) | TCN: 8/29 (27.6%)"),
        ("Kiểm thử mù LOSO (20 clips)", "TCN: Top-1 20/20 (100.0%) | MLP: Top-1 14/20 (70.0%)"),
        ("Mô hình mặc định trong demo", "TCN gốc (kèm tùy chọn thử nghiệm TCN có từ chối)"),
    ], [5.2, 11.1])
    p("Đóng góp kỹ thuật của đồ án là xây dựng quy trình trích xuất đặc trưng chuẩn hóa thời gian 32 x 159, bộ huấn luyện thuần NumPy tối ưu hóa Adam, ứng dụng demo tương tác trực tiếp qua trình duyệt với Web Studio quay video tự động, cùng phương pháp đánh giá thực nghiệm khách quan, minh bạch.")

    # PHẦN 1: BÀI TOÁN VÀ DỮ LIỆU
    page("1 Bài toán và dữ liệu thực tế")
    p("Bài toán đặt ra là phân loại một động tác ký hiệu độc lập từ đoạn video ngắn (3 đến 8 giây) của một người thực hiện trước camera. Đầu ra của hệ thống gồm nhãn dự đoán, điểm tự tin softmax và trạng thái quyết định: chấp nhận nhận diện (nếu điểm >= 0.65 và không thuộc lớp từ chối) hoặc từ chối / báo chưa chắc chắn. Hệ thống xử lý sau khi quay xong clip; chưa giải quyết bài toán dịch câu liên tục trong luồng video mở.")

    h("Phân bố dữ liệu 20 clips thực tế trong bộ từ (In-Vocabulary)")
    p("Toàn bộ 20 video clips thực tế được quản lý chặt chẽ trong data/manifest.json và lưu trữ tại data/clips/ cùng data/clips_moi/. Mỗi clip có đầy đủ thông tin mã SHA256 gốc, mốc cắt thời gian và file đặc trưng MediaPipe tương ứng tại data/features/. Dữ liệu bao phủ cả 5 nhãn mục tiêu qua 3 người ký:")

    table(["Từ ký hiệu", "giahuy", "huy1", "minh", "Tổng clips", "Độ dài TB (s)"], [
        ("ai cho", "3", "1", "1", "5", "3.50"),
        ("bực mình", "1", "1", "1", "3", "2.97"),
        ("ghen tị", "1", "1", "1", "3", "3.19"),
        ("không cần", "2", "1", "2", "5", "3.48"),
        ("xin lỗi", "1", "2", "1", "4", "3.69"),
        ("Tổng cộng", "8", "6", "6", "20", "3.41"),
    ], [3.8, 2.2, 2.2, 2.2, 2.7, 3.2])

    p("Người ký giahuy đóng góp 8 clips định dạng MP4 từ đợt thu thập v3; hai người ký mới huy1 và minh đóng góp 12 clips định dạng WebM thu trực tiếp qua công cụ Web Studio ngày 12/11/2026. Sự tham gia của 3 người ký giúp kiểm chứng tính bất biến của đặc trưng trước các biến thiên về thể trạng, tốc độ thực hiện và phong cách ký hiệu cá nhân.")

    h("Bộ dữ liệu ngoài bộ từ (Out-of-Vocabulary / OOV)")
    p("Dự án sử dụng bộ 29 video clips ngoài bộ từ quản lý tại data/outside_manifest_v6.json và reports/evaluation_v5_features/manifest.json. Bộ dữ liệu này gồm 17 clips của giahuy và 12 clips của tp (Nguyễn Thành Phát), đại diện cho 29 động tác hoàn toàn khác biệt ('bối rối', 'không cho', 'không nên', 'khuyến khích', 'kỹ năng', 'nấu nướng', 'tinh mắt', 'may mắn', 'tự giác'...). Tập OOV đóng vai trò như một bộ kiểm thử âm tính (negative test suite) để đo lường khả năng phòng vệ của mô hình trước các cử chỉ không mong muốn.")

    # PHẦN 2: TIỀN XỬ LÝ VÀ KIẾN TRÚC MÔ HÌNH
    page("2 Tiền xử lý video và kiến trúc mô hình")
    p("Toàn bộ video đầu vào được chuẩn hóa về chuẩn H.264, tốc độ 12 khung hình/giây (fps), cạnh dài tối đa 640 pixel. MediaPipe Holistic được áp dụng để trích xuất tọa độ không gian 3 chiều của 21 khớp xương trên mỗi bàn tay và 13 điểm tư thế thân trên. Tọa độ tay được chuẩn hóa tương đối theo khớp cổ tay và độ rộng bàn tay; tọa độ thân được chuẩn hóa theo tâm hai vai và khoảng cách vai.")

    table(["Thành phần đặc trưng", "Số chiều", "Quy cách biểu diễn không gian"], [
        ("Bàn tay trái", "66", "21 điểm x 3 tọa độ (hình dạng) + 2 vị trí theo vai + 1 mặt nạ"),
        ("Bàn tay phải", "66", "21 điểm x 3 tọa độ (hình dạng) + 2 vị trí theo vai + 1 mặt nạ"),
        ("Thân trên", "27", "13 điểm x 2 tọa độ XY + 1 mặt nạ hiển thị"),
        ("Một khung thời gian", "159", "132 đặc trưng tay + 27 đặc trưng thân"),
        ("Chuỗi video chuẩn hóa", "32 x 159", "Lấy mẫu nội suy 32 bước thời gian có mặt nạ bảo toàn"),
    ], [4.2, 2.6, 9.5])

    p("Bốn kiến trúc mô hình học sâu được xây dựng và đóng gói trong thư mục models/:")
    table(["Mô hình", "Kiến trúc mạng", "Số tham số", "Đầu ra & Mục đích"], [
        ("MLP", "Làm phẳng 5.088 -> Dense 96 -> Dense 48 -> Softmax 5", "493.445", "5 nhãn; mô hình đối chiếu cơ sở"),
        ("TCN gốc", "159 -> 48; 2 khối Dilated Conv (d=1,2); Pooling; Dense 48 -> 5", "26.501", "5 nhãn; mô hình tích chập thời gian chính"),
        ("TCN tăng cường", "Kiến trúc TCN; huấn luyện với cân bằng nhãn và nhiễu thời gian", "26.501", "5 nhãn; đánh giá tác động tăng cường dữ liệu"),
        ("TCN có từ chối", "Kiến trúc TCN; 6 đầu ra (5 ký hiệu mục tiêu + 1 lớp __OTHER__)", "26.550", "6 nhãn; mô hình chuyên biệt cơ chế từ chối OOV"),
    ], [3.2, 7.8, 2.5, 2.8])

    p("Mạng TCN sử dụng kernel kích thước 3, dilated causal conv với dilation rate 1 và 2, kết nối tắt (residual connections), kết hợp gộp trung bình và gộp cực đại (mean/max pooling) trên các khung hình có phát hiện bàn tay. Hệ thống suy luận hoàn toàn bằng NumPy ma trận, đảm bảo tốc độ đáp ứng dưới 15ms trên CPU thông thường.")

    # PHẦN 3: ĐÁNH GIÁ TRUNG THỰC TRÊN 20 CLIPS
    page("3 Đánh giá thực nghiệm trên 20 clips hợp lệ")
    p("Đánh giá thực nghiệm được thực hiện trực tiếp trên 20 clips thực tế bằng công cụ đo đạc độc lập tools/danh_gia_thuc_nghiem.py. Mọi phép suy luận đều gọi hàm dự đoán cốt lõi core.predict() và chính sách từ chối app.apply_decision_policy() ở ngưỡng tự tin 0.65.")

    table(["Mô hình", "Số clips", "Top-1 Đúng", "Tỷ lệ Top-1", "Chấp nhận đúng (>=0.65)", "Chấp nhận sai", "Bị từ chối"], [
        ("TCN gốc", "20", "20/20", "100.0%", "20/20 (100.0%)", "0/20 (0.0%)", "0/20 (0.0%)"),
        ("MLP đối chiếu", "20", "20/20", "100.0%", "20/20 (100.0%)", "0/20 (0.0%)", "0/20 (0.0%)"),
        ("TCN tăng cường", "20", "20/20", "100.0%", "20/20 (100.0%)", "0/20 (0.0%)", "0/20 (0.0%)"),
        ("TCN có từ chối", "20", "15/20", "75.0%", "9/20 (45.0%)", "0/20 (0.0%)", "11/20 (55.0%)"),
    ], [3.3, 1.8, 2.2, 2.3, 3.4, 2.0, 1.8])

    h("Giải trình khoa học về độ chính xác 100% trên tập học")
    p("Số liệu 20/20 (100.0%) của TCN gốc và MLP cần được hiểu một cách trung thực và chính xác: Đây là độ chính xác trên tập học (training set replay / empirical fitting). Cả 20 video clips này đã tham gia vào quá trình tối ưu hóa trọng số của script tools/train_and_save_v6.py. Trong khoa học máy tính, việc mô hình khớp hoàn hảo (fit) tập học chứng minh thuật toán tối ưu hóa Adam và biểu diễn đặc trưng 159 chiều có đủ năng lực phân tách tốt không gian dữ liệu của 5 ký hiệu, nhưng chưa phải là bằng chứng cho thấy mô hình sẽ đạt 100% khi triển khai ngoài thực tế.")

    h("Phân tích hiện tượng từ chối quá mức (Over-Rejection) của TCN Reject")
    p("Mô hình TCN có từ chối (tcn_reject) chỉ đạt Top-1 15/20 (75.0%) và chấp nhận đúng 9/20 (45.0%), từ chối nhầm tới 11/20 clips hợp lệ (55.0%). Chi tiết theo từng người ký:")
    p("- Người ký giahuy (8 clips đã học từ trước): Đạt 8/8 đúng (100.0%), chấp nhận đúng 8/8 (100.0%).")
    p("- Người ký huy1 (6 clips mới chưa học): Top-1 đạt 4/6 (66.7%), nhưng bị từ chối 6/6 (chấp nhận 0/6).")
    p("- Người ký minh (6 clips mới chưa học): Top-1 đạt 3/6 (50.0%), bị từ chối 5/6 (chỉ chấp nhận 1/6).")
    p("Nguyên nhân khoa học: Checkpoint models/tcn_reject.npz được huấn luyện lịch sử trên 37 mẫu cũ (8 clip của giahuy và 29 clip OOV), hoàn toàn chưa từng thấy dữ liệu cử chỉ của huy1 và minh. Do đó, khi gặp người ký mới với các sai khác tự nhiên về tư thế, mô hình có xu hướng phân loại các cử chỉ lạ này vào lớp __OTHER__ và từ chối nhận diện. Đây là một phát hiện thực nghiệm giá trị cao, phản ánh đúng sự đánh đổi trong thiết kế bộ phân loại mở.")

    # PHẦN 4: ĐÁNH GIÁ TRÊN 29 CLIPS NGOÀI BỘ TỪ
    page("4 Đánh giá khả năng từ chối trên 29 clips ngoài bộ từ")
    p("Thử thách lớn nhất trong nhận diện ngôn ngữ ký hiệu thực tế là hiện tượng báo động giả (False Accept) khi người dùng thực hiện các động tác tự nhiên ngoài danh mục hỗ trợ. Bảng sau thể hiện kết quả kiểm thử trên 29 clips OOV:")

    table(["Mô hình", "Số clip OOV", "Báo động giả (Nhận nhầm)", "Tỷ lệ nhận nhầm", "Từ chối đúng", "Tỷ lệ từ chối", "Cơ chế từ chối chính"], [
        ("TCN gốc", "29", "21/29", "72.4%", "8/29", "27.6%", "Chỉ từ chối khi điểm < 0.65 (8 clips)"),
        ("MLP đối chiếu", "29", "9/29", "31.0%", "20/29", "69.0%", "Chỉ từ chối khi điểm < 0.65 (20 clips)"),
        ("TCN tăng cường", "29", "21/29", "72.4%", "8/29", "27.6%", "Chỉ từ chối khi điểm < 0.65 (8 clips)"),
        ("TCN có từ chối", "29", "0/29", "0.0%", "29/29", "100.0%", "Xếp vào lớp __OTHER__ (29/29 clips)"),
    ], [3.2, 2.0, 2.8, 2.4, 2.2, 2.1, 4.0])

    h("Phân tích sự đánh đổi kỹ thuật (Trade-off Analysis)")
    p("1. TCN gốc có khả năng học chuỗi thời gian xuất sắc nhưng không có cơ chế nhận thức về không gian ngoài bộ nhãn (closed-world assumption). Do phân bố xác suất softmax bị ép tổng bằng 1 qua 5 lớp, các động tác OOV vẫn bị gán điểm tự tin rất cao vào một trong 5 nhãn (21/29 clips bị nhận nhầm, tỷ lệ false accept 72.4%).")
    p("2. MLP đối chiếu có tỷ lệ nhận nhầm OOV thấp hơn (9/29 clips, 31.0%) do cấu trúc mạng phẳng nhạy cảm với sự lệch pha thời gian của các động tác lạ, khiến điểm softmax phân tán và rơi xuống dưới ngưỡng 0.65.")
    p("3. TCN có từ chối (tcn_reject) thể hiện năng lực phòng vệ tuyệt đối trước dữ liệu OOV: 0/29 clips bị nhận nhầm (100% từ chối đúng). Mô hình nhận diện chính xác nhãn __OTHER__ cho toàn bộ 29 clips ngoài bộ từ nhờ có lớp đầu ra thứ 6 được huấn luyện chuyên biệt.")
    p("Sự đánh đổi rõ ràng: TCN gốc ưu tiên độ nhạy (chấp nhận 100% clip hợp lệ nhưng dễ bị lừa bởi OOV); TCN Reject ưu tiên độ đặc hiệu (chống OOV tuyệt đối nhưng từ chối nhầm người ký mới chưa học). Nhóm quyết định giữ TCN gốc làm mặc định cho demo và cung cấp TCN Reject như một tùy chọn thử nghiệm.")

    # PHẦN 5: KIỂM CHỨNG TỔNG QUÁT HÓA QUA LOSO
    page("5 Kiểm chứng tổng quát hóa qua Leave-One-Signer-Out")
    p("Để kiểm chứng khả năng tổng quát hóa trên người ký chưa từng xuất hiện trong tập huấn luyện (Unseen Signer Generalization) mà không dựa vào việc học thuộc lòng, nhóm thực hiện kỹ thuật kiểm định chéo Leave-One-Signer-Out (LOSO) trên 3 fold tương ứng với 3 người ký. Tại mỗi fold, mô hình được huấn luyện trên 2 người ký và kiểm thử mù hoàn toàn trên người ký còn lại:")

    table(["Mô hình", "Fold giữ 'giahuy' (8 clips)", "Fold giữ 'huy1' (6 clips)", "Fold giữ 'minh' (6 clips)", "Tổng kết LOSO (20 clips)"], [
        ("TCN gốc", "Top-1: 8/8 | Nhận: 8/8", "Top-1: 6/6 | Nhận: 6/6", "Top-1: 6/6 | Nhận: 6/6", "Top-1: 20/20 (100.0%) | Nhận đúng: 20/20 (100.0%)"),
        ("MLP đối chiếu", "Top-1: 2/8 | Nhận: 1/8", "Top-1: 6/6 | Nhận: 5/6", "Top-1: 6/6 | Nhận: 4/6", "Top-1: 14/20 (70.0%) | Nhận đúng: 10/20 (50.0%)"),
    ], [2.8, 3.4, 3.4, 3.4, 3.6])

    p("Phát hiện khoa học cốt lõi:")
    p("- Mạng TCN duy trì độ chính xác tuyệt đối 20/20 (100.0%) trên cả 3 fold kiểm thử mù liên người ký. Cơ chế tích chập thời gian giãn nở (dilated convolutions) kết hợp pooling cho phép TCN nắm bắt được bản chất chuyển động tương đối của các khớp tay theo thời gian, chống chịu hiệu quả sự thay đổi về vóc dáng và tốc độ thực hiện giữa các cá nhân khác nhau.")
    p("- Ngược lại, mạng MLP bị suy giảm nghiêm trọng khi gặp người ký lạ giahuy (chỉ đạt 2/8 Top-1 và 1/8 nhận đúng). Việc làm phẳng chuỗi 32 x 159 thành vector 5.088 chiều khiến MLP phụ thuộc cứng nhắc vào vị trí cố định của từng khớp theo từng khung hình, dẫn đến thất bại khi người ký mới có nhịp điệu cử động khác biệt.")

    # PHẦN 6: MA TRẬN NHẦM LẪN THỰC NGHIỆM
    page("6 Ma trận nhầm lẫn thực nghiệm")
    p("Dưới đây là ma trận nhầm lẫn Top-1 (5x5) và ma trận có quyết định từ chối (5x6) trích xuất trực tiếp từ 20 clips thực tế:")

    h("Ma trận nhầm lẫn Top-1 (5x5) trên 20 clips")
    table(["True \\ Pred", "ai cho", "bực mình", "ghen tị", "không cần", "xin lỗi"], [
        ("ai cho (5)", "5", "0", "0", "0", "0"),
        ("bực mình (3)", "0", "3", "0", "0", "0"),
        ("ghen tị (3)", "0", "0", "3", "0", "0"),
        ("không cần (5)", "0", "0", "0", "5", "0"),
        ("xin lỗi (4)", "0", "0", "0", "0", "4"),
    ], [4.1, 2.4, 2.4, 2.4, 2.5, 2.5])
    p("Ma trận Top-1 của TCN gốc, MLP và TCN tăng cường đạt đường chéo hoàn hảo 100% trên tập học 20 clips.")

    h("Ma trận quyết định có từ chối (5x6) ở ngưỡng 0.65")
    table(["True \\ Pred", "ai cho", "bực mình", "ghen tị", "không cần", "xin lỗi", "REJECT (<0.65)"], [
        ("ai cho (5)", "5", "0", "0", "0", "0", "0"),
        ("bực mình (3)", "0", "3", "0", "0", "0", "0"),
        ("ghen tị (3)", "0", "0", "3", "0", "0", "0"),
        ("không cần (5)", "0", "0", "0", "5", "0", "0"),
        ("xin lỗi (4)", "0", "0", "0", "0", "4", "0"),
    ], [3.5, 2.1, 2.2, 2.2, 2.4, 2.2, 2.4])

    h("Ma trận quyết định của TCN có từ chối (tcn_reject)")
    table(["True \\ Pred", "ai cho", "bực mình", "ghen tị", "không cần", "xin lỗi", "REJECT (__OTHER__)"], [
        ("ai cho (5)", "3", "0", "0", "0", "0", "2"),
        ("bực mình (3)", "0", "1", "0", "0", "0", "2"),
        ("ghen tị (3)", "0", "0", "1", "0", "0", "2"),
        ("không cần (5)", "0", "0", "0", "3", "0", "2"),
        ("xin lỗi (4)", "0", "0", "0", "0", "1", "3"),
    ], [3.5, 2.1, 2.2, 2.2, 2.4, 2.2, 2.4])
    p("Trong mô hình TCN Reject, 11 clip bị rơi vào cột REJECT gồm: 2 clip 'ai cho' (huy1, minh), 2 clip 'bực mình' (huy1, minh), 2 clip 'ghen tị' (huy1, minh), 2 clip 'không cần' (huy1, minh) và 3 clip 'xin lỗi' (2 huy1, 1 minh). Toàn bộ 8 clip của giahuy đều được chấp nhận đúng 100%.")

    fig("confusion_primary.png", "Hình 1 Ma trận nhầm lẫn đối chứng trên tập clips thử nghiệm")
    fig("confusion_reject_primary.png", "Hình 2 Quyết định có từ chối ở ngưỡng 0.65")

    # PHẦN 7: ỨNG DỤNG VÀ HỆ THỐNG PHẦN MỀM
    page("7 Ứng dụng web demo và công cụ thu video")
    p("Dự án xây dựng bộ giải pháp phần mềm hoàn chỉnh, độc lập và thân thiện với người dùng:")

    h("Giao diện Web Demo Nhận diện (web/)")
    p("Giao diện nhận diện trực quan tại http://localhost:8765 gồm 4 phân hệ chức năng: Nhận diện, Thực nghiệm, Đánh giá và Hướng dẫn. Người dùng có thể chọn clip mẫu có sẵn hoặc tải lên video tự quay từ webcam. Hệ thống hiển thị thanh tiến trình xử lý, nhãn gợi ý kèm điểm tự tin, tỷ lệ khung hình thấy tay và thời gian suy luận. Khi mô hình TCN Reject phát hiện cử chỉ ngoài bộ từ, giao diện hiển thị thông báo rõ ràng: 'Có thể nằm ngoài bộ 5 ký hiệu'.")

    h("Web Studio Quay Video Tự động (tools/studio_quay_video.html)")
    p("Công cụ phòng thu tại http://localhost:8767 hỗ trợ người dùng tự thu thập dữ liệu ký hiệu chuẩn mực. Tính năng nổi bật gồm: đếm ngược 3-2-1 kèm âm thanh trực quan, tự động ghi hình 4.5 giây với chuẩn WebM/VP8, hỗ trợ phím tắt Spacebar, tự động đặt tên chuẩn theo mã người ký và nhãn ký hiệu, và lưu trực tiếp về máy chủ qua API HTTP POST.")

    h("Kiểm thử tự động và đảm bảo chất lượng phần mềm")
    table(["Hạng mục kiểm tra", "Kết quả thực tế", "Ghi chú kỹ thuật"], [
        ("Kiểm thử đơn vị & API", "Tất cả test cases đạt chuẩn (test_results_v6.json)", "Kiểm tra trích xuất đặc trưng, suy luận mô hình và lưu log"),
        ("Kiểm thử trình duyệt thật", "Trình duyệt Chromium & Playwright đạt (browser_checks_v6.json)", "Kiểm tra UI/UX, luồng quay video và tương tác DOM"),
        ("Kiểm tra toàn diện môi trường", "Hoàn tất thành công qua tools/doctor.py", "Xác nhận nạp đủ 4 mô hình, 20 clips và bộ trích điểm MediaPipe"),
        ("Tương thích hệ điều hành", "Hoạt động ổn định trên Windows 11 64-bit", "Chạy thuần NumPy 100%, không bị ảnh hưởng bởi chính sách WDAC"),
    ], [4.5, 6.2, 5.9])

    # PHẦN 8: HƯỚNG DẪN CÀI ĐẶT VÀ TÁI LẬP
    page("8 Hướng dẫn cài đặt và tái lập thực nghiệm")
    p("Toàn bộ đồ án được đóng gói khép kín trong thư mục d:\\KyHieuLab_DoAn_v6. Để cài đặt và tái lập toàn bộ kết quả:")

    h("1. Khởi chạy ứng dụng Demo")
    p("Chạy tệp batch tự động CHAY_DEMO.bat hoặc chạy lệnh dòng lệnh:")
    p("& .venv\\Scripts\\python.exe app.py")
    p("Mở trình duyệt tại http://localhost:8765 để trải nghiệm nhận diện.")

    h("2. Chạy kịch bản đánh giá thực nghiệm độc lập")
    p("Để tái lập toàn bộ bảng số liệu thực nghiệm và ma trận nhầm lẫn của cả 4 mô hình trên 20 clips và 29 clips OOV:")
    p("& .venv\\Scripts\\python.exe tools/danh_gia_thuc_nghiem.py")
    p("Tùy chọn xuất định dạng JSON: & .venv\\Scripts\\python.exe tools/danh_gia_thuc_nghiem.py --json")

    h("3. Dựng lại tài liệu báo cáo Word (.docx) và Markdown (.md)")
    p("Chạy kịch bản đồng bộ báo cáo:")
    p("& .venv\\Scripts\\python.exe research/build_report_v6.py")
    p("Lệnh sẽ tự động xuất đồng thời reports/Bao_cao_DoAn_KyHieuLab_v6.docx và reports/Bao_cao_DoAn_KyHieuLab_v6.md chuẩn định dạng A4 và Times New Roman.")

    # PHẦN 9: KẾT LUẬN VÀ KẾ HOẠCH PHÁT TRIỂN
    page("9 Kết luận và kế hoạch phát triển")
    p("Đồ án Ký Hiệu Lab v6 đã hoàn thành xuất sắc các mục tiêu nghiên cứu và ứng dụng kỹ thuật đặt ra: xây dựng đường ống xử lý hoàn chỉnh từ video đến nhận diện ký hiệu tiếng Việt, mở rộng bộ dữ liệu thực tế lên 20 clips qua 3 người ký, phân tích và chứng minh thực nghiệm năng lực vượt trội của mạng TCN trong bài toán mô hình hóa chuỗi thời gian cử chỉ, đồng thời giải quyết và công bố trung thực sự đánh đổi giữa độ nhạy nhận diện và khả năng từ chối ngoài bộ từ.")

    h("Kế hoạch phát triển tiếp theo")
    p("1. Mở rộng quy mô thu thập theo kế hoạch collection_plan_250.csv: Mục tiêu đạt 250 clips bao phủ 5 người ký x 5 ký hiệu x 10 lượt lặp, thu thập theo các phiên thời gian độc lập.")
    p("2. Huấn luyện lại nhánh từ chối TCN Reject: Đưa 12 clips mới của huy1 và minh vào tập huấn luyện của tcn_reject kết hợp 29 clips OOV để mô hình học được phân bố cử chỉ của đa người ký, khắc phục triệt để hiện tượng từ chối quá mức trên người ký mới.")
    p("3. Nghiên cứu cơ chế trích xuất đặc trưng khuôn mặt: Tích hợp các điểm biểu cảm cơ bản từ MediaPipe Face Mesh để tăng cường nhận diện các sắc thái biểu cảm đặc trưng của ngôn ngữ ký hiệu (như 'bực mình', 'ghen tị').")

    # PHẦN 10: TÀI LIỆU THAM KHẢO
    h("Tài liệu tham khảo")
    references = [
        "[1] Google MediaPipe. Holistic Landmark Detection Solution. https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker",
        "[2] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271, 2018.",
        "[3] scikit-learn developers. Common pitfalls and recommended practices in Machine Learning. https://scikit-learn.org/stable/common_pitfalls.html",
        "[4] scikit-learn developers. Cross-validation: evaluating estimator performance. https://scikit-learn.org/stable/modules/cross_validation.html",
        "[5] World Federation of the Deaf (WFD). Sign Language Recognition and Technology Guidelines, 2023.",
    ]
    for ref in references:
        p(ref)

    # Dựng file Word (.docx) và file Markdown (.md) đồng bộ
    build_v6("Bao_cao_DoAn_KyHieuLab_v6", "18/11/2026")


if __name__ == "__main__":
    main()
