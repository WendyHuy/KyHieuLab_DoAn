"""Scan and verify newly collected sign language video clips."""
from pathlib import Path
import csv
import json
import re
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
INCOMING_DIR = ROOT / "data" / "clips_moi"
PLAN_FILE = ROOT / "data" / "collection_plan_250.csv"

INCOMING_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("KIỂM TRA VÀ NHẬP VIDEO MỚI VÀO DỰ ÁN")
print("=" * 70)
print(f"Thư mục chứa video mới: {INCOMING_DIR}")

# Find video files
video_files = list(INCOMING_DIR.glob("*.webm")) + list(INCOMING_DIR.glob("*.mp4"))
print(f"Số lượng video tìm thấy trong data/clips_moi/: {len(video_files)}")

if not video_files:
    print("\n👉 Hướng dẫn:")
    print("1. Mở file 'MO_STUDIO_QUAY_VIDEO.bat' để bật công cụ quay tự động.")
    print("2. Các video quay xong tải về máy (thư mục Downloads).")
    print(f"3. Copy các video đã quay dán vào thư mục: {INCOMING_DIR}")
    print("4. Chạy lại script này để kiểm tra tính hợp lệ và tự động cập nhật tiến độ 250 clip!")
    sys.exit(0)

# Pattern: S01_B01_ai_cho_R01.webm
PATTERN = re.compile(r"^(S0[1-5])_(B0[1-2])_([a-z_]+)_(R0[1-5])\.(webm|mp4)$", re.IGNORECASE)

LABEL_MAP = {
    "ai_cho": "ai cho",
    "buc_minh": "bực mình",
    "ghen_ti": "ghen tị",
    "khong_can": "không cần",
    "xin_loi": "xin lỗi"
}

valid_clips = []
invalid_clips = []

for vf in sorted(video_files):
    match = PATTERN.match(vf.name)
    if match:
        signer, session, label_slug, rep, ext = match.groups()
        label_norm = label_slug.lower()
        if label_norm in LABEL_MAP:
            valid_clips.append({
                "path": vf,
                "name": vf.name,
                "signer": signer.upper(),
                "session": session.upper(),
                "label": LABEL_MAP[label_norm],
                "rep": int(rep[1:]),
                "size_kb": round(vf.stat().st_size / 1024, 1)
            })
        else:
            invalid_clips.append((vf.name, f"Ký hiệu lạ: {label_slug}"))
    else:
        invalid_clips.append((vf.name, "Tên file không đúng chuẩn (Chuẩn: S01_B01_ai_cho_R01.webm)"))

print(f"\n✅ Số video hợp lệ: {len(valid_clips)}")
if valid_clips:
    by_signer = {}
    for vc in valid_clips:
        s = vc["signer"]
        by_signer[s] = by_signer.get(s, 0) + 1
    print("  Phân bổ theo người ký:")
    for s, count in sorted(by_signer.items()):
        print(f"    - Người {s}: {count}/50 clip")

if invalid_clips:
    print(f"\n⚠️ Số video chưa chuẩn tên ({len(invalid_clips)} file):")
    for name, reason in invalid_clips[:10]:
        print(f"    - {name}: {reason}")

# Update collection_plan_250.csv if present
if PLAN_FILE.exists() and valid_clips:
    with open(PLAN_FILE, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    
    updated_count = 0
    valid_map = {(vc["signer"], vc["session"], vc["label"], str(vc["rep"])): vc for vc in valid_clips}
    
    for r in rows:
        key = (r["signer_code"], r["session_code"], r["label"], r["repetition"])
        if key in valid_map:
            vc = valid_map[key]
            if r["status"] != "collected":
                r["status"] = "collected"
                r["video_path"] = str(vc["path"].relative_to(ROOT))
                updated_count += 1
                
    if updated_count > 0:
        fieldnames = ["signer_code", "session_code", "label", "repetition", "proposed_split", "status", "video_path", "start", "end", "review_status"]
        with open(PLAN_FILE, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n🎉 Đã cập nhật thành công {updated_count} clip vào bảng kế hoạch {PLAN_FILE.name}!")
print("=" * 70)
