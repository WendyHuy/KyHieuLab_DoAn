"""Extract features for new clips and update manifest."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import time

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import core

print("=" * 80)
print("BƯỚC 1: TRÍCH XUẤT ĐIỂM MỐC MEDIAPIPE CHO 12 CLIP MỚI CỦA HUY1 & MINH")
print("=" * 80)

# Process clips in data/clips_moi
clips_moi = sorted(list((ROOT / "data/clips_moi").glob("*.webm")))
print(f"Tìm thấy {len(clips_moi)} video trong data/clips_moi/\n")

new_manifest_entries = []

for c in clips_moi:
    parts = c.stem.split("_")
    signer = parts[0]
    sid = parts[-1]
    
    # Label mapping
    if "aicho" in c.name or "ai_cho" in c.name: label = "ai cho"
    elif "buc_minh" in c.name: label = "bực mình"
    elif "ghen_ti" in c.name: label = "ghen tị"
    elif "khong_can" in c.name: label = "không cần"
    elif "xin_loi" in c.name: label = "xin lỗi"
    else:
        print(f"Bỏ qua file không xác định: {c.name}")
        continue
        
    print(f"[+] Trích xuất: {c.name:<34} (Signer: {signer:<5}, Nhãn: '{label:<10}', ID: {sid})")
    
    start_time = time.perf_counter()
    x, t, quality = core.extract_video(c)
    extract_sec = time.perf_counter() - start_time
    print(f"    -> Đã trích {len(x)} frames | Tỷ lệ tay: {quality['hand_fraction']:.2f} | Thời gian: {extract_sec:.2f}s")
    
    # Save features to data/features/{sid}.npz
    feat_dest = ROOT / f"data/features/{sid}.npz"
    np.savez_compressed(feat_dest, features=x, times=t)
    
    # Copy video to data/clips/{sid}.webm
    clip_dest = ROOT / f"data/clips/{sid}.webm"
    shutil.copy2(c, clip_dest)
    
    video_bytes = c.read_bytes()
    video_sha = hashlib.sha256(video_bytes).hexdigest()
    
    entry = {
        "id": sid,
        "label": label,
        "signer": signer,
        "batch": "2026-11-12",
        "source_file": c.name,
        "video": f"data/clips/{sid}.webm",
        "original_start": 0.0,
        "original_end": float(quality["duration_observed"]),
        "label_source": "user contribution",
        "metadata_match": True,
        "experimental_include": True,
        "reviewStatus": "verified_by_user",
        "trainingEligible": True,
        "boundary_status": "full_clip",
        "source_video_sha256": video_sha,
        "v3_video": f"data/clips/{sid}.webm",
        "v3_quality": quality,
        "normalized_video_sha256": video_sha,
        "normalization": "Raw webm 25fps resized to 640px max",
        "feature_status": "extracted_direct_from_webm",
        "quality": quality,
        "extract_seconds": extract_sec
    }
    new_manifest_entries.append(entry)

print("\n" + "=" * 80)
print(f"Đã trích xuất đặc trưng thành công cho {len(new_manifest_entries)} clip mới!")

# Step 2: Merge with existing manifest
manifest_path = ROOT / "data/manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    existing_manifest = json.load(f)

# Keep existing clips, avoid duplicates by id
merged_manifest = [r for r in existing_manifest if not any(n["id"] == r["id"] for n in new_manifest_entries)]
merged_manifest.extend(new_manifest_entries)

# Save backup and new manifest
if not (ROOT / "data/manifest_backup_v6.json").exists():
    shutil.copy2(manifest_path, ROOT / "data/manifest_backup_v6.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(merged_manifest, f, ensure_ascii=False, indent=2)

print(f"Tổng số clip trong dataset sau khi gộp: {len(merged_manifest)} clip (gốc: {len(existing_manifest)}, mới: {len(new_manifest_entries)})")
from collections import Counter
print("Phân bổ nhãn mới:", dict(Counter(r["label"] for r in merged_manifest)))
print("Phân bổ người ký:", dict(Counter(r["signer"] for r in merged_manifest)))
print("=" * 80)
