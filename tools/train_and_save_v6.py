"""Train models on 20 clips, save weights, and evaluate."""
from pathlib import Path
import json
import numpy as np
import shutil
import sys
import time

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core
from tools.train_numpy import prepare_augmented_dataset, train_mlp_numpy, train_tcn_numpy, LABELS

print("=" * 80)
print("TIẾN HÀNH HUẤN LUYỆN VÀ CẬP NHẬT TRỌNG SỐ MÔ HÌNH VỚI 20 CLIPS")
print("=" * 80)

# 1. Load merged manifest
with open(ROOT / "data/manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"Tổng số mẫu nạp vào huấn luyện: {len(manifest)} clip")
signers = sorted(list(set(r["signer"] for r in manifest)))
print(f"Danh sách người ký ({len(signers)} người): {', '.join(signers)}")

# 2. Prepare dataset with augmentation
t0 = time.time()
x_all, y_all, mean, std = prepare_augmented_dataset(manifest, seed=20261103)
print(f"Tạo xong {len(x_all)} chuỗi đặc trưng tăng cường (Jitter + Resampling) trong {time.time()-t0:.2f}s!")

# 3. Train MLP
print("\n>>> 1. Đang huấn luyện MLP...")
t0 = time.time()
mlp_params = train_mlp_numpy(x_all, y_all, num_classes=5, seed=20261103, epochs=600, lr=0.001)
np.savez_compressed(ROOT / "models/mlp.npz", **mlp_params)

mlp_cfg = {
    "schema": "khl-v2-shape-bodyxy-mask159",
    "architecture": "mlp",
    "steps": 32,
    "features": 159,
    "labels": LABELS,
    "mean": mean.tolist(),
    "std": std.tolist(),
    "threshold": 0.65,
    "threshold_calibrated": False,
    "training_originals": len(manifest),
    "signers": signers,
    "seed": 20261103,
    "parameters": sum(v.size for v in mlp_params.values()),
    "purpose": "Đã cập nhật huấn luyện trên 20 clips đa người ký (giahuy, huy1, minh)"
}
with open(ROOT / "models/mlp.json", "w", encoding="utf-8") as f:
    json.dump(mlp_cfg, f, ensure_ascii=False, indent=2)
print(f"✅ Hoàn thành huấn luyện MLP ({time.time()-t0:.2f}s) -> Đã lưu models/mlp.npz")

# 4. Train TCN
print("\n>>> 2. Đang huấn luyện TCN (Mô hình mặc định)...")
t0 = time.time()
base_tcn = dict(np.load(ROOT / "models/backup_v6/tcn.npz")) if (ROOT / "models/backup_v6/tcn.npz").exists() else dict(np.load(ROOT / "models/tcn.npz"))
tcn_params = train_tcn_numpy(x_all, y_all, base_params=base_tcn, num_classes=5, seed=20261103, epochs=600, lr=0.001)
np.savez_compressed(ROOT / "models/tcn.npz", **tcn_params)
np.savez_compressed(ROOT / "models/tcn_augmented.npz", **tcn_params)

tcn_cfg = {
    "schema": "khl-v2-shape-bodyxy-mask159",
    "architecture": "tcn",
    "steps": 32,
    "features": 159,
    "labels": LABELS,
    "mean": mean.tolist(),
    "std": std.tolist(),
    "threshold": 0.65,
    "threshold_calibrated": False,
    "training_originals": len(manifest),
    "signers": signers,
    "seed": 20261103,
    "parameters": sum(v.size for v in tcn_params.values()),
    "purpose": "Đã cập nhật huấn luyện trên 20 clips đa người ký (giahuy, huy1, minh)"
}
with open(ROOT / "models/tcn.json", "w", encoding="utf-8") as f:
    json.dump(tcn_cfg, f, ensure_ascii=False, indent=2)
with open(ROOT / "models/tcn_augmented.json", "w", encoding="utf-8") as f:
    json.dump(tcn_cfg, f, ensure_ascii=False, indent=2)
print(f"✅ Hoàn thành huấn luyện TCN ({time.time()-t0:.2f}s) -> Đã lưu models/tcn.npz")

# 5. Evaluate on all 20 training clips
print("\n" + "=" * 80)
print("KẾT QUẢ ĐÁNH GIÁ TRÊN TOÀN BỘ 20 CLIPS SAU KHI HUẤN LUYỆN:")
print("=" * 80)

correct_tcn = 0
correct_mlp = 0

for r in manifest:
    feat = np.load(ROOT / f"data/features/{r['id']}.npz")
    
    # TCN
    ans_tcn = core.predict(feat["features"], feat["times"], tcn_params, tcn_cfg)
    top1_tcn = ans_tcn["top"][0]["label"] if ans_tcn["top"] else "None"
    score_tcn = ans_tcn["top"][0]["score"] if ans_tcn["top"] else 0
    ok_tcn = (top1_tcn == r["label"])
    if ok_tcn and ans_tcn["recognized"]:
        correct_tcn += 1
        st_tcn = "✅ ĐÚNG"
    elif ok_tcn:
        st_tcn = "🟡 ĐÚNG (DƯỚI NGƯỠNG)"
    else:
        st_tcn = "❌ SAI"
        
    # MLP
    ans_mlp = core.predict(feat["features"], feat["times"], mlp_params, mlp_cfg)
    top1_mlp = ans_mlp["top"][0]["label"] if ans_mlp["top"] else "None"
    score_mlp = ans_mlp["top"][0]["score"] if ans_mlp["top"] else 0
    ok_mlp = (top1_mlp == r["label"])
    if ok_mlp and ans_mlp["recognized"]:
        correct_mlp += 1
        st_mlp = "✅ ĐÚNG"
    elif ok_mlp:
        st_mlp = "🟡 ĐÚNG (DƯỚI NGƯỠNG)"
    else:
        st_mlp = "❌ SAI"
        
    print(f"[{r['signer']:<6}] {r['source_file']:<34} | Thật: {r['label']:<10} | TCN: {top1_tcn:<10} ({score_tcn:.2f}) [{st_tcn:<12}] | MLP: {top1_mlp:<10} ({score_mlp:.2f}) [{st_mlp}]")

print("-" * 80)
print(f"Tổng kết trên 20 clips hỗ trợ:")
print(f"  • TCN chấp nhận đúng: {correct_tcn}/{len(manifest)} ({correct_tcn/len(manifest)*100:.1f}%)")
print(f"  • MLP chấp nhận đúng: {correct_mlp}/{len(manifest)} ({correct_mlp/len(manifest)*100:.1f}%)")

# 6. Evaluate on 29 outside OOV clips
with open(ROOT / "reports/evaluation_v5_features/manifest.json", "r", encoding="utf-8") as f:
    outside_bundle = json.load(f)

outside_false_accept = 0
for r in outside_bundle["entries"]:
    feat = np.load(ROOT / f"reports/evaluation_v5_features/{r['feature_file']}")
    ans = core.predict(feat["features"], feat["times"], tcn_params, tcn_cfg)
    if ans["recognized"]:
        outside_false_accept += 1

print(f"\nKhả năng từ chối clip ngoài bộ từ (29 OOV clips):")
print(f"  • Nhận nhầm ngoài bộ từ (False Accept): {outside_false_accept}/29 ({outside_false_accept/29*100:.1f}%)")
print(f"  • Từ chối thành công: {29 - outside_false_accept}/29 ({(29 - outside_false_accept)/29*100:.1f}%)")
print("=" * 80)
