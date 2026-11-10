"""Audit frozen v4 OOF provenance and run fixed-threshold retrospective OOV probes.

Examples:
  python research/audit_v5.py --source-v3 /path/to/KHL_NhanDien_v3
  python research/audit_v5.py --source-v3 /path/to/KHL_NhanDien_v3 --work-dir /path/to/cache

This script never trains, selects a threshold, or promotes annotation metadata.
Normalized OOV clips and feature caches stay outside the delivered project.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core import PARTS, extract_video, predict, resample_masked

METHODS = {"mlp": "MLP", "tcn": "TCN gốc", "tcn_augmented": "TCN tăng cường"}
NORMALIZATION = ["fps=12,scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1", "libx264", "20", "yuv420p", "no_audio"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, fields, records):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def fit_fingerprint():
    digest = hashlib.sha256()
    files = [ROOT / "research/protocol.json", ROOT / "core.py", ROOT / "training.py", ROOT / "training_variant.py", *sorted((ROOT / "data/features").glob("*.npz"))]
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def audit_oof(data, records, protocol):
    """Check actual normalizer arrays, not only train/test identifier disjointness."""
    by_id = {r["id"]: r for r in data}
    feature_map = {}
    for row in data:
        path = ROOT / "data/features" / (row["id"] + ".npz")
        with np.load(path) as z:
            features, times = z["features"], z["times"]
            assert features.shape[1] == 159 and len(features) == len(times)
            assert np.isfinite(features).all() and np.all(np.diff(times) > 0)
            assert np.isin(features[:, [65, 131, 158]], [0, 1]).all()
            feature_map[row["id"]] = resample_masked(features, times)
    expected_fingerprint = fit_fingerprint()
    tuple_keys = [(r["method"], r["seed"], r["id"]) for r in records]
    assert len(tuple_keys) == len(set(tuple_keys)) == 45
    normalizer_max_error = 0.0
    for record in records:
        held = by_id[record["id"]]
        expected_ids = set(by_id) - {record["id"]}
        assert set(record["train_ids"]) == expected_ids and len(record["train_ids"]) == 7
        train_hashes = [by_id[i]["source_video_sha256"] for i in record["train_ids"]]
        assert train_hashes == record["train_source_sha256"]
        assert held["source_video_sha256"] not in train_hashes
        assert record["run_fingerprint"] == expected_fingerprint
        x = np.array([feature_map[i] for i in record["train_ids"]])
        mean, std = x.mean(axis=(0, 1)), np.maximum(x.std(axis=(0, 1)), 0.2)
        for _, _, mask in PARTS:
            mean[mask] = 0
            std[mask] = 1
        error = max(float(np.max(np.abs(mean - np.asarray(record["normalization_mean"])))), float(np.max(np.abs(std - np.asarray(record["normalization_std"])))))
        normalizer_max_error = max(normalizer_max_error, error)
        assert error < 1e-6, (record["id"], record["method"], error)
        assert record["top1_correct"] == (record["provided"] == record["predicted"])
        assert len(record["history"]) == 10 and record["history"][-1]["step"] == 500
    assert len({r["id"] for r in records}) == 5
    assert {r["provided"] for r in records} == set(protocol["heldout_labels"])
    by_method = []
    stability_rows = []
    for method, name in METHODS.items():
        per_clip = []
        for clip_id in sorted({r["id"] for r in records}):
            attempts = sorted([r for r in records if r["method"] == method and r["id"] == clip_id], key=lambda r: r["seed"])
            assert len(attempts) == 3
            predictions = [r["predicted"] for r in attempts]
            item = {
                "id": clip_id, "provided": attempts[0]["provided"], "n_seeds": 3,
                "predictions": predictions, "same_prediction_all_seeds": len(set(predictions)) == 1,
                "correct_seed_count": sum(r["top1_correct"] for r in attempts),
                "accepted_seed_count": sum(r["accepted"] for r in attempts),
                "accepted_correct_seed_count": sum(r["accepted"] and r["top1_correct"] for r in attempts),
                "accepted_wrong_seed_count": sum(r["accepted"] and not r["top1_correct"] for r in attempts),
                "top_score_min": min(r["top_score"] for r in attempts),
                "top_score_max": max(r["top_score"] for r in attempts),
            }
            per_clip.append(item)
            stability_rows.append({"method": method, **{key: value for key, value in item.items() if key != "predictions"}, "predictions": " | ".join(predictions)})
        by_method.append({"id": method, "name": name, "per_clip_stability": per_clip, "correct_all_seeds_clips": sum(x["correct_seed_count"] == 3 for x in per_clip), "any_accepted_wrong_clips": sum(x["accepted_wrong_seed_count"] > 0 for x in per_clip)})
    return {"audit_passed": True, "source_exclusion_checked": True, "normalization_train_only_checked": True, "normalizer_arrays_checked": 45, "normalizer_max_abs_error": normalizer_max_error, "checkpoints_checked": 45, "fit_fingerprint": expected_fingerprint}, by_method, stability_rows


def freeze_ood_protocol(source_v3, data, outside):
    weights = {}
    for model in METHODS:
        config = read_json(ROOT / "models" / (model + ".json"))
        assert config["threshold"] == 0.65 and config["training_originals"] == 8
        weights[model] = {"weights_sha256": sha(ROOT / "models" / (model + ".npz")), "config_sha256": sha(ROOT / "models" / (model + ".json")), "threshold": config["threshold"]}
    source_videos = []
    training_hashes = {r["source_video_sha256"] for r in data}
    for row in outside:
        assert row["source_video_sha256"] not in training_hashes
        assert row["experimental_include"] and row["label"] not in {r["label"] for r in data}
        source_videos.append({"id": row["id"], "label": row["label"], "signer": row["signer"], "source_video_sha256": row["source_video_sha256"], "v3_mp4_sha256": sha(source_v3 / row["video"]), "reviewStatus": row.get("reviewStatus"), "trainingEligible": row.get("trainingEligible")})
    payload = {
        "date": "2026-11-04", "kind": "retrospective_outside_vocabulary_diagnostic",
        "source_collection": "KHL_NhanDien_v3", "expected_n": 29,
        "source_selection": "All experimental_include v3 rows whose exact label is outside the frozen v4 five-label vocabulary; two pending metadata-incomplete rows excluded.",
        "normalization": NORMALIZATION, "models": weights,
        "videos": source_videos, "core_sha256": sha(ROOT / "core.py"),
        "decisions_frozen_before_predictions": True,
        "threshold_policy": "Inherited 0.65 for all three models; no threshold fitting or model selection from these outcomes.",
        "labels_and_review": "Labels assumed as submitted by user; original pending review and trainingEligible flags remain unchanged.",
        "claim_limit": "Convenience sample of older recordings, not a calibrated population open-set false acceptance rate or blind new-signer experiment.",
    }
    path = ROOT / "research/evaluation_v5_protocol.json"
    if path.exists():
        old = read_json(path)
        assert {k: v for k, v in old.items() if k != "frozen_at_utc"} == payload, "Frozen input protocol changed; do not silently reuse or retune results."
    else:
        payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, payload)
    return sha(path)


def load_models():
    models = {}
    for name in METHODS:
        with np.load(ROOT / "models" / (name + ".npz")) as z:
            weights = {key: z[key].copy() for key in z.files}
        models[name] = (weights, read_json(ROOT / "models" / (name + ".json")))
    return models


def predict_feature_item(row, features, times, quality, models, protocol_hash):
    item = []
    for model, (weights, config) in models.items():
        answer = predict(features, times, weights, config)
        top = answer["top"]
        item.append({
            "id": row["id"], "provided_label": row["label"], "signer": row["signer"],
            "model": model, "expected_action": "reject_outside_vocabulary",
            "accepted": bool(answer["recognized"]), "false_accept": bool(answer["recognized"]),
            "predicted": top[0]["label"] if top else None,
            "top_score": top[0]["score"] if top else None,
            "hand_fraction": answer.get("hand_fraction", quality["hand_fraction"]),
            "rejection_reason": None if answer["recognized"] else "missing_hands" if not top else "below_threshold",
            "protocol_sha256": protocol_hash,
        })
    return item


def probe_videos(source_v3, outside, workdir, protocol_hash):
    workdir.mkdir(parents=True, exist_ok=True)
    models = load_models()
    output, errors = [], []
    core_hash = sha(ROOT / "core.py")
    for index, row in enumerate(outside, 1):
        clip = workdir / (row["id"] + ".mp4")
        feature_path = workdir / (row["id"] + ".npz")
        metadata_path = workdir / (row["id"] + ".json")
        input_key = hashlib.sha256((sha(source_v3 / row["video"]) + core_hash + json.dumps(NORMALIZATION)).encode()).hexdigest()
        started = time.perf_counter()
        try:
            if feature_path.exists() and metadata_path.exists() and read_json(metadata_path).get("input_key") == input_key:
                with np.load(feature_path) as z:
                    features, times = z["features"], z["times"]
                quality = read_json(metadata_path)["quality"]
            else:
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(source_v3 / row["video"]), "-map", "0:v:0", "-vf", NORMALIZATION[0], "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(clip)], check=True, capture_output=True)
                features, times, quality = extract_video(clip)
                np.savez_compressed(feature_path, features=features, times=times)
                write_json(metadata_path, {"input_key": input_key, "quality": quality})
            output.extend(predict_feature_item(row, features, times, quality, models, protocol_hash))
            print(f"OOV {index}/{len(outside)} {row['id']} {row['label']} " + str({name: output[-3 + j]["accepted"] for j, name in enumerate(METHODS)}) + f" {time.perf_counter()-started:.2f}s", flush=True)
        except Exception as error:
            errors.append({"id": row["id"], "label": row["label"], "error_type": type(error).__name__, "message": str(error)})
            print("PROBE_ERROR", row["id"], str(error), flush=True)
    return output, errors


def create_feature_bundle(outside, workdir, protocol_hash):
    """Ship only compact derived features, with source and extraction provenance."""
    folder = ROOT / "reports/evaluation_v5_features"
    folder.mkdir(exist_ok=True)
    entries = []
    for row in outside:
        feature_path = workdir / (row["id"] + ".npz")
        metadata_path = workdir / (row["id"] + ".json")
        if not feature_path.exists() or not metadata_path.exists():
            continue
        destination = folder / feature_path.name
        shutil.copy2(feature_path, destination)
        metadata = read_json(metadata_path)
        entries.append({
            "id": row["id"], "label": row["label"], "signer": row["signer"],
            "source_video_sha256": row["source_video_sha256"],
            "reviewStatus": row.get("reviewStatus"), "trainingEligible": row.get("trainingEligible"),
            "feature_file": feature_path.name, "feature_sha256": sha(destination),
            "normalized_video_sha256": sha(workdir / (row["id"] + ".mp4")),
            "quality": metadata["quality"], "extraction_input_key": metadata["input_key"],
            "features_written_at_utc": datetime.fromtimestamp(feature_path.stat().st_mtime, timezone.utc).isoformat(),
        })
    bundle = {"kind": "derived_landmark_features_not_new_recordings", "protocol_sha256": protocol_hash, "n": len(entries), "core_sha256": sha(ROOT / "core.py"), "normalization": NORMALIZATION, "entries": entries}
    write_json(folder / "manifest.json", bundle)
    return bundle


def replay_feature_bundle():
    protocol_path = ROOT / "research/evaluation_v5_protocol.json"
    protocol = read_json(protocol_path)
    protocol_hash = sha(protocol_path)
    folder = ROOT / "reports/evaluation_v5_features"
    bundle = read_json(folder / "manifest.json")
    assert bundle["protocol_sha256"] == protocol_hash and bundle["core_sha256"] == sha(ROOT / "core.py")
    assert bundle["n"] == 29
    for model, entry in protocol["models"].items():
        assert sha(ROOT / "models" / (model + ".npz")) == entry["weights_sha256"]
        assert sha(ROOT / "models" / (model + ".json")) == entry["config_sha256"]
    outside = bundle["entries"]
    expected = {row["id"]: row for row in protocol["videos"]}
    assert set(expected) == {row["id"] for row in outside}
    models = load_models()
    output = []
    for row in outside:
        for field in ["label", "signer", "source_video_sha256", "reviewStatus", "trainingEligible"]:
            assert row[field] == expected[row["id"]][field]
        path = folder / row["feature_file"]
        assert path.parent.resolve() == folder.resolve() and sha(path) == row["feature_sha256"]
        with np.load(path) as z:
            features, times = z["features"], z["times"]
        output.extend(predict_feature_item(row, features, times, row["quality"], models, protocol_hash))
    return outside, output, [], protocol_hash


def verify_frozen_predictions(predictions, protocol_hash):
    reference = read_json(ROOT / "research/evaluation_v5_original_predictions.json")
    assert reference["protocol_sha256"] == protocol_hash
    expected = {(row["model"], row["id"]): row for row in reference["records"]}
    actual = {(row["model"], row["id"]): row for row in predictions}
    assert set(actual) == set(expected) and len(actual) == 87
    maximum_error = 0.0
    for key, row in actual.items():
        old = expected[key]
        for field in ["provided_label", "signer", "predicted", "accepted"]:
            assert row[field] == old[field], (key, field)
        for field in ["top_score", "hand_fraction"]:
            if row[field] is None or old[field] is None:
                assert row[field] == old[field]
            else:
                maximum_error = max(maximum_error, abs(row[field] - old[field]))
    # Float32 BLAS implementations can differ slightly across user devices.
    assert maximum_error < 1e-5
    folder = ROOT / "reports/evaluation_v5_features"
    return {"bundled_feature_files": 29, "bundle_bytes": sum(path.stat().st_size for path in folder.iterdir()), "video_extraction_vs_bundled_replay_predictions": len(actual), "max_numeric_difference": maximum_error, "numeric_tolerance": 1e-5, "threshold_and_weights_unchanged": True}


def write_figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    models = summary["outside_vocabulary"]["models"]
    names = [m["name"] for m in models]
    false_accepts = [m["false_accepts"] for m in models]
    rejected = [m["rejected"] for m in models]
    ax.barh(names, false_accepts, color="#cd774f", label="Chấp nhận nhầm vào 5 nhãn")
    ax.barh(names, rejected, left=false_accepts, color="#79a18b", label="Từ chối")
    for i, model in enumerate(models):
        if model["false_accepts"]:
            ax.text(model["false_accepts"] / 2, i, str(model["false_accepts"]), va="center", ha="center", color="white", fontsize=13)
        if model["rejected"]:
            ax.text(model["false_accepts"] + model["rejected"] / 2, i, str(model["rejected"]), va="center", ha="center", fontsize=13)
    ax.set_xlim(0, 29)
    ax.set_xlabel("Số clip trong nhóm thăm dò ngoài bộ từ")
    ax.set_title("29 video cũ ngoài bộ 5 nhãn · ngưỡng cố định 0,65\nThăm dò hồi cứu, chưa phải tỷ lệ lỗi đại diện thực tế", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    (ROOT / "reports/figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "reports/figures/evaluation_v5_outside_vocabulary.png", dpi=220)
    plt.close(fig)


def build_report(summary):
    provenance, scope, oov = summary["provenance"], summary["scope"], summary["outside_vocabulary"]
    lines = ["# Đánh giá bổ sung Ký Hiệu Lab v5", "Ngày 04/11/2026", "Bản v5 mở rộng phép đánh giá và kiểm tra nguồn dữ liệu, giữ mô hình và ngưỡng 0,65 của v4. Đợt này không thu thêm video thực, không huấn luyện thêm và không lựa chọn cấu hình theo kết quả ngoài bộ từ.", "## Phạm vi và kiểm tra tính đúng của thực nghiệm", f"Bộ học có 8 video, 5 nhãn, một mã người ký. Phép giữ từng clip ngoài phần học chỉ bao phủ 5 video nguồn thuộc hai nhãn ai cho và không cần. Ba nhãn bực mình, ghen tị và xin lỗi chưa có mẫu kiểm tra riêng. Ba seed dùng lại cùng năm clip, không tạo thêm mẫu độc lập.", f"Đã đối chiếu {provenance['checkpoints_checked']} checkpoint: đủ 45 tổ hợp phương pháp, seed và clip; mỗi phần học đúng bảy video còn lại; mã băm video giữ riêng không nằm trong phần học. Đã tính lại 45 cặp trung bình và độ lệch chuẩn từ chính video học để so với các mảng lưu; sai khác lớn nhất {provenance['normalizer_max_abs_error']:.3g}. Không phát hiện chuẩn hóa dùng video giữ riêng trong phép kiểm tra này.", "## Kết quả theo seed và theo video nguồn", "| Mô hình | Đúng ở seed chính | Chấp nhận đúng ở seed chính | Clip đúng ở cả 3 seed | Clip có ít nhất một seed chấp nhận sai |", "| --- | --- | --- | --- | --- |"]
    for model in summary["oof"]["models"]:
        lines.append(f"| {model['name']} | {model['primary']['top1_correct']}/5 | {model['primary']['accepted_correct']}/5 | {model['correct_all_seeds_clips']}/5 | {model['any_accepted_wrong_clips']}/5 |")
    lines.extend(["Bảng độ ổn định đếm trên năm video nguồn. Cột có seed chấp nhận sai không phải tỷ lệ lỗi trên người dùng mới. Các dự đoán từng seed và từng clip nằm trong evaluation_v5_clip_stability.csv.", "## Thăm dò video ngoài bộ từ", f"Chọn trước toàn bộ {oov['requested_n']} clip đủ hồ sơ học thử trong bộ v3 có nhãn ngoài năm nhãn hiện tại: 17 clip giahuy và 12 clip tp, 29 nhãn khác nhau. Những video này không thuộc tám nguồn học của mô hình v4. Hai clip thiếu hồ sơ ở trạng thái pending không được dùng. Video được chuẩn hóa theo cùng quy tắc 12 fps và kích thước trước khi trích điểm lại.", "Cả 29 clip thăm dò vẫn có reviewStatus pending và trainingEligible false trong hồ sơ gốc; nhãn được hiểu theo thông tin người thu cung cấp và giả định người thu đã đối chiếu động tác. Không thay đổi các trạng thái này. Đây là tập tiện có từ lần thu cũ, chưa phải test mù hoặc mẫu đại diện cho mọi hoạt động ngoài bộ từ.", "| Mô hình | Đã xử lý | Chấp nhận nhầm | Từ chối | Tỷ lệ chấp nhận nhầm trong nhóm này |", "| --- | --- | --- | --- | --- |"])
    for model in oov["models"]:
        lines.append(f"| {model['name']} | {model['n']} | {model['false_accepts']} | {model['rejected']} | {model['false_accept_rate']*100:.1f}% |")
    lines.extend([f"Số clip lỗi hoặc bỏ qua: {len(oov['errors'])}. Mẫu lỗi được lưu riêng trong JSON, không tính thành từ chối đúng. Các tỷ lệ trên dùng số clip thực sự xử lý được, không được suy rộng thành FAR đã được hiệu chỉnh của sản phẩm.", "Hình trong gói: reports/figures/evaluation_v5_outside_vocabulary.png. Bảng dự đoán từng clip nằm trong reports/evaluation_v5_outside_predictions.csv; ma trận một hàng ngoài bộ từ nằm trong reports/evaluation_v5_outside_confusion.csv.", "## Quyết định sử dụng và việc còn thiếu", "Bản hiện tại dùng cho trình bày và phiên thử có giám sát. Điểm softmax cao chưa đủ để xác nhận ký hiệu thật sự thuộc bộ năm nhãn. Cần luôn có cách ký lại, nhập tay hoặc xác nhận bởi người kiểm tra; kết quả nên được đọc là nhãn gợi ý trong bộ đang học.", "Chưa đáp ứng kiểm chứng độc lập đủ năm nhãn hoặc người ký mới, hiệu chỉnh ngưỡng ngoài bộ từ và điều kiện dịch công khai. Ưu tiên tiếp theo là video thực của nhiều người và phiên, có validation và test tách trước; dữ liệu ngoài bộ từ dùng chọn ngưỡng phải tách khỏi phần dùng kiểm tra cuối. Không tăng ngưỡng chỉ để làm đẹp kết quả của 29 clip vừa xem.", "## Tái lập", "Chạy `python research/audit_v5.py` từ thư mục gốc bản v5 để tính lại từ gói đặc trưng đã trích, có mã băm đối chiếu. Gói reports/evaluation_v5_features chỉ chứa đặc trưng dẫn xuất của 29 video cũ, không phải 29 video mới và không được nhập vào phần học.", "Để trích điểm lại từ video thật, giải nén gói v3 đầy đủ và chạy `python research/audit_v5.py --source-v3 /duong/dan/KHL_NhanDien_v3 --work-dir /thu/muc/tam`. Cần requirements-research.txt và FFmpeg cho chế độ này. Protocol được ghi trước dự đoán trong research/evaluation_v5_protocol.json; mã băm trọng số và cấu hình giữ nguyên. MP4 chuẩn hóa nằm ở work-dir và không được đóng gói vào bản v5."])
    blocks, table_lines = [], []
    for line in lines:
        if line.startswith("|"):
            table_lines.append(line)
        else:
            if table_lines:
                blocks.append("\n".join(table_lines))
                table_lines = []
            blocks.append(line)
    if table_lines:
        blocks.append("\n".join(table_lines))
    (ROOT / "reports/EVALUATION_V5.md").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v3", type=Path, help="Optional original V3 directory. Omit to replay bundled derived features.")
    parser.add_argument("--work-dir", "--workdir", dest="workdir", type=Path, default=Path(tempfile.gettempdir()) / "khl_v5_evaluation")
    args = parser.parse_args()
    data = read_json(ROOT / "data/manifest.json")
    protocol = read_json(ROOT / "research/protocol.json")
    metrics = read_json(ROOT / "reports/metrics.json")
    records = read_json(ROOT / "research/oof_results.json")
    labels = set(protocol["labels"])
    source_v3 = args.source_v3.resolve() if args.source_v3 else None
    all_v3 = read_json(source_v3 / "manifest.json") if source_v3 else read_json(ROOT / "data/source_manifest_v3.json")
    outside = [r for r in all_v3 if r["experimental_include"] and r["label"] not in labels]
    assert len(outside) == 29 and len({r["id"] for r in outside}) == 29
    assert len({r["source_video_sha256"] for r in outside}) == 29
    by_source_id = {r["id"]: r for r in all_v3}
    for row in data:
        for key in ["label", "signer", "source_video_sha256", "reviewStatus", "trainingEligible"]:
            assert row[key] == by_source_id[row["id"]][key]
        assert sha(ROOT / row["video"]) == row["normalized_video_sha256"]
    provenance, stability, stability_rows = audit_oof(data, records, protocol)
    if source_v3:
        protocol_hash = freeze_ood_protocol(source_v3, data, outside)
        print("PROTOCOL_FROZEN", protocol_hash, flush=True)
        predictions, errors = probe_videos(source_v3, outside, args.workdir.resolve(), protocol_hash)
        create_feature_bundle(outside, args.workdir.resolve(), protocol_hash)
        mode = "video_extraction_with_verified_feature_cache"
    else:
        outside, predictions, errors, protocol_hash = replay_feature_bundle()
        mode = "replay_bundled_derived_features"
    reproduction_checks = verify_frozen_predictions(predictions, protocol_hash)
    method_ood = []
    for method, name in METHODS.items():
        selected = [r for r in predictions if r["model"] == method]
        accepted = sum(r["accepted"] for r in selected)
        method_ood.append({"id": method, "name": name, "n": len(selected), "false_accepts": accepted, "rejected": len(selected) - accepted, "false_accept_rate": accepted / len(selected) if selected else None, "missing_hands_rejections": sum(r["rejection_reason"] == "missing_hands" for r in selected), "by_signer": {signer: {"n": sum(r["signer"] == signer for r in selected), "false_accepts": sum(r["signer"] == signer and r["accepted"] for r in selected)} for signer in sorted({r["signer"] for r in selected})}})
    methods = []
    for model in metrics["models"]:
        extra = next(r for r in stability if r["id"] == model["id"])
        methods.append({**model, **extra})
    summary = {
        "schema_version": 1, "date": "2026-11-04", "default_model": "tcn",
        "scope": {"labels": protocol["labels"], "training_originals": 8, "training_signers": 1, "heldout_unique_clips": 5, "heldout_labels": 2, "unmeasured_labels": protocol["unmeasured_labels"], "new_real_videos": 0},
        "provenance": provenance,
        "execution_mode": mode,
        "reproduction_checks": reproduction_checks,
        "oof": {"primary_seed": metrics["primary_seed"], "n_unique_clips": 5, "tested_label_names": protocol["heldout_labels"], "models": methods, "warning": "Three seeds reuse the same five known source videos; no fresh blind or new-signer test."},
        "outside_vocabulary": {"kind": "retrospective_diagnostic", "requested_n": 29, "n": len(predictions) // 3, "signer_counts": dict(collections.Counter(r["signer"] for r in outside)), "models": method_ood, "errors": errors, "skipped": errors, "threshold": 0.65, "threshold_calibrated": False, "protocol_sha256": protocol_hash, "source_review_status": "pending", "source_training_eligible": False, "warning": "Older convenience recordings with different submitted labels, not calibrated or representative open-set FAR."},
        "readiness": {"course_demo": "conditional_supervised_proof_of_concept", "all_five_labels_validated": False, "unseen_signer_validated": False, "public_use": False, "threshold_calibrated": False},
        "limitations": ["No new real recordings collected", "Three supported labels remain singletons with no held-out evaluation", "OOF uses one signer and only two repeated labels", "OOF clips reused over seeds and earlier project iterations", "Outside labels assumed from submitted metadata; records remain pending and unverified", "OOV convenience probes are retrospective and not a representative open-set sample", "No test-guided threshold or model promotion"],
        "environment": {"python": platform.python_version(), "platform": platform.system(), "numpy": np.__version__, "mediapipe": importlib.metadata.version("mediapipe"), "opencv": importlib.metadata.version("opencv-contrib-python")},
    }
    write_json(ROOT / "reports/evaluation_v5.json", summary)
    csv_write(ROOT / "reports/evaluation_v5_clip_stability.csv", ["method", "id", "provided", "n_seeds", "predictions", "same_prediction_all_seeds", "correct_seed_count", "accepted_seed_count", "accepted_correct_seed_count", "accepted_wrong_seed_count", "top_score_min", "top_score_max"], stability_rows)
    csv_write(ROOT / "reports/evaluation_v5_outside_predictions.csv", ["id", "provided_label", "signer", "model", "expected_action", "accepted", "false_accept", "predicted", "top_score", "hand_fraction", "rejection_reason", "protocol_sha256"], predictions)
    confusion = []
    for method in METHODS:
        selected = [r for r in predictions if r["model"] == method]
        count = collections.Counter(r["predicted"] if r["accepted"] else "CHUA_CHAC" for r in selected)
        confusion.append({"model": method, "true_group": "NGOAI_BO_TU", **{label: count[label] for label in protocol["labels"] + ["CHUA_CHAC"]}})
    csv_write(ROOT / "reports/evaluation_v5_outside_confusion.csv", ["model", "true_group"] + protocol["labels"] + ["CHUA_CHAC"], confusion)
    write_figures(summary)
    build_report(summary)
    report_text = (ROOT / "reports/EVALUATION_V5.md").read_text(encoding="utf-8")
    assert "|\n\n|" not in report_text
    summary["reproduction_checks"]["markdown_tables_validated"] = True
    write_json(ROOT / "reports/evaluation_v5.json", summary)
    print("EVALUATION_COMPLETE", json.dumps({"audit": provenance["audit_passed"], "ood_models": method_ood, "errors": errors}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
