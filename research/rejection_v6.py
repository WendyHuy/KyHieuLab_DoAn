"""Retrospective matched development CV for an explicit outside-vocabulary class.

This study is designed after viewing v5 outside-vocabulary diagnostics. It is
neither a fresh blind test nor evidence of generalization to new signers.
Run from the project root: python research/rejection_v6.py
Use --rerun only to intentionally recompute matching cached fits.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import collections
import csv
import hashlib
import json
import os
import platform
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import jax
import jax.numpy as jnp
import training
from core import PARTS, infer_logits, normalize, resample_masked

REJECT = "__OTHER__"
METHODS = {"tcn": "TCN gốc 5 lớp", "tcn_reject": "TCN có lớp ngoài bộ từ"}
SEEDS = [20261103, 20261104, 20261105]
STEPS, BATCH, THRESHOLD = 500, 64, 0.65


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, fields, records):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def load_sources():
    known = read_json(ROOT / "data/manifest.json")
    bundle = read_json(ROOT / "reports/evaluation_v5_features/manifest.json")
    negatives = bundle["entries"]
    supported = sorted({r["label"] for r in known})
    assert len(known) == 8 and len(negatives) == 29 and len(supported) == 5
    rows, features = [], {}
    for row in known:
        item = {"id": row["id"], "source_label": row["label"], "label": row["label"], "kind": "known", "signer": row["signer"], "source_video_sha256": row["source_video_sha256"], "reviewStatus": row["reviewStatus"], "trainingEligible": row["trainingEligible"], "feature_file": "data/features/" + row["id"] + ".npz"}
        rows.append(item)
    for row in negatives:
        path = ROOT / "reports/evaluation_v5_features" / row["feature_file"]
        assert sha(path) == row["feature_sha256"]
        assert row["label"] not in supported
        item = {"id": row["id"], "source_label": row["label"], "label": REJECT, "kind": "outside", "signer": row["signer"], "source_video_sha256": row["source_video_sha256"], "reviewStatus": row["reviewStatus"], "trainingEligible": row["trainingEligible"], "feature_file": "reports/evaluation_v5_features/" + row["feature_file"]}
        rows.append(item)
    assert len({r["id"] for r in rows}) == len({r["source_video_sha256"] for r in rows}) == 37
    for row in rows:
        path = ROOT / row["feature_file"]
        row["feature_sha256"] = sha(path)
        with np.load(path) as z:
            x, t = z["features"].copy(), z["times"].copy()
        assert x.ndim == 2 and x.shape[1] == 159 and len(x) == len(t)
        assert np.isfinite(x).all() and np.all(np.diff(t) > 0)
        assert np.isin(x[:, [65, 131, 158]], [0, 1]).all()
        features[row["id"]] = (x, t)
    return rows[:8], rows[8:], supported, features


def freeze_protocol(known, outside, supported):
    counts = collections.Counter(r["label"] for r in known)
    # Match prior known-source fold order, not an ordering picked by outcomes.
    repeated = [r for r in known if counts[r["label"]] > 1]
    assert len(repeated) == 5
    ordered_outside = sorted(outside, key=lambda r: (r["source_video_sha256"], r["id"]))
    negative_groups = [[r for position, r in enumerate(ordered_outside) if position % 5 == fold] for fold in range(5)]
    folds = []
    for index, held_known in enumerate(repeated):
        held_negative = negative_groups[index]
        training_known = [r for r in known if r["id"] != held_known["id"]]
        training_negative = [r for r in outside if r["id"] not in {n["id"] for n in held_negative}]
        train_hashes = {r["source_video_sha256"] for r in training_known + training_negative}
        test_hashes = {r["source_video_sha256"] for r in [held_known] + held_negative}
        assert train_hashes.isdisjoint(test_hashes)
        folds.append({"fold": index, "known_holdout_id": held_known["id"], "outside_holdout_ids": [r["id"] for r in held_negative], "train_known_ids": [r["id"] for r in training_known], "train_outside_ids": [r["id"] for r in training_negative], "heldout_source_sha256": sorted(test_hashes), "training_source_sha256": sorted(train_hashes)})
    payload = {
        "date": "2026-11-07", "version": "rejection-v6-development-cv", "study_kind": "retrospective_development_cv",
        "motivation": "Designed after seeing v5 false acceptance diagnostics; all 37 source recordings are already known development material.",
        "supported_labels": supported, "classifier_labels": supported + [REJECT], "reject_label": REJECT,
        "known_originals": 8, "outside_originals": 29, "new_real_videos": 0,
        "seeds": SEEDS, "primary_seed": SEEDS[0], "training_seed_rule": "seed + 100 * fold",
        "negative_partition": "Sort unique original source hashes ascending, then assign sorted position modulo 5; every source and all its variants stay in one fold.",
        "folds": folds, "sources": known + outside,
        "architecture": "Same small noncausal TCN as v4/v5; five outputs for baseline, six outputs for candidate.",
        "steps": STEPS, "batch_size": BATCH, "learning_rate": 0.001, "l2_weight_penalty": 0.00001, "label_smoothing": 0.02,
        "augmentation": "The original training.augmentation, 20 training-only variants per original source; no held-out augmentation.",
        "normalization": "Mean and std from unaugmented resampled training originals only; std floor 0.2; validity-mask channels mean0/std1.",
        "baseline_sampling": "Original training.fit uniform sampling over augmented known training data.",
        "candidate_sampling": "Uniform six-class label sampling followed by uniform example sampling within that class.",
        "ablation_limit": "The candidate changes both explicit outside class and class-balanced sampling; it is not an isolated outside-class-only ablation.",
        "threshold": THRESHOLD, "threshold_calibrated": False,
        "acceptance": "At least eight hand-visible resampled steps, raw top class is supported, and raw softmax score >=0.65; top OTHER always rejects. Do not renormalize scores after removing OTHER.",
        "promotion_rule": {"primary_and_mean_known_top1": "candidate >= baseline", "primary_and_mean_known_accepted_correct": "candidate >= baseline", "primary_and_mean_known_accepted_wrong": "candidate <= baseline", "primary_and_mean_outside_false_accepts": "candidate < baseline", "fallback": "retain tcn default if any condition fails"},
        "source_metadata": "Original reviewStatus and trainingEligible fields preserved; experimental use is not expert approval.",
        "core_sha256": sha(ROOT / "core.py"), "training_sha256": sha(ROOT / "training.py"),
        "limitations": ["Not blind; designed after earlier OOV outcomes", "Known-source holdout covers only two repeated labels and one signer", "Three singleton supported labels have no held-out test", "Outside class groups 29 heterogeneous submitted labels", "Same source clips reused across three seeds; no sample-count inflation", "Source-based split is not signer-disjoint", "No threshold sweep or probability calibration", "No universal outside-vocabulary or public-use guarantee"],
    }
    path = ROOT / "research/rejection_v6_protocol.json"
    if path.exists():
        previous = read_json(path)
        assert {k: v for k, v in previous.items() if k != "frozen_at_utc"} == payload, "Frozen protocol inputs differ; do not silently reuse results."
        return previous, sha(path)
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, payload)
    return payload, sha(path)


def balanced_fit(records, labels, feature_map, seed):
    rng = np.random.default_rng(seed)
    augmented, targets, originals = [], [], []
    for row in records:
        sequence, times = feature_map[row["id"]]
        original = resample_masked(sequence, times)
        originals.append(original)
        augmented.append(original)
        for _ in range(20):
            augmented.append(training.augmentation(sequence, times, rng))
        targets.extend([labels.index(row["label"])] * 21)
    originals = np.asarray(originals, np.float32)
    mean = originals.mean(axis=(0, 1))
    std = np.maximum(originals.std(axis=(0, 1)), 0.2)
    for _, _, mask in PARTS:
        mean[mask] = 0
        std[mask] = 1
    x = normalize(np.asarray(augmented, np.float32), mean, std)
    y = np.asarray(targets, np.int32)
    by_label = {label: np.flatnonzero(y == label) for label in range(len(labels))}
    assert all(len(indices) for indices in by_label.values())
    params = training.initial_params("tcn", len(labels), rng)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)

    def loss(weights, batch_x, batch_y):
        logp = jax.nn.log_softmax(training.forward(weights, batch_x, "tcn"))
        ce = -jnp.mean(0.98 * logp[jnp.arange(len(batch_y)), batch_y] + 0.02 * logp.mean(axis=1))
        return ce + 1e-5 * sum(jnp.sum(value * value) for key, value in weights.items() if key.startswith("w"))

    @jax.jit
    def step(weights, m, v, xx, yy, number):
        value, gradient = jax.value_and_grad(loss)(weights, xx, yy)
        m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, gradient)
        v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, gradient)
        weights = jax.tree.map(lambda a, b, c: a - 0.001 * (b / (1 - 0.9 ** number)) / (jnp.sqrt(c / (1 - 0.999 ** number)) + 1e-8), weights, m, v)
        return weights, m, v, value

    history = []
    for number in range(1, STEPS + 1):
        requested_labels = rng.integers(0, len(labels), BATCH)
        indices = np.array([rng.choice(by_label[int(label)]) for label in requested_labels])
        params, first, second, value = step(params, first, second, x[indices], y[indices], number)
        if number % 50 == 0:
            history.append({"step": number, "regularized_training_loss": float(value)})
    params = {key: np.asarray(value) for key, value in params.items()}
    test = x[:3]
    jax_logits = np.asarray(training.forward(params, jnp.asarray(test), "tcn"))
    numpy_logits = np.array([infer_logits(row, params, "tcn") for row in test])
    parity_error = float(np.max(np.abs(jax_logits - numpy_logits)))
    assert parity_error < 0.001, parity_error
    config = {"schema": "khl-v2-shape-bodyxy-mask159", "architecture": "tcn", "method": "tcn_reject", "steps": 32, "features": 159, "labels": labels, "supported_labels": labels[:-1], "reject_label": REJECT, "mean": mean.tolist(), "std": std.tolist(), "threshold": THRESHOLD, "threshold_calibrated": False, "training_originals": len(records), "known_training_originals": sum(r["kind"] == "known" for r in records), "outside_training_originals": sum(r["kind"] == "outside" for r in records), "seed": seed, "training_steps": STEPS, "parameters": sum(value.size for value in params.values()), "sampling": "balanced_label_batches", "augmentation": "original training.augmentation,20 variants/source", "purpose": "Retrospective development experimental model; no independent or new-signer validation.", "expert_verified": False, "source_review_metadata_unchanged": True, "extractor": "MediaPipe Holistic 0.10.21 model_complexity=0", "face_features": False, "raw_softmax_over_all_six_outputs": True, "do_not_renormalize_supported_scores": True}
    return params, config, history, parity_error


def evaluate_one(weights, config, row, feature_map):
    sequence, times = feature_map[row["id"]]
    sampled = resample_masked(sequence, times)
    visible = int(((sampled[:, 65] + sampled[:, 131]) > 0).sum())
    x = normalize(sampled, np.asarray(config["mean"], np.float32), np.asarray(config["std"], np.float32))
    logits = infer_logits(x, weights, "tcn")
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    index = int(probability.argmax())
    decision = config["labels"][index]
    score = float(probability[index])
    accepted = bool(visible >= 8 and decision != REJECT and score >= THRESHOLD)
    return {"id": row["id"], "kind": row["kind"], "provided": row["label"], "source_label": row["source_label"], "signer": row["signer"], "source_video_sha256": row["source_video_sha256"], "raw_decision_class": decision, "raw_top_score": score, "top1_correct": decision == row["label"], "accepted": accepted, "accepted_correct": accepted and row["kind"] == "known" and decision == row["label"], "accepted_wrong": accepted and (row["kind"] == "outside" or decision != row["label"]), "false_reject": row["kind"] == "known" and not accepted, "outside_false_accept": row["kind"] == "outside" and accepted, "rejection_reason": None if accepted else "missing_hands" if visible < 8 else "outside_class" if decision == REJECT else "below_threshold", "visible_hand_steps": visible, "probabilities": {label: float(value) for label, value in zip(config["labels"], probability)}}


def summarize_known(rows, tested_labels):
    from sklearn.metrics import f1_score
    accepted = sum(r["accepted"] for r in rows)
    correct_accepts = sum(r["accepted_correct"] for r in rows)
    return {"n": len(rows), "top1_correct": sum(r["top1_correct"] for r in rows), "top1_accuracy": sum(r["top1_correct"] for r in rows) / len(rows), "accepted": accepted, "accepted_correct": correct_accepts, "accepted_wrong": sum(r["accepted_wrong"] for r in rows), "false_rejects": len(rows) - accepted, "coverage": accepted / len(rows), "correct_accept_rate": correct_accepts / len(rows), "accepted_accuracy": correct_accepts / accepted if accepted else None, "macro_f1_two_tested_labels": float(f1_score([r["provided"] for r in rows], [r["raw_decision_class"] for r in rows], labels=tested_labels, average="macro", zero_division=0))}


def summarize_outside(rows):
    false_accepts = sum(r["accepted"] for r in rows)
    return {"n": len(rows), "false_accepts": false_accepts, "rejected": len(rows) - false_accepts, "false_accept_rate": false_accepts / len(rows), "outside_class_rejections": sum(r["rejection_reason"] == "outside_class" for r in rows), "low_score_rejections": sum(r["rejection_reason"] == "below_threshold" for r in rows), "missing_hands_rejections": sum(r["rejection_reason"] == "missing_hands" for r in rows)}


def summarize_predictions(predictions, protocol):
    methods = []
    for method, name in METHODS.items():
        per_seed = []
        for seed in SEEDS:
            selected = [r for r in predictions if r["method"] == method and r["seed"] == seed]
            known = [r for r in selected if r["kind"] == "known"]
            outside = [r for r in selected if r["kind"] == "outside"]
            assert len(known) == len({r["id"] for r in known}) == 5
            assert len(outside) == len({r["id"] for r in outside}) == 29
            per_seed.append({"seed": seed, "known": summarize_known(known, ["ai cho", "không cần"]), "outside": summarize_outside(outside)})
        mean = {}
        for group in ["known", "outside"]:
            mean[group] = {}
            for field in per_seed[0][group]:
                values = [item[group][field] for item in per_seed]
                mean[group][field] = float(np.mean(values)) if all(value is not None for value in values) else None
        methods.append({"id": method, "name": name, "primary": per_seed[0], "by_seed": per_seed, "mean": mean})
    baseline, candidate = methods
    checks = {}
    for point in ["primary", "mean"]:
        base = baseline[point]
        cand = candidate[point]
        checks[point + "_known_top1_nondecreasing"] = cand["known"]["top1_correct"] >= base["known"]["top1_correct"]
        checks[point + "_known_correct_accepts_nondecreasing"] = cand["known"]["accepted_correct"] >= base["known"]["accepted_correct"]
        checks[point + "_known_wrong_accepts_not_increasing"] = cand["known"]["accepted_wrong"] <= base["known"]["accepted_wrong"]
        checks[point + "_outside_false_accepts_reduced"] = cand["outside"]["false_accepts"] < base["outside"]["false_accepts"]
    passed = all(checks.values())
    return methods, checks, passed


def make_figures(methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), layout="constrained")
    names = [m["name"] for m in methods]
    for ax, kind, title, maximum in [(axes[0], "known", "5 clip nhãn hỗ trợ · seed chính", 5), (axes[1], "outside", "29 clip ngoài bộ từ · seed chính", 29)]:
        if kind == "known":
            first = [m["primary"][kind]["accepted_correct"] for m in methods]
            second = [m["primary"][kind]["accepted_wrong"] for m in methods]
            third = [m["primary"][kind]["false_rejects"] for m in methods]
            parts = [(first, "Chấp nhận đúng", "#6e9d87"), (second, "Chấp nhận sai", "#c66e4e"), (third, "Từ chối", "#c9d4d7")]
        else:
            parts = [([m["primary"][kind]["false_accepts"] for m in methods], "Chấp nhận nhầm", "#c66e4e"), ([m["primary"][kind]["rejected"] for m in methods], "Từ chối", "#6e9d87")]
        left = np.zeros(2)
        for values, label, color in parts:
            ax.barh(names, values, left=left, label=label, color=color)
            for i, value in enumerate(values):
                if value:
                    ax.text(left[i] + value / 2, i, str(value), ha="center", va="center", fontsize=12)
            left += np.asarray(values)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, maximum)
        ax.set_xlabel("Số video nguồn")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9, frameon=False)
    fig.suptitle("Đánh giá phát triển hồi cứu có ghép cặp nguồn · ngưỡng 0,65", fontsize=12)
    fig.savefig(ROOT / "reports/figures/rejection_v6_matched_cv.png", dpi=220)
    plt.close(fig)


def write_report(summary):
    lines = ["# Thử nghiệm lớp ngoài bộ từ Ký Hiệu Lab v6", "Ngày 07/11/2026", "Đợt này bổ sung học có giám sát với một lớp ngoài bộ từ, sau khi đã thấy kết quả thăm dò v5. Đây là nghiên cứu phát triển hồi cứu, không phải test mù mới. Không có video thực mới được quay.", "## Thiết kế ghép cặp trước khi học", "Protocol được đóng băng trước học: cùng năm lượt giữ clip nhãn hỗ trợ, cùng ba seed, 500 bước, batch 64, learning rate 0,001, ngưỡng 0,65. Mỗi lượt giữ riêng một trong năm clip thuộc ai cho và không cần; 29 nguồn ngoài bộ từ được xếp theo SHA256 nguồn rồi chia vòng vào năm nhóm. Toàn bộ biến thể của nguồn giữ riêng đứng ngoài phần học.", "TCN gốc học trên bảy clip hỗ trợ còn lại. Ứng viên sáu lớp học trên cùng bảy clip đó cộng 23 hoặc 24 nguồn ngoài bộ từ thuộc phần học, với lấy mẫu cân bằng sáu lớp. Mọi tăng cường và chuẩn hóa chỉ tạo từ phần học. Ứng viên thay đổi đồng thời lớp ngoài bộ từ và lấy mẫu cân bằng; phép so sánh chưa tách riêng tác dụng từng thay đổi.", "## Kết quả theo seed", "| Phương pháp | Seed | Hỗ trợ gợi ý đầu đúng /5 | Hỗ trợ chấp nhận đúng /5 | Hỗ trợ chấp nhận sai | Hỗ trợ bị từ chối | Ngoài bộ từ nhận nhầm /29 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for method in summary["methods"]:
        for seed in method["by_seed"]:
            k, o = seed["known"], seed["outside"]
            lines.append(f"| {method['name']} | {seed['seed']} | {k['top1_correct']} | {k['accepted_correct']} | {k['accepted_wrong']} | {k['false_rejects']} | {o['false_accepts']} |")
    lines.extend(["Các hàng seed dùng lại cùng 5 nguồn nhãn hỗ trợ và cùng 29 nguồn ngoài bộ từ; không cộng thành 15 hoặc 87 video độc lập. Số bị từ chối ở nhóm hỗ trợ là false rejection trong nhóm này; chấp nhận nhưng sai nhãn được ghi riêng.", "Không ghép 5 clip nhãn hỗ trợ và 29 clip ngoài bộ từ thành một accuracy chung. Cột ngoài bộ từ là kết quả giữ nguồn riêng bằng mô hình học trong từng lượt, khác phép thăm dò v5 dùng một mô hình học đủ tám clip hỗ trợ.", "## Quyết định mô hình", "Quy tắc thay mặc định đã chốt trước học: ở seed chính và trung bình ba seed, ứng viên phải không giảm gợi ý đầu đúng, không giảm chấp nhận đúng, không tăng chấp nhận sai của nhóm hỗ trợ, đồng thời giảm số nhận nhầm ngoài bộ từ. Không quét hoặc hạ ngưỡng theo tập giữ riêng.", "Kết quả quy tắc: " + ("đạt; ứng viên được khuyến nghị làm mặc định cho bản đồ án." if summary["promotion_passed"] else "chưa đạt; giữ TCN gốc làm mặc định và để ứng viên ở chế độ thực nghiệm."), "| Điều kiện | Đạt |", "| --- | --- |"])
    for key, value in summary["promotion_rule_checks"].items():
        lines.append(f"| {key} | {'Có' if value else 'Chưa'} |")
    lines.extend(["## Trọng số đủ dữ liệu và giới hạn", "Trọng số ứng viên bàn giao được học lại trên đủ 37 nguồn cũ: 8 nguồn của năm nhãn hỗ trợ và 29 nguồn được gán lớp __OTHER__. Đây là trọng số demo sau nghiên cứu, khác các trọng số dùng giữ nguồn riêng. Các JSON nguồn gốc vẫn giữ nguyên pending/trainingEligible false; dùng học thử không biến thành nhãn đã được chuyên gia duyệt.", "Xác suất softmax tính trên đủ sáu đầu ra. Khi lớp cao nhất là __OTHER__, hệ thống từ chối dù điểm cao; nếu lớp cao nhất thuộc bộ hỗ trợ, cần qua ngưỡng 0,65 và đủ tay. Không bỏ lớp ngoài rồi chuẩn hóa lại điểm năm lớp hỗ trợ.", "Chưa có kiểm tra độc lập cho ba nhãn đơn mẫu hoặc người ký mới. Ngoài bộ từ là một nhóm dị thể của các nhãn đã biết từ hồ sơ; độ bao phủ ngoài đời còn hạn chế. Kể cả đạt quy tắc kỹ thuật nội bộ, kết quả không bảo đảm dùng dịch công khai.", "## Tái lập", "Từ thư mục gốc chạy `python research/rejection_v6.py`. Checkpoint có cùng dấu vân tay mã, protocol và đặc trưng sẽ được dùng lại. `--rerun` học lại có chủ đích. Không cần video thô ngoài bộ từ; gói đặc trưng dẫn xuất v5 đã kèm mã băm. Cần requirements-research.txt."])
    blocks, table = [], []
    for line in lines:
        if line.startswith("|"):
            table.append(line)
        else:
            if table:
                blocks.append("\n".join(table)); table = []
            blocks.append(line)
    if table:
        blocks.append("\n".join(table))
    (ROOT / "reports/rejection_v6.md").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    known, outside, supported, features = load_sources()
    all_rows = known + outside
    by_id = {r["id"]: r for r in all_rows}
    protocol, protocol_hash = freeze_protocol(known, outside, supported)
    implementation_hash = sha(Path(__file__))
    run_fingerprint = hashlib.sha256((protocol_hash + implementation_hash).encode()).hexdigest()
    checkpoint_dir = ROOT / "research/rejection_v6_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    print("PROTOCOL_FROZEN", protocol_hash, "IMPLEMENTATION", implementation_hash, flush=True)
    predictions, audits = [], []
    for seed in SEEDS:
        for fold in protocol["folds"]:
            number = fold["fold"]
            held = [by_id[fold["known_holdout_id"]]] + [by_id[i] for i in fold["outside_holdout_ids"]]
            for method in METHODS:
                checkpoint = checkpoint_dir / f"{method}_{seed}_{number}.json"
                if checkpoint.exists() and not args.rerun:
                    saved = read_json(checkpoint)
                    if saved.get("run_fingerprint") == run_fingerprint:
                        predictions.extend(saved["predictions"])
                        audits.append(saved["audit"])
                        continue
                train_ids = fold["train_known_ids"] + (fold["train_outside_ids"] if method == "tcn_reject" else [])
                records = [by_id[i] for i in train_ids]
                source_hashes = {r["source_video_sha256"] for r in records}
                assert source_hashes.isdisjoint({r["source_video_sha256"] for r in held})
                started = time.perf_counter()
                training_seed = seed + number * 100
                if method == "tcn":
                    weights, config, history, parity = training.fit(records, supported, "tcn", training_seed)
                else:
                    weights, config, history, parity = balanced_fit(records, supported + [REJECT], features, training_seed)
                expected = np.array([resample_masked(*features[r["id"]]) for r in records])
                mean, std = expected.mean(axis=(0, 1)), np.maximum(expected.std(axis=(0, 1)), 0.2)
                for _, _, mask in PARTS:
                    mean[mask] = 0; std[mask] = 1
                normalization_error = max(float(np.max(np.abs(mean - np.asarray(config["mean"])))), float(np.max(np.abs(std - np.asarray(config["std"])))))
                assert normalization_error < 1e-6
                results = [evaluate_one(weights, config, row, features) for row in held]
                for result in results:
                    result.update({"method": method, "seed": seed, "fold": number, "training_seed": training_seed})
                audit = {"method": method, "seed": seed, "fold": number, "training_seed": training_seed, "train_ids": train_ids, "train_source_sha256": sorted(source_hashes), "test_ids": [r["id"] for r in held], "test_source_sha256": sorted(r["source_video_sha256"] for r in held), "source_disjoint": True, "train_known_n": sum(r["kind"] == "known" for r in records), "train_outside_n": sum(r["kind"] == "outside" for r in records), "normalizer_max_abs_error": normalization_error, "jax_numpy_max_logit_difference": parity, "elapsed_seconds": time.perf_counter() - started}
                write_json(checkpoint, {"run_fingerprint": run_fingerprint, "protocol_sha256": protocol_hash, "implementation_sha256": implementation_hash, "audit": audit, "predictions": results, "history": history, "normalization_mean": config["mean"], "normalization_std": config["std"]})
                predictions.extend(results); audits.append(audit)
                k = [r for r in results if r["kind"] == "known"][0]
                print("FOLD", seed, number, method, "known", k["raw_decision_class"], k["accepted"], "outside_false_accepts", sum(r["outside_false_accept"] for r in results), "/", len(results) - 1, f"{audit['elapsed_seconds']:.2f}s", flush=True)
    assert len(predictions) == 204 and len(audits) == 30
    methods, promotion_checks, passed = summarize_predictions(predictions, protocol)
    model_path = ROOT / "models/tcn_reject.npz"
    config_path = ROOT / "models/tcn_reject.json"
    if args.rerun or not config_path.exists() or not model_path.exists() or read_json(config_path).get("run_fingerprint") != run_fingerprint:
        weights, config, history, parity = balanced_fit(all_rows, supported + [REJECT], features, SEEDS[0])
        known_manifest = read_json(ROOT / "data/manifest.json")
        negative_manifest = read_json(ROOT / "reports/evaluation_v5_features/manifest.json")["entries"]
        normalized_hashes = {row["id"]: row["normalized_video_sha256"] for row in known_manifest + negative_manifest}
        assert len(normalized_hashes) == 37
        config.update({"protocol_sha256": protocol_hash, "run_fingerprint": run_fingerprint, "implementation_sha256": implementation_hash, "train_ids": [r["id"] for r in all_rows], "train_source_sha256": [r["source_video_sha256"] for r in all_rows], "train_normalized_video_sha256": [normalized_hashes[r["id"]] for r in all_rows], "study_kind": "retrospective_development_cv", "promotion_rule_passed": passed, "recommended_default_from_this_study": "tcn_reject" if passed else "tcn", "jax_numpy_max_logit_difference": parity})
        staged_weights = model_path.with_suffix(".npz.tmp")
        with staged_weights.open("wb") as file:
            np.savez_compressed(file, **weights)
        config["weights_sha256"] = sha(staged_weights)
        # Publish complete config first, then atomically expose the complete NPZ.
        # The report's full_model.sha256 also gates any default promotion.
        write_json(config_path, config)
        staged_weights.replace(model_path)
        write_json(ROOT / "reports/rejection_v6_full_history.json", history)
        write_json(ROOT / "reports/rejection_v6_training_replay.json", [evaluate_one(weights, config, row, features) for row in all_rows])
    summary = {"schema_version": 1, "date": "2026-11-07", "study_kind": "retrospective_development_cv", "primary_seed": SEEDS[0], "recommended_default": "tcn_reject" if passed else "tcn", "promotion_passed": passed, "promotion_rule_checks": promotion_checks, "scope": {"supported_labels": supported, "known_originals": 8, "negative_originals": 29, "total_originals": 37, "new_real_videos": 0, "known_holdout_unique_clips": 5, "outside_holdout_unique_clips": 29, "known_tested_labels": ["ai cho", "không cần"], "unmeasured_labels": [label for label in supported if label not in ["ai cho", "không cần"]], "seeds": SEEDS, "folds": 5}, "units": {"n_top1_accepted_false_rejects_false_accepts": "counts of unique held-out source videos per seed", "rates_and_scores": "proportions from0to1; softmax is not calibrated confidence", "means": "arithmetic mean of three seed metrics on the same sources; not additional independent videos"}, "threshold": THRESHOLD, "threshold_calibrated": False, "reject_label": REJECT, "methods": methods, "provenance": {"protocol_sha256": protocol_hash, "implementation_sha256": implementation_hash, "run_fingerprint": run_fingerprint, "matched_folds": 5, "fits": 30, "checkpoints": len(audits), "all_sources_disjoint_within_fold": all(row["source_disjoint"] for row in audits), "normalizer_max_abs_error": max(row["normalizer_max_abs_error"] for row in audits), "jax_numpy_max_logit_difference": max(row["jax_numpy_max_logit_difference"] for row in audits), "outside_group_sizes": [len(row["outside_holdout_ids"]) for row in protocol["folds"]], "fullfit_candidate_training_originals": 37, "source_metadata_preserved": True, "fullfit_weights_sha256": sha(model_path), "fullfit_config_sha256": sha(config_path)}, "limitations": protocol["limitations"], "ablation_limit": protocol["ablation_limit"], "environment": {"python": platform.python_version(), "platform": platform.system(), "numpy": np.__version__, "jax": jax.__version__}}
    full_config = read_json(config_path)
    assert full_config["weights_sha256"] == sha(model_path)
    summary["full_model"] = {"id": "tcn_reject", "sha256": sha(model_path), "config_sha256": sha(config_path), "supported_labels": supported, "labels": supported + [REJECT], "reject_label": REJECT, "training_originals": 37, "known_training_originals": 8, "outside_training_originals": 29, "seed": SEEDS[0], "raw_softmax_outputs": 6}
    write_json(ROOT / "reports/rejection_v6.json", summary)
    write_json(ROOT / "reports/rejection_v6_predictions.json", predictions)
    write_json(ROOT / "reports/rejection_v6_fold_audit.json", audits)
    fields = ["method", "seed", "fold", "training_seed", "id", "kind", "provided", "source_label", "signer", "source_video_sha256", "raw_decision_class", "raw_top_score", "top1_correct", "accepted", "accepted_correct", "accepted_wrong", "false_reject", "outside_false_accept", "rejection_reason", "visible_hand_steps"]
    csv_write(ROOT / "reports/rejection_v6_predictions.csv", fields, predictions)
    metric_rows = []
    for model in methods:
        for seed in model["by_seed"]:
            for kind in ["known", "outside"]:
                metric_rows.append({"method": model["id"], "seed": seed["seed"], "group": kind, **seed[kind]})
    metric_fields = ["method", "seed", "group", "n", "top1_correct", "top1_accuracy", "accepted", "accepted_correct", "accepted_wrong", "false_rejects", "coverage", "correct_accept_rate", "accepted_accuracy", "macro_f1_two_tested_labels", "false_accepts", "rejected", "false_accept_rate", "outside_class_rejections", "low_score_rejections", "missing_hands_rejections"]
    csv_write(ROOT / "reports/rejection_v6_metrics.csv", metric_fields, metric_rows)
    for kind in ["known", "outside"]:
        matrix_rows = []
        labels = supported if kind == "known" else [REJECT]
        columns = supported + ["__REJECT__"]
        for method in METHODS:
            selected = [r for r in predictions if r["method"] == method and r["seed"] == SEEDS[0] and r["kind"] == kind]
            for label in labels:
                count = collections.Counter(r["raw_decision_class"] if r["accepted"] else "__REJECT__" for r in selected if r["provided"] == label)
                matrix_rows.append({"method": method, "true_label": label, **{column: count[column] for column in columns}})
        csv_write(ROOT / "reports" / f"rejection_v6_{kind}_decision_confusion.csv", ["method", "true_label"] + columns, matrix_rows)
    make_figures(methods)
    write_report(summary)
    print("STUDY_COMPLETE", json.dumps({"promotion_passed": passed, "recommended_default": summary["recommended_default"], "methods": [{"id": m["id"], "primary": m["primary"], "mean": m["mean"]} for m in methods]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
