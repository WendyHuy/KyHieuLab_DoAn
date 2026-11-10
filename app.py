"""Local course demo. Run inference on video bytes; labels never enter inference."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import re
import tempfile
import threading
import time
import uuid
import numpy as np
from core import extract_video, predict
from tools.summarize_trials import summarize

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = {"tcn": "TCN gốc", "tcn_augmented": "TCN tăng cường", "mlp": "MLP đối chiếu", "tcn_reject": "TCN học thêm lớp ngoài bộ từ"}
OUTSIDE = "__OTHER__"
MAX_VIDEO_BYTES = 64 * 1024 * 1024
TRIAL_COLUMNS = ["trial_id", "time_utc", "inference_id", "signer_code", "session_code", "actual_label", "predicted_label", "accepted", "top1_correct", "accepted_correct", "outside_false_accept", "top_score", "hand_fraction", "model", "model_sha256", "video_sha256", "start", "end", "elapsed_seconds", "annotation_status"]
OPTIONAL_TRIAL_COLUMNS = ["rejection_reason", "rejected_as_outside"]


def apply_decision_policy(answer, config):
    """Keep raw class decisions for auditing; show only supported suggestions."""
    answer = dict(answer)
    top = answer.get("top", [])
    raw_label = top[0]["label"] if top else ""
    raw_score = top[0]["score"] if top else None
    reject_label = config.get("reject_label")
    outside = bool(reject_label and raw_label == reject_label)
    answer.update({"raw_decision_class": raw_label, "raw_decision_score": raw_score,
                   "rejected_as_outside": outside,
                   "rejection_reason": "insufficient_hands" if not top else
                       "outside_vocabulary" if outside else
                       "below_threshold" if not answer["recognized"] else ""})
    answer["top"] = [dict(item) for item in top if item["label"] != reject_label]
    answer["recognized"] = bool(answer["recognized"] and not outside)
    if outside:
        answer["message"] = "Có thể nằm ngoài bộ 5 ký hiệu"
    return answer


class InvalidTrial(ValueError):
    pass


class State:
    def __init__(self, root=ROOT, output_dir=None):
        self.root = Path(root)
        self.output_dir = Path(output_dir) if output_dir else self.root / "outputs"
        self.rows = json.loads((self.root / "data/manifest.json").read_text(encoding="utf-8"))
        self.labels = sorted({r["label"] for r in self.rows})
        self.samples = {r["id"]: r for r in self.rows}
        outside_manifest = self.root / "data/outside_manifest_v6.json"
        self.outside_rows = []
        self.outside_samples = {}
        if outside_manifest.exists():
            self.outside_rows = json.loads(outside_manifest.read_text(encoding="utf-8"))["entries"]
            self.outside_samples = {r["id"]: r for r in self.outside_rows if r["id"] in {"c3b59369", "b93f88f2"}}
        self.models = {}
        for name in MODEL_NAMES:
            path = self.root / "models" / (name + ".npz")
            if name == "tcn_reject" and not path.exists():
                continue
            with np.load(path) as z:
                weights = {key: z[key].copy() for key in z.files}
            config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            expected_labels = self.labels + [OUTSIDE] if name == "tcn_reject" else self.labels
            if config["labels"] != expected_labels:
                raise ValueError("Bộ nhãn của trọng số không khớp dữ liệu.")
            if name == "tcn_reject" and (config.get("supported_labels") != self.labels or config.get("reject_label") != OUTSIDE):
                raise ValueError("Cấu hình từ chối phải tách riêng năm nhãn hỗ trợ.")
            self.models[name] = (weights, config, hashlib.sha256(path.read_bytes()).hexdigest())
        self.default_model = "tcn"
        recommendation = self.root / "reports/rejection_v6.json"
        if recommendation.exists():
            report = json.loads(recommendation.read_text(encoding="utf-8"))
            candidate = self.models.get("tcn_reject")
            if (candidate and report.get("promotion_passed") is True and report.get("recommended_default") == "tcn_reject"
                    and report.get("full_model", {}).get("sha256") == candidate[2]):
                self.default_model = "tcn_reject"
        self.inference_lock = threading.Lock()
        self.trial_lock = threading.Lock()
        self.cache = {}

    def infer(self, video_bytes, model=None, start=0.0, end=0.0):
        model = model or self.default_model
        if model not in self.models:
            raise ValueError("Mô hình không hợp lệ.")
        if not np.isfinite([start, end]).all() or start < 0 or end < 0 or (end > 0 and end <= start):
            raise ValueError("Mốc kết thúc phải lớn hơn mốc bắt đầu; dùng 0 để lấy đến cuối.")
        if end > 0 and end - start > 15:
            raise ValueError("Chọn đoạn ký tối đa 15 giây.")
        if not video_bytes or len(video_bytes) > MAX_VIDEO_BYTES:
            raise ValueError("Video phải có dữ liệu và nhỏ hơn 64 MB.")
        started = time.perf_counter()
        weights, config, model_hash = self.models[model]
        with self.inference_lock:
            with tempfile.TemporaryDirectory(prefix="khl_video_") as directory:
                path = Path(directory) / "input_video"
                path.write_bytes(video_bytes)
                x, t, quality = extract_video(path, start=start, end=end)
            answer = apply_decision_policy(predict(x, t, weights, config), config)
        answer.update({
            "inference_id": uuid.uuid4().hex, "model": model,
            "model_sha256": model_hash, "video_sha256": hashlib.sha256(video_bytes).hexdigest(),
            "start": start, "end": end, "quality": quality,
            "elapsed_seconds": time.perf_counter() - started,
            "threshold": config["threshold"], "threshold_calibrated": False,
            "scope": "five_label_course_poc", "annotation_status": "not_annotated",
        })
        # Keep results only; raw user videos are deleted with the temporary directory.
        with self.trial_lock:
            if len(self.cache) >= 64:
                self.cache.pop(next(iter(self.cache)))
            self.cache[answer["inference_id"]] = {"answer": answer, "saved": False, "created": time.monotonic()}
        return answer

    def save_trial(self, payload):
        for field in ("signer_code", "session_code"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", str(payload.get(field, ""))):
                raise ValueError("Mã người và phiên bắt đầu bằng chữ hoặc số, tối đa 32 ký tự; ví dụ S01, S01_B01.")
        actual = payload.get("actual_label")
        if actual not in self.labels + [OUTSIDE]:
            raise ValueError("Chọn nhãn trong danh sách hoặc Ngoài bộ ký hiệu.")
        with self.trial_lock:
            cached = self.cache.get(payload.get("inference_id"))
            if not cached or time.monotonic() - cached["created"] > 1800:
                raise InvalidTrial("Kết quả đã hết hạn. Nhận diện lại video rồi ghi phiếu.")
            if cached["saved"]:
                raise InvalidTrial("Lượt này đã được ghi. Nhận diện một lượt mới để tiếp tục.")
            answer = cached["answer"]
            top = answer["top"]
            predicted = answer.get("raw_decision_class", top[0]["label"] if top else "")
            correct = predicted == actual
            record = {
                "trial_id": uuid.uuid4().hex, "time_utc": datetime.now(timezone.utc).isoformat(),
                "inference_id": answer["inference_id"], "signer_code": payload["signer_code"],
                "session_code": payload["session_code"], "actual_label": actual,
                "predicted_label": predicted, "accepted": answer["recognized"],
                "top1_correct": correct, "accepted_correct": answer["recognized"] and correct,
                "outside_false_accept": actual == OUTSIDE and answer["recognized"],
                "top_score": answer.get("raw_decision_score", top[0]["score"] if top else ""), "hand_fraction": answer.get("hand_fraction", 0),
                "annotation_status": "user_reported_unverified",
                "rejection_reason": answer.get("rejection_reason", ""),
                "rejected_as_outside": answer.get("rejected_as_outside", False),
            }
            for key in ("model", "model_sha256", "video_sha256", "start", "end", "elapsed_seconds"):
                record[key] = answer[key]
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "trials.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            cached["saved"] = True
        return {"saved": True, "trial_id": record["trial_id"], "message": "Đã ghi phiếu. Nhãn phản hồi chưa được duyệt và chưa dùng để tự học."}

    def trial_csv(self):
        rows = self.trial_records()
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=TRIAL_COLUMNS + OPTIONAL_TRIAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return ("\ufeff" + buffer.getvalue()).encode("utf-8")

    def trial_records(self):
        with self.trial_lock:
            path = self.output_dir / "trials.jsonl"
            if not path.exists():
                return []
            records = []
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict) or any(key not in record for key in TRIAL_COLUMNS):
                        raise ValueError("Thiếu trường dữ liệu.")
                    if not isinstance(record["accepted"], bool) or not isinstance(record["top1_correct"], bool):
                        raise ValueError("Trạng thái lượt phải là true hoặc false.")
                    if record["actual_label"] not in self.labels + [OUTSIDE]:
                        raise ValueError("Nhãn không nằm trong phạm vi.")
                    records.append(record)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Phiếu trong trials.jsonl bị lỗi ở dòng {number}. Giữ tệp gốc và kiểm tra lại trước khi tổng hợp.") from error
            return records

    def trial_summary(self):
        records = self.trial_records()
        training_hashes = {row[key] for row in self.rows for key in ("normalized_video_sha256", "source_video_sha256") if key in row}
        by_model = {name: training_hashes | set(config.get("train_source_sha256", [])) | set(config.get("train_normalized_video_sha256", []))
                    for name, (_, config, _) in self.models.items()}
        for name, (_, config, _) in self.models.items():
            source_hashes = set(config.get("train_source_sha256", []))
            training_ids = set(config.get("train_ids", []))
            by_model[name].update(row["normalized_video_sha256"] for row in self.outside_rows
                                  if row["id"] in training_ids and row["source_video_sha256"] in source_hashes)
        summary = summarize(records, training_hashes, self.labels, by_model)
        summary["models"] = [{"id": key, "name": MODEL_NAMES.get(key, key), **value} for key, value in sorted(summary["models"].items())]
        return summary


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        server_version = "KyHieuLab/6"

        def send(self, status, body, content_type="application/json; charset=utf-8", attachment=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; img-src 'self' data:; connect-src 'self'")
            if attachment:
                self.send_header("Content-Disposition", 'attachment; filename="' + attachment + '"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/health":
                return self.send(200, {"ready": True, "version": 6, "models": list(state.models)})
            if route == "/api/config":
                return self.send(200, {"labels": state.labels, "default_model": state.default_model, "models": {key: MODEL_NAMES[key] for key in state.models}, "samples": [{"id": r["id"], "label": r["label"]} for r in state.rows], "outside_samples": [{"id": r["id"], "label": r["label"], "url": "/outside-samples/" + r["id"], "training_sample": True} for r in state.outside_samples.values()], "original_videos": 8, "heldout_unique_clips": 5, "heldout_labels": ["ai cho", "không cần"], "outside_label": OUTSIDE})
            if route == "/api/trials.csv":
                try:
                    return self.send(200, state.trial_csv(), "text/csv; charset=utf-8", "KHL_phieu_thuc_nghiem.csv")
                except (ValueError, OSError) as error:
                    return self.send(400, {"error": str(error)})
            if route == "/api/trial-summary":
                try:
                    return self.send(200, state.trial_summary())
                except (ValueError, OSError) as error:
                    return self.send(400, {"error": str(error)})
            if route.startswith("/samples/"):
                row = state.samples.get(route.removeprefix("/samples/"))
                if row:
                    return self.send(200, (state.root / row["video"]).read_bytes(), "video/mp4")
            if route.startswith("/outside-samples/"):
                row = state.outside_samples.get(route.removeprefix("/outside-samples/"))
                if row:
                    return self.send(200, (state.root / row["video"]).read_bytes(), "video/mp4")
            static = {
                "/": ("web/index.html", "text/html; charset=utf-8"),
                "/app.js": ("web/app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("web/style.css", "text/css; charset=utf-8"),
                "/api/evaluation": ("reports/metrics.json", "application/json; charset=utf-8"),
                "/api/evaluation-v5": ("reports/evaluation_v5.json", "application/json; charset=utf-8"),
                "/api/rejection-v6": ("reports/rejection_v6.json", "application/json; charset=utf-8"),
                "/reports/confusion.png": ("reports/figures/confusion_primary.png", "image/png"),
                "/reports/confusion_reject.png": ("reports/figures/confusion_reject_primary.png", "image/png"),
                "/reports/report.md": ("reports/Bao_cao_DoAn_KyHieuLab_v6.md", "text/markdown; charset=utf-8"),
                "/reports/report.pdf": ("reports/Bao_cao_DoAn_KyHieuLab_v6.pdf", "application/pdf"),
                "/reports/evaluation.md": ("reports/EVALUATION_V5.md", "text/markdown; charset=utf-8"),
            }
            if route in static:
                relative, mime = static[route]
                path = state.root / relative
                if path.is_file():
                    return self.send(200, path.read_bytes(), mime)
            return self.send(404, {"error": "Không tìm thấy."})

        def do_POST(self):
            route = urlparse(self.path).path
            if route not in ("/api/predict", "/api/trials"):
                return self.send(404, {"error": "Không tìm thấy."})
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.send(403, {"error": "Chỉ nhận thao tác từ giao diện trên máy này."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                limit = MAX_VIDEO_BYTES if route == "/api/predict" else 16384
                if not 0 < length <= limit:
                    raise ValueError("Kích thước dữ liệu không hợp lệ.")
                self.connection.settimeout(60)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Dữ liệu truyền lên chưa đầy đủ.")
                if route == "/api/trials":
                    payload = json.loads(body)
                    if not isinstance(payload, dict):
                        raise ValueError("Phiếu phải là một đối tượng JSON.")
                    return self.send(200, state.save_trial(payload))
                args = parse_qs(urlparse(self.path).query)
                answer = state.infer(body, model=args.get("model", [state.default_model])[0], start=float(args.get("start", ["0"])[0]), end=float(args.get("end", ["0"])[0]))
                return self.send(200, answer)
            except InvalidTrial as error:
                return self.send(409, {"error": str(error)})
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                return self.send(400, {"error": str(error)})
            except Exception as error:
                print(type(error).__name__, str(error), flush=True)
                return self.send(500, {"error": "Không xử lý được video. Thử clip MP4 ngắn và chạy tools/doctor.py để kiểm tra cài đặt."})
    return Handler


def create_server(port=8765, state=None):
    state = state or State()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    from run import main
    main()
