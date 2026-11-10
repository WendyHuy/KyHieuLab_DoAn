"""Check the demo installation without training or changing user files."""
from pathlib import Path
import argparse
import importlib
import importlib.metadata
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", action="store_true", help="Run one actual sample to prepare MediaPipe assets before the demonstration.")
    args = parser.parse_args()
    failures = []
    print("Ký Hiệu Lab v6 - Kiểm tra cài đặt", flush=True)
    print("Python:", sys.version.split()[0])
    for module, distribution, expected in [("numpy", "numpy", "1.26.4"), ("cv2", "opencv-contrib-python", "4.11.0.86"), ("mediapipe", "mediapipe", "0.10.21")]:
        try:
            importlib.import_module(module)
            version = importlib.metadata.version(distribution)
            print(distribution + ":", version)
            if version != expected:
                failures.append(distribution + " cần bản " + expected)
        except Exception as error:
            failures.append(distribution + ": " + str(error))
    if not failures:
        try:
            from app import State
            state = State()
            print("Đã đọc", len(state.samples), "clip và", len(state.models), "mô hình.")
            print("Nhãn:", ", ".join(state.labels))
            from app import MODEL_NAMES
            print("Cấu hình mặc định:", MODEL_NAMES[state.default_model], "; ngưỡng 0,65 chưa hiệu chỉnh.")
            if args.warmup:
                print("Đang chuẩn bị bộ trích điểm bằng một clip mẫu. Lần đầu có thể cần mạng để tải mô hình MediaPipe…", flush=True)
                sample = state.rows[0]
                answer = state.infer((ROOT / sample["video"]).read_bytes())
                print("Đường xử lý video đã sẵn sàng;", answer["quality"]["sampled_frames"], "khung lấy mẫu.", flush=True)
                print("Đây là kiểm tra kỹ thuật bằng mẫu học, không đo độ chính xác độc lập.")
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            failures.append(str(error))
    if failures:
        print("\nCần xử lý:")
        for message in failures:
            print("-", message)
        print("Cài lại bằng Python 3.11 64-bit trong môi trường riêng với requirements.txt.")
        return 1
    print("Cài đặt đủ để chạy demo. Kiểm tra này không đo độ chính xác hoặc webcam vật lý.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
