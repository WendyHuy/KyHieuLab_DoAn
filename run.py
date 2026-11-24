"""Start the local demo after the server has bound its port."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import webbrowser

# Auto-relaunch inside .venv if executed with an external Python (e.g. system Python 3.14)
_ROOT = Path(__file__).resolve().parent
_VENV_PYTHON = _ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

if _VENV_PYTHON.is_file():
    try:
        if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
            sys.exit(subprocess.call([str(_VENV_PYTHON)] + sys.argv))
    except Exception:
        pass


def main():
    try:
        import mediapipe
    except ImportError:
        print("\n[LỖI CÀI ĐẶT] Không tìm thấy thư viện MediaPipe trong môi trường Python hiện tại.")
        print(f"  Python đang chạy: {sys.executable} ({sys.version.split()[0]})")
        print("  Giải pháp: Hãy chạy ứng dụng bằng tệp 'CHAY_DEMO.bat'")
        print("  Hoặc gõ lệnh: .venv\\Scripts\\python.exe run.py\n")
        raise SystemExit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    from app import create_server
    try:
        server = create_server(args.port)
    except OSError:
        raise SystemExit("Cổng đang được dùng. Thử: python run.py --port 8766")
    url = f"http://localhost:{server.server_port}"
    print("ViSign-Hub v6 sẵn sàng tại " + url, flush=True)
    print("Chỉ chạy trên máy này. Nhấn Ctrl+C để dừng.", flush=True)
    if not args.no_browser:
        timer = threading.Timer(0.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
