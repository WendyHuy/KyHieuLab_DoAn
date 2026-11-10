"""Start the local demo after the server has bound its port."""
import argparse
import threading
import webbrowser


def main():
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
    print("Ký Hiệu Lab v5 sẵn sàng tại " + url, flush=True)
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
