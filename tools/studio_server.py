"""Lightweight recording studio server for collecting new sign clips directly to dataset."""
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import csv
import json
import os
import sys
import threading
import time
import webbrowser

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
CLIPS_MOI_DIR = ROOT / "data" / "clips_moi"
PLAN_FILE = ROOT / "data" / "collection_plan_250.csv"
CLIPS_MOI_DIR.mkdir(parents=True, exist_ok=True)

PORT = 8767

LABEL_MAP = {
    "ai_cho": "ai cho",
    "buc_minh": "bực mình",
    "ghen_ti": "ghen tị",
    "khong_can": "không cần",
    "xin_loi": "xin lỗi"
}

class StudioHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/tools/studio_quay_video.html")
            self.end_headers()
            return
            
        if parsed.path == "/api/studio-status":
            files = list(CLIPS_MOI_DIR.glob("*.webm")) + list(CLIPS_MOI_DIR.glob("*.mp4"))
            file_names = [f.name for f in files]
            
            # Count by signer
            stats = {}
            for name in file_names:
                parts = name.split("_")
                if len(parts) >= 4:
                    s = parts[0]
                    stats[s] = stats.get(s, 0) + 1

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            resp = {
                "total_collected": len(files),
                "target": 250,
                "by_signer": stats,
                "files": file_names
            }
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/save-clip":
            query = parse_qs(parsed.query)
            filename = query.get("filename", [""])[0]
            if not filename or ".." in filename or "/" in filename or "\\" in filename:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "Tên file không hợp lệ"}).encode("utf-8"))
                return

            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 64 * 1024 * 1024:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "Kích thước video không hợp lệ (tối đa 64MB)"}).encode("utf-8"))
                return

            video_bytes = self.rfile.read(length)
            dest_path = CLIPS_MOI_DIR / filename
            dest_path.write_bytes(video_bytes)

            # Update collection_plan_250.csv if present
            parts = filename.replace(".webm", "").replace(".mp4", "").split("_")
            if len(parts) == 4 and PLAN_FILE.exists():
                signer, session, label_slug, rep_str = parts
                label_norm = LABEL_MAP.get(label_slug, label_slug)
                rep_num = str(int(rep_str.replace("R", "")))

                try:
                    with open(PLAN_FILE, "r", encoding="utf-8-sig") as f:
                        rows = list(csv.DictReader(f))
                    
                    updated = False
                    for r in rows:
                        if (r["signer_code"].upper() == signer.upper() and
                            r["session_code"].upper() == session.upper() and
                            r["label"] == label_norm and
                            str(r["repetition"]) == rep_num):
                            r["status"] = "collected"
                            r["video_path"] = str(dest_path.relative_to(ROOT))
                            updated = True
                            break

                    if updated:
                        fieldnames = ["signer_code", "session_code", "label", "repetition", "proposed_split", "status", "video_path", "start", "end", "review_status"]
                        with open(PLAN_FILE, "w", encoding="utf-8-sig", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)
                except Exception as e:
                    print("Lỗi cập nhật plan CSV:", e)

            files = list(CLIPS_MOI_DIR.glob("*.webm")) + list(CLIPS_MOI_DIR.glob("*.mp4"))

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": True,
                "saved_path": str(dest_path.relative_to(ROOT)),
                "size_kb": round(len(video_bytes) / 1024, 1),
                "total_collected": len(files)
            }, ensure_ascii=False).encode("utf-8"))
            print(f"✅ Đã lưu trực tiếp: {filename} ({round(len(video_bytes)/1024, 1)} KB). Tổng cộng: {len(files)} clip.", flush=True)
            return

        self.send_response(404)
        self.end_headers()

def main():
    server = ThreadingHTTPServer(("localhost", PORT), StudioHandler)
    url = f"http://localhost:{PORT}/tools/studio_quay_video.html"
    print("=" * 70, flush=True)
    print("  🎥 KÝ HIỆU LAB - MÁY CHỦ QUAY VÀ LƯU VIDEO TRỰC TIẾP", flush=True)
    print("=" * 70, flush=True)
    print(f"Địa chỉ Studio: {url}", flush=True)
    print(f"Thư mục lưu video tự động: {CLIPS_MOI_DIR}", flush=True)
    print("Mỗi khi bạn quay xong 1 clip, video sẽ tự động lưu thẳng vào thư mục trên!", flush=True)
    print("Nhấn Ctrl+C để tắt máy chủ.", flush=True)
    print("-" * 70, flush=True)

    timer = threading.Timer(0.8, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng máy chủ studio.")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
