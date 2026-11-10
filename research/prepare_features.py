"""Extract the same landmark schema from the normalized course videos."""
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import extract_video


def main():
    manifest_path = ROOT / "data/manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in rows:
        started = time.perf_counter()
        x, t, quality = extract_video(ROOT / row["video"])
        assert np.isfinite(x).all() and np.all(np.diff(t) > 0)
        np.savez_compressed(ROOT / "data/features" / (row["id"] + ".npz"), features=x, times=t)
        row["quality"] = quality
        row["extract_seconds"] = time.perf_counter() - started
        row["feature_status"] = "extracted_normalized_v4_video"
        print(row["id"], row["label"], quality, flush=True)
    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
