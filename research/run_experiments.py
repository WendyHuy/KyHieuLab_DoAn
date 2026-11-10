"""Reproduce the frozen five-label protocol; never select by held-out scores."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import training
from training_variant import fit_candidate
from core import predict, resample_masked


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint():
    digest = hashlib.sha256()
    for path in [ROOT / "research/protocol.json", ROOT / "core.py", ROOT / "training.py", ROOT / "training_variant.py", *sorted((ROOT / "data/features").glob("*.npz"))]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def score(weights, config, row):
    with np.load(ROOT / "data/features" / (row["id"] + ".npz")) as z:
        result = predict(z["features"], z["times"], weights, config)
    top = result["top"]
    return {
        "id": row["id"], "source_sha256": row["source_video_sha256"],
        "provided": row["label"], "predicted": top[0]["label"] if top else None,
        "top_score": top[0]["score"] if top else None,
        "top3": [r["label"] for r in top],
        "top1_correct": bool(top and top[0]["label"] == row["label"]),
        "top3_correct": row["label"] in [r["label"] for r in top],
        "accepted": result["recognized"], "hand_fraction": result.get("hand_fraction", 0),
    }


def fit(records, labels, method, seed):
    if method == "tcn_augmented":
        weights, config, history = fit_candidate(records, seed)
    else:
        weights, config, history, _ = training.fit(records, labels, method, seed)
    config.update({
        "purpose": "course proof of concept; not independently validated",
        "vocabulary_size": 5, "expert_verified": False, "review_metadata_unchanged": True,
        "method": method, "threshold": 0.65, "threshold_calibrated": False,
    })
    return weights, config, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="Recompute cached fits and replace their outputs.")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "research/protocol.json").read_text(encoding="utf-8"))
    rows = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    labels = protocol["labels"]
    assert len(rows) == 8 and len(labels) == 5
    assert all(r["experimental_include"] for r in rows)
    counts = collections.Counter(r["label"] for r in rows)
    eligible = [r for r in rows if counts[r["label"]] > 1]
    run_fingerprint = fingerprint()
    checkpoint_dir = ROOT / "research/checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    results = []
    for seed in protocol["seeds"]:
        for fold, held in enumerate(eligible):
            train_rows = [r for r in rows if r["id"] != held["id"]]
            assert len(train_rows) == 7
            assert held["source_video_sha256"] not in {r["source_video_sha256"] for r in train_rows}
            for method in protocol["models"]:
                checkpoint = checkpoint_dir / f"{method}_{seed}_{held['id']}.json"
                if checkpoint.exists() and not args.rerun:
                    record = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if record.get("run_fingerprint") == run_fingerprint:
                        results.append(record)
                        continue
                started = time.perf_counter()
                weights, config, history = fit(train_rows, labels, method, seed + fold * 100)
                record = score(weights, config, held)
                record.update({
                    "method": method, "seed": seed, "fold": fold,
                    "training_seed": seed + fold * 100,
                    "train_ids": [r["id"] for r in train_rows],
                    "train_source_sha256": [r["source_video_sha256"] for r in train_rows],
                    "normalization_mean": config["mean"], "normalization_std": config["std"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "run_fingerprint": run_fingerprint, "history": history,
                })
                write_json(checkpoint, record)
                results.append(record)
                print("HELD_OUT", method, seed, held["id"], record["predicted"], record["accepted"], flush=True)
    write_json(ROOT / "research/oof_results.json", results)
    # Fit demonstration weights on all eight originals, distinct from OOF fits.
    for method in protocol["models"]:
        config_path = ROOT / "models" / (method + ".json")
        if config_path.exists() and not args.rerun:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("run_fingerprint") == run_fingerprint:
                continue
        weights, config, history = fit(rows, labels, method, protocol["seeds"][0])
        config["run_fingerprint"] = run_fingerprint
        config["train_ids"] = [r["id"] for r in rows]
        np.savez_compressed(ROOT / "models" / (method + ".npz"), **weights)
        write_json(config_path, config)
        write_json(ROOT / "research" / (method + "_full_history.json"), history)
        write_json(ROOT / "research" / (method + "_training_replay.json"), [score(weights, config, r) for r in rows])
    print("COMPLETE", len(results), "OOF predictions; 5 unique clips, not 15 per method.", flush=True)


if __name__ == "__main__":
    main()
