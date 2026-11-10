"""Summarize recorded attempts; annotations are user reports, not expert truth."""
from pathlib import Path
import argparse
import collections
import json

ROOT = Path(__file__).resolve().parents[1]


def summarize(records, training_hashes=None, labels=None, training_hashes_by_model=None):
    training_hashes = set(training_hashes or [])
    training_hashes_by_model = {key: set(value) for key, value in (training_hashes_by_model or {}).items()}
    def is_training_replay(record):
        return record["video_sha256"] in training_hashes_by_model.get(record["model"], training_hashes)
    labels = list(labels or [])
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["model"]].append(record)
    models = {}
    for name, attempts in groups.items():
        inside = [r for r in attempts if r["actual_label"] != "__OTHER__"]
        outside = [r for r in attempts if r["actual_label"] == "__OTHER__"]
        accepted_inside = [r for r in inside if r["accepted"]]
        versions = collections.Counter(r.get("model_sha256", "unknown") for r in attempts)
        models[name] = {
            "attempts": len(attempts), "unique_video_byte_hashes": len({r["video_sha256"] for r in attempts}),
            "in_vocabulary_attempts": len(inside), "outside_attempts": len(outside),
            "top1_accuracy_in_vocabulary": sum(r["top1_correct"] for r in inside) / len(inside) if inside else None,
            "coverage_in_vocabulary": len(accepted_inside) / len(inside) if inside else None,
            "accepted_accuracy_in_vocabulary": sum(r["top1_correct"] for r in accepted_inside) / len(accepted_inside) if accepted_inside else None,
            "outside_false_accept_rate": sum(r["accepted"] for r in outside) / len(outside) if outside else None,
            "signer_codes": sorted({r["signer_code"] for r in attempts}),
            "training_replay_attempts": sum(is_training_replay(r) for r in attempts),
            "model_versions": [{"model_sha256": key, "attempts": count} for key, count in sorted(versions.items())],
        }
        if labels:
            truth = labels + ["__OTHER__"]
            predictions = labels + ["__REJECT__"]
            matrix = [[0] * len(predictions) for _ in truth]
            for record in attempts:
                actual = record["actual_label"]
                predicted = record["predicted_label"] if record["accepted"] else "__REJECT__"
                if actual in truth and predicted in predictions:
                    matrix[truth.index(actual)][predictions.index(predicted)] += 1
            models[name]["decision_confusion"] = {"truth_labels": truth, "prediction_labels": predictions, "counts": matrix}
    return {
        "attempts": len(records),
        "unique_videos": len({r["video_sha256"] for r in records}),
        "signers": len({r["signer_code"] for r in records}),
        "training_replay_attempts": sum(is_training_replay(r) for r in records),
        "models": models,
        "annotation_status": "user_reported_unverified",
        "notes": ["Attempts can reuse the same video; they are not necessarily independent samples.", "A different byte hash does not prove a different source performance.", "Signer codes are user-entered, not independently verified identities.", "Results are not automatically an independent test or expert verification.", "Metrics can combine model versions; inspect model_versions before comparing."]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/trials.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/trial_summary.json")
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit("Chưa có phiếu thực nghiệm. Chạy demo và ghi phiếu trước.")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    training_hashes = {r[key] for r in manifest for key in ("normalized_video_sha256", "source_video_sha256") if key in r}
    by_model = {}
    outside_path = ROOT / "data/outside_manifest_v6.json"
    outside_rows = json.loads(outside_path.read_text(encoding="utf-8"))["entries"] if outside_path.exists() else []
    for config_path in (ROOT / "models").glob("*.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        by_model[config_path.stem] = training_hashes | set(config.get("train_source_sha256", [])) | set(config.get("train_normalized_video_sha256", []))
        source_hashes = set(config.get("train_source_sha256", []))
        training_ids = set(config.get("train_ids", []))
        by_model[config_path.stem].update(row["normalized_video_sha256"] for row in outside_rows
                                         if row["id"] in training_ids and row["source_video_sha256"] in source_hashes)
    result = summarize(records, training_hashes, sorted({r["label"] for r in manifest}), by_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
