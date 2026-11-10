"""Count results without treating repeated seeds as new source videos."""
from pathlib import Path
import collections
import csv
import json
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"
METHODS = {"mlp": "MLP", "tcn": "TCN gốc", "tcn_augmented": "TCN tăng cường"}


def csv_write(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    protocol = json.loads((ROOT / "research/protocol.json").read_text(encoding="utf-8"))
    data = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "research/oof_results.json").read_text(encoding="utf-8"))
    primary_seed = protocol["seeds"][0]
    labels = protocol["labels"]
    tested = protocol["heldout_labels"]
    models = []
    assert len(results) == 45 and len({r["id"] for r in results}) == 5
    for method, name in METHODS.items():
        per_seed = []
        for seed in protocol["seeds"]:
            records = [r for r in results if r["method"] == method and r["seed"] == seed]
            assert len(records) == 5
            for record in records:
                assert record["id"] not in record["train_ids"]
                assert record["source_sha256"] not in record["train_source_sha256"]
            true = [r["provided"] for r in records]
            predicted = [r["predicted"] for r in records]
            accepted = [r for r in records if r["accepted"]]
            correct = sum(r["top1_correct"] for r in records)
            accepted_correct = sum(r["top1_correct"] for r in accepted)
            per_seed.append({"seed": seed, "n_unique_clips": 5, "top1_correct": correct, "top1_accuracy": correct / 5, "macro_f1_two_tested_labels": float(f1_score(true, predicted, labels=tested, average="macro", zero_division=0)), "top3_correct": sum(r["top3_correct"] for r in records), "accepted": len(accepted), "accepted_correct": accepted_correct, "accepted_wrong": len(accepted) - accepted_correct, "coverage": len(accepted) / 5, "accepted_accuracy": accepted_correct / len(accepted) if accepted else None})
        model = {"id": method, "name": name, "primary": per_seed[0], "per_seed": per_seed, "mean_top1_accuracy_over_seeds": float(np.mean([r["top1_accuracy"] for r in per_seed])), "std_top1_accuracy_over_seeds_population": float(np.std([r["top1_accuracy"] for r in per_seed])), "note": "Variability over three seeds on the same five clips; not independent samples or a confidence interval."}
        models.append(model)
        records = [r for r in results if r["method"] == method and r["seed"] == primary_seed]
        for reject in [False, True]:
            columns = labels + (["CHUA_CHAC"] if reject else [])
            matrix = np.zeros((len(labels), len(columns)), int)
            for r in records:
                predicted = "CHUA_CHAC" if reject and not r["accepted"] else r["predicted"]
                matrix[labels.index(r["provided"]), columns.index(predicted)] += 1
            csv_write(OUT / f"{method}_confusion{'_reject' if reject else ''}.csv", ["true_label"] + columns, [[label] + row.tolist() for label, row in zip(labels, matrix)])
        precision, recall, f1, support = precision_recall_fscore_support([r["provided"] for r in records], [r["predicted"] for r in records], labels=tested, zero_division=0)
        csv_write(OUT / f"{method}_per_label_primary.csv", ["label", "precision", "recall", "f1", "support"], list(zip(tested, precision, recall, f1, support)))
    summary = {
        "date": "2026-11-03", "primary_seed": primary_seed, "default_model": "tcn",
        "data": {"labels": labels, "original_videos": 8, "people": ["giahuy"], "unique_heldout_clips": 5, "tested_labels": tested, "untested_labels": protocol["unmeasured_labels"], "new_real_videos": 0},
        "models": models, "threshold": 0.65, "threshold_calibrated": False,
        "decision": "TCN baseline remains the preselected default; augmented candidate is comparison only.",
        "readiness": {"course_proof_of_concept": "implemented with explicit data limitations", "independent_five_label_validation": False, "public_translation": False},
        "limitations": protocol["limitations"],
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_write(OUT / "metrics_by_seed.csv", ["method", "seed", "n_unique_clips", "top1_correct", "top1_accuracy", "macro_f1_two_tested_labels", "accepted", "accepted_correct", "accepted_wrong", "coverage"], [[model["id"]] + [r[k] for k in ["seed", "n_unique_clips", "top1_correct", "top1_accuracy", "macro_f1_two_tested_labels", "accepted", "accepted_correct", "accepted_wrong", "coverage"]] for model in models for r in model["per_seed"]])
    fields = ["method", "seed", "fold", "id", "provided", "predicted", "top_score", "top1_correct", "top3_correct", "accepted", "hand_fraction"]
    csv_write(OUT / "oof_predictions.csv", fields, [[r[k] for k in fields] for r in results])
    counts = collections.Counter(r["label"] for r in data)
    csv_write(OUT / "label_coverage.csv", ["label", "original_videos", "signers", "heldout_unique_clips", "evaluation_status"], [[label, counts[label], "giahuy", counts[label] if counts[label] > 1 else 0, "retrospective_same_signer" if counts[label] > 1 else "no_heldout_sample"] for label in labels])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    for reject in [False, True]:
        fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.6 if not reject else 3.9), layout="constrained", sharey=True)
        for ax, (method, name) in zip(axes, METHODS.items()):
            matrix = np.zeros((2, 4 if reject else 3), int)
            for r in [r for r in results if r["method"] == method and r["seed"] == primary_seed]:
                col = 3 if reject and not r["accepted"] else tested.index(r["predicted"]) if r["predicted"] in tested else 2
                matrix[tested.index(r["provided"]), col] += 1
            ax.imshow(matrix, cmap="YlOrBr" if reject else "Greens", vmin=0, vmax=3)
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, str(value), ha="center", va="center", fontsize=18, color="white" if value >= 2 else "black")
            ax.set_title(name, fontsize=13)
            ax.set_xticks(range(matrix.shape[1]), ["ai cho", "không cần", "khác"] + (["từ chối"] if reject else []), rotation=45, ha="right", fontsize=11)
            ax.set_yticks(range(2), tested, fontsize=11)
            ax.set_xlabel("Kết quả" if reject else "Nhãn dự đoán", fontsize=11)
        axes[0].set_ylabel("Nhãn cung cấp", fontsize=11)
        fig.suptitle(f"5 clip giữ ngoài phần học · seed {primary_seed}" + (" · ngưỡng 0,65" if reject else ""), fontsize=13)
        fig.savefig(FIG / ("confusion_reject_primary.png" if reject else "confusion_primary.png"), dpi=220)
        plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
