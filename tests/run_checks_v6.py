"""Regression and v6 reject-class provenance checks; all logs are temporary."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import State, apply_decision_policy, create_server
from test_project import ProjectTests
from run_checks_v5 import V5SummaryTests

REQUIRE_CANDIDATE = False


class BaselineProjectTests(ProjectTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Preserve the original three-model regression scope. New candidate and
        # promotion rules are checked separately below against a fresh State.
        cls.state.models = {name: model for name, model in cls.state.models.items() if name != "tcn_reject"}
        cls.state.default_model = "tcn"


class V6DecisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="khl_v6_policy_")
        self.temp = Path(self.temporary.name)
        self.server = None

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self.temporary.cleanup()

    def staged_root(self, candidate=True):
        target = self.temp / "project"
        (target / "models").mkdir(parents=True)
        (target / "data").mkdir()
        (target / "reports").mkdir()
        shutil.copy2(ROOT / "data/manifest.json", target / "data/manifest.json")
        for name in ("mlp", "tcn", "tcn_augmented"):
            for suffix in (".npz", ".json"):
                shutil.copy2(ROOT / "models" / (name + suffix), target / "models" / (name + suffix))
        if candidate:
            with np.load(ROOT / "models/tcn.npz") as original:
                weights = {key: original[key].copy() for key in original.files}
            weights["w4"] = np.zeros((48, 6), dtype=np.float32)
            # This is a controlled software fixture, not a trained quality claim.
            weights["b4"] = np.array([-8, -7, -6, -5, -4, 10], dtype=np.float32)
            np.savez_compressed(target / "models/tcn_reject.npz", **weights)
            config = json.loads((ROOT / "models/tcn.json").read_text(encoding="utf-8"))
            config.update({"supported_labels": config["labels"].copy(), "labels": config["labels"] + ["__OTHER__"], "reject_label": "__OTHER__", "train_source_sha256": ["candidate_only_source"], "train_normalized_video_sha256": ["candidate_only_normalized"]})
            (target / "models/tcn_reject.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return target

    def recommendation(self, target, passed=True, digest=None):
        path = target / "models/tcn_reject.npz"
        digest = digest if digest is not None else hashlib.sha256(path.read_bytes()).hexdigest()
        report = {"promotion_passed": passed, "recommended_default": "tcn_reject", "full_model": {"sha256": digest}}
        (target / "reports/rejection_v6.json").write_text(json.dumps(report), encoding="utf-8")

    def serve(self, state):
        self.server = create_server(0, state)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def get(self, endpoint):
        return urllib.request.urlopen(self.url + endpoint, timeout=120).read()

    def post(self, endpoint, body):
        body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.url + endpoint, data=body, method="POST")
        return json.loads(urllib.request.urlopen(request, timeout=120).read())

    def sample_bytes(self):
        rows = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
        return (ROOT / next(r["video"] for r in rows if r["id"] == "3d328846")).read_bytes()

    def test_29_raw_other_forces_rejection_without_renormalizing_suggestions(self):
        original = {"recognized": True, "message": "__OTHER__", "top": [{"label": "__OTHER__", "score": .8}, {"label": "ai cho", "score": .1}, {"label": "không cần", "score": .05}]}
        result = apply_decision_policy(original, {"reject_label": "__OTHER__"})
        self.assertFalse(result["recognized"])
        self.assertTrue(result["rejected_as_outside"])
        self.assertEqual(result["raw_decision_class"], "__OTHER__")
        self.assertEqual(result["raw_decision_score"], .8)
        self.assertEqual(result["rejection_reason"], "outside_vocabulary")
        self.assertEqual(result["top"], [{"label": "ai cho", "score": .1}, {"label": "không cần", "score": .05}])
        self.assertEqual(original["top"][0]["label"], "__OTHER__")

    def test_30_supported_low_score_is_not_promoted_after_filtering_other(self):
        result = apply_decision_policy({"recognized": False, "message": "Chưa chắc chắn", "top": [{"label": "ai cho", "score": .6}, {"label": "__OTHER__", "score": .3}, {"label": "xin lỗi", "score": .05}]}, {"reject_label": "__OTHER__"})
        self.assertFalse(result["recognized"])
        self.assertFalse(result["rejected_as_outside"])
        self.assertEqual(result["rejection_reason"], "below_threshold")
        self.assertEqual(result["raw_decision_score"], .6)
        self.assertEqual(result["top"][0]["score"], .6)

    def test_31_no_hand_decision_has_no_invented_raw_class(self):
        result = apply_decision_policy({"recognized": False, "message": "Chưa thấy đủ bàn tay", "top": []}, {"reject_label": "__OTHER__"})
        self.assertFalse(result["recognized"])
        self.assertFalse(result["rejected_as_outside"])
        self.assertEqual(result["raw_decision_class"], "")
        self.assertIsNone(result["raw_decision_score"])
        self.assertEqual(result["rejection_reason"], "insufficient_hands")

    def test_32_candidate_is_optional_without_weights(self):
        target = self.staged_root(candidate=False)
        state = State(root=target, output_dir=self.temp / "outputs")
        self.assertEqual(set(state.models), {"mlp", "tcn", "tcn_augmented"})
        self.assertEqual(state.default_model, "tcn")

    def test_33_promotion_requires_matching_hash_and_keeps_five_public_labels(self):
        target = self.staged_root()
        self.recommendation(target)
        state = State(root=target, output_dir=self.temp / "outputs")
        self.assertEqual(state.default_model, "tcn_reject")
        self.serve(state)
        config = json.loads(self.get("/api/config"))
        self.assertEqual(len(config["labels"]), 5)
        self.assertNotIn("__OTHER__", config["labels"])
        self.assertEqual(set(config["models"]), {"mlp", "tcn", "tcn_augmented", "tcn_reject"})
        self.assertEqual(config["default_model"], "tcn_reject")
        answer = self.post("/api/predict", self.sample_bytes())
        self.assertEqual(answer["model"], "tcn_reject")
        self.assertEqual(answer["raw_decision_class"], "__OTHER__")
        self.assertFalse(answer["recognized"])

    def test_34_failed_or_stale_promotion_retains_baseline_default(self):
        target = self.staged_root()
        self.recommendation(target, passed=False)
        self.assertEqual(State(root=target).default_model, "tcn")
        self.recommendation(target, passed=True, digest="stale-model-hash")
        self.assertEqual(State(root=target).default_model, "tcn")

    def test_35_other_annotation_preserves_raw_decision_not_filtered_suggestion(self):
        target = self.staged_root()
        state = State(root=target, output_dir=self.temp / "outputs")
        self.serve(state)
        answer = self.post("/api/predict?model=tcn_reject", self.sample_bytes())
        self.assertTrue(answer["rejected_as_outside"])
        self.assertNotIn("__OTHER__", [r["label"] for r in answer["top"]])
        self.assertGreater(answer["raw_decision_score"], max(r["score"] for r in answer["top"]))
        result = self.post("/api/trials", {"inference_id": answer["inference_id"], "actual_label": "__OTHER__", "signer_code": "QA_V6", "session_code": "QA_V6_B01", "predicted_label": answer["top"][0]["label"], "top_score": 0, "accepted": True})
        self.assertTrue(result["saved"])
        rows = list(csv.DictReader(io.StringIO(self.get("/api/trials.csv").decode("utf-8-sig"))))
        row = rows[0]
        self.assertEqual(row["predicted_label"], "__OTHER__")
        self.assertAlmostEqual(float(row["top_score"]), answer["raw_decision_score"])
        self.assertEqual(row["top1_correct"], "True")
        self.assertEqual(row["accepted"], "False")
        self.assertEqual(row["rejected_as_outside"], "True")
        self.assertEqual(row["rejection_reason"], "outside_vocabulary")

    def test_36_other_raw_decision_stays_wrong_for_supported_truth(self):
        target = self.staged_root()
        state = State(root=target, output_dir=self.temp / "outputs")
        self.serve(state)
        answer = self.post("/api/predict?model=tcn_reject", self.sample_bytes())
        supported_suggestion = answer["top"][0]["label"]
        self.post("/api/trials", {"inference_id": answer["inference_id"], "actual_label": supported_suggestion, "signer_code": "QA_V6", "session_code": "QA_V6_B01"})
        rows = list(csv.DictReader(io.StringIO(self.get("/api/trials.csv").decode("utf-8-sig"))))
        self.assertEqual(rows[0]["predicted_label"], "__OTHER__")
        self.assertEqual(rows[0]["top1_correct"], "False")
        self.assertEqual(rows[0]["accepted_correct"], "False")

    def test_37_replay_tracking_is_specific_to_model_training_sources(self):
        target = self.staged_root()
        state = State(root=target, output_dir=self.temp / "outputs")
        base = dict.fromkeys(__import__("app").TRIAL_COLUMNS, "")
        base.update({"actual_label": "__OTHER__", "predicted_label": "__OTHER__", "accepted": False, "top1_correct": True, "signer_code": "QA_V6", "session_code": "QA_V6_B01", "video_sha256": "candidate_only_normalized", "model_sha256": "fixture"})
        state.output_dir.mkdir()
        records = [{**base, "model": name} for name in ["tcn", "tcn_reject"]]
        (state.output_dir / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        summary = state.trial_summary()
        models = {model["id"]: model for model in summary["models"]}
        self.assertEqual(summary["unique_videos"], 1)
        self.assertEqual(summary["training_replay_attempts"], 1)
        self.assertEqual(models["tcn"]["training_replay_attempts"], 0)
        self.assertEqual(models["tcn_reject"]["training_replay_attempts"], 1)

    def test_38_production_candidate_loads_and_default_matches_frozen_promotion(self):
        state = State(output_dir=self.temp / "outputs")
        if "tcn_reject" not in state.models:
            if REQUIRE_CANDIDATE:
                self.fail("The release requires candidate weights but they are not present.")
            self.skipTest("Candidate training is not complete yet.")
        report = json.loads((ROOT / "reports/rejection_v6.json").read_text(encoding="utf-8"))
        should_promote = (report.get("promotion_passed") is True and report.get("recommended_default") == "tcn_reject" and report.get("full_model", {}).get("sha256") == state.models["tcn_reject"][2])
        self.assertEqual(state.default_model, "tcn_reject" if should_promote else "tcn")
        config = state.models["tcn_reject"][1]
        self.assertEqual(config["labels"], state.labels + ["__OTHER__"])
        self.assertEqual(config["supported_labels"], state.labels)
        self.assertEqual(config["reject_label"], "__OTHER__")
        self.serve(state)
        answer = self.post("/api/predict?model=tcn_reject", self.sample_bytes())
        self.assertEqual(answer["model"], "tcn_reject")
        self.assertTrue(all(row["label"] in state.labels for row in answer["top"]))
        self.assertIn(answer["raw_decision_class"], config["labels"])
        if answer["raw_decision_class"] == "__OTHER__":
            self.assertFalse(answer["recognized"])
            self.assertTrue(answer["rejected_as_outside"])


    def test_39_reencoded_negative_replay_requires_source_id_and_hash_match(self):
        target = self.staged_root()
        config_path = target / "models/tcn_reject.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["train_ids"] = config.get("train_ids", []) + ["qa_negative"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        rows = [{"id": "qa_negative", "label": "outside fixture", "source_video_sha256": "candidate_only_source", "normalized_video_sha256": "new_encoder_hash"},
                {"id": "wrong_id", "label": "outside fixture", "source_video_sha256": "candidate_only_source", "normalized_video_sha256": "wrong_id_hash"},
                {"id": "qa_negative", "label": "outside fixture", "source_video_sha256": "wrong_source", "normalized_video_sha256": "wrong_source_hash"}]
        (target / "data/outside_manifest_v6.json").write_text(json.dumps({"entries": rows}), encoding="utf-8")
        state = State(root=target, output_dir=self.temp / "outputs")
        base = dict.fromkeys(__import__("app").TRIAL_COLUMNS, "")
        base.update({"actual_label": "__OTHER__", "predicted_label": "__OTHER__", "accepted": False, "top1_correct": True, "signer_code": "QA_V6", "session_code": "QA_V6_B01", "model_sha256": "fixture"})
        records = [{**base, "model": model, "video_sha256": digest} for model, digest in [("tcn", "new_encoder_hash"), ("tcn_reject", "new_encoder_hash"), ("tcn_reject", "wrong_id_hash"), ("tcn_reject", "wrong_source_hash")]]
        state.output_dir.mkdir()
        (state.output_dir / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        summary = state.trial_summary()
        models = {row["id"]: row for row in summary["models"]}
        self.assertEqual(summary["training_replay_attempts"], 1)
        self.assertEqual(models["tcn"]["training_replay_attempts"], 0)
        self.assertEqual(models["tcn_reject"]["training_replay_attempts"], 1)

    def test_40_outside_demo_routes_are_allowlisted_existing_performances(self):
        state = State(output_dir=self.temp / "outputs")
        self.serve(state)
        config = json.loads(self.get("/api/config"))
        self.assertEqual({r["id"] for r in config["outside_samples"]}, {"c3b59369", "b93f88f2"})
        self.assertTrue(all(r["training_sample"] for r in config["outside_samples"]))
        for sample in config["outside_samples"]:
            original = state.outside_samples[sample["id"]]
            received = self.get(sample["url"])
            self.assertEqual(hashlib.sha256(received).hexdigest(), original["normalized_video_sha256"])
        for endpoint in ["/outside-samples/missing", "/outside-samples/%2e%2e/models/tcn.npz"]:
            with self.assertRaises(__import__("urllib.error").error.HTTPError) as captured:
                self.get(endpoint)
            self.assertEqual(captured.exception.code, 404)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-candidate", action="store_true", help="Fail instead of skipping if candidate weights are still absent.")
    REQUIRE_CANDIDATE = parser.parse_args().require_candidate
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([loader.loadTestsFromTestCase(BaselineProjectTests), loader.loadTestsFromTestCase(V5SummaryTests), loader.loadTestsFromTestCase(V6DecisionTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "passed": result.wasSuccessful(), "candidate_required": REQUIRE_CANDIDATE, "isolated_output_directory": True, "baseline_regressions": 28, "v6_decision_provenance_tests": 12, "physical_webcam_tested": False, "notes": ["Forced OTHER fixtures validate software decisions and annotations, not model quality.", "Legacy OOF source data and three-model baselines are unchanged.", "No QA annotations or raw uploads remain in user outputs."]}
    (ROOT / "reports/test_results_v6.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
