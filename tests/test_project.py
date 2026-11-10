"""Meaningful data, masking, HTTP inference and trial provenance checks."""
from pathlib import Path
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import State, create_server
from core import resample_masked, normalize, predict, PARTS
from research.run_experiments import fingerprint


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="khl_tests_")
        cls.state = State(output_dir=Path(cls.temporary.name) / "outputs")
        cls.server = create_server(0, cls.state)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.rows = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
        cls.sample = next(r for r in cls.rows if r["id"] == "3d328846")
        cls.video = (ROOT / cls.sample["video"]).read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.temporary.cleanup()

    def get(self, path):
        return urllib.request.urlopen(self.url + path, timeout=120).read()

    def post(self, path, body, headers=None):
        request = urllib.request.Request(self.url + path, data=body, headers=headers or {}, method="POST")
        return json.loads(urllib.request.urlopen(request, timeout=120).read())

    def json_post(self, path, payload):
        return self.post(path, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})

    def assert_status(self, status, function):
        with self.assertRaises(urllib.error.HTTPError) as captured:
            function()
        self.assertEqual(captured.exception.code, status)

    def test_01_five_labels_eight_distinct_source_videos(self):
        self.assertEqual(len(self.rows), 8)
        self.assertEqual(len({r["label"] for r in self.rows}), 5)
        self.assertEqual(len({r["source_video_sha256"] for r in self.rows}), 8)

    def test_02_normalized_video_hashes_match(self):
        for row in self.rows:
            self.assertEqual(hashlib.sha256((ROOT / row["video"]).read_bytes()).hexdigest(), row["normalized_video_sha256"])

    def test_03_feature_shapes_masks_and_times(self):
        for row in self.rows:
            with np.load(ROOT / "data/features" / (row["id"] + ".npz")) as z:
                x, t = z["features"], z["times"]
                self.assertEqual(x.shape[1], 159)
                self.assertTrue(np.isfinite(x).all())
                self.assertTrue(np.all(np.diff(t) > 0))
                self.assertTrue(np.allclose(np.diff(t), 1 / 12, atol=0.002))
                self.assertTrue(np.isin(x[:, [65, 131, 158]], [0, 1]).all())

    def test_04_each_fold_excludes_source_before_augmentation(self):
        records = json.loads((ROOT / "research/oof_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(records), 45)
        self.assertEqual(len({r["id"] for r in records}), 5)
        for row in records:
            self.assertNotIn(row["id"], row["train_ids"])
            self.assertNotIn(row["source_sha256"], row["train_source_sha256"])
            self.assertEqual(len(row["train_ids"]), 7)
            self.assertEqual(row["run_fingerprint"], fingerprint())

    def test_05_missing_hand_does_not_become_interpolated_hand(self):
        x = np.ones((3, 159), dtype=np.float32)
        x[1, :66] = 0
        result = resample_masked(x, [0, 0.1, 0.2], [0.1])
        self.assertEqual(result[0, 65], 0)
        self.assertTrue(np.all(result[0, :65] == 0))

    def test_06_missing_body_zeros_body_relative_wrist_positions(self):
        x = np.ones((2, 159), dtype=np.float32)
        x[:, 158] = 0
        result = resample_masked(x, [0, 0.1])
        self.assertTrue(np.all(result[:, [63, 64, 129, 130]] == 0))

    def test_07_timestamps_must_increase(self):
        with self.assertRaises(ValueError):
            resample_masked(np.zeros((2, 159)), [0, 0])

    def test_08_no_hand_feature_sequence_rejected(self):
        weights, config, _ = self.state.models["tcn"]
        result = predict(np.zeros((20, 159), dtype=np.float32), np.arange(20) / 12, weights, config)
        self.assertFalse(result["recognized"])
        self.assertEqual(result["top"], [])

    def test_09_health_config_and_ui(self):
        self.assertTrue(json.loads(self.get("/api/health"))["ready"])
        self.assertEqual(len(json.loads(self.get("/api/config"))["samples"]), 8)
        self.assertIn("Nhận diện", self.get("/").decode("utf-8"))

    def test_10_sample_route_serves_only_manifest_ids(self):
        self.assertEqual(self.get("/samples/3d328846"), self.video)
        self.assert_status(404, lambda: self.get("/samples/../models/tcn.npz"))
        self.assert_status(404, lambda: self.get("/models/tcn.npz"))

    def test_11_all_models_predict_actual_video_bytes(self):
        for model in self.state.models:
            answer = self.post("/api/predict?model=" + model, self.video)
            self.assertEqual(answer["top"][0]["label"], "không cần")
            self.assertTrue(answer["recognized"])
            self.assertEqual(answer["video_sha256"], hashlib.sha256(self.video).hexdigest())

    def test_12_crop_is_applied_before_feature_prediction(self):
        answer = self.post("/api/predict?start=1&end=3", self.video)
        self.assertGreaterEqual(answer["quality"]["sampled_frames"], 20)
        self.assertLessEqual(answer["quality"]["sampled_frames"], 25)
        self.assertLessEqual(answer["quality"]["duration_observed"], 2.01)

    def test_13_invalid_crop_rejected(self):
        self.assert_status(400, lambda: self.post("/api/predict?start=3&end=2", self.video))
        self.assert_status(400, lambda: self.post("/api/predict?start=nan", self.video))

    def test_14_long_crop_rejected(self):
        self.assert_status(400, lambda: self.post("/api/predict?end=16", self.video))

    def test_15_unknown_model_rejected(self):
        self.assert_status(400, lambda: self.post("/api/predict?model=unknown", self.video))

    def test_16_invalid_video_rejected(self):
        self.assert_status(400, lambda: self.post("/api/predict", b"not-a-video"))

    def test_17_nonperson_video_not_accepted(self):
        path = Path(self.temporary.name) / "black.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2:r=12", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)], check=True)
        answer = self.post("/api/predict", path.read_bytes())
        self.assertFalse(answer["recognized"])
        self.assertEqual(answer["top"], [])

    def test_18_cross_origin_writes_rejected(self):
        self.assert_status(403, lambda: self.post("/api/trials", b"{}", {"Origin": "https://outside.example"}))

    def test_19_trial_uses_server_prediction_and_prevents_duplicate(self):
        answer = self.post("/api/predict", self.video)
        payload = {"inference_id": answer["inference_id"], "actual_label": "không cần", "signer_code": "QA01", "session_code": "QA01_B01", "top_score": 0, "predicted_label": "ai cho", "accepted": False}
        result = self.json_post("/api/trials", payload)
        self.assertTrue(result["saved"])
        rows = list(csv.DictReader(io.StringIO(self.get("/api/trials.csv").decode("utf-8-sig"))))
        saved = next(r for r in rows if r["trial_id"] == result["trial_id"])
        self.assertEqual(saved["predicted_label"], answer["top"][0]["label"])
        self.assertAlmostEqual(float(saved["top_score"]), answer["top"][0]["score"])
        self.assertEqual(saved["accepted"], "True")
        self.assertEqual(saved["annotation_status"], "user_reported_unverified")
        self.assert_status(409, lambda: self.json_post("/api/trials", payload))

    def test_20_outside_label_is_recorded_as_false_accept(self):
        answer = self.post("/api/predict", self.video)
        self.json_post("/api/trials", {"inference_id": answer["inference_id"], "actual_label": "__OTHER__", "signer_code": "QA02", "session_code": "QA02_B01"})
        rows = list(csv.DictReader(io.StringIO(self.get("/api/trials.csv").decode("utf-8-sig"))))
        saved = next(r for r in rows if r["inference_id"] == answer["inference_id"])
        self.assertEqual(saved["outside_false_accept"], "True")

    def test_21_trial_without_prediction_or_valid_label_rejected(self):
        self.assert_status(409, lambda: self.json_post("/api/trials", {"inference_id": "absent", "actual_label": "ai cho", "signer_code": "QA01", "session_code": "QA01_B01"}))
        self.assert_status(400, lambda: self.json_post("/api/trials", {"actual_label": "unsupported", "signer_code": "QA01", "session_code": "QA01_B01"}))

    def test_22_no_raw_video_in_trial_output(self):
        self.assertEqual([p.name for p in self.state.output_dir.iterdir()], ["trials.jsonl"])

    def test_23_default_is_preselected_tcn_and_results_have_correct_denominators(self):
        result = json.loads(self.get("/api/evaluation"))
        self.assertEqual(result["default_model"], "tcn")
        self.assertEqual(result["data"]["unique_heldout_clips"], 5)
        self.assertEqual(len(result["data"]["tested_labels"]), 2)
        for model in result["models"]:
            self.assertEqual(len(model["per_seed"]), 3)
            self.assertTrue(all(seed["n_unique_clips"] == 5 for seed in model["per_seed"]))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProjectTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(), "isolated_output_directory": True, "physical_webcam_tested": False, "ffmpeg_required_for_synthetic_probe": True}
    (ROOT / "reports/test_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
