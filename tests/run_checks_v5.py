"""Run v4 regressions plus v5 provenance/summary checks using isolated outputs."""
from pathlib import Path
import concurrent.futures
import csv
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import State, TRIAL_COLUMNS, create_server
from test_project import ProjectTests


class V5SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="khl_v5_api_")
        self.state = State(output_dir=Path(self.temporary.name) / "outputs")
        self.server = create_server(0, self.state)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temporary.cleanup()

    def get(self, route):
        return urllib.request.urlopen(self.url + route, timeout=120).read()

    def post(self, route, payload):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.url + route, data=body, method="POST")
        return json.loads(urllib.request.urlopen(request, timeout=120).read())

    def fixture(self, **values):
        row = dict.fromkeys(TRIAL_COLUMNS, "")
        row.update({"trial_id": "fixture", "time_utc": "2026-11-04T00:00:00+00:00", "inference_id": "fixture", "model": "tcn", "signer_code": "QA01", "session_code": "QA01_B01", "actual_label": "ai cho", "predicted_label": "ai cho", "accepted": True, "top1_correct": True, "accepted_correct": True, "outside_false_accept": False, "top_score": .9, "hand_fraction": 1, "model_sha256": "version_a", "video_sha256": self.state.rows[0]["normalized_video_sha256"], "start": 0, "end": 0, "elapsed_seconds": 2, "annotation_status": "user_reported_unverified"})
        row.update(values)
        return row

    def write_records(self, records):
        self.state.output_dir.mkdir(exist_ok=True)
        (self.state.output_dir / "trials.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    def test_24_empty_summary_has_no_invented_zero_accuracy(self):
        result = json.loads(self.get("/api/trial-summary"))
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(result["unique_videos"], 0)
        self.assertEqual(result["signers"], 0)
        self.assertEqual(result["models"], [])
        self.assertEqual(result["training_replay_attempts"], 0)

    def test_25_summary_denominators_repeats_versions_and_reject_matrix(self):
        self.write_records([
            self.fixture(),
            self.fixture(accepted=False, accepted_correct=False),
            self.fixture(actual_label="__OTHER__", predicted_label="bực mình", top1_correct=False, accepted_correct=False, outside_false_accept=True, video_sha256="outside", model_sha256="version_b", signer_code="QA02"),
            self.fixture(actual_label="__OTHER__", predicted_label="ghen tị", accepted=False, top1_correct=False, accepted_correct=False, video_sha256="outside", model_sha256="version_b", signer_code="QA02"),
            self.fixture(model="mlp", actual_label="không cần", predicted_label="không cần", video_sha256="new", model_sha256="version_m", signer_code="QA03"),
        ])
        result = json.loads(self.get("/api/trial-summary"))
        self.assertEqual((result["attempts"], result["unique_videos"], result["signers"]), (5, 3, 3))
        self.assertEqual(result["training_replay_attempts"], 2)
        model = next(m for m in result["models"] if m["id"] == "tcn")
        self.assertEqual((model["attempts"], model["in_vocabulary_attempts"], model["outside_attempts"]), (4, 2, 2))
        self.assertEqual(model["top1_accuracy_in_vocabulary"], 1)
        self.assertEqual(model["coverage_in_vocabulary"], .5)
        self.assertEqual(model["accepted_accuracy_in_vocabulary"], 1)
        self.assertEqual(model["outside_false_accept_rate"], .5)
        self.assertEqual(model["model_versions"], [{"model_sha256": "version_a", "attempts": 2}, {"model_sha256": "version_b", "attempts": 2}])
        matrix = model["decision_confusion"]
        self.assertEqual(sum(sum(row) for row in matrix["counts"]), 4)
        ai = matrix["truth_labels"].index("ai cho")
        other = matrix["truth_labels"].index("__OTHER__")
        reject = matrix["prediction_labels"].index("__REJECT__")
        self.assertEqual(matrix["counts"][ai][matrix["prediction_labels"].index("ai cho")], 1)
        self.assertEqual(matrix["counts"][ai][reject], 1)
        self.assertEqual(matrix["counts"][other][reject], 1)
        self.assertEqual(sum(matrix["counts"][other]), 2)

    def test_26_corrupt_log_is_reported_without_silently_skipping_rows(self):
        self.state.output_dir.mkdir(exist_ok=True)
        path = self.state.output_dir / "trials.jsonl"
        malformed = ["{broken", "[]", json.dumps({"model": "tcn"}), json.dumps(self.fixture(accepted="False")), json.dumps(self.fixture(actual_label="unknown-label"))]
        for bad in malformed:
            path.write_text(json.dumps(self.fixture()) + "\n" + bad + "\n", encoding="utf-8")
            for endpoint in ["/api/trial-summary", "/api/trials.csv"]:
                with self.assertRaises(urllib.error.HTTPError) as captured:
                    self.get(endpoint)
                self.assertEqual(captured.exception.code, 400)
                self.assertIn("dòng 2", json.loads(captured.exception.read())["error"])
            self.assertIn(bad, path.read_text(encoding="utf-8"))

    def test_27_expired_inference_id_cannot_be_recorded(self):
        video = ROOT / next(r["video"] for r in self.state.rows if r["id"] == "3d328846")
        answer = self.post("/api/predict", video.read_bytes())
        self.state.cache[answer["inference_id"]]["created"] -= 1801
        with self.assertRaises(urllib.error.HTTPError) as captured:
            self.post("/api/trials", {"inference_id": answer["inference_id"], "actual_label": "không cần", "signer_code": "QA01", "session_code": "QA01_B01"})
        self.assertEqual(captured.exception.code, 409)
        self.assertFalse(self.state.output_dir.exists())

    def test_28_concurrent_duplicate_annotation_has_one_saved_record(self):
        video = ROOT / next(r["video"] for r in self.state.rows if r["id"] == "3d328846")
        answer = self.post("/api/predict", video.read_bytes())
        payload = {"inference_id": answer["inference_id"], "actual_label": "không cần", "signer_code": "QA01", "session_code": "QA01_B01"}
        def submit():
            try:
                self.post("/api/trials", payload)
                return 200
            except urllib.error.HTTPError as error:
                return error.code
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            status = list(pool.map(lambda _: submit(), range(2)))
        self.assertEqual(sorted(status), [200, 409])
        rows = list(csv.DictReader(io.StringIO(self.get("/api/trials.csv").decode("utf-8-sig"))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["inference_id"], answer["inference_id"])


if __name__ == "__main__":
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([loader.loadTestsFromTestCase(ProjectTests), loader.loadTestsFromTestCase(V5SummaryTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(), "isolated_output_directory": True, "physical_webcam_tested": False, "regression_tests": 23, "v5_summary_provenance_tests": 5, "notes": ["Synthetic annotations used only inside temporary directories.", "No QA annotations or raw uploads remain in user outputs."]}
    (ROOT / "reports/test_results_v5.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
