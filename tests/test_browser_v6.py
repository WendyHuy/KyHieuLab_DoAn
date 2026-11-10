"""Real browser checks with actual inference and temporary trial outputs.

Install playwright and its Chromium for the optional browser checks. You may
provide an existing trusted Chromium via --executable or CHROMIUM_EXECUTABLE.
Synthetic camera input is explicitly marked and is not a physical webcam test.
"""
from pathlib import Path
import argparse
import csv
import io
import json
import os
import platform
import sys
import tempfile
import threading
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import State, create_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default=os.environ.get("CHROMIUM_EXECUTABLE"))
    parser.add_argument("--screenshots-dir", type=Path, help="Optional QA images; keep outside shared deliverables.")
    parser.add_argument("--require-candidate", action="store_true", help="Require the trained candidate and real outside replay checks.")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright, expect

    report = {
        "passed": False, "checks_passed": 0, "checks": [],
        "browser": {"name": "Chromium", "version": None, "automation": "Playwright", "headless": True},
        "desktop_mobile_viewports": [{"width": 1440, "height": 1100}, {"width": 390, "height": 844}],
        "real_video_inference": False, "synthetic_camera_capture_tested": False,
        "physical_webcam_tested": False, "dom_mock_only": False, "isolated_outputs": True, "candidate_required": args.require_candidate, "candidate_outside_replay_tested": False,
        "environment": {"platform": platform.system(), "python": platform.python_version()},
        "console_page_errors": [], "csp_errors": [], "validation_errors": [],
        "limitations": ["Headless Linux browser; this does not test the user's physical webcam, Windows installation or mobile device hardware.", "A prerecorded training video verifies software flow, not independent recognition accuracy.", "Camera capture uses a synthetic source (native fake device when supported, otherwise an injected canvas MediaStream); permission denial is injected explicitly."],
    }

    def checked(name, detail=None):
        item = {"name": name, "passed": True}
        if detail is not None:
            item["detail"] = detail
        report["checks"].append(item)
        print("PASS", name, flush=True)

    with tempfile.TemporaryDirectory(prefix="khl_v6_browser_") as temporary:
        temporary = Path(temporary)
        state = State(output_dir=temporary / "outputs")
        server = create_server(0, state)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}"
        page = None
        try:
            with sync_playwright() as playwright:
                options = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage", "--single-process", "--no-zygote", "--disable-gpu", "--use-fake-device-for-media-stream"]}
                if args.executable:
                    options["executable_path"] = str(Path(args.executable).resolve())
                browser = playwright.chromium.launch(**options)
                report["browser"]["version"] = browser.version
                context = browser.new_context(viewport={"width": 1440, "height": 1100}, permissions=["camera"], accept_downloads=True)
                page = context.new_page()
                page.set_default_timeout(30000)
                page.on("pageerror", lambda error: report["console_page_errors"].append(str(error)))
                page.on("console", lambda message: report["csp_errors"].append(message.text) if message.type == "error" and ("Content Security Policy" in message.text or "Refused to" in message.text) else None)
                page.on("console", lambda message: report["validation_errors"].append(message.text) if "Pattern attribute value" in message.text else None)
                page.goto(url, wait_until="networkidle")
                expect(page.locator("#label-list > *")).to_have_count(5)
                expect(page.locator("#tab-recognize")).to_have_attribute("aria-selected", "true")
                expect(page.locator("#panel-recognize")).to_be_visible()
                checked("Desktop page loads five labels and the recognition workspace")
                with urllib.request.urlopen(url + "/api/config") as response:
                    api_config = json.load(response)
                option_values = page.locator("#model option").evaluate_all("options => options.map(option=>option.value)")
                assert set(option_values) == set(api_config["models"]) == set(state.models)
                assert page.locator("#model").input_value() == api_config["default_model"] == state.default_model
                assert len(api_config["labels"]) == 5 and "__OTHER__" not in api_config["labels"]
                if args.require_candidate:
                    assert "tcn_reject" in option_values
                checked("Model selector reflects loaded models and the hash-gated default with five public labels", {"models": option_values, "default": state.default_model})
                page.locator("#model").select_option("tcn")

                if args.screenshots_dir:
                    args.screenshots_dir.mkdir(parents=True, exist_ok=True)
                    page.evaluate("window.scrollTo(0,0)")
                    page.wait_for_function("() => window.scrollY === 0")
                    page.screenshot(path=str(args.screenshots_dir / "desktop-initial.png"), full_page=True, animations="disabled")

                for tab in ("experiment", "evaluation", "guide", "recognize"):
                    page.locator("#tab-" + tab).click()
                    expect(page.locator("#tab-" + tab)).to_have_attribute("aria-selected", "true")
                    expect(page.locator("#panel-" + tab)).to_be_visible()
                    for inactive in ("recognize", "experiment", "evaluation", "guide"):
                        if inactive != tab:
                            expect(page.locator("#panel-" + inactive)).not_to_be_visible()
                checked("All four tabs expose the matching labelled panel")

                page.locator("#tab-recognize").focus()
                page.keyboard.press("ArrowRight")
                expect(page.locator("#tab-experiment")).to_have_attribute("aria-selected", "true")
                page.keyboard.press("Home")
                expect(page.locator("#tab-recognize")).to_have_attribute("aria-selected", "true")
                checked("Tab list supports ArrowRight and Home keyboard navigation")

                page.locator("#examples").select_option("3d328846")
                expect(page.locator("#predict")).to_be_enabled()
                page.wait_for_function("() => document.getElementById('video').readyState >= 2")
                video_info = page.locator("#video").evaluate("video => ({duration:video.duration,width:video.videoWidth,height:video.videoHeight})")
                assert 0 < video_info["duration"] < 15 and video_info["width"] > 0
                checked("MP4 sample preview decodes in the real browser", video_info)

                page.locator("#start").fill("3")
                page.locator("#end").fill("2")
                predict_calls = []
                page.on("request", lambda request: predict_calls.append(request.url) if "/api/predict" in request.url else None)
                before = len(predict_calls)
                page.locator("#predict").click()
                expect(page.locator("#status")).to_contain_text("Mốc")
                assert len(predict_calls) == before
                checked("Invalid crop is explained before a video request is sent")
                page.locator("#start").fill("0")
                page.locator("#end").fill("0")

                with page.expect_response(lambda response: "/api/predict?" in response.url and response.request.method == "POST") as prediction_response:
                    page.locator("#predict").click()
                actual = prediction_response.value.json()
                assert prediction_response.value.status == 200 and actual["top"][0]["label"] == "không cần" and actual["recognized"]
                expect(page.locator("#result")).to_contain_text("không cần")
                report["real_video_inference"] = True
                checked("Actual sample bytes are processed by MediaPipe and the TCN; result is shown")
                if args.screenshots_dir:
                    args.screenshots_dir.mkdir(parents=True, exist_ok=True)
                    page.evaluate("window.scrollTo(0,0)")
                    page.wait_for_function("() => window.scrollY === 0")
                    page.screenshot(path=str(args.screenshots_dir / "desktop-recognition.png"), full_page=True, animations="disabled")


                page.locator("#go-experiment").click()
                expect(page.locator("#panel-experiment")).to_be_visible()
                page.locator("#signer").fill("bad+code")
                assert not page.locator("#signer").evaluate("input => input.checkValidity()")
                page.locator("#session").fill("bad+session")
                assert not page.locator("#session").evaluate("input => input.checkValidity()")
                checked("Native HTML validation rejects invalid signer and session codes")
                page.locator("#signer").fill("QA_BROWSER")
                page.locator("#session").fill("QA_BROWSER_B01")
                page.locator("#actual-label").select_option("không cần")
                page.locator("#save-trial").click()
                expect(page.locator("#trial-status")).to_contain_text("Đã ghi")
                expect(page.locator("#save-trial")).to_be_disabled()
                with urllib.request.urlopen(url + "/api/trial-summary") as response:
                    summary = json.load(response)
                assert summary["attempts"] == 1 and summary["unique_videos"] == 1 and summary["training_replay_attempts"] == 1
                checked("A prediction-backed annotation is saved once and its training replay is identified")

                with page.expect_download() as download_info:
                    page.locator("#panel-experiment a[href='/api/trials.csv']").first.click()
                destination = temporary / "download.csv"
                download_info.value.save_as(destination)
                rows = list(csv.DictReader(io.StringIO(destination.read_text(encoding="utf-8-sig"))))
                assert len(rows) == 1 and rows[0]["inference_id"] == actual["inference_id"] and rows[0]["predicted_label"] == "không cần"
                checked("CSV download preserves the server prediction and Unicode label")

                page.locator("#tab-recognize").click()
                page.locator("#model").select_option("mlp")
                page.locator("#tab-experiment").click()
                expect(page.locator("#save-trial")).to_be_disabled()
                checked("Changing the model invalidates the previous result for annotation")

                page.locator("#tab-recognize").click()
                sample = ROOT / next(row["video"] for row in state.rows if row["id"] == "3d328846")
                page.locator("#file").set_input_files(sample)
                expect(page.locator("#predict")).to_be_enabled()
                with page.expect_response(lambda response: "/api/predict?" in response.url) as uploaded_response:
                    page.locator("#predict").click()
                assert uploaded_response.value.status == 200
                expect(page.locator("#result")).to_contain_text("không cần")
                assert next(reversed(state.cache.values()))["answer"]["top"][0]["label"] == "không cần"
                page.locator("#start").fill("1")
                page.locator("#tab-experiment").click()
                expect(page.locator("#save-trial")).to_be_disabled()
                checked("File upload uses actual video and crop changes invalidate the result")

                page.locator("#tab-recognize").click()
                page.locator("#start").fill("0")
                def stale_response(route):
                    response = route.fetch(timeout=120000)
                    page.locator("#model").evaluate("el => {el.value='tcn';el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));}")
                    route.fulfill(response=response)
                page.route("**/api/predict?*", stale_response, times=1)
                page.locator("#predict").click()
                expect(page.locator("#predict")).to_be_enabled(timeout=120000)
                page.locator("#tab-experiment").click()
                expect(page.locator("#save-trial")).to_be_disabled()
                checked("An older in-flight response cannot enable annotation after configuration changes")

                page.locator("#tab-recognize").click()
                if "tcn_reject" in state.models:
                    page.locator("#model").select_option("tcn_reject")
                    page.locator("#examples").select_option("outside:c3b59369")
                    expect(page.locator("#predict")).to_be_enabled()
                    with page.expect_response(lambda response: "/api/predict?" in response.url) as outside_response:
                        page.locator("#predict").click()
                    expect(page.locator("#go-experiment")).to_be_enabled()
                    outside_answer = next(reversed(state.cache.values()))["answer"]
                    assert outside_response.value.status == 200 and outside_answer["model"] == "tcn_reject"
                    assert outside_answer["raw_decision_class"] == "__OTHER__" and outside_answer["rejected_as_outside"] and not outside_answer["recognized"]
                    assert all(row["label"] in state.labels for row in outside_answer["top"])
                    expect(page.locator("#result-state")).to_contain_text("Ngoài")
                    assert "__OTHER__" not in page.locator("#result").inner_text() + page.locator("#top").inner_text()
                    report["candidate_outside_replay_tested"] = True
                    checked("Actual six-output candidate rejects an existing outside replay without showing its internal class token", {"source_id": "c3b59369", "training_replay_only": True, "raw_score": outside_answer["raw_decision_score"]})
                    if args.screenshots_dir:
                        page.evaluate("window.scrollTo(0,0)")
                        page.wait_for_function("() => window.scrollY === 0")
                        page.screenshot(path=str(args.screenshots_dir / "desktop-candidate-outside.png"), full_page=True, animations="disabled")
                    page.locator("#go-experiment").click()
                    page.locator("#actual-label").select_option("__OTHER__")
                    page.locator("#save-trial").click()
                    expect(page.locator("#trial-status")).to_contain_text("Đã ghi")
                    csv_rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(url + "/api/trials.csv").read().decode("utf-8-sig"))))
                    outside_trial = next(row for row in csv_rows if row["inference_id"] == outside_answer["inference_id"])
                    assert outside_trial["predicted_label"] == "__OTHER__" and outside_trial["accepted"] == "False" and outside_trial["rejected_as_outside"] == "True"
                    assert abs(float(outside_trial["top_score"]) - outside_answer["raw_decision_score"]) < 1e-7
                    with urllib.request.urlopen(url + "/api/trial-summary") as response:
                        outside_summary = json.load(response)
                    candidate_summary = next(model for model in outside_summary["models"] if model["id"] == "tcn_reject")
                    assert candidate_summary["outside_attempts"] == 1 and candidate_summary["training_replay_attempts"] == 1 and candidate_summary["outside_false_accept_rate"] == 0
                    checked("Outside replay annotation keeps the raw decision and is counted as training replay for this model")
                    page.locator("#tab-evaluation").click()
                    expect(page.locator("#rejection-v6-section")).to_be_visible()
                    expect(page.locator("#rejection-v6-models tr")).to_have_count(2)
                    with urllib.request.urlopen(url + "/api/rejection-v6") as response:
                        matched = json.load(response)
                    assert matched["scope"]["known_holdout_unique_clips"] == 5 and matched["scope"]["outside_holdout_unique_clips"] == 29
                    checked("Matched source-excluded comparison is shown separately from historical v5 full-fit probes")
                    page.locator("#tab-recognize").click()
                    page.locator("#model").select_option("tcn")
                elif args.require_candidate:
                    raise AssertionError("Candidate browser tests required but the model was not loaded.")
                page.locator("#camera").click()
                try:
                    expect(page.locator("#record")).to_be_enabled()
                    report["synthetic_camera_source"] = "chromium_fake_device"
                    report["native_camera_emulation_supported"] = True
                except AssertionError:
                    report["native_camera_emulation_supported"] = False
                    report["native_camera_emulation_error"] = page.locator("#status").inner_text()
                    if page.locator("#stop-camera").is_enabled():
                        page.locator("#stop-camera").click()
                    page.evaluate("""() => {
                      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=360;
                      const context=canvas.getContext('2d');let frame=0;
                      const timer=setInterval(()=>{context.fillStyle='#153932';context.fillRect(0,0,640,360);context.fillStyle='#dfbc76';context.fillRect((frame++*7)%600,120,40,60);},80);
                      const stream=canvas.captureStream(12);window.__qaSyntheticCamera={canvas,stream,timer};
                      Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:async()=>stream});
                    }""")
                    page.locator("#camera").click()
                    expect(page.locator("#record")).to_be_enabled()
                    report["synthetic_camera_source"] = "injected_canvas_media_stream"
                expect(page.locator("#predict")).to_be_disabled()
                page.locator("#record").click()
                expect(page.locator("#predict")).to_be_enabled(timeout=15000)
                page.wait_for_function("() => document.getElementById('video').srcObject === null")
                if report["synthetic_camera_source"] == "injected_canvas_media_stream":
                    assert page.evaluate("window.__qaSyntheticCamera.stream.getTracks().every(track=>track.readyState==='ended')")
                    page.evaluate("clearInterval(window.__qaSyntheticCamera.timer)")
                with page.expect_response(lambda response: "/api/predict?" in response.url) as camera_response:
                    page.locator("#predict").click()
                expect(page.locator("#go-experiment")).to_be_enabled()
                camera_answer = next(reversed(state.cache.values()))["answer"]
                assert camera_response.value.status == 200 and not camera_answer["recognized"]
                assert camera_answer["quality"]["sampled_frames"] >= 8
                report["synthetic_camera_capture_tested"] = True
                checked("Synthetic browser media records real WebM, stops tracks and processes a rejection", {"physical_camera": False, "source": report["synthetic_camera_source"], "sampled_frames": camera_answer["quality"]["sampled_frames"]})
                page.locator("#go-experiment").click()
                page.locator("#actual-label").select_option("__OTHER__")
                page.locator("#save-trial").click()
                expect(page.locator("#trial-status")).to_contain_text("Đã ghi")
                with urllib.request.urlopen(url + "/api/trial-summary") as response:
                    rejected_summary = json.load(response)
                model_summary = next(model for model in rejected_summary["models"] if model["id"] == camera_answer["model"])
                assert model_summary["outside_attempts"] == 1 and model_summary["outside_false_accept_rate"] == 0
                checked("Rejected no-hand capture can be annotated and counted in outside denominators")
                page.locator("#tab-recognize").click()

                if args.screenshots_dir:
                    args.screenshots_dir.mkdir(parents=True, exist_ok=True)
                    page.evaluate("window.scrollTo(0,0)")
                    page.wait_for_function("() => window.scrollY === 0")
                    page.screenshot(path=str(args.screenshots_dir / "desktop-no-hands.png"), full_page=True, animations="disabled")
                    page.locator("#tab-evaluation").click()
                    page.evaluate("window.scrollTo(0,0)")
                    page.wait_for_function("() => window.scrollY === 0")
                    page.screenshot(path=str(args.screenshots_dir / "desktop-evaluation.png"), full_page=True, animations="disabled")
                page.set_viewport_size({"width": 390, "height": 844})
                overflow = {}
                for tab in ("recognize", "experiment", "evaluation", "guide"):
                    page.locator("#tab-" + tab).click()
                    expect(page.locator("#panel-" + tab)).to_be_visible()
                    dimensions = page.evaluate("({viewport:innerWidth,scroll:document.documentElement.scrollWidth,body:document.body.scrollWidth})")
                    assert max(dimensions["scroll"], dimensions["body"]) <= dimensions["viewport"] + 1, (tab, dimensions)
                    overflow[tab] = dimensions
                    if args.screenshots_dir:
                        page.evaluate("window.scrollTo(0,0)")
                        page.wait_for_function("() => window.scrollY === 0")
                        page.screenshot(path=str(args.screenshots_dir / ("mobile-" + tab + ".png")), full_page=True, animations="disabled")
                checked("All four views fit a 390px mobile viewport without horizontal page overflow", overflow)

                page.locator("#tab-recognize").click()
                page.evaluate("Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:()=>Promise.reject(new DOMException('Permission denied by QA fixture','NotAllowedError'))});")
                page.locator("#camera").click()
                expect(page.locator("#status")).to_contain_text("camera")
                expect(page.locator("#record")).to_be_disabled()
                checked("Injected camera permission denial is explained and recording remains disabled", {"permission_failure_injected": True})

                assert not report["console_page_errors"], report["console_page_errors"]
                assert not report["csp_errors"], report["csp_errors"]
                assert not report["validation_errors"], report["validation_errors"]
                checked("Real page has no uncaught JavaScript, CSP or invalid HTML pattern errors")
                assert [p.name for p in state.output_dir.iterdir()] == ["trials.jsonl"]
                checked("Only an isolated annotation log remains; raw uploads are not retained")
                report["passed"] = True
                browser.close()
        except Exception as error:
            report["failure"] = {"type": type(error).__name__, "message": str(error)}
            traceback.print_exc()
            if page is not None and args.screenshots_dir:
                try:
                    args.screenshots_dir.mkdir(parents=True, exist_ok=True)
                    page.evaluate("window.scrollTo(0,0)")
                    page.wait_for_function("() => window.scrollY === 0")
                    page.screenshot(path=str(args.screenshots_dir / "failure.png"), full_page=True, animations="disabled")
                except Exception:
                    pass
        finally:
            server.shutdown()
            server.server_close()
    report["checks_passed"] = sum(check["passed"] for check in report["checks"])
    (ROOT / "reports/browser_checks_v6.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
