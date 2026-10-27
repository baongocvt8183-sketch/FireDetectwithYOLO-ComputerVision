"""Kiểm thử backend: model thật, lifecycle, API và validation.

Chạy: .venv/bin/python -m unittest discover -s tests -v
Các test cần camera thật được bỏ qua có chủ đích vì môi trường CI không có thiết bị.
"""
from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import create_app
from app.annotate import draw_detections
from app.config import SettingsStore
from app.events import DetectionLedger


class UnitTests(unittest.TestCase):
    def test_settings_validation_and_reset(self) -> None:
        store = SettingsStore()
        store.update({"inference": {"conf": 0.55, "imgsz": 640, "classes": [0, 1]}}, {0, 1})
        inference, _ = store.snapshot()
        self.assertAlmostEqual(inference.conf, 0.55)
        self.assertEqual(inference.classes, [0, 1])

        with self.assertRaises(ValueError):
            store.update({"inference": {"conf": 1.5}})
        with self.assertRaises(ValueError):
            store.update({"inference": {"imgsz": 600}})
        with self.assertRaises(ValueError):
            store.update({"inference": {"classes": [5]}}, {0, 1})
        with self.assertRaises(ValueError):
            store.update({"system": {"target_fps": 0}})

        store.reset("all")
        inference, system = store.snapshot()
        self.assertEqual(inference.classes, None)
        self.assertEqual(system.camera_index, 0)

    def test_alert_cooldown_and_analytics(self) -> None:
        ledger = DetectionLedger()
        detections = [
            {
                "class_id": 1,
                "name": "fire",
                "confidence": 0.91,
                "track_id": 4,
                "box": [1, 2, 3, 4],
            }
        ]
        first = ledger.observe(detections, "Camera 0", 30, None, 200, {1: "fire"})
        second = ledger.observe(detections, "Camera 0", 30, None, 200, {1: "fire"})
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

        history = ledger.history(10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["name"], "fire")

        analytics = ledger.analytics({1: "fire"})
        self.assertEqual(analytics["frames_processed"], 2)
        self.assertEqual(analytics["total_detections"], 2)
        self.assertEqual(analytics["alerts_total"], 1)
        self.assertAlmostEqual(analytics["average_confidence"], 0.91)

        ledger.clear()
        self.assertEqual(ledger.history(), [])
        self.assertEqual(ledger.status()["total"], 0)

    def test_annotate_preserves_shape_and_draws_box(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        detection = {
            "class_id": 1,
            "name": "fire",
            "confidence": 0.87,
            "box": [20, 20, 100, 90],
            "track_id": 7,
        }
        output = draw_detections(frame, [detection])
        self.assertEqual(output.shape, (120, 160, 3))
        self.assertNotEqual(output[20, 20].tolist(), [0, 0, 0])
        self.assertEqual(output[5, 5].tolist(), [0, 0, 0])


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="firevision-test-")
        cls.app = create_app(upload_dir=cls.tmp.name, warmup=False)
        cls.client = cls.app.test_client()
        cls.services = cls.app.extensions["firevision"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.services.shutdown()
        cls.tmp.cleanup()

    def setUp(self) -> None:
        try:
            self.services.pipeline.stop()
        except Exception:
            pass
        self.services.settings.reset("all")
        self.services.ledger.clear()

    # ------------------------------------------------------------------ API cơ bản
    def test_health_and_status(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        payload = health.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["weights"], "best.pt")

        status = self.client.get("/api/status")
        self.assertEqual(status.status_code, 200)
        body = status.get_json()
        self.assertEqual(body["state"], "idle")
        self.assertIn("model", body)
        self.assertIn("alerts", body)

    def test_patch_settings_validation(self) -> None:
        invalid = self.client.patch("/api/settings", json={"inference": {"conf": 1.4}})
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("error", invalid.get_json())

        invalid_imgsz = self.client.patch("/api/settings", json={"inference": {"imgsz": 600}})
        self.assertEqual(invalid_imgsz.status_code, 400)

        valid = self.client.patch("/api/settings", json={"inference": {"conf": 0.5}})
        self.assertEqual(valid.status_code, 200)
        self.assertAlmostEqual(valid.get_json()["inference"]["conf"], 0.5)

    def test_upload_rejects_unknown_extension(self) -> None:
        response = self.client.post(
            "/api/source/upload",
            data={"file": (io.BytesIO(b"hello"), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_upload_rejects_corrupt_image(self) -> None:
        response = self.client.post(
            "/api/source/upload",
            data={"file": (io.BytesIO(b"not-an-image"), "broken.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_history_clear_endpoint(self) -> None:
        self.services.ledger.observe(
            [{"class_id": 0, "name": "smoke", "confidence": 0.7, "track_id": None, "box": [0, 0, 1, 1]}],
            "test",
            0,
            None,
            200,
            {0: "smoke"},
        )
        self.assertEqual(len(self.client.get("/api/history").get_json()["events"]), 1)
        self.assertEqual(self.client.delete("/api/history").status_code, 200)
        self.assertEqual(len(self.client.get("/api/history").get_json()["events"]), 0)

    # ------------------------------------------------------------------ model thật
    def test_model_checkpoint_is_preserved_and_loads(self) -> None:
        info = self.services.model.info(load=True)
        self.assertTrue(info["loaded"], info["load_error"])
        self.assertEqual(info["task"], "detect")
        self.assertEqual(info["names"], {"0": "smoke", "1": "fire"})
        self.assertEqual(self.services.model.weights_path.name, "best.pt")

        self.services.settings.update({"inference": {"tracking": False, "imgsz": 320}})
        detections, latency = self.services.model.infer(
            np.zeros((240, 320, 3), dtype=np.uint8)
        )
        self.assertIsInstance(detections, list)
        self.assertGreater(latency, 0)

    def test_image_pipeline_end_to_end(self) -> None:
        frame = np.full((240, 320, 3), 70, dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)

        response = self.client.post(
            "/api/source/upload",
            data={"file": (io.BytesIO(buffer.tobytes()), "sample.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 202, response.get_json())

        deadline = time.time() + 90
        status = {}
        while time.time() < deadline:
            status = self.client.get("/api/status").get_json()
            if status["state"] == "error":
                self.fail(f"Pipeline lỗi: {status['message']}")
            if status["state"] == "running" and status["frame"]["seq"] > 0:
                break
            time.sleep(0.2)

        self.assertEqual(status.get("state"), "running", status)
        self.assertEqual(status["source"]["type"], "image")
        self.assertGreater(status["frame"]["width"], 0)
        self.assertGreater(status["frame"]["height"], 0)

        screenshot = self.client.get("/api/screenshot")
        self.assertEqual(screenshot.status_code, 200)
        self.assertEqual(screenshot.mimetype, "image/jpeg")
        self.assertTrue(screenshot.data.startswith(b"\xff\xd8"))

        stopped = self.client.post("/api/stop")
        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(self.client.get("/api/status").get_json()["state"], "idle")

    def test_video_pipeline_end_to_end(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            clip = Path(handle.name)
        try:
            writer = cv2.VideoWriter(
                str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120)
            )
            self.assertTrue(writer.isOpened(), "OpenCV không tạo được video test")
            for index in range(8):
                frame = np.full((120, 160, 3), 40 + index * 10, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            response = self.client.post(
                "/api/source/upload",
                data={"file": (io.BytesIO(clip.read_bytes()), "clip.mp4")},
                content_type="multipart/form-data",
            )
            self.assertEqual(response.status_code, 202, response.get_json())

            deadline = time.time() + 120
            status = {}
            while time.time() < deadline:
                status = self.client.get("/api/status").get_json()
                if status["state"] in {"finished", "error"}:
                    break
                time.sleep(0.2)

            self.assertEqual(status.get("state"), "finished", status)
            self.assertEqual(status["source"]["type"], "video")
            self.assertGreater(status["frame"]["width"], 0)
            self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

            stopped = self.client.post("/api/stop")
            self.assertEqual(stopped.status_code, 200)
            self.assertEqual(self.client.get("/api/status").get_json()["state"], "idle")
        finally:
            clip.unlink(missing_ok=True)

    def test_stop_clears_active_alerts(self) -> None:
        self.services.ledger.observe(
            [{"class_id": 0, "name": "smoke", "confidence": 0.7, "track_id": None, "box": [0, 0, 1, 1]}],
            "test",
            0,
            None,
            200,
            {0: "smoke"},
        )
        self.assertEqual(self.client.get("/api/status").get_json()["alerts"]["active"], ["smoke"])
        self.assertEqual(self.client.post("/api/stop").status_code, 200)
        self.assertEqual(self.client.get("/api/status").get_json()["alerts"]["active"], [])

    def test_api_404_is_json(self) -> None:
        response = self.client.get("/api/khong-ton-tai")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
