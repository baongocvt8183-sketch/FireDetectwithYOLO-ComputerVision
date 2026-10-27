"""HTTP API của dashboard."""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
from flask import Blueprint, Response, current_app, jsonify, render_template, request, send_file
from PIL import Image

from .config import (
    IMAGE_EXTENSIONS,
    MAX_IMAGE_BYTES,
    MAX_VIDEO_BYTES,
    VIDEO_EXTENSIONS,
)
from .model_service import ModelUnavailable
from .pipeline import PipelineBusy, PipelineError
from .services import Services

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__)


def services() -> Services:
    return current_app.extensions["firevision"]


def _json_error(message: str, status: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status


@api.errorhandler(PipelineBusy)
def _handle_busy(exc: PipelineBusy):
    return _json_error(str(exc), 409)


@api.errorhandler(PipelineError)
def _handle_pipeline_error(exc: PipelineError):
    return _json_error(str(exc), 400)


@api.errorhandler(ModelUnavailable)
def _handle_model_unavailable(exc: ModelUnavailable):
    return _json_error(str(exc), 503)


@api.errorhandler(ValueError)
def _handle_value_error(exc: ValueError):
    return _json_error(str(exc), 400)


@api.get("/")
def index():
    response = current_app.make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-store"
    return response


@api.get("/api/health")
def health():
    svc = services()
    info = svc.model.info(load=False)
    return jsonify(
        {
            "ok": True,
            "app": "firevision",
            "version": current_app.config.get("APP_VERSION", "1.0.0"),
            "weights": info["weights"],
            "weights_exists": info["weights_exists"],
            "model_loaded": info["loaded"],
            "model_error": info["load_error"],
        }
    )


@api.get("/api/model")
def model_info():
    info = services().model.info(load=True)
    return jsonify(info)


@api.get("/api/status")
def status():
    payload = services().pipeline.status()
    payload["server_time"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return jsonify(payload)


@api.get("/api/settings")
def get_settings():
    svc = services()
    info = svc.model.info(load=True)
    names = {int(key): value for key, value in info["names"].items()}
    return jsonify(svc.settings.serialized(names))


@api.patch("/api/settings")
def update_settings():
    svc = services()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Body phải là JSON object")

    inference_patch = payload.get("inference") or {}
    system_patch = payload.get("system") or {}
    if not isinstance(inference_patch, dict):
        raise ValueError("inference phải là object")
    if not isinstance(system_patch, dict):
        raise ValueError("system phải là object")

    names: dict[int, str] = {}
    needs_names = "classes" in inference_patch or "alert_classes" in system_patch
    if needs_names or svc.model.loaded:
        info = svc.model.info(load=True)
        if needs_names and not info["loaded"]:
            raise ModelUnavailable(info["load_error"] or "Chưa nạp được model để xác thực lớp.")
        names = {int(key): value for key, value in info["names"].items()}

    requested_device = inference_patch.get("device")
    if requested_device is not None:
        svc.model.check_device(str(requested_device))

    svc.settings.update(payload, set(names) if names else None)
    fresh_names = {int(key): value for key, value in svc.model.names.items()}
    return jsonify(svc.settings.serialized(fresh_names))


@api.post("/api/settings/reset")
def reset_settings():
    svc = services()
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        raise ValueError("Body phải là JSON object")
    scope = payload.get("scope", "all")
    if scope not in {"inference", "system", "all"}:
        raise ValueError("scope phải là inference, system hoặc all")
    svc.settings.reset(scope)
    info = svc.model.info(load=False)
    names = {int(key): value for key, value in info["names"].items()}
    return jsonify(svc.settings.serialized(names))


@api.get("/api/analytics")
def analytics():
    svc = services()
    return jsonify(svc.ledger.analytics(svc.model.names))


@api.get("/api/history")
def history():
    svc = services()
    limit = request.args.get("limit", default=100, type=int)
    return jsonify({"events": svc.ledger.history(limit or 100)})


@api.delete("/api/history")
def clear_history():
    services().ledger.clear()
    return jsonify({"ok": True})


@api.get("/api/alerts")
def alerts():
    svc = services()
    limit = request.args.get("limit", default=20, type=int)
    return jsonify(svc.ledger.status(limit=limit or 20))


@api.post("/api/source/webcam")
def start_webcam():
    svc = services()
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        raise ValueError("Body phải là JSON object")
    index = payload.get("index")
    if index is not None:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 16:
            raise ValueError("index camera phải là số nguyên 0–16")
    svc.pipeline.start_webcam(index)
    return jsonify({"ok": True, "status": svc.pipeline.status()}), 202


@api.post("/api/source/upload")
def upload_source():
    svc = services()
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ValueError("Thiếu tệp tải lên")

    original = Path(upload.filename).name
    display_name = re.sub(r"[\x00-\x1f\x7f]", "", original).strip()[:120] or "tệp tải lên"
    extension = Path(original).suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        kind = "image"
    elif extension in VIDEO_EXTENSIONS:
        kind = "video"
    else:
        raise ValueError("Chỉ hỗ trợ ảnh JPG/PNG/BMP/WEBP và video MP4/AVI/MOV/MKV/WEBM")

    target = svc.upload_dir / f"{uuid.uuid4().hex}{extension}"
    upload.save(target)

    try:
        size = target.stat().st_size
        if size == 0:
            raise ValueError("Tệp tải lên rỗng")
        limit = MAX_IMAGE_BYTES if kind == "image" else MAX_VIDEO_BYTES
        if size > limit:
            raise ValueError(f"Tệp vượt quá giới hạn {limit // (1024 * 1024)} MB")

        if kind == "image":
            try:
                with Image.open(target) as probe:
                    probe.verify()
            except Exception as exc:
                # Ultralytics monkeypatch PIL.Image.open và có thể ném
                # ModuleNotFoundError (pi_heif) khi tệp không phải ảnh.
                raise ValueError("Ảnh không hợp lệ hoặc bị hỏng") from exc
        else:
            capture = cv2.VideoCapture(str(target))
            try:
                if not capture.isOpened():
                    raise ValueError("Video không hợp lệ hoặc codec không được hỗ trợ")
            finally:
                capture.release()

        if svc.pipeline.status()["state"] in {"starting", "running", "paused", "stopping"}:
            svc.pipeline.stop()

        svc.pipeline.start_file(target, kind, display_name)
        return (
            jsonify(
                {
                    "ok": True,
                    "kind": kind,
                    "name": display_name,
                    "size": size,
                    "status": svc.pipeline.status(),
                }
            ),
            202,
        )
    except Exception:
        target.unlink(missing_ok=True)
        raise


@api.post("/api/pause")
def pause():
    services().pipeline.pause()
    return jsonify({"ok": True, "status": services().pipeline.status()})


@api.post("/api/resume")
def resume():
    services().pipeline.resume()
    return jsonify({"ok": True, "status": services().pipeline.status()})


@api.post("/api/stop")
def stop():
    services().pipeline.stop()
    return jsonify({"ok": True, "status": services().pipeline.status()})


@api.get("/api/stream")
def stream():
    svc = services()

    def generate():
        last_seq = 0
        while True:
            jpeg, sequence = svc.pipeline.wait_frame(last_seq, timeout=5.0)
            if jpeg is None:
                continue
            last_seq = sequence
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

    response = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@api.get("/api/screenshot")
def screenshot():
    jpeg = services().pipeline.screenshot()
    if not jpeg:
        raise PipelineError("Chưa có frame để chụp. Hãy khởi động nguồn nhận diện trước.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return send_file(
        BytesIO(jpeg),
        mimetype="image/jpeg",
        as_attachment=True,
        download_name=f"detection-{stamp}.jpg",
        max_age=0,
    )


def register_error_handlers(app: Any) -> None:
    @app.errorhandler(413)
    def too_large(_exc):
        limits = f"ảnh tối đa {MAX_IMAGE_BYTES // (1024 * 1024)} MB, video tối đa {MAX_VIDEO_BYTES // (1024 * 1024)} MB"
        return _json_error(f"Tệp tải lên quá lớn ({limits})", 413)

    @app.errorhandler(404)
    def not_found(_exc):
        if request.path.startswith("/api/"):
            return _json_error("Không tìm thấy endpoint", 404)
        return render_template("index.html"), 404

    @app.errorhandler(Exception)
    def unhandled(exc: Exception):
        if request.path.startswith("/api/"):
            logger.exception("Lỗi xử lý API %s", request.path)
            return _json_error("Máy chủ gặp lỗi không mong muốn. Xem log máy chủ.", 500)
        logger.exception("Lỗi xử lý trang %s", request.path)
        return Response("Máy chủ gặp lỗi không mong muốn.", status=500, mimetype="text/plain")
