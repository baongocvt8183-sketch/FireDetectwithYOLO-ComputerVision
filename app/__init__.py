"""Khởi tạo Flask app cho Fire & Smoke Vision dashboard."""
from __future__ import annotations

import atexit
import logging
import os
import tempfile
from pathlib import Path

from flask import Flask

from .config import DEFAULT_WEIGHTS, MAX_UPLOAD_BYTES, SettingsStore
from .events import DetectionLedger
from .model_service import ModelService
from .pipeline import SourceManager
from .services import Services

APP_VERSION = "1.0.0"
logger = logging.getLogger(__name__)

# Không để ultralytics tự cài package trong lúc chạy (thay đổi môi trường/nguồn mạng).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def create_app(
    weights_path: str | Path | None = None,
    upload_dir: str | Path | None = None,
    warmup: bool = True,
) -> Flask:
    base_dir = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        SEND_FILE_MAX_AGE_DEFAULT=0,
        JSON_SORT_KEYS=False,
        APP_VERSION=APP_VERSION,
    )

    # Matplotlib (kéo theo ultralytics) không nên ghi cache vào HOME khi chạy dashboard.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "firevision-mpl"))

    settings = SettingsStore()
    model = ModelService(Path(weights_path or DEFAULT_WEIGHTS), settings)
    ledger = DetectionLedger()
    pipeline = SourceManager(model, settings, ledger)

    owns_upload_dir = upload_dir is None
    storage = (
        Path(tempfile.mkdtemp(prefix="firevision-uploads-"))
        if upload_dir is None
        else Path(upload_dir)
    )
    storage.mkdir(parents=True, exist_ok=True)

    services = Services(
        settings=settings,
        model=model,
        ledger=ledger,
        pipeline=pipeline,
        upload_dir=storage,
        owns_upload_dir=owns_upload_dir,
    )
    app.extensions["firevision"] = services

    from .routes import api, register_error_handlers

    app.register_blueprint(api)
    register_error_handlers(app)

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "media-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'self'",
        )
        return response

    atexit.register(services.shutdown)
    logger.info("Fire & Smoke Vision %s khởi tạo (weights: %s)", APP_VERSION, model.weights_path.name)

    if warmup:
        # Nạp + warmup model nền để frame đầu tiên không phải chờ khởi tạo torch/tracker.
        # Đây là frame rỗng, không đi qua pipeline và không hiển thị như detection.
        def _warmup() -> None:
            try:
                import numpy as np

                model.info(load=True)
                model.infer(np.zeros((480, 640, 3), dtype=np.uint8))
                logger.info("Model đã sẵn sàng (warmup xong).")
            except Exception:
                logger.warning("Warmup model thất bại; sẽ thử lại khi có nguồn.", exc_info=True)

        import threading

        threading.Thread(target=_warmup, name="firevision-warmup", daemon=True).start()

    return app
