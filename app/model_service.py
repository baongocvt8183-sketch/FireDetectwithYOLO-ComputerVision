"""Nạp và chạy model YOLO duy nhất cho toàn ứng dụng.

Model được nạp lazy, giữ nguyên file weights gốc. Không có thao tác ghi,
đổi tên, train lại hay thay đổi class mapping ở bất kỳ đâu trong module này.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import SettingsStore


class ModelUnavailable(RuntimeError):
    """Model chưa sẵn sàng hoặc môi trường thiếu thư viện."""


class ModelService:
    def __init__(self, weights_path: Path, settings: SettingsStore) -> None:
        self.weights_path = Path(weights_path)
        self._settings = settings
        self._model: Any | None = None
        self._names: dict[int, str] = {}
        self._task: str | None = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    def _resolve_device(self, preference: str) -> str:
        if preference == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise ModelUnavailable(
                "Thiếu PyTorch. Hãy cài đặt theo hướng dẫn trong README."
            ) from exc

        if preference == "auto":
            if torch.cuda.is_available():
                return "0"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if preference.isdigit():
            index = int(preference)
            count = torch.cuda.device_count()
            if index >= count:
                raise ModelUnavailable(
                    f"GPU {index} không khả dụng (máy có {count} GPU)."
                )
            return preference
        return "cpu"

    def load(self) -> Any:
        """Nạp model một lần. Ném ModelUnavailable nếu không thể nạp."""
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_error:
                raise ModelUnavailable(self._load_error)
            if not self.weights_path.is_file():
                self._load_error = (
                    f"Không tìm thấy weights '{self.weights_path.name}'. "
                    "Đặt file weights vào thư mục dự án hoặc đặt biến môi trường FIREVISION_WEIGHTS."
                )
                raise ModelUnavailable(self._load_error)

            try:
                from ultralytics import YOLO
            except ImportError as exc:
                self._load_error = (
                    "Thiếu thư viện ultralytics/torch. "
                    "Chạy: .venv/bin/pip install -r requirements.txt"
                )
                raise ModelUnavailable(self._load_error) from exc

            try:
                model = YOLO(str(self.weights_path))
                raw_names = getattr(model, "names", {}) or {}
                if isinstance(raw_names, dict):
                    names = {int(key): str(value) for key, value in raw_names.items()}
                else:
                    names = {index: str(value) for index, value in enumerate(raw_names)}
                self._names = dict(sorted(names.items()))
                self._task = getattr(model, "task", None)
                self._model = model
                self._load_error = None
            except Exception as exc:  # lỗi nạp checkpoint cần được báo cho UI
                self._load_error = f"Không nạp được model: {type(exc).__name__}: {exc}"
                raise ModelUnavailable(self._load_error) from exc

        return self._model

    def info(self, load: bool = False) -> dict[str, Any]:
        if load and self._model is None:
            try:
                self.load()
            except ModelUnavailable:
                pass

        inference, _ = self._settings.snapshot()
        device: str | None = None
        device_error: str | None = None
        try:
            device = self._resolve_device(inference.device)
        except ModelUnavailable as exc:
            device_error = str(exc)

        return {
            "loaded": self.loaded,
            "weights": self.weights_path.name,
            "weights_exists": self.weights_path.is_file(),
            "task": self._task,
            "nc": len(self._names) if self._names else None,
            "names": {str(key): value for key, value in self._names.items()},
            "preferred_device": inference.device,
            "device": device,
            "device_error": device_error,
            "load_error": self._load_error,
        }

    def check_device(self, preference: str) -> str:
        """Xác thực lựa chọn thiết bị trước khi lưu cấu hình."""
        return self._resolve_device(preference)

    def reset_tracking(self) -> None:
        """Bắt đầu tracker mới khi đổi nguồn, tránh ID cũ lẫn sang nguồn mới."""
        with self._infer_lock:
            predictor = getattr(self._model, "predictor", None) if self._model else None
            if predictor is not None and hasattr(predictor, "trackers"):
                predictor.trackers = []

    def infer(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        model = self.load()
        inference, _ = self._settings.snapshot()
        device = self._resolve_device(inference.device)

        kwargs: dict[str, Any] = {
            "conf": inference.conf,
            "iou": inference.iou,
            "imgsz": inference.imgsz,
            "max_det": inference.max_det,
            "device": device,
            "verbose": False,
        }
        if inference.classes is not None:
            kwargs["classes"] = list(inference.classes)

        started = time.perf_counter()
        with self._infer_lock:
            if inference.tracking:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    **kwargs,
                )
            else:
                results = model.predict(source=frame, **kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0

        detections: list[dict[str, Any]] = []
        if not results:
            return detections, latency_ms
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections, latency_ms

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        for index in range(len(xyxy)):
            class_id = int(class_ids[index])
            track_id = int(track_ids[index]) if track_ids is not None else None
            x1, y1, x2, y2 = (float(value) for value in xyxy[index])
            detections.append(
                {
                    "class_id": class_id,
                    "name": self._names.get(class_id, str(class_id)),
                    "confidence": round(float(confidences[index]), 4),
                    "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "track_id": track_id,
                }
            )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections, latency_ms
