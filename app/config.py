"""Cấu hình và xác thực tham số cho Fire & Smoke Vision dashboard."""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = Path(
    os.environ.get("FIREVISION_WEIGHTS", str(BASE_DIR / "best.pt"))
).expanduser()

# Giới hạn upload: ảnh nhỏ, video lớn hơn nhưng vẫn có trần để tránh đầy đĩa.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 300 * 1024 * 1024
MAX_UPLOAD_BYTES = MAX_VIDEO_BYTES


@dataclass
class InferenceSettings:
    conf: float = 0.35
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 100
    device: str = "auto"
    tracking: bool = True
    # None = dùng toàn bộ lớp của model; tuyệt đối không suy đoán class mapping.
    classes: list[int] | None = None


@dataclass
class CaptureSettings:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 24
    mirror: bool = False
    jpeg_quality: int = 82
    alert_cooldown: float = 10.0
    alert_classes: list[int] | None = None
    max_history: int = 200


def _number(value: Any, name: str, low: float, high: float, *, integer: bool = False) -> float | int:
    if isinstance(value, bool):
        raise ValueError(f"{name} phải là số")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} phải là số") from exc
    if math.isnan(parsed) or math.isinf(parsed):
        raise ValueError(f"{name} không hợp lệ")
    if not low <= parsed <= high:
        raise ValueError(f"{name} phải nằm trong khoảng {low:g} – {high:g}")
    return int(parsed) if integer else parsed


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} phải là true/false")


def _class_list(value: Any, name: str, known: set[int] | None) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} phải là danh sách class id")
    try:
        ids = {int(item) for item in value}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} chứa class id không hợp lệ") from exc
    if not ids:
        raise ValueError(f"{name} phải có ít nhất một lớp")
    if known is not None and not ids.issubset(known):
        raise ValueError(f"{name} có class id không tồn tại trong model")
    return sorted(ids)


def _device(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("device phải là chuỗi (auto, cpu hoặc số GPU)")
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu"}:
        return normalized
    if normalized.isdigit() and int(normalized) <= 7:
        return normalized
    raise ValueError("device chỉ nhận auto, cpu hoặc chỉ số GPU 0–7")


class SettingsStore:
    """Cấu hình dùng chung giữa request handler và worker inference.

    Mọi thay đổi được gán nguyên khối dưới lock để worker luôn thấy một
    snapshot nhất quán, không bị nửa cũ nửa mới.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._inference = InferenceSettings()
        self._system = CaptureSettings()

    def snapshot(self) -> tuple[InferenceSettings, CaptureSettings]:
        with self._lock:
            return (
                replace(self._inference, classes=list(self._inference.classes) if self._inference.classes else None),
                replace(
                    self._system,
                    alert_classes=list(self._system.alert_classes) if self._system.alert_classes else None,
                ),
            )

    def update(self, patch: Any, class_ids: set[int] | None = None) -> tuple[InferenceSettings, CaptureSettings]:
        if not isinstance(patch, dict):
            raise ValueError("Dữ liệu cấu hình phải là object JSON")

        with self._lock:
            inference = replace(self._inference)
            system = replace(self._system)

            raw_inference = patch.get("inference", {})
            raw_system = patch.get("system", {})
            if raw_inference is not None and not isinstance(raw_inference, dict):
                raise ValueError("inference phải là object")
            if raw_system is not None and not isinstance(raw_system, dict):
                raise ValueError("system phải là object")

            for key, value in (raw_inference or {}).items():
                if key == "conf":
                    inference.conf = _number(value, "conf", 0.01, 0.99)
                elif key == "iou":
                    inference.iou = _number(value, "iou", 0.01, 0.99)
                elif key == "imgsz":
                    size = _number(value, "imgsz", 320, 1280, integer=True)
                    if size % 32:
                        raise ValueError("imgsz phải là bội số của 32")
                    inference.imgsz = int(size)
                elif key == "max_det":
                    inference.max_det = int(_number(value, "max_det", 1, 300, integer=True))
                elif key == "device":
                    inference.device = _device(value)
                elif key == "tracking":
                    inference.tracking = _boolean(value, "tracking")
                elif key == "classes":
                    inference.classes = _class_list(value, "classes", class_ids)
                else:
                    raise ValueError(f"Không hỗ trợ cài đặt inference '{key}'")

            for key, value in (raw_system or {}).items():
                if key == "camera_index":
                    system.camera_index = int(_number(value, "camera_index", 0, 16, integer=True))
                elif key == "width":
                    system.width = int(_number(value, "width", 160, 3840, integer=True))
                elif key == "height":
                    system.height = int(_number(value, "height", 120, 2160, integer=True))
                elif key == "target_fps":
                    system.target_fps = int(_number(value, "target_fps", 1, 60, integer=True))
                elif key == "mirror":
                    system.mirror = _boolean(value, "mirror")
                elif key == "jpeg_quality":
                    system.jpeg_quality = int(_number(value, "jpeg_quality", 40, 95, integer=True))
                elif key == "alert_cooldown":
                    system.alert_cooldown = float(_number(value, "alert_cooldown", 0, 600))
                elif key == "alert_classes":
                    system.alert_classes = _class_list(value, "alert_classes", class_ids)
                elif key == "max_history":
                    system.max_history = int(_number(value, "max_history", 20, 1000, integer=True))
                else:
                    raise ValueError(f"Không hỗ trợ cài đặt hệ thống '{key}'")

            self._inference = inference
            self._system = system
            return self.snapshot()

    def reset(self, scope: str = "all") -> tuple[InferenceSettings, CaptureSettings]:
        if scope not in {"inference", "system", "all"}:
            raise ValueError("scope phải là inference, system hoặc all")
        with self._lock:
            if scope in {"inference", "all"}:
                self._inference = InferenceSettings()
            if scope in {"system", "all"}:
                self._system = CaptureSettings()
            return self.snapshot()

    def serialized(self, class_names: dict[int, str]) -> dict[str, Any]:
        inference, system = self.snapshot()
        return {
            "inference": {
                "conf": inference.conf,
                "iou": inference.iou,
                "imgsz": inference.imgsz,
                "max_det": inference.max_det,
                "device": inference.device,
                "tracking": inference.tracking,
                "classes": inference.classes,
            },
            "system": {
                "camera_index": system.camera_index,
                "width": system.width,
                "height": system.height,
                "target_fps": system.target_fps,
                "mirror": system.mirror,
                "jpeg_quality": system.jpeg_quality,
                "alert_cooldown": system.alert_cooldown,
                "alert_classes": system.alert_classes,
                "max_history": system.max_history,
            },
            "class_names": {str(key): name for key, name in class_names.items()},
        }
