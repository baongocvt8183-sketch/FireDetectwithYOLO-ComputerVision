"""Vẽ bounding box lên frame và mã hoá JPEG cho luồng MJPEG."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# BGR. Màu theo tên lớp chỉ phục vụ hiển thị, không ảnh hưởng tới model.
_FALLBACK_COLORS = [
    (255, 123, 155),  # #9B7BFF
    (121, 168, 67),   # #43A879
    (59, 169, 242),   # #F2A93B
    (117, 93, 232),   # #E85D75
    (255, 189, 71),   # #47BD9B
]


def color_for(name: str, index: int) -> tuple[int, int, int]:
    lowered = name.lower()
    if "fire" in lowered or "lửa" in lowered or "cháy" in lowered:
        return (117, 93, 232)  # danger #E85D75
    if "smoke" in lowered or "khói" in lowered:
        return (255, 123, 155)  # light purple #9B7BFF
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Vẽ trực tiếp lên frame gốc, giữ nguyên hệ toạ độ của model."""
    height, width = frame.shape[:2]
    thickness = max(1, round(min(width, height) / 400))
    font_scale = max(0.42, min(0.9, width / 1600))

    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["box"])
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        color = color_for(detection["name"], detection["class_id"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = f"{detection['name']} {detection['confidence'] * 100:.0f}%"
        if detection.get("track_id") is not None:
            label += f"  #{detection['track_id']}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_top = y1 - text_height - baseline - 8
        if label_top < 0:
            label_top = y1
        label_bottom = label_top + text_height + baseline + 8
        right = min(width - 1, x1 + text_width + 12)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, label_top), (right, label_bottom), color, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.putText(
            frame,
            label,
            (x1 + 6, label_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 82) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("Không mã hoá được frame JPEG")
    return buffer.tobytes()
