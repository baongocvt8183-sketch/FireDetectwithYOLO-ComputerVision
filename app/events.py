"""Nhật ký cảnh báo và số liệu đo được từ pipeline thật.

Không sinh dữ liệu mẫu: mọi sự kiện và số liệu đều đến từ các frame đã
chạy inference trong phiên hiện tại.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from datetime import datetime
from typing import Any


class DetectionLedger:
    def __init__(self, max_history: int = 200) -> None:
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=1000)
        self._max_history = max(20, min(1000, max_history))
        self._last_alert_at: dict[int, float] = {}
        self._active: Counter[str] = Counter()
        self._next_id = 1
        self._alerts_total = 0

        self._session_started = time.time()
        self._frames = 0
        self._detections = 0
        self._class_counts: Counter[int] = Counter()
        self._confidence_sum = 0.0
        self._confidence_count = 0

    # ------------------------------------------------------------------ ghi
    def observe(
        self,
        detections: list[dict[str, Any]],
        source_label: str,
        cooldown_seconds: float,
        alert_classes: list[int] | None,
        max_history: int,
        class_names: dict[int, str],
    ) -> list[dict[str, Any]]:
        """Ghi nhận một frame và trả về các cảnh báo mới vừa phát sinh."""
        now = time.time()
        new_events: list[dict[str, Any]] = []

        with self._lock:
            self._max_history = max(20, min(1000, max_history))
            self._frames += 1
            self._detections += len(detections)
            self._active = Counter(item["name"] for item in detections)

            for item in detections:
                class_id = int(item["class_id"])
                self._class_counts[class_id] += 1
                self._confidence_sum += float(item["confidence"])
                self._confidence_count += 1

            allowed = set(alert_classes) if alert_classes is not None else None
            best_by_class: dict[int, dict[str, Any]] = {}
            for item in detections:
                class_id = int(item["class_id"])
                if allowed is not None and class_id not in allowed:
                    continue
                current = best_by_class.get(class_id)
                if current is None or item["confidence"] > current["confidence"]:
                    best_by_class[class_id] = item

            for class_id, item in best_by_class.items():
                last = self._last_alert_at.get(class_id, 0.0)
                if cooldown_seconds > 0 and now - last < cooldown_seconds:
                    continue
                self._last_alert_at[class_id] = now
                event = {
                    "id": self._next_id,
                    "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "class_id": class_id,
                    "name": item["name"],
                    "confidence": float(item["confidence"]),
                    "track_id": item.get("track_id"),
                    "source": source_label,
                }
                self._next_id += 1
                self._alerts_total += 1
                self._events.append(event)
                new_events.append(event)

            while len(self._events) > self._max_history:
                self._events.popleft()

        return new_events

    # ------------------------------------------------------------------ đọc
    def status(self, limit: int = 5) -> dict[str, Any]:
        with self._lock:
            latest = list(self._events)[-limit:][::-1]
            return {
                "total": self._alerts_total,
                "active": sorted(self._active),
                "latest": latest,
            }

    def clear_active(self) -> None:
        """Không còn frame nào đang chạy thì không còn cảnh báo 'active'."""
        with self._lock:
            self._active.clear()

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._events)[-max(1, min(limit, 1000)):]
            return items[::-1]

    def analytics(self, class_names: dict[int, str]) -> dict[str, Any]:
        with self._lock:
            uptime = max(0.0, time.time() - self._session_started)
            per_class = [
                {
                    "class_id": class_id,
                    "name": class_names.get(class_id, str(class_id)),
                    "count": count,
                }
                for class_id, count in self._class_counts.most_common()
            ]
            return {
                "session_started_at": datetime.fromtimestamp(self._session_started)
                .astimezone()
                .isoformat(timespec="seconds"),
                "uptime_seconds": round(uptime, 1),
                "frames_processed": self._frames,
                "total_detections": self._detections,
                "average_confidence": round(self._confidence_sum / self._confidence_count, 4)
                if self._confidence_count
                else None,
                "alerts_total": self._alerts_total,
                "per_class": per_class,
                "detections_per_minute": round(self._detections / (uptime / 60.0), 2)
                if uptime > 0
                else 0,
            }

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._alerts_total = 0
            self._last_alert_at.clear()
