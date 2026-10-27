"""Vòng đời nguồn capture + inference duy nhất cho dashboard.

Chỉ tồn tại một worker tại một thời điểm. Mở/đóng panel UI không gọi vào
module này; camera chỉ được mở/đóng bởi start/stop đổi nguồn thực sự.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .annotate import draw_detections, encode_jpeg
from .config import SettingsStore
from .events import DetectionLedger
from .model_service import ModelService, ModelUnavailable

ACTIVE_STATES = {"starting", "running", "paused", "stopping"}


class PipelineError(RuntimeError):
    """Lỗi có thể hiển thị trực tiếp cho người dùng."""


class PipelineBusy(PipelineError):
    """Thao tác không hợp lệ với trạng thái pipeline hiện tại."""


class SourceManager:
    def __init__(
        self,
        model: ModelService,
        settings: SettingsStore,
        ledger: DetectionLedger,
    ) -> None:
        self._model = model
        self._settings = settings
        self._ledger = ledger

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0

        self._state = "idle"
        self._message = "Chưa có nguồn. Chọn webcam hoặc tải ảnh/video."
        self._source_type: str | None = None
        self._source_label: str | None = None
        self._frame_jpeg: bytes | None = None
        self._frame_seq = 0
        self._frame_size = [0, 0]
        self._detections: list[dict[str, Any]] = []
        self._fps: float | None = None
        self._latency_ms: float | None = None
        self._started_at: float | None = None
        self._last_frame_at: float | None = None
        self._temp_path: Path | None = None

    # --------------------------------------------------------------- trạng thái
    def status(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            counts = Counter(item["name"] for item in self._detections)
            return {
                "state": self._state,
                "message": self._message,
                "source": {"type": self._source_type, "label": self._source_label},
                "fps": round(self._fps, 1) if self._fps is not None else None,
                "latency_ms": self._latency_ms,
                "objects": len(self._detections),
                "counts": dict(counts),
                "detections": list(self._detections[:20]),
                "frame": {
                    "width": self._frame_size[0],
                    "height": self._frame_size[1],
                    "seq": self._frame_seq,
                    "age_ms": round((now - self._last_frame_at) * 1000, 0)
                    if self._last_frame_at
                    else None,
                },
                "started_at": datetime.fromtimestamp(self._started_at)
                .astimezone()
                .isoformat(timespec="seconds")
                if self._started_at
                else None,
                "uptime_seconds": round(now - self._started_at, 1) if self._started_at else None,
                "model": self._model.info(load=False),
                "alerts": self._ledger.status(),
            }

    def wait_frame(self, last_seq: int, timeout: float = 5.0) -> tuple[bytes | None, int]:
        with self._condition:
            if self._frame_seq == last_seq:
                self._condition.wait(timeout=timeout)
            if self._frame_seq != last_seq and self._frame_jpeg is not None:
                return self._frame_jpeg, self._frame_seq
            return None, last_seq

    def screenshot(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg

    # ------------------------------------------------------------------ điều khiển
    def start_webcam(self, index: int | None = None) -> None:
        _, system = self._settings.snapshot()
        camera_index = system.camera_index if index is None else index
        label = f"Camera {camera_index}"
        self._begin(
            target=self._worker_webcam,
            args=(camera_index,),
            source_type="webcam",
            label=label,
            message=f"Đang mở {label.lower()}…",
        )

    def start_file(self, path: Path, kind: str, display_name: str) -> None:
        if kind not in {"image", "video"}:
            raise PipelineError("Loại tệp không được hỗ trợ")
        temp_path = Path(path)
        if not temp_path.is_file():
            raise PipelineError("Không tìm thấy tệp tải lên")
        label = display_name or temp_path.name
        self._begin(
            target=self._worker_file,
            args=(temp_path, kind),
            source_type=kind,
            label=label,
            message=f"Đang xử lý {kind}: {label}",
            temp_path=temp_path,
        )

    def pause(self) -> None:
        with self._lock:
            if self._state not in {"running", "starting"}:
                raise PipelineBusy("Chỉ có thể tạm dừng khi pipeline đang chạy")
            self._pause_event.set()
            self._state = "paused"
            self._message = "Đã tạm dừng. Nguồn capture vẫn được giữ mở."
            if self._frame_seq is not None:
                self._frame_seq += 1
            self._condition.notify_all()

    def resume(self) -> None:
        with self._lock:
            if self._state != "paused":
                raise PipelineBusy("Pipeline không ở trạng thái tạm dừng")
            self._pause_event.clear()
            self._state = "running"
            self._message = "Đang nhận diện…"
            self._fps = None
            self._condition.notify_all()

    def stop(self, timeout: float = 10.0) -> None:
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._thread = None
                self._reset_locked("Đã dừng nhận diện.")
                return
            self._state = "stopping"
            self._message = "Đang dừng nguồn…"
            self._stop_event.set()
            self._condition.notify_all()

        thread.join(timeout)
        if thread.is_alive():
            raise PipelineBusy("Pipeline chưa dừng kịp. Vui lòng thử lại sau vài giây.")

        with self._lock:
            if self._thread is thread:
                self._thread = None
                self._reset_locked("Đã dừng nhận diện.")

    # ------------------------------------------------------------------ nội bộ
    def _begin(
        self,
        target: Any,
        args: tuple[Any, ...],
        source_type: str,
        label: str,
        message: str,
        temp_path: Path | None = None,
    ) -> None:
        with self._lock:
            if self._state in ACTIVE_STATES or (self._thread is not None and self._thread.is_alive()):
                raise PipelineBusy("Pipeline đang chạy. Hãy dừng nguồn hiện tại trước khi đổi nguồn.")

            self._generation += 1
            generation = self._generation
            self._stop_event = threading.Event()
            self._pause_event = threading.Event()
            self._temp_path = temp_path

            self._state = "starting"
            self._message = message
            self._source_type = source_type
            self._source_label = label
            self._frame_jpeg = None
            self._detections = []
            self._frame_size = [0, 0]
            self._fps = None
            self._latency_ms = None
            self._started_at = time.time()
            self._last_frame_at = None
            self._frame_seq += 1
            self._ledger.clear_active()
            self._condition.notify_all()

            self._model.reset_tracking()
            self._thread = threading.Thread(
                target=self._run_guarded,
                args=(generation, target, args),
                name=f"firevision-{source_type}",
                daemon=True,
            )
            self._thread.start()

    def _run_guarded(self, generation: int, target: Any, args: tuple[Any, ...]) -> None:
        try:
            target(generation, *args)
        finally:
            self._finish_worker(generation)

    def _finish_worker(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            if self._thread is threading.current_thread():
                self._thread = None
            self._pause_event.clear()
            self._cleanup_temp_locked()
            self._ledger.clear_active()
            if self._state == "stopping":
                # stop() đã đặt trạng thái này trước khi join; worker kết thúc thì
                # phải đưa về idle, nếu không UI sẽ kẹt ở "đang dừng".
                self._reset_locked("Đã dừng nhận diện.")
            elif self._state in {"starting", "running", "paused"}:
                self._state = "idle"
                self._message = "Pipeline đã dừng."
            self._fps = None
            self._condition.notify_all()

    def _set_state(self, generation: int, state: str, message: str) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._state = state
            self._message = message
            self._condition.notify_all()

    def _publish(
        self,
        generation: int,
        jpeg: bytes,
        detections: list[dict[str, Any]],
        latency_ms: float,
        shape: tuple[int, ...],
        fps: float | None,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._frame_jpeg = jpeg
            self._frame_seq += 1
            self._detections = detections
            self._frame_size = [int(shape[1]), int(shape[0])]
            self._latency_ms = round(latency_ms, 1)
            if fps is not None:
                self._fps = fps
            self._last_frame_at = time.time()
            self._condition.notify_all()

    def _record(
        self,
        generation: int,
        detections: list[dict[str, Any]],
        label: str,
    ) -> None:
        _, system = self._settings.snapshot()
        with self._lock:
            if generation != self._generation:
                return
        self._ledger.observe(
            detections=detections,
            source_label=label,
            cooldown_seconds=system.alert_cooldown,
            alert_classes=system.alert_classes,
            max_history=system.max_history,
            class_names=self._model.names,
        )

    def _reset_locked(self, message: str) -> None:
        self._state = "idle"
        self._message = message
        self._source_type = None
        self._source_label = None
        self._frame_jpeg = None
        self._detections = []
        self._frame_size = [0, 0]
        self._fps = None
        self._latency_ms = None
        self._started_at = None
        self._last_frame_at = None
        self._pause_event.clear()
        self._stop_event.set()
        self._ledger.clear_active()
        self._cleanup_temp_locked()
        self._frame_seq += 1
        self._condition.notify_all()

    def _cleanup_temp_locked(self) -> None:
        path, self._temp_path = self._temp_path, None
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Tệp tạm sẽ được dọn khi thoát ứng dụng; không chặn pipeline vì lỗi FS.
            pass

    # --------------------------------------------------------------- workers
    def _worker_webcam(self, generation: int, camera_index: int) -> None:
        _, system = self._settings.snapshot()
        capture: cv2.VideoCapture | None = None
        label = f"Camera {camera_index}"
        try:
            capture = cv2.VideoCapture(camera_index)
            if not capture.isOpened():
                raise PipelineError(
                    f"Không mở được {label.lower()}. Kiểm tra thiết bị hoặc ứng dụng đang chiếm dụng camera."
                )
            if system.width > 0:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, system.width)
            if system.height > 0:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, system.height)

            self._set_state(generation, "running", f"{label} đang nhận diện")
            self._process_capture(generation, capture, "webcam", label, system)
        except PipelineError as exc:
            self._set_state(generation, "error", str(exc))
        except ModelUnavailable as exc:
            self._set_state(generation, "error", str(exc))
        except cv2.error:
            self._set_state(generation, "error", "OpenCV gặp lỗi khi đọc nguồn camera.")
        finally:
            if capture is not None:
                capture.release()

    def _worker_file(self, generation: int, path: Path, kind: str) -> None:
        _, system = self._settings.snapshot()
        with self._lock:
            label = self._source_label or path.name
        capture: cv2.VideoCapture | None = None
        try:
            if kind == "image":
                frame = cv2.imread(str(path))
                if frame is None:
                    raise PipelineError("Không đọc được tệp ảnh. Định dạng có thể không được hỗ trợ.")
                if system.mirror:
                    frame = cv2.flip(frame, 1)
                detections, latency = self._model.infer(frame)
                annotated = draw_detections(frame, detections)
                jpeg = encode_jpeg(annotated, system.jpeg_quality)
                self._publish(generation, jpeg, detections, latency, annotated.shape, None)
                self._record(generation, detections, label)
                self._set_state(generation, "running", f"Đã xử lý ảnh: {label}")
                # Ảnh tĩnh: phát lại frame định kỳ để trình duyệt nhận được
                # boundary kế tiếp trong luồng MJPEG và render ảnh ra màn hình.
                # Chrome chỉ vẽ frame hiện tại khi thấy đầu frame kế tiếp.
                while not self._stop_event.wait(1.0):
                    self._publish(generation, jpeg, detections, latency, annotated.shape, None)
                return

            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise PipelineError("Không mở được video. Định dạng có thể không được hỗ trợ.")
            self._set_state(generation, "running", f"Đang phát video: {label}")
            finished = self._process_capture(generation, capture, "video", label, system)
            if finished:
                self._set_state(generation, "finished", f"Video đã phát xong: {label}")
        except PipelineError as exc:
            self._set_state(generation, "error", str(exc))
        except ModelUnavailable as exc:
            self._set_state(generation, "error", str(exc))
        except cv2.error:
            self._set_state(generation, "error", "OpenCV gặp lỗi khi đọc tệp media.")
        finally:
            if capture is not None:
                capture.release()

    def _process_capture(
        self,
        generation: int,
        capture: cv2.VideoCapture,
        source_type: str,
        label: str,
        system: Any,
    ) -> bool:
        webcam_interval = 1.0 / max(1, system.target_fps)
        if source_type == "video":
            native_fps = capture.get(cv2.CAP_PROP_FPS)
            native_interval = 1.0 / native_fps if native_fps and native_fps > 0.1 else 0.0
            interval = max(native_interval, webcam_interval) if native_interval else webcam_interval
        else:
            interval = webcam_interval

        failures = 0
        frame_times: deque[float] = deque(maxlen=40)

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                if self._state != "paused":
                    self._set_state(generation, "paused", "Đã tạm dừng. Nguồn capture vẫn được giữ mở.")
                self._stop_event.wait(0.1)
                frame_times.clear()
                continue

            loop_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                if source_type == "video":
                    return True
                failures += 1
                if failures >= 8:
                    raise PipelineError("Mất kết nối camera sau nhiều lần đọc thất bại.")
                self._stop_event.wait(0.15)
                continue

            failures = 0
            if system.mirror:
                frame = cv2.flip(frame, 1)

            detections, latency = self._model.infer(frame)
            annotated = draw_detections(frame, detections)
            jpeg = encode_jpeg(annotated, system.jpeg_quality)

            frame_times.append(time.perf_counter())
            measured_fps = None
            if len(frame_times) >= 3:
                span = frame_times[-1] - frame_times[0]
                if span > 0:
                    measured_fps = (len(frame_times) - 1) / span

            self._publish(generation, jpeg, detections, latency, annotated.shape, measured_fps)
            self._record(generation, detections, label)

            elapsed = time.perf_counter() - loop_started
            remaining = interval - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

        return False
