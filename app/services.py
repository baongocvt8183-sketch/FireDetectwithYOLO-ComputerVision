"""Bó các service dùng chung cho Flask app."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import SettingsStore
from .events import DetectionLedger
from .model_service import ModelService
from .pipeline import SourceManager

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: SettingsStore
    model: ModelService
    ledger: DetectionLedger
    pipeline: SourceManager
    upload_dir: Path
    owns_upload_dir: bool = False

    def shutdown(self) -> None:
        try:
            self.pipeline.stop(timeout=3.0)
        except Exception:
            # Đây là đường thoát tiến trình: ghi log thay vì che mất lỗi.
            logger.warning("Không dừng được pipeline sạch sẽ khi thoát", exc_info=True)
        if self.owns_upload_dir:
            shutil.rmtree(self.upload_dir, ignore_errors=True)
