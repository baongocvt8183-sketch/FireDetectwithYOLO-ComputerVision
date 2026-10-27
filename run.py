#!/usr/bin/env python3
"""Entry point: chạy dashboard Fire & Smoke Vision."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Đảm bảo UTF-8 trên Windows console khi in tiếng Việt
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config import DEFAULT_WEIGHTS
from app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fire & Smoke Vision dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Địa chỉ lắng nghe (mặc định chỉ local)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng HTTP (mặc định 8000)")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Đường dẫn weights YOLO (mặc định best.pt ở thư mục dự án)",
    )
    parser.add_argument("--upload-dir", default=None, help="Thư mục tạm cho file tải lên")
    parser.add_argument("--debug", action="store_true", help="Bật debug Flask (không dùng reloader)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    weights = Path(args.weights).expanduser()
    if not weights.is_file():
        print(
            f"Không tìm thấy weights: {weights.name}. "
            "Đặt file vào thư mục dự án hoặc dùng --weights.",
            file=sys.stderr,
        )
        return 1

    app = create_app(weights_path=weights, upload_dir=args.upload_dir)
    print("Fire & Smoke Vision dashboard")
    print(f"  Weights : {weights.name}")
    print(f"  URL     : http://{args.host}:{args.port}")
    print("  Ctrl+C để dừng.")
    try:
        # use_reloader=False: reloader sẽ tạo tiến trình thứ hai và nạp model/camera lần nữa.
        app.run(host=args.host, port=args.port, threaded=True, debug=args.debug, use_reloader=False)
    finally:
        app.extensions["firevision"].shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
