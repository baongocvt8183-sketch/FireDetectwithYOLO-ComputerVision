# Fire & Smoke Vision Dashboard

Dashboard nhận diện cháy/khói theo phong cách **fullscreen detection + floating glass panels**.
Camera/video là vùng hiển thị chính; menu và cấu hình nằm trong drawer/panel nổi, mở bằng nút
hamburger `☰` ở góc trên bên trái.

> Ứng dụng dùng lại weights đã train sẵn `best.pt`. Không có bước train lại, không tải model
> thay thế, không đổi tên hay ghi đè file weights.

---

## 1. Trạng thái dự án

Trước khi dashboard được thêm vào, repository chỉ có:

- `best.pt` — weights thật (không sửa đổi).
- `fire_smoke_dataset/data.yaml`, `fire_smoke_dataset/configs/data.yaml` — cấu hình dataset.
- `fire_smoke_dataset/scripts/*.py` — các file rỗng, chưa có pipeline.
- `AGENTS.md`, `SKILL.md`, `README.md` — gói hướng dẫn cho coding agent.

Hiện tại backend + frontend dashboard đã được bổ sung. Các file script/dataset cũ vẫn giữ nguyên.

### Model

| Thuộc tính | Giá trị |
|---|---|
| File | `best.pt` (5.4 MB, không bị thay đổi) |
| Kiến trúc | YOLO26n detection (`yolo26n.yaml`) |
| Task | `detect` |
| Số lớp | 2 |
| Class names đọc từ checkpoint | `0: smoke`, `1: fire` |
| `imgsz` lúc train | 640 |

**Lưu ý class mapping:** `fire_smoke_dataset/data.yaml` ghi `0: fire`, `1: smoke`, trong khi
checkpoint `best.pt` lưu `0: smoke`, `1: fire`. Dashboard **đọc tên lớp trực tiếp từ checkpoint**
(`model.names`) nên nhãn hiển thị khớp với weights đã train. Không sửa file `data.yaml` cũng như
không suy đoán lại mapping.

---

## 2. Kiến trúc

```
run.py                     # entry point, chạy Flask server
app/
  __init__.py              # create_app, security headers, warmup model
  config.py                # settings + validation, giới hạn upload
  model_service.py         # nạp YOLO một lần, inference, tracking, model info
  pipeline.py              # lifecycle nguồn capture (webcam/image/video), FPS/latency
  annotate.py              # vẽ box/label/ID lên frame, encode JPEG
  events.py                # alert cooldown + history + analytics thật
  services.py              # bó service dùng chung
  routes.py                # REST API + MJPEG stream + screenshot + upload
templates/index.html       # giao diện dashboard
static/css/app.css         # glassmorphism, responsive, reduced-motion
static/js/app.js           # state UI, drawer/panel, polling, form, alert
tests/test_app.py          # unittest cho API, model thật, lifecycle
requirements.txt
```

- **Một** source worker tại một thời điểm. Mở/đóng drawer hay panel **không** gọi start/stop,
  không remount `<img>` stream và không nạp lại model.
- Inference chạy trong worker thread; Flask server chạy `threaded=True`.
- Frame được annotate ngay trên server nên bounding box luôn khớp với ảnh đã hiển thị, kể cả khi
  browser co giãn `object-fit: contain`.
- FPS/latency/analytics lấy từ phép đo thật trong pipeline; không có số liệu hard-code.
- Không ghi frame hay sự kiện nhận diện ra đĩa. Ảnh/video upload chỉ lưu tạm để xử lý và bị xoá
  khi dừng/đổi nguồn.

---

## 3. Cài đặt

Yêu cầu: Python 3.11+ (đã kiểm thử với Python 3.14.4), pip. Với webcam cần quyền truy cập thiết bị
video của user đang chạy server.

```bash
cd /home/bngoccuti0067/Project/ComputerVisionPJ/firesmokedetect
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# CPU-only (khuyến nghị nếu không có GPU NVIDIA đã cấu hình CUDA)
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

.venv/bin/pip install -r requirements.txt
```

Nếu dùng GPU, cài torch/torchvision theo hướng dẫn của PyTorch cho CUDA tương ứng, sau đó cài
`requirements.txt`. Chọn `device = auto` trong YOLO Settings để dùng GPU nếu có.

---

## 4. Chạy dashboard

```bash
.venv/bin/python run.py
```

Mở `http://127.0.0.1:8000`.

Tùy chọn:

```bash
.venv/bin/python run.py --host 127.0.0.1 --port 8000 --weights best.pt
```

- Mặc định chỉ lắng nghe `127.0.0.1`. **Không** dùng `--host 0.0.0.0` khi chưa có lớp xác thực
  hoặc mạng tin cậy, vì API điều khiển camera không có đăng nhập.
- `--upload-dir <dir>`: đổi thư mục tạm cho file tải lên.
- `--debug`: bật debug Flask; reloader luôn tắt để tránh hai tiến trình cùng mở camera/model.

Webcam được mở **trên máy chủ**, không phải webcam của trình duyệt. Trên Linux, nếu không mở được
camera, kiểm tra user thuộc nhóm `video` và thiết bị `/dev/video*` không bị ứng dụng khác chiếm.

---

## 5. Sử dụng

- `☰` mở drawer; chọn **Live Detection** để đóng panel và quay lại khung nhận diện toàn màn hình.
- Chọn **Image / Video** để tải ảnh/video; tệp được xử lý ngay sau khi bấm *Bắt đầu xử lý*.
- Thanh điều khiển nổi: Bật webcam, Tạm dừng/Tiếp tục, Dừng, Ảnh chụp, Toàn màn hình.
- **YOLO Settings**: confidence, IoU, `imgsz`, max detections, device, tracking, lớp inference.
  Thay đổi áp dụng cho frame kế tiếp, không reload model.
- **System Settings**: camera index, độ phân giải, FPS mục tiêu, gương ảnh, chất lượng JPEG,
  cooldown cảnh báo, số sự kiện lưu, lớp phát cảnh báo. Các thay đổi camera áp dụng từ lần
  bắt đầu nguồn tiếp theo.
- Drawer/panel đóng bằng nút close, `Escape`, click backdrop hoặc nút `☰`. Trạng thái cấu hình và
  nguồn đang chạy không bị reset khi đóng panel.
- Ảnh chụp tải về là frame đã annotate ở độ phân giải gốc.

### Giới hạn upload

| Loại | Định dạng | Tối đa |
|---|---|---|
| Ảnh | JPG, JPEG, PNG, BMP, WEBP | 25 MB |
| Video | MP4, AVI, MOV, MKV, WEBM | 300 MB |

Tệp được kiểm tra extension, dung lượng; ảnh được verify bằng Pillow, video được probe bằng OpenCV
trước khi đưa vào pipeline. Nội dung lỗi/không đọc được sẽ trả lỗi rõ ràng, không giả lập detection.

---

## 6. API

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/api/health` | Trạng thái server/weights, không nạp model |
| GET | `/api/model` | Nạp (lazy) và trả metadata model, class names |
| GET | `/api/status` | State, FPS, latency, detections, alerts hiện tại |
| GET | `/api/stream` | MJPEG stream đã annotate |
| GET | `/api/screenshot` | Tải JPEG frame hiện tại |
| POST | `/api/source/webcam` | Bắt đầu webcam (`{"index": 0}` tùy chọn) |
| POST | `/api/source/upload` | Upload `file` (ảnh/video), tự dừng nguồn cũ |
| POST | `/api/pause` / `/api/resume` / `/api/stop` | Điều khiển lifecycle |
| GET | `/api/settings` | Cấu hình hiện tại + class names |
| PATCH | `/api/settings` | Cập nhật `inference`/`system`, có validation |
| POST | `/api/settings/reset` | Reset `scope`: `inference`, `system`, `all` |
| GET | `/api/history` | Sự kiện cảnh báo (mới nhất trước) |
| DELETE | `/api/history` | Xoá sự kiện cảnh báo trong bộ nhớ |
| GET | `/api/alerts` | Cảnh báo gần đây + lớp đang xuất hiện |
| GET | `/api/analytics` | Số liệu phiên hiện tại |

Cấu hình mặc định nằm trong `app/config.py`. Có thể đổi weights bằng biến môi trường
`FIREVISION_WEIGHTS` hoặc `--weights`.

---

## 7. Kiểm thử

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Các test bao gồm:

- Validation settings và reset.
- Cooldown cảnh báo, history, analytics.
- Vẽ bounding box trên frame.
- API health/status/settings/history, lỗi JSON 404.
- Từ chối extension lạ và ảnh hỏng, dọn file tạm.
- **Nạp weights thật**: kiểm tra `task=detect`, class names `{0: smoke, 1: fire}`, chạy inference.
- **End-to-end ảnh**: upload ảnh PNG → pipeline chạy → `/api/screenshot` trả JPEG → stop về idle.
- **End-to-end video**: upload clip MP4 → phát hết frame → state `finished`, file tạm được xoá.

Bộ test tự động không mở webcam để tránh phụ thuộc thiết bị. Khi chạy trên máy có
camera, xác nhận thủ công: bật webcam, mở/đóng drawer nhiều lần, kiểm tra FPS và box vẫn cập nhật
liên tục (không restart camera).

---

## 8. Quyền riêng tư và an toàn

- Không có detection/KPI giả; dashboard chỉ hiển thị dữ liệu từ inference thật.
- Không ghi ảnh/video/sự kiện nhận diện ra đĩa. History/analytics nằm trong RAM và mất khi restart.
- File upload nằm trong thư mục tạm với tên UUID; xoá khi dừng/đổi nguồn và khi thoát.
- Không hard-code credential/API key; không mở CORS; CSP và security headers được bật.
- `best.pt` cùng các định dạng weights đã được đưa vào `.gitignore`.

---

## 9. Hạn chế đã biết

- Webcam là thiết bị của **máy chủ**; chưa hỗ trợ webcam trình duyệt từ xa.
- Chưa có xác thực người dùng; chỉ nên chạy trong mạng tin cậy/localhost.
- Settings/history chưa được lưu qua lần restart (theo chủ đích để tránh ghi dữ liệu nhận diện).
- Tracking dùng ByteTrack của Ultralytics; ID có thể thay đổi khi đổi nguồn (tracker được reset).
- Video phát không nhanh hơn FPS gốc hoặc FPS mục tiêu (lấy giá trị chậm hơn); hết video sẽ dừng
  ở trạng thái `finished` và giữ frame cuối cho ảnh chụp.
- Tốc độ phụ thuộc CPU/GPU; trên CPU 4 nhân, YOLO26n @640 có thể mất vài trăm ms/frame.

---

## 10. Gói hướng dẫn coding agent

`AGENTS.md` và `SKILL.md` vẫn được giữ ở thư mục gốc làm quy ước cho các phiên làm việc với agent.
