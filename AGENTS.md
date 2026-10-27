# AI Vision Dashboard — Coding Agent Instructions

> **Vai trò:** Senior Full-Stack Engineer, Computer Vision Engineer và UI/UX Engineer.  
> **Mục tiêu:** Nâng cấp dự án YOLO hiện có thành dashboard nhận diện toàn màn hình, giao diện glassmorphism cao cấp, panel chức năng ẩn/mở bằng nút hamburger.  
> **Nguyên tắc quan trọng:** Kiểm tra dự án trước khi sửa; giữ nguyên model weights và pipeline đang hoạt động; không giả vờ rằng tính năng đã chạy nếu chưa kiểm chứng.

## 1. Mục tiêu sản phẩm

Xây dựng dashboard AI Vision theo phong cách **Fullscreen Detection View + Floating Glass Panels**:

- Mặc định, vùng camera/video nhận diện chiếm tối đa diện tích cửa sổ.
- Nút hamburger `☰` luôn có thể truy cập ở góc trên bên trái.
- Nhấn hamburger sẽ mở sidebar trượt từ trái.
- Sidebar có các mục điều hướng: Live Detection, Image/Video, Detection History, Analytics, Alerts, YOLO Settings và System Settings.
- Các cài đặt nâng cao mở trong panel phụ hoặc nhóm accordion, không chiếm màn hình thường trực.
- Đóng panel sẽ trả giao diện về chế độ nhận diện gần như toàn màn hình.
- Việc mở/đóng panel **không được khởi động lại camera, reset model hoặc dừng inference**.
- Hiển thị bounding boxes, class labels, confidence và tracking IDs (nếu pipeline hỗ trợ).
- Status overlay gọn gàng: camera status, model status, FPS, inference latency, số đối tượng.
- Thanh điều khiển nổi có Start/Pause/Stop, Screenshot và Fullscreen nếu phù hợp.

## 2. Thiết kế UI

### Phong cách
- Premium glassmorphism.
- Bố cục tối giản, ưu tiên camera làm hero element.
- Panel nổi, nền bán trong suốt, viền sáng nhẹ, bóng đổ mềm.
- Bo góc lớn, khoảng cách nhất quán.
- Chuyển động tinh tế, không gây mất tập trung.
- Tông nền lavender/gray, tím làm điểm nhấn.

### Bảng màu tham khảo
- Background: `#F0EDF7`
- Primary purple: `#7547FF`
- Light purple: `#9B7BFF`
- Primary text: `#171717`
- Secondary text: `#777777`
- Success: `#43A879`
- Warning: `#F2A93B`
- Danger: `#E85D75`

Màu sắc có thể điều chỉnh dựa trên hệ thống hiện tại, nhưng phải giữ tính nhất quán và độ tương phản đủ tốt.

### Hành vi responsive
- Desktop: camera chiếm toàn màn hình; sidebar là overlay drawer.
- Tablet: drawer có chiều rộng hợp lý, controls có thể cuộn.
- Mobile: sidebar gần toàn chiều rộng; controls dạng accordion; không gây horizontal overflow.
- Hỗ trợ `prefers-reduced-motion`.
- Nút điều khiển cần có trạng thái focus/hover/disabled và nhãn accessibility.

## 3. Model weights đã train

Người dùng cho biết đã có file model đã train (có thể là `.pt`; người dùng ghi `.bt`). **Không được giả định phần mở rộng hoặc đổi tên/xóa/ghi đè file.**

Thực hiện:
1. Kiểm tra repository và tìm các file model hiện có, ví dụ `.pt`, `.onnx`, `.engine`, `.weights` hoặc tên do dự án sử dụng.
2. Xác định đúng file weights từ cấu hình hoặc đường dẫn đang dùng.
3. Nếu file không có trong workspace, hỏi người dùng vị trí; không tự tạo model giả.
4. Không train lại, tải model khác, thay weights hoặc sửa class mapping nếu không được yêu cầu.
5. Không đưa weights vào Git; kiểm tra `.gitignore` và đề xuất bỏ qua thư mục model nếu cần.
6. Không in đường dẫn nhạy cảm hoặc nội dung bí mật ra log.
7. Giữ nguyên pipeline inference hiện có nếu đang hoạt động; chỉ refactor có lý do và kiểm thử phù hợp.

## 4. Quy trình bắt buộc trước khi sửa code

### Bước A — Khảo sát
- Đọc README, cấu hình dependency, entry points và cấu trúc thư mục.
- Xác định frontend framework (nếu có), backend, cách chạy ứng dụng.
- Tìm code khởi tạo YOLO, weights path, class names, camera capture, image/video upload, tracking, alerts.
- Xác định môi trường: OS, Python/Node version, requirements/package manager nếu thể hiện trong repository.
- Kiểm tra test/lint hiện có.
- Ghi nhận các chức năng đang hoạt động và các rủi ro.

### Bước B — Đề xuất ngắn trước khi sửa
Tóm tắt:
- Kiến trúc hiện tại.
- File dự kiến thay đổi.
- Cách bảo toàn model/pipeline.
- Các bước kiểm thử.
Không dừng lại để hỏi những câu đã có thể trả lời từ repository. Chỉ hỏi khi thiếu thông tin thực sự chặn công việc hoặc có rủi ro phá hủy dữ liệu.

### Bước C — Triển khai theo thay đổi nhỏ
- Ưu tiên tái sử dụng kiến trúc hiện có.
- Không tự ý migrate framework.
- Không xóa tính năng cũ nếu chưa hiểu tác dụng.
- Tách UI, inference, media capture và state management theo mức độ phù hợp với dự án.
- Tránh refactor diện rộng không liên quan.
- Không thêm dependency nếu API/browser/CSS hiện có đã đủ.
- Nếu buộc phải thêm dependency, giải thích lý do và cập nhật manifest/lockfile đúng cách.

### Bước D — Kiểm thử và báo cáo
- Chạy các test/lint/build có sẵn nếu môi trường cho phép.
- Kiểm tra lỗi import, type errors và đường dẫn assets.
- Kiểm tra trạng thái rỗng, lỗi camera, file không hợp lệ và backend không kết nối.
- Không tuyên bố đã test webcam thật nếu môi trường không có camera.
- Nếu không thể chạy một bài test, nói rõ lý do và đưa lệnh để người dùng tự chạy.
- Cung cấp lệnh khởi động chính xác dựa trên repository, không đoán tùy tiện.

## 5. Yêu cầu kỹ thuật cho camera và YOLO

- Không chạy inference nặng trên UI/main thread nếu có thể làm UI bị treo.
- Không tạo nhiều camera capture loop khi người dùng chuyển tab hoặc mở/đóng panel.
- Dùng một nguồn capture/inference lifecycle rõ ràng.
- Dừng và giải phóng camera khi người dùng thực sự stop/thoát, không phải khi đóng sidebar.
- Xử lý camera không tồn tại, bị ứng dụng khác chiếm dụng, mất kết nối và video lỗi.
- Khi đổi input source, giải phóng tài nguyên nguồn cũ trước khi mở nguồn mới.
- Có trạng thái `idle`, `starting`, `running`, `paused`, `stopping`, `error` nếu phù hợp.
- Đảm bảo bounding boxes khớp với hình ảnh sau resize/crop/letterboxing.
- Giữ nguyên tọa độ gốc và phép biến đổi hiển thị một cách nhất quán.
- Không giả lập detections khi inference chưa chạy.
- Không dùng số liệu analytics giả làm số liệu thật; nếu có mock data cho UI, phải ghi rõ và tách khỏi luồng production.
- Tính FPS/latency bằng dữ liệu đo thực tế; không hard-code số liệu.
- Không ghi ảnh/video hoặc dữ liệu nhận diện nhạy cảm ra đĩa nếu không có yêu cầu.
- Xác thực loại file và giới hạn kích thước upload phù hợp.
- Đảm bảo model inference và frontend không chặn nhau quá mức; chỉ gửi dữ liệu cần thiết qua API/WebSocket.

## 6. Yêu cầu UI/UX tương tác

### Sidebar
- Toggle bằng nút hamburger.
- Đóng bằng nút close, hamburger, Escape; click bên ngoài có thể đóng khi phù hợp.
- Khi mở sidebar, overlay có z-index đúng và không che mất nút đóng.
- Không làm remount component video/inference chỉ vì state menu thay đổi.
- Giữ trạng thái chọn menu khi chuyển giữa các panel.

### Detection viewport
- Chiếm `100dvh`/không gian ứng dụng còn lại, tránh thanh cuộn không cần thiết.
- Dùng `object-fit: contain` hoặc phương pháp phù hợp để không méo video.
- Overlay box phải sử dụng cùng hệ tọa độ với ảnh đã hiển thị.
- Nút hamburger và trạng thái quan trọng luôn có độ tương phản tốt trên nền sáng/tối.
- Có loading/empty/error states rõ ràng.

### Controls
- Nút Start/Stop phải liên kết đúng với backend/pipeline thật.
- Chặn thao tác không hợp lệ, ví dụ Start nhiều lần khi đang chạy.
- Khi thay đổi confidence/IoU, cập nhật inference theo cách phù hợp với pipeline hiện có.
- Không tự động reset settings khi đóng panel.
- Có xác nhận nếu thao tác có thể xóa lịch sử hoặc kết thúc phiên đang chạy.

## 7. Bảo mật và độ tin cậy

- Không hard-code API keys, credentials hoặc secrets.
- Không thêm lệnh shell nguy hiểm hoặc xóa hàng loạt.
- Không xóa model/data/user files.
- Tránh `eval`, xử lý upload thiếu kiểm tra và đường dẫn file không an toàn.
- Không tắt xác thực/CORS một cách rộng rãi chỉ để “cho chạy”.
- Giữ thông báo lỗi hữu ích nhưng không lộ thông tin nhạy cảm.
- Không thực hiện Git commit/push/deploy trừ khi người dùng yêu cầu.
- Không thay đổi môi trường toàn cục nếu có thể dùng virtual environment.

## 8. Chất lượng mã nguồn

- Tên biến/hàm rõ nghĩa; tránh code lặp.
- Tách component khi nó có trách nhiệm độc lập, nhưng không tạo abstraction quá mức.
- TypeScript: tránh `any` nếu có thể xác định kiểu.
- Python: dùng type hints ở nơi hữu ích, xử lý exception cụ thể, quản lý resource bằng context manager/finally khi phù hợp.
- Không nuốt exception bằng `except: pass`.
- Không để TODO giả vờ là chức năng đã hoàn thiện.
- Comment giải thích lý do/logic khó hiểu, không diễn giải từng dòng hiển nhiên.
- Cập nhật README khi cách chạy hoặc cấu hình thay đổi.
- Giữ diff nhỏ, dễ review.

## 9. Definition of Done

Chỉ xem nhiệm vụ hoàn tất khi:
- [ ] Sidebar mở/đóng đúng.
- [ ] Đóng panel trả camera về chế độ fullscreen.
- [ ] Mở/đóng panel không khởi động lại camera hoặc model.
- [ ] Webcam, ảnh và video hoạt động theo những gì dự án hỗ trợ.
- [ ] Bounding boxes và labels căn đúng với media.
- [ ] Start/Stop và lỗi nguồn đầu vào được xử lý.
- [ ] Model weights và class mapping ban đầu được bảo toàn.
- [ ] Không hiển thị dữ liệu nhận diện giả như dữ liệu thật.
- [ ] Layout responsive và không tràn ngang.
- [ ] Test/build khả dụng đã được chạy hoặc có lý do rõ ràng nếu không chạy được.
- [ ] Có hướng dẫn chạy và danh sách file đã thay đổi.

## 10. Format báo cáo cuối cùng

Sau khi hoàn thành, báo cáo ngắn gọn theo cấu trúc:

1. **Đã làm:** những thay đổi chính.
2. **File thay đổi:** liệt kê file và mục đích.
3. **Model:** đường dẫn/tên weights được sử dụng, không sửa weights.
4. **Kiểm thử:** lệnh đã chạy và kết quả thật.
5. **Cách chạy:** lệnh chính xác.
6. **Còn thiếu:** việc chưa kiểm chứng hoặc cần người dùng cung cấp.

Không khẳng định thành công nếu chỉ mới viết code mà chưa chạy/kiểm thử.
