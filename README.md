# TPV-UAV-AN — ver3OBB

Phân tích video UAV hoặc RTMP/HLS bằng **YOLO26m OBB → tracking → grouping → hành vi → alert**. Nhánh hiện tại dùng một detector OBB; local chạy checkpoint `.pt`, server dùng TensorRT FP16 960×960. MobileNetV3 Small phân loại rửa/sửa xe trên crop người–xe.

Tài liệu chi tiết: [Kiến trúc, logic, input/output và vai trò từng file](docs/kien_truc_he_thong.md).

## 1. Môi trường

Python ≥3.10. Local có thể tạo môi trường bằng:

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install -e .
```

Server cần PyTorch CUDA, TensorRT và FFmpeg NVENC tương thích driver. Không cài đè PyTorch server tùy tiện. Engine phải export trên môi trường tương thích với GPU/TensorRT dùng inference.

Thư mục server đang dùng:

```text
/home/abc/yuu_workspace/TPV-UAV-AN_ver3OBB
```

Bản `ver2` là deployment riêng, không tự đổi theo nhánh OBB.

## 2. Model và cấu hình

| File | Vai trò |
|---|---|
| `configs/detection.yaml` | OBB `.pt` local, mặc định MPS |
| `configs/detection.server.yaml` | OBB TensorRT/CUDA/NVENC, pipeline đầy đủ |
| `configs/detection.obb.yaml` | Config riêng test OBB local |
| `configs/detection.obb.server.yaml` | Config riêng test OBB trên server |
| `configs/bbox_limits.yaml` | Giới hạn kích thước đối tượng trước tracking |
| `configs/tracking.yaml` | ID, ghép OBB, camera-motion/optical-flow |
| `configs/tracking.detection_only.yaml` | Tắt tracking trong test detection-only |
| `configs/grouping.yaml` | Nhóm, dwell, xe đứng yên, grace |
| `configs/behavior.yaml` | MobileNet, crop, confidence/margin/cache |
| `configs/alerts.yaml` | Xác nhận/chống lặp, clip và ảnh minh chứng |
| `configs/stream.yaml` | Nguồn, FPS output, reconnect hai mức, publish viz |
| `configs/backend.yaml` | Bật/tắt upload, API URL, GPS và camera ID |

Đường dẫn model/output trong detection config tính từ thư mục YAML. Các file detection độc lập: sửa `detect_fps` ở file nào phải truyền đúng `--config` file đó. `null` detect mọi frame pipeline xử lý; số dương giới hạn lượt detect/giây timeline, không phải FPS xử lý.

Hệ thống dùng 13 nhãn nghiệp vụ; bỏ lớp xe hàng rong. Alias checkpoint như Biển báo/Cây/Hàng hóa/Thiết bị máy móc/Xe rác được ánh xạ trong `src/detection/labels.py`. NMS bổ sung OBB dùng giao polygon cùng nhãn theo IoU/IoS.

Export model:

```bash
.venv/bin/python scripts/export_tensorrt.py \
  --model models/yolo26m_obb.pt --imgsz 960
```

## 3. Biến môi trường

```bash
cp .env.example .env
```

Điền các biến cần dùng:

```dotenv
TPV_UPLOAD_API_KEY=
UAV_STREAM_URL=
UAV_VIZ_URL=
```

Hai runner thật tự nạp `.env` ở gốc dự án, ưu tiên biến môi trường đã đặt. Không commit key hoặc `.env`; key trên server cần cấu hình riêng. Biến URL trống dùng fallback YAML.

Nguồn: `--input` → `UAV_STREAM_URL` → nguồn RTMP/HLS trong `stream.yaml`. `UAV_STREAM_URL` được đặt sẽ ưu tiên cả khi có `--source hls`. URL có query xác thực cần giữ kín.

## 4. Chạy pipeline thật

Các lệnh server dưới đây chạy tại thư mục `TPV-UAV-AN_ver3OBB`. Nếu đã activate `.venv`, có thể thay `.venv/bin/python` bằng `python`.

### Video → chỉ alert

```bash
.venv/bin/python run_video.py \
  --config configs/detection.server.yaml \
  --input data/video_test02.mp4 --no-upload
```

Không ghi video tổng/JSONL từng frame. Video output alert giữ độ phân giải/FPS nguồn. Có thể truyền `--max-frames 600`. Nếu gửi BE và biết thời điểm quay file, thêm `--recording-start '2026-11-23T10:00:00+07:00'`; mặc định lấy lúc chạy làm mốc.

### RTMP/HLS → alert

```bash
.venv/bin/python run_stream.py \
  --config configs/detection.obb.server.yaml --no-upload

# Nguồn HLS thay thế.
.venv/bin/python run_stream.py \
  --config configs/detection.obb.server.yaml --source hls --no-upload
```

HTTP URL dạng thư mục player được nối `index.m3u8`; media server phải thực sự cung cấp playlist đó. Dừng bằng Ctrl+C và chờ đóng writer/upload.

### Phát live đã viz

```bash
.venv/bin/python run_stream.py \
  --config configs/detection.obb.server.yaml --no-upload --publish
```

- Bật bằng `--publish` hoặc `stream.publish.enabled`; tắt bằng `--no-publish`.
- HLS player: `http://14.232.161.78:8888/live/stream_ai_obb/`.
- HLS playlist: `http://14.232.161.78:8888/live/stream_ai_obb/index.m3u8`.
- VLC RTMP: `rtmp://14.232.161.78/live/stream_ai_obb`, thêm query auth đọc nếu server yêu cầu.
- Link có hình khi pipeline có frame và media server cho phép publish/read đường này. Nguồn `/live/stream/` là video gốc, không phải luồng viz.

Publisher dùng worker riêng, queue 2, bỏ hình xem cũ nếu nghẽn; không áp dụng bỏ frame cho clip minh chứng. Mặc định publish 15 FPS, lặp hình mới nhất khi thiếu; quá 2 giây không có viz mới thì ngừng encoder/chờ. Mốc 2 giây này thuộc publisher, độc lập PAUSE/RESET của quan sát.

Publisher mặc định dùng FFmpeg bundled/libx264 ultrafast, 2 threads vì FFmpeg NVENC ghi MP4 hiện tại không có FLV. Clip alert vẫn NVENC. Muốn NVENC cho publish, đặt `publish.encoder: nvenc` và `publish.ffmpeg` tới binary hỗ trợ cả NVENC/FLV. Không tự dùng binary ghi MP4 nếu thiếu FLV.

## 5. Stream không ổn định và FPS

```yaml
pause_gap_seconds: 1.0
reset_gap_seconds: 10.0
```

| Trạng thái | Xử lý |
|---|---|
| Bình thường | Nhận frame mới nhất, tracking/grouping tích lũy quan sát thật |
| PAUSE: quá gap ngắn hoặc reconnect | Giữ lịch sử, không cộng khoảng mất nguồn; frame trở lại buộc detect để kiểm tra ghép ID |
| RESUME: ghép được ID cũ | Tiếp tục trên đồng hồ quan sát đã trừ gap; chỉ track được detection mới ghép lại làm bằng chứng ở lượt resume |
| RESET: quá gap dài, đổi cảnh hoặc không ghép lại được | Reset tracking/grouping/cache hành vi/lịch cảnh báo; model Detector và MobileNet đã nạp vẫn giữ |

Mất nguồn quá ngưỡng reset được kiểm tra cả khi reader chưa trả frame. Reset phiên do gap dài đóng clip cũ và tạo thư mục phiên mới; scene cut hoặc không ghép lại được reset lịch sử tại chỗ, clip đang mở có thể tiếp tục qua ranh giới cảnh.

`timeout_ms` là thời gian chờ mở/đọc nguồn; `reconnect_seconds` là khoảng chờ sau lỗi trước thử lại. Không dùng hai tham số này để thay ngưỡng PAUSE/RESET. Code ghi `[PAUSE]`, `[RESUME]`, `[RESET]` kèm lý do.

Các FPS khác nhau:

| Giá trị | Ý nghĩa |
|---|---|
| FPS nguồn khai báo | Metadata OpenCV đọc từ nguồn; có thể không rõ hoặc không chính xác |
| FPS nhận (5s) | Đếm frame thực nhận trước khi queue bỏ cũ; cửa sổ 5 giây, về 0 sau >2 giây không nhận frame |
| FPS xử lý | Frame được AI pipeline xử lý/wall time, gồm chờ nguồn |
| `output_fps` | FPS cố định của clip stream; file video dùng FPS file |
| `publish.fps` | Nhịp encoder luồng xem trực tiếp |
| FPS detector | Nghịch đảo thời gian riêng detector, không phải FPS toàn pipeline |

Reader chỉ giữ frame mới nhất; nguồn nhanh hơn output sẽ bỏ slot dư, nguồn/AI chậm hơn thì clip lặp frame viz. Frame lặp **không** tạo event hoặc cộng thời gian minh chứng. Tracking/grouping/behavior dùng đồng hồ quan sát; clip và detected_at vẫn dùng timeline media. Không dùng timestamp capture/PTS camera: mạng trả frame theo đợt vẫn có thể ảnh hưởng phép đo.

## 6. Nhóm và hành vi

Nhóm đồ dùng gồm xô, hàng hoá, bàn, ghế, xe đẩy và người đủ quan hệ. Một ghế, bàn hoặc xô đơn lẻ không tạo hành vi kinh doanh. Biển có ≥50% diện tích trong bao lồi đồ dùng được gộp, không tạo nhóm biển riêng.

Nhóm khác: người–xe máy đứng yên, người–máy móc, xe rác với/không có người, vật liệu và biển/cây/cột. Xe đứng yên dùng tâm OBB đã bù camera, ngưỡng tốc độ 0.08/0.2, xác nhận dừng 1 giây và chuyển động cao 1 giây để mất trạng thái. Tham khảo giá trị thực trong YAML.

Text hiện tại: `Kinh doanh/Bán hàng`, `Đổ nước`, `Sửa chữa máy móc`, `Tập kết rác`, `Tập kết vật liệu xây dựng`, `Đặt biển báo`. Các rule thành phần chưa kiểm chứng vỉa hè hoặc quy định pháp lý.

MobileNet chỉ chạy cho nhóm người–xe confirmed/đứng yên. Crop AABB padding 0.1, RGB letterbox 224, normalize ImageNet; không xoay crop theo OBB. Khi đủ confidence YAML (mốc 0.6), margin 0.2 và không phải other_activity:

- Confidence **>0.85**: `Rửa xe` hoặc `Sửa xe`.
- Confidence ≤0.85 nhưng đủ ngưỡng: `Rửa xe / Sửa xe`.

Model/classifier được giữ qua reconnect; cache kết quả reset khi reset lịch sử. ID nhóm/hành vi giữ khi còn liên hệ thành viên, không có ReID dài hạn.

Viz: bbox/chữ **vàng**, font **30**, text trên bbox. Runner thật chỉ vẽ nhóm có hành vi (`B{id}`), không vẽ mọi object detect được. Không có hành vi thì luồng viz có thể trông như nguồn gốc.

## 7. Alert, minh chứng và BE

Clip cố định từ 2 giây trước đến 8 giây sau event mở cửa sổ; event khác trong khoảng được thêm nhưng không kéo dài clip. Giữ nguyên độ phân giải, bỏ audio. Chống lặp theo nhóm + tên hành vi, confirmation 0.5 giây, rearm 2 giây theo đồng hồ bằng chứng.

Ảnh minh chứng là **nguyên frame đã viz**, mỗi nhóm một ảnh dùng chung nhiều hành vi. Chỉ xét hành vi có tổng thời gian quan sát ≥1.5 giây; chọn giữa đoạn thành viên cùng tồn tại liên tục lâu nhất, bằng nhau ưu tiên nhiều thành viên. Không tính frame lặp là bằng chứng.

Có lưu clip: chỉ giữ metadata theo frame, đóng clip rồi giải mã clip ngắn để nén ảnh đã chọn. Ảnh chịu nén H.264 từ clip. Tắt clip trong test: vẫn nén JPEG các frame quan sát thật có hành vi để xuất ảnh độc lập.

Bật upload: điền GPS thực trong `backend.yaml`, đặt key trong môi trường/.env và thay `--no-upload` bằng `--upload`. Client dùng `/initiate` → PUT ảnh → `/part-urls` → PUT video part → `/complete`. Không tạo danh tính hoặc confidence giả.

Upload tắt giữ alert tại workspace. Upload bật vẫn dùng file tạm, chỉ xóa folder khi BE xác nhận đúng ID, `success=true`, `status=ready`; lỗi giữ clip/ảnh/`upload.json`. Chưa có tự reconcile/replay/abort phiên chưa rõ kết quả; không tự tạo lại phiên upload có thể trùng.

```text
outputs/alerts/<thời gian>/
  summary.json
  camera_<frame>/alerts/alert_000001/   # Stream; video không có cấp camera
    clip.mp4
    clip.encoder.log                   # NVENC
    group_000001.jpg                   # Nếu đủ ngưỡng minh chứng
    metadata.json
    upload.json                        # Nếu đã thử upload/lỗi
```

## 8. Test ảnh/video

```bash
# Chỉ detection OBB, không tracking/grouping.
.venv/bin/python test/test_obb_detection.py --mode video \
  --config configs/detection.obb.server.yaml \
  --input data/video_test02.mp4 --output outputs/obb_detect

# Toàn pipeline alert; chỉ lưu clip alert, không video tổng.
.venv/bin/python test/test_alert.py --mode video \
  --config configs/detection.obb.server.yaml \
  --input data/video_test04.mp4 --output outputs/alert_test \
  --no-save-video --save-alert-video
```

`--save-video` lưu thêm `detected.mp4`; `--no-save-alert-video` tắt clip alert độc lập. Runner test ghi `detections.jsonl`, `timing.json`, `alerts/index.json`, không gọi BE. Dùng thư mục output mới để tránh lẫn kết quả.

Kiểm tra không mạng/GPU:

```bash
.venv/bin/python test/test_tracking.py --mode unit
.venv/bin/python test/test_grouping.py --mode unit
.venv/bin/python test/test_behavior.py --mode unit
.venv/bin/python test/test_alert.py --mode unit
.venv/bin/python test/test_obb_rules.py
.venv/bin/python test/test_pipeline.py
.venv/bin/python test/test_stream_reconnect.py
.venv/bin/python test/test_publish.py
```

## 9. Hiệu năng và giới hạn

Tracking OBB cache hình học một lần/đối tượng/frame, loại cặp không thể ghép trước giao OpenCV; giữ gate IoU hoặc tâm gần. Shapely vẫn dùng trong grouping. Camera flow vẫn chạy CPU.

`alerts` gộp buffer/copy/chờ queue/đóng clip/trích ảnh. `alert_export` nhỏ không có nghĩa toàn bộ cảnh báo rẻ. Clip writer không bỏ frame; queue đầy phải chờ. Một frame 4K BGR khoảng 23.7 MiB, ring buffer 2 giây ở 30 FPS khoảng 1.39 GiB chưa tính copy/queue.

Runner thật ghi stage và `alert_stages` vào summary. FPS từng phiên tính cả chờ nguồn và finalize clip, chưa gồm toàn bộ chờ upload cuối. Publisher/drop/encoder chạy nền nên không cộng thời gian worker trực tiếp vào wall time luồng chính.

Chưa có quản lý nhiều camera, HTTP job API, ReID dài hạn, pose hay phân vùng vỉa hè. Đổi độ phân giải giữa clip đang mở chưa tự reset. Dữ liệu nguồn/mạng và tải server ảnh hưởng độ chính xác/thời gian, cần so sánh trên cùng video và cấu hình.

Log stream bổ sung `FPS xử lý (5s)`: số frame thực xử lý xong trong 5 giây gần nhất / 5. Không đếm frame lặp để ghi/phát video. `FPS xử lý TB` vẫn là trung bình từ lúc bắt đầu vòng stream; khi tạm mất nguồn, cửa sổ 5 giây giảm về 0 dù trung bình tích luỹ còn cao.


### Thử giới hạn CPU threads

Bốn detection YAML có `runtime.cv_threads: 4` và `runtime.torch_threads: 2`. Cả runner thật và runner test áp dụng một lần trước nạp model/worker, in giá trị hiệu lực. Điều chỉnh ở đúng file `--config`, khởi động lại để áp dụng. OpenCV có thể thử 1/2/4/8 trên cùng video; đây là mốc thử, không đảm bảo nhiều thread hơn sẽ nhanh hơn. Không đổi CUDA/TensorRT, FFmpeg, ONNX Runtime hay rule. `timing.json.cpu_threads` (test) và `summary.json.cpu_threads` (runner thật) ghi giá trị hiệu lực. Không dùng `torch.set_num_interop_threads` để tránh yêu cầu chỉ gọi một lần và ảnh hưởng tiến trình đã chạy.
