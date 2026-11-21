# TPV-UAV-AN — Nhận diện, tracking và gộp nhóm đối tượng

Phân tích ảnh/video UAV bằng hai model YOLOE. Hệ thống nhận diện đối tượng, theo dõi ID qua từng frame và gộp các đối tượng liên quan thành vùng nhóm. Video đầu ra giữ độ phân giải và FPS nguồn.

Đây là bản refactor đang phát triển. Điểm chạy hiện tại nằm trong `test/`; chưa triển khai livestream hay pipeline dịch vụ.

## Chức năng hiện có

- Detect 12 nhãn tiếng Việt; bỏ các lớp ngoài danh sách.
- Lọc box trùng cùng nhãn bằng IoU và IoS, giữ box có confidence cao hơn.
- Tracking bằng Kalman, Hungarian, optical flow và bù chuyển động camera.
- Optical flow riêng từng vật chỉ chạy trên frame không detect; frame có detect vẫn tính bù camera.
- Vẽ bbox bằng OpenCV, cache vùng chữ Unicode tạo bằng PIL.
- Server hỗ trợ TensorRT FP16, hai model chạy song song trên CUDA stream riêng và encoder H.264 NVENC bất đồng bộ.
- Đo thời gian từng bước, xuất kết quả JSONL và thống kê JSON.

## Nhãn nhận diện

| ID danh mục | Nhãn đầu ra |
|---|---|
| 1 | người |
| 2 | xe máy |
| 4 | bàn nhựa |
| 5 | ghế nhựa |
| 6 | xe đẩy bán hàng |
| 7 | biển quảng cáo |
| 8 | cột điện |
| 9 | thân cây |
| 10 | hàng hoá |
| 11 | vật liệu xây dựng — chỉ đống gạch/gạch xếp chồng |
| 13 | xô chậu |
| 14 | xe chở rác |

ID danh mục không phải chỉ số kênh đầu ra của checkpoint. `src/detection/labels.py` ánh xạ cụm từ của checkpoint sang nhãn tiếng Việt; thêm cụm từ vào đây không huấn luyện model nhận lớp mới.

Model chính `yoloe-v8s-single` cung cấp các lớp chính; model bổ sung `yoloe-sidewalk-extra-v5` cung cấp hàng hoá, đống gạch, xô chậu và xe chở rác. Hai model cùng hỗ trợ hàng hoá; kết quả được gộp trước khi tracking.

## Cấu trúc

```text
configs/
  detection.yaml          # Local: checkpoint .pt, MPS/CPU
  detection.server.yaml   # Server: TensorRT, CUDA, NVENC
  bbox_limits.yaml        # Giới hạn bbox riêng từng nhãn theo tỷ lệ ảnh
  tracking.yaml           # Vòng đời ID, motion và ngưỡng ghép box
  grouping.yaml           # Quan hệ, khoảng cách và thời gian xác nhận nhóm
src/
  core/                   # Đọc cấu hình, đo thời gian, ghi video
  detection/              # Nhãn, detector, CUDA song song và render
  tracking/               # Kalman/Hungarian và optical flow
  grouping/               # Tạo nhóm từ các track
scripts/
  export_tensorrt.py       # Tạo engine trên GPU server
  benchmark_parallel.py   # So sánh tuần tự/song song và kiểm tra kết quả
test/                    # Các lệnh chạy thử và kiểm tra quy tắc
models/                   # Checkpoint, engine và metadata
data/                    # Video và thư mục ảnh datatest
outputs/                  # Kết quả chạy, không đưa vào Git
docs/server.md           # Ghi chú môi trường server
```

## Cài đặt local

Yêu cầu Python >= 3.10; ví dụ dùng Python 3.11 và `uv`:

```bash
cd TPV-UAV-AN_final
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

Chuẩn bị hai checkpoint sau trong `models/`:

```text
models/yoloe-v8s-single.pt
models/yoloe-sidewalk-extra-v5.pt
```

Model, video và môi trường không được tải tự động từ repository. Trên Apple Silicon, dùng `device: mps` nếu PyTorch hỗ trợ; nếu không, đổi thành `device: cpu` trong `configs/detection.yaml`.

```bash
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

## Chạy thử

Các lệnh dưới đây chạy tại thư mục gốc dự án.

### Ảnh trong datatest

```bash
.venv/bin/python test/test_obj_detection.py --mode images
```

Mặc định đọc toàn bộ ảnh trực tiếp trong `data/datatest/`. Chọn thư mục khác bằng `--input data/thu_muc_anh`.

### Video local

```bash
.venv/bin/python test/test_obj_detection.py \
  --mode video \
  --config configs/detection.yaml \
  --input data/video_test03.mp4 \
  --output outputs/local_video03
```

Chế độ video tự bật tracking và grouping nếu `enabled: true` trong cấu hình tương ứng. Có thể chọn file khác bằng `--tracking-config` và `--grouping-config`.

### Video grouping trên server

```bash
cd /home/abc/yuu_workspace/TPV-UAV-AN_final

.venv/bin/python test/test_grouping.py \
  --mode video \
  --config configs/detection.server.yaml \
  --input data/video_test03.mp4 \
  --output outputs/grouping_video03
```

`test_grouping.py --mode video` dùng chung runner detect/tracking để tránh lặp code. Thêm `--max-frames 600` để giới hạn đoạn đo; `0` hoặc bỏ tham số nghĩa là đọc hết video. Mỗi lần chạy nên dùng thư mục output riêng vì file kết quả cùng tên sẽ được ghi lại.

## Cấu hình detect và FPS

Đường dẫn tương đối trong YAML detection tính từ vị trí file YAML. Đường dẫn `--input`, `--output` tương đối tính từ thư mục terminal hiện tại.

```yaml
inference:
  device: mps           # Local: mps/cpu; server: "0"
  imgsz: 960
  detect_fps: 10        # Số lượt detect mỗi giây video nguồn
  nms_iou: 0.5
  merge_iou: 0.5
  merge_ios: 0.9        # null để tắt lọc độ chứa
```

- `detect_fps: null`: detect mọi frame. Giá trị cao hơn FPS nguồn được giới hạn bằng FPS nguồn.
- `detect_fps` không phải giới hạn tốc độ xử lý thực tế. Video 30 FPS vẫn xuất 30 FPS dù máy xử lý chậm hơn thời gian thực.
- Khi bật tracking, các frame không detect dùng bbox dự đoán/optical flow. Khi tắt tracking, giữ bbox của lần detect gần nhất.
- `thresholds` đặt ngưỡng riêng cho từng nhãn, ghi đúng tên tiếng Việt.
- IoS = diện tích giao / diện tích box nhỏ hơn; lọc cùng nhãn khi IoS đạt ngưỡng. Box khác nhãn không bị quy tắc này loại.
- `output.font: null` tự tìm font Unicode. Có thể đặt đường dẫn `.ttf` nếu máy không có font phù hợp.

## Tracking

`configs/tracking.yaml`:

| Tham số | Ý nghĩa |
|---|---|
| `enabled` | Bật/tắt tracking |
| `min_hits` | Số lần detect liên tiếp để xác nhận track |
| `max_prediction_seconds` | Thời gian tối đa vẫn vẽ bbox kể từ detection cuối |
| `max_lost_seconds` | Thời gian giữ ID để ghép lại |
| `camera_motion` | Bù camera và optical flow |
| `motion_max_side` | Kích thước tối đa ảnh thu nhỏ tính flow |
| `flow_refresh_interval` | Chu kỳ tìm lại điểm đặc trưng |
| `min_iou`, `max_center_distance` | Điều kiện ghép detection với track |

Thời gian dùng timestamp video. Chuyển cảnh xóa track nhưng không tái sử dụng ID trong cùng lượt chạy. Tracker chưa có ReID; che khuất lâu hoặc các vật cùng nhãn đứng sát nhau vẫn có thể đổi ID.

## Grouping

`configs/grouping.yaml` định nghĩa năm nhóm:

| Nhóm | Quy tắc chính |
|---|---|
| Đồ dùng và hàng hoá | Xô chậu, hàng hoá, bàn ghế, xe đẩy; người đủ gần cùng cụm đủ thời gian |
| Người và xe máy đứng yên | Xe phải đứng yên sau bù camera, rồi người phải ở gần đủ thời gian |
| Xe rác và người lái ứng viên | Vùng người phù hợp phần trên bbox xe; chỉ là ứng viên hình học |
| Vật liệu xây dựng | Vùng đống gạch theo track hiện có |
| Biển treo trên cây hoặc cột ứng viên | Biển gần/giao thân cây hoặc cột điện; chưa xác minh quan hệ treo thực tế |

Bàn/quầy làm neo; ghế có khoảng ghép riêng và cụm có giới hạn độ trải để tránh nối dây chuyền. Người được chọn vào một cụm gần nhất, không dùng người làm cầu nối gộp cụm.

Các ngưỡng chính:

```yaml
chair_gap_ratio: 1.5
person_confirm_seconds: 1.0
stationary_seconds: 1.0
motion_window_seconds: 1.5
stationary_speed_ratio: 0.08
moving_speed_ratio: 0.2
moving_confirm_seconds: 1.0
relation_grace_seconds: 1.0
```

Khoảng cách là tỷ lệ kích thước ảnh, không phải mét. Trạng thái đứng yên cần detection thực và bù camera đáng tin cậy; mất bù camera ngắn được giữ lịch sử tối đa `relation_grace_seconds`, không tích luỹ bằng chứng mới trong khoảng đó. Xe chưa đủ bằng chứng đứng yên không được xuất nhóm người–xe. Việc gộp nhóm chưa kết luận sửa/rửa xe hoặc vi phạm; detection sai lớp vẫn có thể gây nhóm sai.

## TensorRT và NVENC trên server

Server hiện dùng hai engine FP16 với đầu vào cố định 960×960:

```text
models/yoloe-v8s-single.engine
models/yoloe-sidewalk-extra-v5.engine
```

Engine phải được xây trên môi trường GPU/TensorRT tương thích. `requirements.txt` chưa bao gồm TensorRT và công cụ export; môi trường server đã cài riêng các gói này.

```bash
# Khi cần chuẩn bị lại môi trường export CUDA 12
uv pip install "tensorrt-cu12>=10.8,<11" "onnx>=1.12,<2" onnxslim onnxruntime

# Export lại hai checkpoint .pt
.venv/bin/python scripts/export_tensorrt.py --imgsz 960
```

Đổi `imgsz` cần export lại engine tương ứng.

```yaml
inference:
  device: "0"
  parallel_models: true
output:
  video:
    encoder: nvenc
    preset: p1
    cq: 20
    async_encode: true
    queue_size: 2
    ffmpeg: ../.tools/nvenc/bin/ffmpeg
```

Chế độ song song chỉ hỗ trợ engine TensorRT 10+ cố định trên CUDA: mỗi model có context, buffer và stream riêng; chờ cả hai trước khi gộp kết quả. Đặt `parallel_models: false` để chạy tuần tự. Local mặc định tuần tự.

NVENC cần FFmpeg được build có `h264_nvenc` tương thích driver. Server đang dùng bản riêng `.tools/nvenc/bin/ffmpeg`; chỉ cài dependencies Python không tự tạo executable này. Worker giữ tối đa số frame trong `queue_size`, chờ khi đầy và chờ ghi hết lúc đóng; không tự bỏ frame. Đầu vào NVENC yuv420p cần kích thước chẵn. Luồng xuất hiện tại chỉ ghi hình, không giữ âm thanh.

Không ghi đè cấu hình local bằng cấu hình CUDA. Luôn truyền `--config` rõ ràng khi so sánh hiệu năng vì cấu hình local/server có thể khác `detect_fps`.

## Kết quả và thời gian

```text
outputs/<lan_chay>/
  detected.mp4          # Video có object ID và bbox nhóm
  detections.jsonl      # Một dòng JSON mỗi frame
  timing.json           # Thống kê toàn lượt
  detected.encoder.log  # Log FFmpeg khi dùng NVENC
```

Chế độ ảnh xuất `<ten_anh>.detected.jpg` thay cho video. JSONL video có:

- `frame_index`, `timestamp_seconds`, `detector_ran`, `detect_ms`.
- `detections`: kết quả detect gần nhất, có thể được giữ lại ở frame không detect.
- `tracks`: bbox từng frame, `track_id`, `position_source`, tuổi detection.
- `groups`: `group_id`, loại, danh sách `track_ids`, bbox và trạng thái nhóm.

`timing.json`:

| Trường/bước | Ý nghĩa |
|---|---|
| `model_load_seconds` | Nạp model, tách khỏi xử lý video |
| `average_detect_ms` | Toàn bộ detector, gồm lượt đầu và hai model |
| `average_detect_ms_excluding_first` | Trung bình bỏ lượt detect đầu |
| `decode`, `tracking`, `grouping`, `render`, `jsonl` | Thời gian từng bước |
| `encode_write` | Ghi trực tiếp hoặc đẩy/chờ hàng đợi khi async |
| `encoder_worker` | Thời gian ghi của worker, có thể chồng với bước khác |
| `finalize_video` | Đóng video, chờ hàng đợi và FFmpeg hoàn tất |
| `processing_fps_including_finalize` | FPS toàn pipeline, tính cả hoàn tất file |

Detector được đồng bộ CUDA/MPS khi đo. FPS detector không phải FPS toàn pipeline. Không cộng thời gian worker với các bước chính để suy ra tổng vì chúng chạy chồng nhau.

## Kiểm tra

```bash
.venv/bin/python test/test_detection_nms.py
.venv/bin/python test/test_tracking.py
.venv/bin/python test/test_grouping.py --mode unit
```

Kiểm tra tốc độ và độ khớp hai model tuần tự/song song trên server:

```bash
.venv/bin/python scripts/benchmark_parallel.py
```

Bài này dùng các ảnh PNG trong `data/datatest`, warmup và đo luân phiên hai chế độ; báo cáo tại `outputs/parallel_benchmark.json`. Đây là so sánh backend, không thay thế đánh giá bằng dữ liệu gán nhãn.

## Giới hạn hiện tại

- Chưa triển khai segmentation, luật xác nhận vi phạm, upload backend hoặc livestream trong bản refactor này.
- Chưa có classifier xác nhận rửa xe/sửa xe, ReID hoặc EMA/hysteresis cho nhãn.
- Box khác lớp có thể cùng khoanh một vật; IoS cùng nhãn cũng có thể gộp nhầm vật thật nằm trong nhau.
- Tốc độ phụ thuộc kích thước video, tần suất detect, số track và tải CPU/GPU chung của server.
