# Chạy detect trên server

Server: `abc@100.105.11.124`.
Thư mục: `/home/abc/yuu_workspace/TPV-UAV-AN_final`.
Môi trường Python riêng: `.venv`.

`configs/detection.yaml` ở local giữ cấu hình MPS. Bản server dùng
`configs/detection.server.yaml` với `device: "0"` (CUDA).
Server dùng hai engine TensorRT FP16 (960×960), NVENC preset p1/CQ20 và worker hàng đợi 2 frame. Các ngưỡng và detect_fps riêng trên server được giữ khi chuyển backend.

FFmpeg riêng: `.tools/nvenc/bin/ffmpeg`. Engine được xây trên GPU server, không dùng cho Mac.
`timing.json` có `processing_fps_including_finalize` tính cả chờ worker/FFmpeg hoàn tất. `encode_write` khi async là thời gian đẩy/chờ hàng đợi; thời gian worker có thể chồng với các bước khác.

Export lại khi đổi kích thước suy luận: `.venv/bin/python scripts/export_tensorrt.py --imgsz 960`.
Cấu hình trước chuyển được sao lưu thành `configs/*.before_tensorrt_nvenc.yaml`.

```bash
ssh abc@100.105.11.124
cd /home/abc/yuu_workspace/TPV-UAV-AN_final
.venv/bin/python test/test_obj_detection.py \
  --config configs/detection.server.yaml \
  --mode video --input data/video_test01.mp4
```

Chạy thử ngắn: thêm `--max-frames 60`.
Kết quả mặc định nằm trong `outputs/detect/video/`.
FPS video xuất ra bằng FPS nguồn; `detect_fps` quyết định tần suất gọi AI.

Đồng bộ lại từ local, giữ cấu hình mặc định riêng của server:

```bash
rsync -az --exclude .venv --exclude .git --exclude outputs \
  --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store \
  --exclude /configs/detection.yaml --exclude /configs/detection.server.yaml \
  ./ abc@100.105.11.124:/home/abc/yuu_workspace/TPV-UAV-AN_final/
```

Chạy lệnh đồng bộ tại thư mục gốc `TPV-UAV-AN_final` của local.
