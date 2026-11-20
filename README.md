src/main: gọi pipeline của hệ thống, có các chế độ như realtime livestream, chạy inputvideo
src/pipeline: tổng hợp các module thành một pipeline

Tracking video
---------------
Chế độ video tự đọc `configs/tracking.yaml`. Tracking chạy trên từng frame;
AI vẫn chạy theo `detect_fps`. ID chỉ xác nhận sau `min_hits` lần detect liên tiếp.
Bbox mất detection chỉ được vẽ đến `max_prediction_seconds`; ID được giữ để
ghép lại đến `max_lost_seconds`. Đổi cảnh sẽ xóa track, không tái sử dụng ID.
Optical flow và bù chuyển động camera chạy trên ảnh thu nhỏ.

Tắt tracking: đặt `enabled: false` trong YAML để dùng lại cách giữ bbox gần nhất.
Chọn cấu hình khác: `--tracking-config configs/tracking.yaml`.
JSONL có `tracks` (bbox/ID từng frame), `position_source` và tuổi detection.
`timing.json` có bước `tracking`. Nhãn tiếng Việt giữ nguyên, thêm tiền tố ID.

Kiểm tra module: `.venv/bin/python test/test_tracking.py`.
Bộ này chưa gồm ReID, ByteTrack hai ngưỡng hoặc EMA/hysteresis/dwell-time nhãn.
