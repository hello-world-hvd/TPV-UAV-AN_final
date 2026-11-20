"""Chạy thử detect ảnh/video; video giữ FPS gốc và giữ bbox gần nhất giữa các lần detect."""
"""
Detect toàn bộ ảnh trong data/datatest:
    .venv/bin/python test/test_obj_detection.py --mode images
Detect video:
    .venv/bin/python test/test_obj_detection.py \
    --mode video \
    --input data/video_test01.mp4
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection.detector import Detector
from detection.render import draw_detections, load_font
from core.config import load_config


def show_progress(index, total, detect_count, started, final=False):
    """In tiến trình theo frame đã ghi, tốc độ thực tế và thời gian còn lại ước tính."""
    elapsed = max(time.perf_counter() - started, 1e-9)
    speed = index / elapsed
    if total:
        percentage = min(index / total, 1) * 100
        remaining = max(total - index, 0) / speed if speed else 0
        progress = f"{index}/{total} frame ({percentage:.1f}%)"
        eta = f"{remaining:.0f}s"
    else:
        progress, eta = f"{index} frame", "chưa rõ"
    print(f"\r{progress} | Detect: {detect_count} lượt | Xử lý: {speed:.2f} FPS"
          f" | Đã chạy: {elapsed:.1f}s | Còn lại ~{eta}",
          end="\n" if final else "", flush=True)


def main():
    root = ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["images", "video"])
    parser.add_argument("--input", type=Path, help="Thư mục ảnh hoặc đường dẫn video")
    parser.add_argument("--max-frames", type=int, default=0, help="Giới hạn frame video; 0: toàn bộ")
    parser.add_argument("--config", type=Path, default=root / "configs" / "detection.yaml")
    parser.add_argument("--output", type=Path, help="Ghi đè thư mục đầu ra trong YAML")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames phải >= 0")
    if args.input is None:
        if args.mode == "video":
            parser.error("Chế độ video cần --input đường/dẫn/video.mp4")
        args.input = root / "data" / "datatest"
    if args.mode == "images":
        if not args.input.is_dir():
            parser.error(f"Không tìm thấy thư mục ảnh: {args.input}")
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in extensions and p.is_file())
        if not images:
            parser.error(f"Thư mục không có ảnh: {args.input}")
    elif not args.input.is_file():
        parser.error(f"Không tìm thấy video: {args.input}")
    config = load_config(args.config)
    args.output = args.output or config["output"]["directory"] / args.mode
    font = load_font(config["output"]["font"])
    detector = Detector(config)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "detections.jsonl").open("w", encoding="utf-8") as records:
        if args.mode == "images":
            for path in images:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Không đọc được ảnh: {path}")
                detections = detector.detect(image)
                rendered = draw_detections(image, detections, font)
                output_path = args.output / f"{path.name}.detected.jpg"
                if not cv2.imwrite(str(output_path), rendered):
                    raise RuntimeError(f"Không ghi được ảnh: {output_path}")
                records.write(json.dumps({"source": str(path), "detections": detections}, ensure_ascii=False) + "\n")
                print(f"{path.name}: {len(detections)} đối tượng")
        else:
            capture = cv2.VideoCapture(str(args.input))
            writer = None
            try:
                if not capture.isOpened():
                    raise ValueError("Không đọc được ảnh/video đầu vào.")
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not 0 < fps < float("inf"):
                    raise ValueError("Video không có FPS hợp lệ.")
                index = 0
                frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                total = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else None
                if args.max_frames:
                    total = min(total, args.max_frames) if total else args.max_frames
                detect_count = 0
                started = time.perf_counter()
                last_report = started
                configured_fps = config["inference"]["detect_fps"]
                detect_fps = min(configured_fps, fps) if configured_fps is not None else fps
                print(f"FPS video/output: {fps:g}; FPS detect theo video nguồn: {detect_fps:g}")
                last_slot = -1
                detection_frame_index = 0
                detections = []
                while args.max_frames == 0 or index < args.max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(str(args.output / "detected.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("Không mở được bộ ghi video.")
                    # Lập lịch theo thời gian video, không theo tốc độ xử lý của máy.
                    slot = int(index * detect_fps / fps + 1e-9)
                    detector_ran = slot > last_slot
                    if detector_ran:
                        detections = detector.detect(frame)
                        detect_count += 1
                        detection_frame_index = index
                        last_slot = slot
                    # Vẽ box cũ lên frame hiện tại; lần detect rỗng sẽ xóa các box cũ.
                    writer.write(draw_detections(frame, detections, font))
                    records.write(json.dumps({"frame_index": index, "timestamp_seconds": index / fps,
                                              "detector_ran": detector_ran,
                                              "detection_frame_index": detection_frame_index,
                                              "detections": detections}, ensure_ascii=False) + "\n")
                    index += 1
                    now = time.perf_counter()
                    if now - last_report >= 1:
                        show_progress(index, total, detect_count, started)
                        last_report = now
                if index == 0:
                    raise ValueError("Video không có frame đọc được.")
                show_progress(index, total, detect_count, started, final=True)
            finally:
                capture.release()
                if writer is not None:
                    writer.release()
    print(f"Đã lưu kết quả: {args.output.resolve()}")


if __name__ == "__main__":
    main()
