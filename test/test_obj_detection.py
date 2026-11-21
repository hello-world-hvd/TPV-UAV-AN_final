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
from contextlib import contextmanager

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection.detector import Detector
from detection.render import draw_detections, load_font
from core.config import load_config, load_tracking_config
from tracking.tracker import Tracker
from grouping.groups import Grouper
from behavior.engine import BehaviorEngine
from core.timing import DetectorTimer
from core.video_writer import open_writer, ThreadedWriter


class StageTimer:
    """Cộng thời gian và số lượt từng bước, bao gồm bước đóng video để hoàn tất file."""

    def __init__(self):
        self.steps = {}

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            step = self.steps.setdefault(name, {"seconds": 0.0, "calls": 0})
            step["seconds"] += time.perf_counter() - started
            step["calls"] += 1

    def summary(self, elapsed):
        return {name: {**step, "average_ms": step["seconds"] / step["calls"] * 1000,
                       "wall_percent": step["seconds"] / elapsed * 100 if elapsed else 0}
                for name, step in self.steps.items()}


def show_progress(index, total, detect_count, started, timer, stages, final=False):
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
    stats = timer.summary()
    average_ms = stats["average_detect_ms"] or 0
    detector_fps = stats["detector_fps"] or 0
    slowest = max(stages.steps, key=lambda name: stages.steps[name]["seconds"], default="-")
    print(f"\r{progress} | Detect: {detect_count} lượt | Xử lý: {speed:.2f} FPS"
          f" | Detector TB: {average_ms:.1f} ms/lượt ({detector_fps:.2f} FPS)"
          f" | Bước tốn thời gian nhất: {slowest}"
          f" | Đã chạy: {elapsed:.1f}s | Còn lại ~{eta}",
          end="\n" if final else "", flush=True)


def main():
    root = ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visualization', choices=['all', 'group_ids'], default='all')
    parser.add_argument("--mode", required=True, choices=["images", "video"])
    parser.add_argument("--input", type=Path, help="Thư mục ảnh hoặc đường dẫn video")
    parser.add_argument("--max-frames", type=int, default=0, help="Giới hạn frame video; 0: toàn bộ")
    parser.add_argument("--config", type=Path, default=root / "configs" / "detection.yaml")
    parser.add_argument("--grouping-config", type=Path, default=root / "configs" / "grouping.yaml")
    parser.add_argument("--behavior-config", type=Path, default=root / "configs" / "behavior.yaml")
    parser.add_argument("--tracking-config", type=Path, default=root / "configs" / "tracking.yaml")
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
    model_started = time.perf_counter()
    detector = Detector(config)
    model_load_seconds = time.perf_counter() - model_started
    print(f"Nạp model: {model_load_seconds:.2f}s (tách riêng thời gian xử lý)")
    timer = DetectorTimer(config["inference"]["device"])
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "detections.jsonl").open("w", encoding="utf-8") as records:
        if args.mode == "images":
            for path in images:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Không đọc được ảnh: {path}")
                detections, detect_seconds = timer.detect(detector, image)
                rendered = draw_detections(image, detections, font)
                output_path = args.output / f"{path.name}.detected.jpg"
                if not cv2.imwrite(str(output_path), rendered):
                    raise RuntimeError(f"Không ghi được ảnh: {output_path}")
                records.write(json.dumps({"source": str(path), "detect_ms": detect_seconds * 1000,
                                          "detections": detections}, ensure_ascii=False) + "\n")
                print(f"{path.name}: {len(detections)} đối tượng | Detector: {detect_seconds * 1000:.1f} ms")
        else:
            tracking_config = load_tracking_config(args.tracking_config)
            tracker = Tracker(tracking_config) if tracking_config["enabled"] else None
            group_config = yaml.safe_load(args.grouping_config.read_text(encoding="utf-8"))
            grouper = Grouper(group_config) if tracker is not None and group_config['enabled'] else None
            behavior = BehaviorEngine(args.behavior_config, config['inference']['device']) if grouper else None
            stages = StageTimer()
            video_started = time.perf_counter()
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
                    with stages.measure("decode"):
                        ok, frame = capture.read()
                    if not ok:
                        break
                    if writer is None:
                        height, width = frame.shape[:2]
                        with stages.measure("open_writer"):
                            writer = open_writer(args.output / "detected.mp4", fps, (width, height),
                                                 config["output"]["video"])
                    # Lập lịch theo thời gian video, không theo tốc độ xử lý của máy.
                    slot = int(index * detect_fps / fps + 1e-9)
                    detector_ran = slot > last_slot
                    detect_seconds = None
                    if detector_ran:
                        with stages.measure("detect"):
                            detections, detect_seconds = timer.detect(detector, frame)
                        detect_count += 1
                        detection_frame_index = index
                        last_slot = slot
                    if tracker is not None:
                        with stages.measure("tracking"):
                            tracks = tracker.update(frame, index / fps, detections if detector_ran else None)
                    else:
                        tracks = detections
                    groups = []
                    if grouper is not None:
                        with stages.measure("grouping"):
                            groups = grouper.update(tracks, index / fps, tracker.last_motion)
                    if behavior is not None:
                        with stages.measure('behavior'):
                            behavior.update(frame, tracks, groups, index / fps)
                    with stages.measure("render"):
                        rendered = draw_detections(frame, [] if args.visualization == 'group_ids' else tracks,
                                                   font, groups, group_ids_only=args.visualization == 'group_ids')
                    with stages.measure("encode_write"):
                        writer.write(rendered)
                    with stages.measure("jsonl"):
                        records.write(json.dumps({"frame_index": index, "timestamp_seconds": index / fps,
                                              "detector_ran": detector_ran,
                                              "detect_ms": detect_seconds * 1000 if detect_seconds is not None else None,
                                              "detection_frame_index": detection_frame_index,
                                              "tracks": tracks if tracker is not None else [],
                                              "groups": groups,
                                              "detections": detections}, ensure_ascii=False) + "\n")
                    index += 1
                    now = time.perf_counter()
                    if now - last_report >= 1:
                        show_progress(index, total, detect_count, started, timer, stages)
                        last_report = now
                if index == 0:
                    raise ValueError("Video không có frame đọc được.")
                show_progress(index, total, detect_count, started, timer, stages, final=True)
            finally:
                with stages.measure("finalize_video"):
                    capture.release()
                    if writer is not None:
                        writer.release()
                with stages.measure("flush_jsonl"):
                    records.flush()
            video_elapsed = time.perf_counter() - video_started
            stage_stats = stages.summary(video_elapsed)
            print("\nThời gian từng bước (TB tính trên số lượt thực hiện của bước):")
            for name, step in sorted(stage_stats.items(), key=lambda item: item[1]["seconds"], reverse=True):
                print(f"  {name:16s}: {step['seconds']:8.3f}s | {step['average_ms']:8.2f} ms/lượt"
                      f" | {step['wall_percent']:5.1f}% | {step['calls']} lượt")
    stats = timer.summary()
    stats["model_load_seconds"] = model_load_seconds
    detector.close()
    stats['parallel_models'] = config['inference'].get('parallel_models', False)
    if args.mode == "video":
        stats["video_processing_seconds"] = video_elapsed
        stats["tracking"] = tracking_config
        stats["processing_fps_including_finalize"] = index / video_elapsed
        stats["output_video_bytes"] = (args.output / "detected.mp4").stat().st_size
        stats["video_encoder"] = config["output"]["video"]
        if isinstance(writer, ThreadedWriter):
            stats["encoder_worker"] = {"write_seconds": writer.write_seconds,
                                       "frames_written": writer.frames_written}
            print("encode_write = đẩy/chờ hàng đợi; finalize_video gồm chờ worker và FFmpeg ghi hết.")
        print(f"Tổng kể cả hoàn tất file: {video_elapsed:.2f}s | {index / video_elapsed:.2f} FPS")
        stats["stages"] = stage_stats
        stats["other_seconds"] = max(0, video_elapsed - sum(step["seconds"] for step in stage_stats.values()))
    (args.output / "timing.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detector [{stats['device']}]: {stats['average_detect_ms']:.1f} ms/lượt"
          f" | {stats['detector_fps']:.2f} FPS | Lượt đầu: {stats['first_detect_ms']:.1f} ms")
    if stats["average_detect_ms_excluding_first"] is not None:
        print(f"Detector TB bỏ lượt đầu: {stats['average_detect_ms_excluding_first']:.1f} ms/lượt")
    print(f"Đã lưu kết quả: {args.output.resolve()}")


if __name__ == "__main__":
    main()
