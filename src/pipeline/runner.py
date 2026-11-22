"""Chạy pipeline thật cho file video hoặc RTMP/HLS; chỉ xuất alert với bbox/chữ hành vi đỏ, hỗ trợ gửi BE."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import cv2
import yaml
from dotenv import load_dotenv

from alerts.manager import AlertManager
from alerts.uploader import AlertDelivery
from behavior.engine import BehaviorEngine
from core.config import load_config, load_tracking_config
from core.timing import DetectorTimer
from detection.detector import Detector
from detection.render import draw_detections, load_font
from grouping.groups import Grouper
from pipeline.stream import StreamReader
from tracking.tracker import Tracker

ROOT = Path(__file__).resolve().parents[2]


class AlertPipeline:
    def __init__(self, config, alert_config, output, fps, delivery):
        self.config, self.fps = config, fps
        self.detector = Detector(config)
        self.timer = DetectorTimer(config['inference']['device'])
        self.tracker = Tracker(load_tracking_config(ROOT/'configs/tracking.yaml'))
        self.grouper = Grouper(yaml.safe_load((ROOT/'configs/grouping.yaml').read_text()))
        self.behavior = BehaviorEngine(ROOT/'configs/behavior.yaml', config['inference']['device'])
        self.font = load_font(config['output']['font'])
        self.alerts = AlertManager(output, alert_config, True, config['output']['video'], delivery.submit)
        self.last_slot = -1
        self.index = 0
        self.previous = None
        self.processed = 0
        self.started = time.perf_counter()
        self.steps = {}

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            step = self.steps.setdefault(name, dict(seconds=0., calls=0))
            step['seconds'] += time.perf_counter()-started
            step['calls'] += 1

    def process(self, frame, index):
        timestamp = index/self.fps
        detect_fps = self.config['inference']['detect_fps']
        slot = math.floor(timestamp * min(detect_fps or self.fps, self.fps) + 1e-8)
        detections = None
        if slot > self.last_slot:
            with self.measure('detect'):
                detections, _ = self.timer.detect(self.detector, frame)
            self.last_slot = slot
        with self.measure('tracking'):
            tracks = self.tracker.update(frame, timestamp, detections)
        with self.measure('grouping'):
            groups = self.grouper.update(tracks, timestamp, self.tracker.last_motion)
        with self.measure('behavior'):
            self.behavior.update(frame, tracks, groups, timestamp)
        # Minh chứng chỉ vẽ B{id} + tên hành vi; không vẽ object/nhóm chưa có hành vi.
        groups = [g for g in groups if g.get('behaviors')]
        with self.measure('render'):
            rendered = draw_detections(frame, [], self.font, groups, group_ids_only=True)
        # Stream bỏ frame cũ ở khâu đọc để không trễ dần. Clip CFR lặp viz gần nhất
        # ở các slot thiếu, giữ đúng thời lượng theo thời gian nguồn thay vì tốc độ AI.
        with self.measure('alerts'):
            if self.previous is not None:
                old_frame, old_groups = self.previous
                for missing in range(self.index, index):
                    self.alerts.update(old_frame, missing, self.fps, old_groups)
            self.alerts.update(rendered, index, self.fps, groups, self.tracker.last_motion.get('scene_cut', False))
        self.index = index+1
        self.previous = (rendered, groups)
        self.processed += 1
        if self.processed % 100 == 0:
            print(f'Đã xử lý {self.processed} frame | {self.processed/(time.perf_counter()-self.started):.2f} FPS'
                  f' | {self.alerts.package_count} alert', flush=True)

    def close(self):
        try:
            with self.measure('finalize_alerts'):
                result = self.alerts.finalize(None, self.fps, self.index)
            for name, step in sorted(self.steps.items(), key=lambda item: -item[1]['seconds']):
                print(f"{name:16s}: {step['seconds']:.3f}s | {step['seconds']*1000/step['calls']:.2f} ms/lượt", flush=True)
            print(f'Chi tiết alert (giây): {self.alerts.timings}', flush=True)
            return dict(frames_processed=self.processed, output_frames=self.index,
                        elapsed_seconds=time.perf_counter()-self.started, events=result['event_count'],
                        alerts=result['package_count'], detector=self.timer.summary(), stages=self.steps, alert_stages=self.alerts.timings)
        finally:
            self.alerts.close()
            self.detector.close()


def main(mode):
    # Nạp .env ở gốc dự án, không ghi đè biến môi trường đã đặt trên máy/server.
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='File video hoặc URL stream; bỏ trống dùng stream.yaml')
    parser.add_argument('--source', choices=['rtmp','hls'], default='rtmp')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/detection.yaml')
    parser.add_argument('--alert-config', type=Path, default=ROOT/'configs/alerts.yaml')
    parser.add_argument('--backend-config', type=Path, default=ROOT/'configs/backend.yaml')
    parser.add_argument('--stream-config', type=Path, default=ROOT/'configs/stream.yaml')
    parser.add_argument('--upload', action=argparse.BooleanOptionalAction, default=None,
                        help='Ghi đè backend.enabled; --no-upload lưu alert ở workspace')
    parser.add_argument('--recording-start', help='Thời điểm bắt đầu video ISO8601 có múi giờ; mặc định lúc chạy')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/alerts')
    parser.add_argument('--max-frames', type=int, default=0, help='0: không giới hạn; stream dừng bằng Ctrl+C')
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error('--max-frames phải >= 0')
    if mode == 'video' and (not args.input or not Path(args.input).is_file()):
        parser.error('Cần --input trỏ tới file video.')
    started_at = datetime.fromisoformat(args.recording_start) if args.recording_start else datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
    if started_at.tzinfo is None:
        parser.error('--recording-start cần múi giờ, ví dụ +07:00')
    config = load_config(args.config)
    output = args.output / datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    delivery = AlertDelivery(args.backend_config, started_at, args.upload)
    capture = pipeline = None
    results = []
    try:
        if mode == 'video':
            capture = cv2.VideoCapture(args.input)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError('Không đọc được video/FPS.')
            pipeline = AlertPipeline(config, args.alert_config, output, fps, delivery)
            index = 0
            while not args.max_frames or index < args.max_frames:
                with pipeline.measure('decode'):
                    ok, frame = capture.read()
                if not ok:
                    break
                pipeline.process(frame, index)
                index += 1
        else:
            stream = yaml.safe_load(args.stream_config.read_text())
            if not math.isfinite(stream['output_fps']) or stream['output_fps'] <= 0 or stream['timeout_ms'] <= 0 or stream['reconnect_seconds'] <= 0:
                raise ValueError('FPS/timeout/reconnect stream phải dương.')
            url = args.input or os.environ.get(stream['input_env']) or stream['hls' if args.source == 'hls' else 'input']
            if not url.startswith(('rtmp://','rtmps://','http://','https://')):
                raise ValueError('Nguồn phải là RTMP/HLS.')
            capture = StreamReader(url, stream)
            epoch, origin, processed = None, None, 0
            while not args.max_frames or processed < args.max_frames:
                item = capture.read()
                if item is None:
                    break
                current_epoch, moment, frame = item
                # Không kéo dài frame cũ qua thời gian mất camera; kết thúc clip thiếu post-roll.
                if current_epoch != epoch or (pipeline is not None and (moment-origin)*stream['output_fps']-pipeline.index > stream['output_fps']*2):
                    if pipeline is not None:
                        results.append(pipeline.close())
                        pipeline = None
                    epoch, origin = current_epoch, moment
                    delivery.started_at = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
                    pipeline = AlertPipeline(config, args.alert_config, output/f'camera_{processed:08d}', stream['output_fps'], delivery)
                index = round((moment-origin)*stream['output_fps'])
                if index < pipeline.index:
                    continue
                pipeline.process(frame, index)
                processed += 1
    except KeyboardInterrupt:
        print('Đang đóng clip và chờ upload hoàn tất...', flush=True)
    finally:
        try:
            if capture is not None:
                capture.close() if mode == 'stream' else capture.release()
            if pipeline is not None:
                results.append(pipeline.close())
        finally:
            delivery.close()
        (output/'summary.json').write_text(json.dumps(dict(sessions=results, delivery=delivery.counts), ensure_ascii=False, indent=2))
    print(f'Hoàn tất: {output} | {delivery.counts}', flush=True)
