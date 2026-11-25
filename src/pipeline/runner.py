"""Chạy pipeline thật cho file video hoặc RTMP/HLS; chỉ xuất alert với bbox/chữ hành vi đỏ, hỗ trợ gửi BE."""

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit, urlunsplit
import imageio_ffmpeg
from zoneinfo import ZoneInfo

import cv2
import yaml
from dotenv import load_dotenv

from alerts.manager import AlertManager
from alerts.uploader import AlertDelivery
from behavior.engine import BehaviorEngine
from core.config import load_config, load_tracking_config, apply_cpu_threads
from core.timing import DetectorTimer
from detection.detector import Detector
from detection.render import draw_detections, load_font
from grouping.groups import Grouper
from pipeline.stream import StreamReader
from pipeline.publish import RTMPPublisher
from tracking.tracker import Tracker

ROOT = Path(__file__).resolve().parents[2]


class AlertPipeline:
    def __init__(self, config, alert_config, output, fps, delivery, publisher=None, models=None):
        self.config, self.fps = config, fps
        self.alert_config, self.delivery = alert_config, delivery
        self.publisher = publisher
        self.detector = models[0] if models else Detector(config)
        self.timer = DetectorTimer(config['inference']['device'])
        self.tracker = Tracker(load_tracking_config(ROOT/'configs/tracking.yaml'))
        self.grouper = Grouper(yaml.safe_load((ROOT/'configs/grouping.yaml').read_text()))
        self.behavior = models[1] if models else BehaviorEngine(ROOT/'configs/behavior.yaml', config['inference']['device'])
        self.behavior.cache.clear()
        self.font = load_font(config['output']['font'])
        self.alerts = AlertManager(output, alert_config, True, config['output']['video'], delivery.submit)
        self.last_slot = -1
        self.index = 0
        self.previous = None
        self.processed = 0
        self.started = time.perf_counter()
        self.steps = {}
        self.observation_time = None
        self.last_observation_index = None

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            step = self.steps.setdefault(name, dict(seconds=0., calls=0))
            step['seconds'] += time.perf_counter()-started
            step['calls'] += 1

    def process(self, frame, index, resume=False):
        media_time = index/self.fps
        if self.last_observation_index is None:
            delta = 1/self.fps
            self.observation_time = media_time
        else:
            delta = 1/self.fps if resume else (index-self.last_observation_index)/self.fps
            self.observation_time += delta
        timestamp = self.observation_time
        previous_ids = {t['id'] for t in self.tracker.tracks} if resume else set()
        detect_fps = self.config['inference']['detect_fps']
        slot = math.floor(timestamp * min(detect_fps or self.fps, self.fps) + 1e-8)
        detections = None
        if resume or slot > self.last_slot:
            with self.measure('detect'):
                detections, _ = self.timer.detect(self.detector, frame)
            self.last_slot = slot
        with self.measure('tracking'):
            tracks = self.tracker.update(frame, timestamp, detections)
        scene_cut = self.tracker.last_motion.get('scene_cut', False)
        if scene_cut:
            self.behavior.cache.clear()
            print('[RESET] Đổi cảnh: reset tracking/grouping/cache hành vi/lịch cảnh báo; giữ model.', flush=True)
        if resume and previous_ids and not scene_cut:
            matched = {r['track_id'] for r in tracks if r['position_source']=='detection'} & previous_ids
            if not matched:
                self.tracker = Tracker(load_tracking_config(ROOT/'configs/tracking.yaml'))
                self.grouper = Grouper(yaml.safe_load((ROOT/'configs/grouping.yaml').read_text()))
                self.behavior.cache.clear()
                self.alerts.states.clear()
                self.alerts.pending.clear()
                tracks = self.tracker.update(frame,timestamp,detections)
                scene_cut = True
                print('[RESET] Không ghép lại được track sau tạm ngưng: reset tracking/grouping/cache hành vi/lịch cảnh báo.', flush=True)
            else:
                # Chỉ cho ID được detection mới ghép lại làm bằng chứng khi tiếp tục.
                for t in self.tracker.tracks:
                    if t['id'] in previous_ids and t['id'] not in matched:
                        t['last_seen'] = timestamp-self.tracker.config['max_prediction_seconds']-.001
                tracks = [r for r in tracks if r['position_source']=='detection']
                for value in self.behavior.cache.values():
                    value['timestamp'] = timestamp-self.behavior.config['sample_interval_seconds']
                print(f'[RESUME] Ghép lại {len(matched)} ID; giữ lịch sử, không cộng khoảng mất nguồn.', flush=True)
        with self.measure('grouping'):
            groups = self.grouper.update(tracks, timestamp, self.tracker.last_motion)
        with self.measure('behavior'):
            self.behavior.update(frame, tracks, groups, timestamp)
        # Minh chứng chỉ vẽ B{id} + tên hành vi; không vẽ object/nhóm chưa có hành vi.
        groups = [g for g in groups if g.get('behaviors')]
        with self.measure('render'):
            rendered = draw_detections(frame, [], self.font, groups, group_ids_only=True, tracks=tracks)
        # Stream bỏ frame cũ ở khâu đọc để không trễ dần. Clip CFR lặp viz gần nhất
        # ở các slot thiếu, giữ đúng thời lượng theo thời gian nguồn thay vì tốc độ AI.
        if self.publisher is not None:
            with self.measure('publish_enqueue'):
                self.publisher.write(rendered)
        with self.measure('alerts'):
            if self.previous is not None:
                old_frame, old_groups = self.previous
                for missing in range(self.index, index):
                    self.alerts.update(old_frame, missing, self.fps, old_groups, observed=False, observation_time=timestamp)
            self.alerts.update(rendered, index, self.fps, groups, scene_cut, observation_time=timestamp, observation_seconds=delta)
        self.last_observation_index = index
        self.index = index+1
        self.previous = (rendered, groups)
        self.processed += 1
        if self.processed % 100 == 0:
            print(f'Đã xử lý {self.processed} frame | {self.processed/(time.perf_counter()-self.started):.2f} FPS'
                  f' | {self.alerts.package_count} alert', flush=True)

    def reset_session(self, output, reason="gián đoạn dài"):
        """Đóng bằng chứng phiên cũ, giữ model và reset lịch sử thời gian/quan hệ."""
        print(f'[RESET] {reason}: reset tracking/grouping/cache hành vi/lịch cảnh báo; giữ Detector/MobileNet.', flush=True)
        report = self.close(close_models=False)
        models = (self.detector, self.behavior)
        self.__init__(self.config, self.alert_config, output, self.fps, self.delivery, self.publisher, models)
        return report

    def close(self, close_models=True):
        try:
            with self.measure('finalize_alerts'):
                result = self.alerts.finalize(None, self.fps, self.index)
            for name, step in sorted(self.steps.items(), key=lambda item: -item[1]['seconds']):
                print(f"{name:16s}: {step['seconds']:.3f}s | {step['seconds']*1000/step['calls']:.2f} ms/lượt", flush=True)
            self.tracker.timings.show('Chi tiết tracking')
            self.grouper.timings.show('Chi tiết grouping (geometry nằm trong các bước khác)')
            print(f'Chi tiết alert (giây): {self.alerts.timings}', flush=True)
            return dict(frames_processed=self.processed, output_frames=self.index,
                        elapsed_seconds=time.perf_counter()-self.started, events=result['event_count'],
                        alerts=result['package_count'], detector=self.timer.summary(), stages=self.steps, alert_stages=self.alerts.timings,
                        tracking_details=self.tracker.timings.summary(), grouping_details=self.grouper.timings.summary())
        finally:
            self.alerts.close()
            if close_models:
                self.detector.close()


def main(mode):
    # Nạp .env ở gốc dự án, không ghi đè biến môi trường đã đặt trên máy/server.
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='File video hoặc URL stream; bỏ trống dùng stream.yaml')
    parser.add_argument('--publish', action=argparse.BooleanOptionalAction, default=None, help='Bật/tắt phát stream đã viz (chỉ stream)')
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
    cpu_threads = apply_cpu_threads(config)
    output = args.output / datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    delivery = AlertDelivery(args.backend_config, started_at, args.upload)
    capture = pipeline = publisher = None
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
            stream.setdefault('pause_gap_seconds', 1.)
            stream.setdefault('reset_gap_seconds', 10.)
            if not 0 < stream['pause_gap_seconds'] < stream['reset_gap_seconds']:
                raise ValueError('Cần 0 < pause_gap_seconds < reset_gap_seconds.')
            if not isinstance(stream['reset_gap_seconds'], (int,float)) or not math.isfinite(stream['reset_gap_seconds']) or stream['reset_gap_seconds'] <= 0:
                raise ValueError('reset_gap_seconds phải dương và hữu hạn.')
            url = args.input or os.environ.get(stream['input_env']) or stream['hls' if args.source == 'hls' else 'input']
            if not url.startswith(('rtmp://','rtmps://','http://','https://')):
                raise ValueError('Nguồn phải là RTMP/HLS.')
            publish = stream.get('publish', {})
            if args.publish if args.publish is not None else publish.get('enabled', False):
                destination = os.environ.get(publish.get('url_env','UAV_VIZ_URL')) or publish['url']
                # Ưu tiên URL publish đầy đủ trong env; nếu chưa có query, lấy auth của nguồn RTMP cấu hình.
                parsed = urlsplit(destination)
                if not parsed.query:
                    credentials = urlsplit(os.environ.get(stream['input_env']) or stream['input']).query
                    destination = urlunsplit(parsed._replace(query=credentials))
                publisher = RTMPPublisher(destination, publish.get('fps',15),
                    publish.get('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe(),
                    encoder=publish.get('encoder','libx264'))
                print('Đã bật phát viz; trạng thái sending chỉ xác nhận gửi encoder, chưa xác nhận người xem.',flush=True)
            capture = StreamReader(url, stream)
            run_started = time.monotonic()
            completed_frames = deque()
            epoch, origin, processed, last_received = None, None, 0, None
            paused, last_status = False, time.monotonic()
            while not args.max_frames or processed < args.max_frames:
                item = capture.read(max_age_seconds=stream['pause_gap_seconds'], poll_seconds=.5)
                now = time.monotonic()
                if now-last_status >= 5:
                    declared, received = capture.fps_status()
                    declared_text = f'{declared:.2f}' if declared else 'không rõ'
                    while completed_frames and completed_frames[0] <= now-5:
                        completed_frames.popleft()
                    recent_fps = len(completed_frames)/min(5., max(now-run_started, .001))
                    print(f'FPS nguồn khai báo: {declared_text} | FPS nhận (5s): {received:.2f}'
                          f' | FPS xử lý (5s): {recent_fps:.2f}'
                          f' | FPS xử lý TB: {processed/max(now-run_started, .001):.2f}', flush=True)
                    last_status = now
                if item is None:
                    if last_received is not None and now-last_received > stream['pause_gap_seconds'] and not paused:
                        paused = True
                        print('[PAUSE] Tạm ngưng quan sát: giữ ID/nhóm, dừng cộng bằng chứng.', flush=True)
                    if pipeline is not None and last_received is not None and now-last_received > stream['reset_gap_seconds']:
                        results.append(pipeline.reset_session(output/f'camera_{processed:08d}', f'mất nguồn > {stream["reset_gap_seconds"]}s'))
                        origin, last_received, paused = None, None, False
                    continue
                current_epoch, moment, frame = item
                gap = moment-last_received if last_received is not None else 0.
                if pipeline is None:
                    pipeline = AlertPipeline(config, args.alert_config, output/f'camera_{processed:08d}', stream['output_fps'], delivery, publisher)
                    epoch, origin = current_epoch, None
                    continue  # Lấy lại frame mới nhất sau nạp model.
                if last_received is not None and gap > stream['reset_gap_seconds']:
                    results.append(pipeline.reset_session(output/f'camera_{processed:08d}', f'gap {gap:.2f}s > {stream["reset_gap_seconds"]}s'))
                    epoch, origin, last_received, paused = current_epoch, None, None, False
                    continue
                resume = last_received is not None and (paused or gap > stream['pause_gap_seconds'] or current_epoch != epoch)
                if resume and not paused:
                    print(f'[PAUSE] Gián đoạn/reconnect {gap:.2f}s: kiểm tra ghép lại trước khi tiếp tục.', flush=True)
                paused = False
                epoch = current_epoch
                if origin is None:
                    origin = moment
                    delivery.started_at = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
                last_received = moment
                index = round((moment-origin)*stream['output_fps'])
                if index < pipeline.index:
                    continue
                pipeline.process(frame, index, resume=resume)
                processed += 1
                completed = time.monotonic()
                completed_frames.append(completed)
                while completed_frames and completed_frames[0] <= completed-5:
                    completed_frames.popleft()
    except KeyboardInterrupt:
        print('Đang đóng clip và chờ upload hoàn tất...', flush=True)
    finally:
        try:
            if capture is not None:
                capture.close() if mode == 'stream' else capture.release()
            if pipeline is not None:
                results.append(pipeline.close())
        finally:
            try:
                if publisher is not None:
                    publisher.close()
            finally:
                delivery.close()
        (output/'summary.json').write_text(json.dumps(dict(cpu_threads=cpu_threads, sessions=results, delivery=delivery.counts, publish=publisher.stats if publisher else None), ensure_ascii=False, indent=2))
    print(f'Hoàn tất: {output} | {delivery.counts}', flush=True)
