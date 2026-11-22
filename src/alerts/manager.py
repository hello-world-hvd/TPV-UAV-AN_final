"""Gom cảnh báo, giữ clip -2/+8s và chọn một frame nguyên đã viz hành vi tại khoảng ổn định lâu nhất cho mỗi nhóm trong clip."""

import json
import math
import time
from pathlib import Path
from collections import deque

import cv2
from core.video_writer import open_writer
import yaml


class AlertManager:
    def __init__(self, output, config_path, save_video=True, video_config=None, on_package=None):
        self.output = Path(output) / 'alerts'
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        for key in ('pre_seconds', 'post_seconds', 'rearm_seconds', 'confirmation_seconds'):
            value = self.config.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} phải là số không âm hữu hạn.')
        if self.config['post_seconds'] <= 0 or type(self.config.get('require_confirmed_group')) is not bool:
            raise ValueError('post_seconds phải >0; require_confirmed_group phải là boolean.')
        self.config.setdefault('evidence_min_seconds', 1.5)
        minimum = self.config['evidence_min_seconds']
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or minimum < 0:
            raise ValueError('evidence_min_seconds phải là số không âm hữu hạn.')
        self.on_package = on_package
        self.event_count = self.package_count = 0
        self.save_video = save_video
        self.states, self.events, self.packages = {}, [], []
        self.pending = {}
        self.previous_frame = -1
        self.history = deque()
        self.frames = deque()
        self.writers = {}
        self.video_config = {"encoder": "opencv", "preset": "p1", "cq": 20,
                             "queue_size": 2, **(video_config or {}), "async_encode": True}
        self.video_stats = []
        self.timings = dict(open_encoder=0., enqueue_video=0., close_encoder=0., evidence=0.)

    def update(self, frame, frame_index, fps, groups, scene_cut=False):
        if frame_index <= self.previous_frame:
            raise ValueError('Frame cảnh báo phải tăng liên tục.')
        self.previous_frame = frame_index
        timestamp = frame_index / fps
        if scene_cut:
            self.states.clear()
            self.pending.clear()
        self.states = {key: last for key, last in self.states.items()
                       if timestamp - last <= self.config['rearm_seconds']}
        # Clip đã giữ nguyên frame viz: chỉ lưu metadata để chọn ảnh khi đóng clip.
        # Khi tắt clip, vẫn giữ JPEG để chế độ chỉ ảnh hoạt động độc lập.
        samples = {}
        visible = [group for group in groups if group.get('behaviors')]
        image = None
        if visible and not self.save_video:
            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError('Không mã hoá được ảnh minh chứng.')
            image = jpeg.tobytes()
        for group in visible:
            samples[group['group_id']] = dict(signature=tuple(sorted(group['track_ids'])),
                behaviors=tuple(b['name'] for b in group['behaviors']), jpeg=image,
                bbox=list(group.get('bbox_xyxy', [0, 0, frame.shape[1], frame.shape[0]])))
        self.history.append((frame_index, timestamp, samples))
        if self.save_video:
            self.frames.append((frame_index, frame.copy()))
            first_frame = math.ceil((timestamp - self.config['pre_seconds']) * fps - 1e-7)
            while self.frames and self.frames[0][0] < first_frame:
                self.frames.popleft()
        observed = set()
        for group in groups:
            if self.config['require_confirmed_group'] and group['status'] != 'confirmed':
                continue
            for behavior in group.get('behaviors', []):
                key = (group['group_id'], behavior['name'])
                if key in observed:
                    continue
                observed.add(key)
                first = self.pending.setdefault(key, timestamp)
                if timestamp - first + 1e-8 < self.config['confirmation_seconds']:
                    continue
                previous = self.states.get(key)
                self.states[key] = timestamp
                if previous is not None:
                    continue
                self.event_count += 1
                event_id = self.event_count
                event = dict(event_id=event_id, group_id=group['group_id'], track_ids=list(group['track_ids']),
                             behavior=behavior['name'], source=behavior['source'], status=behavior.get('status', 'suspected'),
                             confidence=behavior.get('confidence'), alert_frame=frame_index, alert_time=timestamp,
                             snapshot=None)
                self.events.append(event)
                # Trong cửa sổ đã mở: chỉ thêm sự kiện, không kéo dài clip.
                if not any(p['start_frame'] <= frame_index < p['stop_frame'] for p in self.packages):
                    self.package_count += 1
                    self.packages.append(dict(package_id=self.package_count, trigger_event_id=event_id,
                        requested_start=timestamp-self.config['pre_seconds'], start=max(0., timestamp-self.config['pre_seconds']),
                        end=timestamp+self.config['post_seconds']))
                    package = self.packages[-1]
                    # Dùng chỉ số nguyên cho cả ghi và kiểm tra; tránh ghi thừa 1 frame do sai số float.
                    package['start_frame'] = max(0, math.ceil(package['start']*fps - 1e-7))
                    package['stop_frame'] = math.ceil(package['end']*fps - 1e-7)
                    package['samples'] = [sample for sample in self.history
                                          if sample[0] >= package['start_frame']]
                    if self.save_video:
                        package = self.packages[-1]
                        folder = self.output / f"alert_{package['package_id']:06d}"
                        folder.mkdir(exist_ok=True)
                        height, width = frame.shape[:2]
                        started = time.perf_counter()
                        writer = open_writer(folder / 'clip.mp4', fps, (width, height), self.video_config)
                        self.timings['open_encoder'] += time.perf_counter()-started
                        self.writers[package['package_id']] = writer
                        start = package['start_frame']
                        package['video_frames'] = 0
                        for index, buffered in self.frames:
                            if index >= start:
                                self._write_video(writer, buffered)
                                package['video_frames'] += 1
                        package['last_video_frame'] = frame_index
        self.pending = {key: value for key, value in self.pending.items() if key in observed}
        for package in self.packages:
            if 'evidence' in package:
                continue
            if frame_index < package['stop_frame']:
                writer = self.writers.get(package['package_id'])
                if writer is not None and package['last_video_frame'] != frame_index:
                    self._write_video(writer, frame)
                    package['video_frames'] += 1
                    package['last_video_frame'] = frame_index
                if not package['samples'] or package['samples'][-1][0] != frame_index:
                    package['samples'].append(self.history[-1])
            else:
                self._close_video(package['package_id'])
                self._seal(package, fps)
                if self.on_package is not None:
                    report = self._report(package, fps, frame_index)
                    self.on_package(self.output / f"alert_{package['package_id']:06d}", report)
                    package['delivered'] = True
        if self.on_package is not None:
            self.packages = [p for p in self.packages if not p.get('delivered')]
            cutoff = min([p['start'] for p in self.packages] + [timestamp-self.config['pre_seconds']])
            self.events = [e for e in self.events if e['alert_time'] >= cutoff]
            self.video_stats = self.video_stats[-2:]
        while self.history and self.history[0][1] < timestamp - self.config['pre_seconds']:
            self.history.popleft()

    def _write_video(self, writer, frame):
        started = time.perf_counter()
        writer.write(frame)
        self.timings['enqueue_video'] += time.perf_counter()-started

    def _close_video(self, package_id):
        writer = self.writers.pop(package_id, None)
        if writer is not None:
            started = time.perf_counter()
            writer.release()
            self.timings['close_encoder'] += time.perf_counter()-started
            self.video_stats.append(dict(package_id=package_id,
                frames_written=writer.frames_written, write_seconds=writer.write_seconds))

    def close(self):
        """Chờ mọi encoder ghi hết, kể cả khi pipeline gặp lỗi."""
        error = None
        for package_id in list(self.writers):
            try:
                self._close_video(package_id)
            except Exception as exc:
                error = error or exc
        self.frames.clear()
        if error is not None:
            raise error

    def _seal(self, package, fps):
        started = time.perf_counter()
        folder = self.output / f"alert_{package['package_id']:06d}"
        folder.mkdir(exist_ok=True)
        ids = {e['group_id'] for e in self.events if package['start_frame'] <= e['alert_frame'] < package['stop_frame']}
        evidence, selected = {}, {}
        for gid in ids:
            durations = {}
            for _, _, samples in package['samples']:
                for name in samples.get(gid, {}).get('behaviors', ()):
                    durations[name] = durations.get(name, 0.) + 1/fps
            accepted = {name for name, seconds in durations.items()
                        if seconds + 1e-8 >= self.config['evidence_min_seconds']}
            if not accepted:
                continue
            runs = []
            for index, timestamp, samples in package['samples']:
                sample = samples.get(gid)
                if sample is None or not accepted.intersection(sample['behaviors']):
                    continue
                if (runs and runs[-1][-1][0] + 1 == index
                        and runs[-1][-1][2]['signature'] == sample['signature']):
                    runs[-1].append((index, timestamp, sample))
                else:
                    runs.append([(index, timestamp, sample)])
            if not runs:
                continue
            # Dài nhất trước; nếu bằng nhau chọn tổ hợp nhiều thành viên hơn.
            run = max(runs, key=lambda r: (len(r), len(r[0][2]['signature'])))
            index, timestamp, sample = run[(len(run)-1)//2]
            name = f'group_{gid:06d}.jpg'
            selected[gid] = (index, sample['jpeg'])
            evidence[gid] = dict(group_id=gid, snapshot=name, frame=index, time=timestamp,
                bbox_xyxy=sample['bbox'], image_scope='full_frame', image_source='clip' if self.save_video else 'frame', track_ids=list(sample['signature']),
                stable_start=run[0][1], stable_end=run[-1][1] + 1/fps,
                stable_seconds=len(run)/fps, behavior_seconds={name: durations[name] for name in sorted(accepted)},
                behavior_names=list(dict.fromkeys(e['behavior'] for e in self.events
                    if e['group_id'] == gid and e['behavior'] in accepted
                    and package['start_frame'] <= e['alert_frame'] < package['stop_frame'])))
        if self.save_video and selected:
            # Chỉ giải mã clip alert (~10 giây), một lượt tới frame cuối cần lấy.
            # grab() tránh chuyển mọi frame sang BGR; chỉ retrieve/nén những ảnh được chọn.
            needed = {index-package['start_frame'] for index, _ in selected.values()}
            images = self._read_evidence(folder/'clip.mp4', needed)
            for gid, (index, _) in selected.items():
                (folder/evidence[gid]['snapshot']).write_bytes(images[index-package['start_frame']])
        else:
            for gid, (_, jpeg) in selected.items():
                (folder/evidence[gid]['snapshot']).write_bytes(jpeg)
        package['evidence_export_seconds'] = time.perf_counter()-started
        self.timings['evidence'] += package['evidence_export_seconds']
        package['evidence'] = evidence
        del package['samples']

    @staticmethod
    def _read_evidence(clip, needed):
        capture = cv2.VideoCapture(str(clip))
        images = {}
        try:
            if not capture.isOpened():
                raise RuntimeError(f'Không mở được clip minh chứng: {clip}')
            for index in range(max(needed)+1):
                if not capture.grab():
                    raise RuntimeError(f'Clip thiếu frame minh chứng {index}: {clip}')
                if index in needed:
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise RuntimeError(f'Không giải mã được frame minh chứng {index}: {clip}')
                    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise RuntimeError('Không mã hoá được ảnh minh chứng.')
                    images[index] = jpeg.tobytes()
        finally:
            capture.release()
        return images

    def _report(self, package, fps, frame_count):
        duration = frame_count / fps
        folder = self.output / f"alert_{package['package_id']:06d}"
        folder.mkdir(exist_ok=True)
        # Cắt theo chỉ số frame để không lệch keyframe; không lấy frame ngoài khoảng yêu cầu.
        start = package['start_frame']
        stop = min(frame_count, package['stop_frame'])
        events = [e for e in self.events if start <= e['alert_frame'] < stop]
        package_events = []
        for event in events:
            evidence = package['evidence'].get(event['group_id'])
            if evidence:
                reference = str((folder / evidence['snapshot']).relative_to(self.output))
                if event['snapshot'] is None:
                    event['snapshot'] = reference
            package_events.append({**event, 'snapshot': evidence['snapshot'] if evidence else None})
        report = {**{k: v for k, v in package.items() if k != 'evidence'},
                  'evidence': list(package['evidence'].values()), 'actual_start': start/fps, 'actual_end': stop/fps,
                  'missing_pre_seconds': max(0., -package['requested_start']),
                  'missing_post_seconds': max(0., package['end']-duration),
                  'fps': fps, 'behavior_names': list(dict.fromkeys(e['behavior'] for e in events)),
                  'events': package_events,
                  'clip': None, 'video_saved': self.save_video}
        if self.save_video and stop > start:
            if package['video_frames'] != stop - start:
                raise RuntimeError(f"Clip {package['package_id']}: đã ghi {package['video_frames']} frame, cần {stop-start} frame [{start}, {stop}).")
            report['clip'] = 'clip.mp4'
        (folder / 'metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report

    def finalize(self, video_path, fps, frame_count):
        self.close()
        reports = []
        for package in self.packages:
            if 'evidence' not in package:
                self._seal(package, fps)
            report = self._report(package, fps, frame_count)
            if self.on_package is not None:
                self.on_package(self.output / f"alert_{package['package_id']:06d}", report)
            reports.append(report)
        summary = dict(event_count=self.event_count, package_count=self.package_count, save_video=self.save_video,
                       events=self.events, packages=reports, encoder_workers=self.video_stats)
        if self.on_package is None:
            (self.output/'index.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary
