"""Gom cảnh báo, giữ clip -2/+8s và chọn một crop viz ổn định lâu nhất cho mỗi nhóm trong clip."""

import json
import math
from pathlib import Path
from collections import deque

import cv2
from core.video_writer import open_writer
import yaml


class AlertManager:
    def __init__(self, output, config_path, save_video=True, video_config=None):
        self.output = Path(output) / 'alerts'
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        for key in ('pre_seconds', 'post_seconds', 'rearm_seconds', 'confirmation_seconds'):
            value = self.config.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} phải là số không âm hữu hạn.')
        if self.config['post_seconds'] <= 0 or type(self.config.get('require_confirmed_group')) is not bool:
            raise ValueError('post_seconds phải >0; require_confirmed_group phải là boolean.')
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
        # Chỉ giữ crop JPEG trong cửa sổ đang mở và phần đệm trước cảnh báo.
        samples = {}
        for group in groups:
            box = group.get('bbox_xyxy', [0, 0, frame.shape[1], frame.shape[0]])
            x1, y1, x2, y2 = box
            x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
            x2, y2 = min(frame.shape[1], math.ceil(x2)), min(frame.shape[0], math.ceil(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            ok, jpeg = cv2.imencode('.jpg', frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError('Không mã hoá được ảnh minh chứng.')
            samples[group['group_id']] = dict(signature=tuple(sorted(group['track_ids'])),
                jpeg=jpeg.tobytes(), bbox=[x1, y1, x2, y2])
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
                event_id = len(self.events) + 1
                event = dict(event_id=event_id, group_id=group['group_id'], track_ids=list(group['track_ids']),
                             behavior=behavior['name'], source=behavior['source'], status=behavior.get('status', 'suspected'),
                             confidence=behavior.get('confidence'), alert_frame=frame_index, alert_time=timestamp,
                             snapshot=None)
                self.events.append(event)
                # Trong cửa sổ đã mở: chỉ thêm sự kiện, không kéo dài clip.
                if not any(p['start'] <= timestamp < p['end'] for p in self.packages):
                    self.packages.append(dict(package_id=len(self.packages)+1, trigger_event_id=event_id,
                        requested_start=timestamp-self.config['pre_seconds'], start=max(0., timestamp-self.config['pre_seconds']),
                        end=timestamp+self.config['post_seconds']))
                    self.packages[-1]['samples'] = [sample for sample in self.history
                        if sample[1] >= self.packages[-1]['start']]
                    if self.save_video:
                        package = self.packages[-1]
                        folder = self.output / f"alert_{package['package_id']:06d}"
                        folder.mkdir(exist_ok=True)
                        height, width = frame.shape[:2]
                        writer = open_writer(folder / 'clip.mp4', fps, (width, height), self.video_config)
                        self.writers[package['package_id']] = writer
                        start = math.ceil(package['start'] * fps - 1e-7)
                        package['video_frames'] = 0
                        for index, buffered in self.frames:
                            if index >= start:
                                writer.write(buffered)
                                package['video_frames'] += 1
                        package['last_video_frame'] = frame_index
        self.pending = {key: value for key, value in self.pending.items() if key in observed}
        for package in self.packages:
            if 'evidence' in package:
                continue
            if timestamp < package['end']:
                writer = self.writers.get(package['package_id'])
                if writer is not None and package['last_video_frame'] != frame_index:
                    writer.write(frame)
                    package['video_frames'] += 1
                    package['last_video_frame'] = frame_index
                if not package['samples'] or package['samples'][-1][0] != frame_index:
                    package['samples'].append(self.history[-1])
            else:
                self._close_video(package['package_id'])
                self._seal(package, fps)
        while self.history and self.history[0][1] < timestamp - self.config['pre_seconds']:
            self.history.popleft()

    def _close_video(self, package_id):
        writer = self.writers.pop(package_id, None)
        if writer is not None:
            writer.release()
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
        folder = self.output / f"alert_{package['package_id']:06d}"
        folder.mkdir(exist_ok=True)
        ids = {e['group_id'] for e in self.events if package['start'] <= e['alert_time'] < package['end']}
        evidence = {}
        for gid in ids:
            runs = []
            for index, timestamp, samples in package['samples']:
                sample = samples.get(gid)
                if sample is None:
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
            (folder / name).write_bytes(sample['jpeg'])
            evidence[gid] = dict(group_id=gid, snapshot=name, frame=index, time=timestamp,
                bbox_xyxy=sample['bbox'], track_ids=list(sample['signature']),
                stable_start=run[0][1], stable_end=run[-1][1] + 1/fps,
                stable_seconds=len(run)/fps,
                behavior_names=list(dict.fromkeys(e['behavior'] for e in self.events
                    if e['group_id'] == gid and package['start'] <= e['alert_time'] < package['end'])))
        package['evidence'] = evidence
        del package['samples']

    def finalize(self, video_path, fps, frame_count):
        self.close()
        duration = frame_count / fps
        reports = []
        for package in self.packages:
            if 'evidence' not in package:
                self._seal(package, fps)
            folder = self.output / f"alert_{package['package_id']:06d}"
            folder.mkdir(exist_ok=True)
            # Cắt theo chỉ số frame để không lệch keyframe; không lấy frame ngoài khoảng yêu cầu.
            start = max(0, math.ceil(package['start'] * fps - 1e-7))
            stop = min(frame_count, math.ceil(package['end'] * fps - 1e-7))
            events = [e for e in self.events if package['start'] <= e['alert_time'] < min(package['end'], duration)]
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
                    raise RuntimeError('Số frame clip cảnh báo không khớp cửa sổ yêu cầu.')
                report['clip'] = 'clip.mp4'
            (folder / 'metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            reports.append(report)
        summary = dict(event_count=len(self.events), package_count=len(reports), save_video=self.save_video,
                       events=self.events, packages=reports, encoder_workers=self.video_stats)
        (self.output/'index.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary
