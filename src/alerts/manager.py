"""Chống lặp cảnh báo; gom sự kiện trong cửa sổ cố định và cắt clip viz sau khi video ghi xong."""

import json
import math
from pathlib import Path
import subprocess
import shutil

import cv2
import imageio_ffmpeg
import yaml


class AlertManager:
    def __init__(self, output, config_path, save_video=True):
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
                image = self.output / 'events' / f'event_{event_id:06d}.jpg'
                image.parent.mkdir(exist_ok=True)
                if not cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f'Không lưu được ảnh cảnh báo: {image}')
                event = dict(event_id=event_id, group_id=group['group_id'], track_ids=list(group['track_ids']),
                             behavior=behavior['name'], source=behavior['source'], status=behavior.get('status', 'suspected'),
                             confidence=behavior.get('confidence'), alert_frame=frame_index, alert_time=timestamp,
                             snapshot=str(image.relative_to(self.output)))
                self.events.append(event)
                # Trong cửa sổ đã mở: chỉ thêm sự kiện, không kéo dài clip.
                if not any(p['start'] <= timestamp < p['end'] for p in self.packages):
                    self.packages.append(dict(package_id=len(self.packages)+1, trigger_event_id=event_id,
                        requested_start=timestamp-self.config['pre_seconds'], start=max(0., timestamp-self.config['pre_seconds']),
                        end=timestamp+self.config['post_seconds']))
        self.pending = {key: value for key, value in self.pending.items() if key in observed}

    def finalize(self, video_path, fps, frame_count):
        duration = frame_count / fps
        reports = []
        for package in self.packages:
            folder = self.output / f"alert_{package['package_id']:06d}"
            folder.mkdir(exist_ok=True)
            # Cắt theo chỉ số frame để không lệch keyframe; không lấy frame ngoài khoảng yêu cầu.
            start = max(0, math.ceil(package['start'] * fps - 1e-7))
            stop = min(frame_count, math.ceil(package['end'] * fps - 1e-7))
            events = [e for e in self.events if package['start'] <= e['alert_time'] < min(package['end'], duration)]
            package_events = []
            for event in events:
                snapshot = Path(event['snapshot']).name
                shutil.copy2(self.output / event['snapshot'], folder / snapshot)
                package_events.append({**event, 'snapshot': snapshot})
            report = {**package, 'actual_start': start/fps, 'actual_end': stop/fps,
                      'missing_pre_seconds': max(0., -package['requested_start']),
                      'missing_post_seconds': max(0., package['end']-duration),
                      'fps': fps, 'behavior_names': list(dict.fromkeys(e['behavior'] for e in events)),
                      'events': package_events,
                      'clip': None, 'video_saved': self.save_video}
            if self.save_video and stop > start:
                target = folder / 'clip.mp4'
                command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                           '-i', str(video_path), '-vf', f'trim=start_frame={start}:end_frame={stop},setpts=PTS-STARTPTS',
                           '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', str(target)]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f'Cắt clip thất bại: {result.stderr[-2000:]}')
                report['clip'] = 'clip.mp4'
            (folder / 'metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            reports.append(report)
        summary = dict(event_count=len(self.events), package_count=len(reports), save_video=self.save_video,
                       events=self.events, packages=reports)
        (self.output/'index.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary
