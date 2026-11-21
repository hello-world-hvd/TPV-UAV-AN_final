"""Xác nhận xe đứng yên bằng chuyển động trong cửa sổ, hysteresis và thời gian giữ qua gián đoạn."""
from collections import deque
import numpy as np
from tracking.motion import warp_box


class StationaryState:
    def __init__(self, config):
        self.config = config
        self.states = {}

    def update(self, tracks, timestamp, motion):
        c = self.config
        if motion.get('scene_cut'):
            self.states.clear()
        active = {r['track_id'] for r in tracks if r['label'] == 'xe máy'}
        for tid in list(self.states):
            if tid not in active and timestamp - self.states[tid]['last_valid'] > c['relation_grace_seconds']:
                del self.states[tid]
        for row in tracks:
            if row['label'] != 'xe máy':
                continue
            s = self.states.setdefault(row['track_id'], dict(history=deque(), low=0., high=0.,
                stopped=False, last_valid=timestamp, last_detection=None, speed=None))
            valid = (motion.get('reliable', False) and row['detection_age_seconds'] <= c['max_observation_age']
                     and row['position_source'] in ('detection', 'optical_flow', 'camera_motion'))
            reason = 'collecting_observations'
            if not valid:
                s['history'].clear()
                s['last_detection'] = None
                reason = 'camera_unreliable' if not motion.get('reliable') else 'observation_unavailable'
                if timestamp - s['last_valid'] > c['relation_grace_seconds']:
                    s.update(stopped=False, low=0., high=0., speed=None)
            else:
                s['last_valid'] = timestamp
                s['history'] = deque((t, warp_box(box, motion['matrix'])) for t, box in s['history'])
                if row['position_source'] == 'detection':
                    dt = timestamp - s['last_detection'] if s['last_detection'] is not None else 0.
                    if dt > c['max_observation_age']:
                        dt = 0.
                        s['history'].clear()
                    s['last_detection'] = timestamp
                    box = np.asarray(row.get('observed_bbox_xyxy', row['bbox_xyxy']), float)
                    s['history'].append((timestamp, box.copy()))
                    while len(s['history']) > 2 and timestamp - s['history'][0][0] > c['motion_window_seconds']:
                        s['history'].popleft()
                    samples = list(s['history'])
                    # Trung vị hai nửa cửa sổ giảm ảnh hưởng một bbox rung; không cộng dồn đường đi.
                    if len(samples) >= 3 and samples[-1][0] - samples[0][0] >= .3:
                        half = len(samples) // 2
                        first, last = samples[:half], samples[-half:]
                        center = lambda items: np.median([(b[:2] + b[2:]) / 2 for _, b in items], axis=0)
                        elapsed = np.median([t for t, _ in last]) - np.median([t for t, _ in first])
                        scale = max(1., float(np.median([np.linalg.norm(b[2:] - b[:2]) for _, b in samples])))
                        s['speed'] = float(np.linalg.norm(center(last) - center(first)) / scale / max(elapsed, .001))
                        if s['speed'] <= c['stationary_speed_ratio']:
                            s['low'] += dt
                            s['high'] = 0.
                            if s['low'] >= c['stationary_seconds']:
                                s['stopped'] = True
                        elif s['speed'] >= c['moving_speed_ratio']:
                            s['high'] += dt
                            s['low'] = 0.
                            if s['high'] >= c['moving_confirm_seconds']:
                                s['stopped'] = False
                        else:
                            s['high'] = 0.
                            s['low'] = 0.
            status = 'stationary' if s['stopped'] else 'pending' if valid else 'unknown'
            if not s['stopped'] and valid and s['high'] >= c['moving_confirm_seconds']:
                status = 'motion_threshold_exceeded'
                reason = 'sustained_motion'
            elif s['stopped'] and valid:
                reason = 'confirmed' if not s['high'] else 'checking_motion'
            row['stationary_info'] = dict(status=status, reason=reason, last_reset_reason=reason, stationary_verified=s['stopped'],
                observed_seconds=round(s['low'], 3), required_seconds=c['stationary_seconds'],
                detection_samples=len(s['history']), camera_reliable=bool(motion.get('reliable')),
                speed_ratio=s['speed'], moving_seconds=round(s['high'], 3))
