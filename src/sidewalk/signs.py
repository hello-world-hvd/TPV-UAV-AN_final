"""Xác nhận biển treo cây/cột trước; biển đặt cần chân chạm vỉa hè. Cache/warp mask, không dùng vùng chưa chắc làm bằng chứng."""
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.geometry import polygon, shape, diagonal
from sidewalk.detector import SidewalkDetector

SIGN_KINDS = {'Biển quảng cáo', 'biển treo trên cây hoặc cột ứng viên'}


class SignVerifier:
    def __init__(self, config_path):
        self.path = Path(config_path)
        self.config = yaml.safe_load(self.path.read_text())['signs']
        c = self.config
        for key in ('segment_fps','max_mask_age_seconds','confirm_seconds','contact_radius_ratio',
                    'min_contact_ratio','hanging_gap_ratio'):
            if not np.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f'sidewalk.signs.{key} phải dương hữu hạn.')
        if c['min_contact_ratio'] > 1:
            raise ValueError('min_contact_ratio phải <=1.')
        self.model = None
        self.reset()

    def reset(self):
        """Chỉ xoá lịch sử, giữ model đã nạp qua reconnect/reset."""
        self.labels = None
        self.mask_time = None
        self.last_inference = None
        self.relations = {}
        self.previous_time = None

    def _hanging(self, sign, supports):
        s = shape(sign)
        center_y = polygon(sign).mean(0)[1]
        for support in supports:
            # Gần cây/cột chỉ là ứng viên: yêu cầu sát thực và biển cao hơn chân giá đỡ.
            if (s.distance(shape(support)) <= self.config['hanging_gap_ratio']*diagonal(sign)
                    and center_y < support['bbox_xyxy'][3]-.15*diagonal(sign)):
                return True
        return False

    def _contact(self, sign):
        if self.labels is None:
            return None
        points = polygon(sign)
        edges = [(points[i],points[(i+1)%4]) for i in range(4)]
        a,b = max(edges,key=lambda pair: float((pair[0][1]+pair[1][1])/2))
        # Dải quanh cạnh dưới, cho phép biển che mask; ngoài ảnh không coi là vỉa hè.
        radius = max(2,round(diagonal(sign)*self.config['contact_radius_ratio']))
        x1,y1 = np.floor(np.minimum(a,b)-radius).astype(int)
        x2,y2 = np.ceil(np.maximum(a,b)+radius+1).astype(int)
        h,w = self.labels.shape
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
        if x2 <= x1 or y2 <= y1:
            return None
        band = np.zeros((y2-y1,x2-x1),np.uint8)
        cv2.line(band,tuple(np.rint(a-[x1,y1]).astype(int)),
                 tuple(np.rint(b-[x1,y1]).astype(int)),255,2*radius+1)
        values = self.labels[y1:y2,x1:x2][band != 0]
        if not len(values):
            return None
        ratio = float(np.count_nonzero(values == 2)/len(values))
        if ratio >= self.config['min_contact_ratio']:
            return True
        # Không đủ bằng chứng khi vùng quanh chân phần lớn chưa chắc chắn.
        return False if np.count_nonzero(values != 255)/len(values) >= .5 else None

    def update(self, frame, tracks, groups, timestamp, motion=None, resume=False):
        motion = motion or {}
        if resume or motion.get('scene_cut'):
            self.reset()
        by_id = {r['track_id']:r for r in tracks}
        candidates = []
        seen = set()
        for group in groups:
            if group['kind'] not in SIGN_KINDS:
                continue
            members = [by_id[i] for i in group['track_ids'] if i in by_id]
            sign = next((r for r in members if r['label'] == 'Biển quảng cáo'),None)
            if (sign is None or sign.get('position_source') not in ('detection','camera_motion','optical_flow')
                    or sign.get('detection_age_seconds',0) > .5):
                continue  # Không dùng prediction thuần hoặc track quá cũ.
            supports = [r for r in members if r['label'] in ('Cột điện','Thân cây')
                        and r.get('position_source') in ('detection','camera_motion','optical_flow')
                        and r.get('detection_age_seconds',0) <= .5]
            candidates.append((group,sign,self._hanging(sign,supports)))
        needs_mask = any(not hung for _,_,hung in candidates)
        if not needs_mask:
            self.labels = None  # Không tích luỹ warp khi không có biển cần kiểm tra.
        elif self.labels is not None:
            matrix = np.asarray(motion.get('matrix',[]),dtype=float)
            if (self.labels.shape != frame.shape[:2] or not motion.get('reliable')
                    or matrix.shape != (2,3) or not np.isfinite(matrix).all()):
                self.labels = None
            else:
                self.labels = cv2.warpAffine(self.labels,matrix,(frame.shape[1],frame.shape[0]),
                    flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        if needs_mask:
            due = self.last_inference is None or timestamp-self.last_inference >= 1/self.config['segment_fps']-1e-8
            if due:
                if self.model is None:
                    self.model = SidewalkDetector(self.path)
                self.labels = self.model.predict(frame)['labels']
                self.mask_time = self.last_inference = timestamp
            elif self.mask_time is None or timestamp-self.mask_time > self.config['max_mask_age_seconds']:
                self.labels = None
        decisions = {}
        for group,sign,hung in candidates:
            contact = None if hung else self._contact(sign)
            name = 'Treo biển báo' if hung else 'Đặt biển báo' if contact is True else None
            group['sign_verification'] = dict(hanging=hung,sidewalk_contact=contact,
                mask_age_seconds=None if self.labels is None else timestamp-self.mask_time)
            key = group['group_id']
            seen.add(key)
            previous = self.relations.get(key)
            seconds = 0.
            if (name is not None and previous is not None and previous['name'] == name
                    and previous['last'] == self.previous_time):
                seconds = previous['seconds']+timestamp-previous['last']
            self.relations[key] = dict(name=name,seconds=seconds,last=timestamp)
            if name and seconds+1e-8 >= self.config['confirm_seconds']:
                decisions[key] = name
                group['status'] = 'confirmed'
        self.relations = {key:value for key,value in self.relations.items() if key in seen}
        self.previous_time = timestamp
        return decisions
