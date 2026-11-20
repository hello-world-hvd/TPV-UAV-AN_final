"""Tạo 5 nhóm theo quan hệ ảnh; xe máy đứng yên phải có detection và bù camera tin cậy."""

import numpy as np
from tracking.motion import warp_box


def envelope(rows):
    boxes = np.asarray([r['bbox_xyxy'] for r in rows], float)
    return np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)]


def gap(a, b):
    return float(np.linalg.norm(np.maximum(0, np.maximum(a[:2] - b[2:], b[:2] - a[2:]))))


class Grouper:
    def __init__(self, config):
        for name in ('confirm_seconds', 'max_observation_age', 'object_gap_ratio', 'cluster_span_ratio',
                     'person_gap_height_ratio', 'stationary_seconds', 'stationary_displacement_ratio', 'attachment_gap_ratio'):
            value = config.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} phải là số dương hữu hạn.')
        for name, default in {'chair_gap_ratio': 1.5, 'person_confirm_seconds': 1.0,
                              'person_foot_height_ratio': 1.5, 'stationary_step_ratio': .1,
                              'stationary_path_ratio': .3}.items():
            config.setdefault(name, default)
            if isinstance(config[name], bool) or not isinstance(config[name], (int, float)) or not np.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f'{name} phải là số dương hữu hạn.')
        self.person_links = {}
        self.config = config
        self.groups = {}
        self.stationary = {}
        self.next_id = 1

    def update(self, tracks, timestamp, motion):
        if motion.get('scene_cut'):
            self.person_links.clear()
            self.groups.clear()
            self.stationary.clear()
        valid = [r for r in tracks if r['detection_age_seconds'] <= self.config['max_observation_age']
                 and r['position_source'] in ('detection', 'optical_flow', 'camera_motion')]
        by_id = {r['track_id']: r for r in valid}
        # Tích luỹ bù camera lên vị trí neo; prediction không được tự xác nhận đứng yên.
        for tid in list(self.stationary):
            state = self.stationary[tid]
            if tid not in by_id or not motion.get('reliable'):
                del self.stationary[tid]
            else:
                state['anchor'] = warp_box(state['anchor'], motion['matrix'])
                state['previous'] = warp_box(state['previous'], motion['matrix'])
        for row in valid:
            if row['label'] != 'xe máy' or row['position_source'] != 'detection':
                continue
            tid = row['track_id']
            box = np.asarray(row.get('observed_bbox_xyxy', row['bbox_xyxy']), float)
            state = self.stationary.get(tid)
            if state is not None:
                anchor = state['anchor']
                scale = max(np.linalg.norm(anchor[2:] - anchor[:2]), 1)
                shift = np.linalg.norm((box[:2] + box[2:] - anchor[:2] - anchor[2:]) / 2) / scale
                delta = np.linalg.norm((box[:2] + box[2:] - state['previous'][:2] - state['previous'][2:]) / 2) / scale
                state['path'] += delta
                if (shift > self.config['stationary_displacement_ratio'] or delta > self.config['stationary_step_ratio']
                        or state['path'] > self.config['stationary_path_ratio']):
                    state = None
            if state is None:
                self.stationary[tid] = {'anchor': box.copy(), 'previous': box.copy(), 'path': 0., 'since': timestamp, 'samples': 1, 'last_sample': timestamp}
            else:
                state['samples'] += 1
                state['previous'] = box.copy()
                state['last_sample'] = timestamp

        people = [r for r in valid if r['label'] == 'người']
        candidates = []
        def add(kind, rows, anchor, eligible=True):
            candidates.append({'kind': kind, 'rows': rows, 'anchor': anchor, 'eligible': eligible})

        # Cụm bàn ghế/hàng hoá: giới hạn đường kính để tránh nối dây chuyền cả khu vực.
        anchors = {'bàn nhựa', 'xe đẩy bán hàng'}
        remaining = sorted([r for r in valid if r['label'] in
                            {'xô chậu', 'hàng hoá', 'bàn nhựa', 'ghế nhựa', 'xe đẩy bán hàng'}],
                           key=lambda r: (r['label'] not in anchors, r['bbox_xyxy'][0], r['bbox_xyxy'][1], r['track_id']))
        while remaining:
            members = [remaining.pop(0)]
            anchor = members[0] if members[0]['label'] in anchors else None
            changed = True
            while changed:
                changed = False
                for row in remaining[:]:
                    if anchor is not None and row['label'] in anchors:
                        continue
                    box = np.asarray(row['bbox_xyxy'])
                    boxes = np.asarray([r['bbox_xyxy'] for r in members + [row]])
                    diagonals = np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1)
                    scale = max(float(np.median(diagonals)), 1)
                    def near(member):
                        other = np.asarray(member['bbox_xyxy'])
                        chair = 'ghế nhựa' in (row['label'], member['label'])
                        ratio = self.config['chair_gap_ratio'] if chair else self.config['object_gap_ratio']
                        reference = np.asarray(anchor['bbox_xyxy']) if anchor is not None else None
                        local_scale = np.linalg.norm(reference[2:] - reference[:2]) if reference is not None else scale
                        return gap(box, other) <= ratio * max(1, local_scale)
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                    if (any(near(m) for m in members) and (anchor is None or near(anchor)) and
                            np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max() <= self.config['cluster_span_ratio'] * scale):
                        members.append(row)
                        remaining.remove(row)
                        changed = True
            add('đồ dùng và hàng hoá', members, anchor['track_id'] if anchor else min(r['track_id'] for r in members))

        for row in valid:
            tid, label = row['track_id'], row['label']
            if label == 'xe máy':
                state = self.stationary.get(tid)
                stopped = bool(state and state['samples'] >= 2 and timestamp - state['since'] >= self.config['stationary_seconds']
                               and timestamp - state['last_sample'] <= self.config['max_observation_age'])
                if stopped:
                    add('người và xe máy đứng yên', [row], tid)
            elif label == 'xe chở rác':
                add('xe rác và người lái ứng viên', [row], tid, False)
            elif label == 'vật liệu xây dựng':
                add('vật liệu xây dựng', [row], tid)
            elif label == 'biển quảng cáo':
                box = np.asarray(row['bbox_xyxy'])
                supports = []
                for support in valid:
                    if support['label'] not in ('thân cây', 'cột điện'):
                        continue
                    other = np.asarray(support['bbox_xyxy'])
                    distance = gap(box, other) / max(1, np.linalg.norm(box[2:] - box[:2]))
                    if distance <= self.config['attachment_gap_ratio']:
                        supports.append((distance, support['track_id'], support))
                if supports:
                    support = min(supports, key=lambda item: item[:2])[2]
                    add('biển treo trên cây hoặc cột ứng viên', [row, support], tid, False)

        # Một người chỉ ghép vào một nhóm gần nhất; không dùng người làm cầu nối cụm.
        next_links = {}
        for person in people:
            p = np.asarray(person['bbox_xyxy'])
            height = max(p[3] - p[1], 1)
            choices = []
            for i, candidate in enumerate(candidates):
                if candidate['kind'] not in ('đồ dùng và hàng hoá', 'người và xe máy đứng yên', 'xe rác và người lái ứng viên'):
                    continue
                objects = [r for r in candidate['rows'] if r['label'] != 'người']
                box = envelope(objects)
                distance = min(gap(p, np.asarray(r['bbox_xyxy'])) for r in objects) / height
                feet_close = any(abs(p[3] - r['bbox_xyxy'][3]) / height <= self.config['person_foot_height_ratio'] for r in objects)
                if candidate['kind'] == 'xe rác và người lái ứng viên':
                    # Chỉ ứng viên khi tâm người nằm trong phần trên bbox xe, không khẳng định người lái.
                    center = (p[:2] + p[2:]) / 2
                    feet_close = box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[1] + .7 * (box[3] - box[1])
                if distance <= self.config['person_gap_height_ratio'] and feet_close:
                    choices.append((distance, i))
            if choices:
                candidate = candidates[min(choices)[1]]
                key = (person['track_id'], candidate['kind'], candidate['anchor'])
                first = self.person_links.get(key, timestamp)
                next_links[key] = first
                if timestamp - first >= self.config['person_confirm_seconds']:
                    candidate['rows'].append(person)
        self.person_links = next_links

        previous = self.groups
        current, output = {}, []
        for candidate in candidates:
            rows = candidate['rows']
            if candidate['kind'] == 'người và xe máy đứng yên' and not any(r['label'] == 'người' for r in rows):
                continue
            key = (candidate['kind'], candidate['anchor'])
            old = previous.get(key)
            if old is None:
                old = {'id': self.next_id, 'first': timestamp}
                self.next_id += 1
            current[key] = old
            confirmed = candidate['eligible'] and timestamp - old['first'] >= self.config['confirm_seconds']
            output.append({'group_id': old['id'], 'kind': candidate['kind'], 'track_ids': sorted(r['track_id'] for r in rows),
                           'bbox_xyxy': envelope(rows).tolist(),
                           'status': 'confirmed' if confirmed else 'candidate',
                           'stationary_verified': candidate['eligible'] if candidate['kind'] == 'người và xe máy đứng yên' else None})
        self.groups = current
        return output
