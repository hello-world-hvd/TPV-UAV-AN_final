"""Tạo 5 nhóm theo quan hệ ảnh; xe máy đứng yên phải có detection và bù camera tin cậy."""

import numpy as np
from grouping.stationary import StationaryState


def envelope(rows):
    boxes = np.asarray([r['bbox_xyxy'] for r in rows], float)
    return np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)]


def gap(a, b):
    return float(np.linalg.norm(np.maximum(0, np.maximum(a[:2] - b[2:], b[:2] - a[2:]))))


class Grouper:
    def __init__(self, config):
        for name in ('confirm_seconds', 'max_observation_age', 'object_gap_ratio', 'cluster_span_ratio',
                     'person_gap_height_ratio', 'stationary_seconds', 'attachment_gap_ratio'):
            value = config.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} phải là số dương hữu hạn.')
        for name, default in {'chair_gap_ratio': 1.5, 'person_confirm_seconds': 1.0,
                              'person_foot_height_ratio': 1.5, 'motion_window_seconds': 1.5, 'stationary_speed_ratio': .08,
                              'moving_speed_ratio': .2, 'moving_confirm_seconds': .5, 'relation_grace_seconds': .5}.items():
            config.setdefault(name, default)
            if isinstance(config[name], bool) or not isinstance(config[name], (int, float)) or not np.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f'{name} phải là số dương hữu hạn.')
        self.person_links = {}
        self.last_timestamp = None
        self.config = config
        self.groups = {}
        self.stationary = StationaryState(config)
        if config["stationary_speed_ratio"] >= config["moving_speed_ratio"]:
            raise ValueError("Ngưỡng đứng yên phải thấp hơn ngưỡng di chuyển.")
        self.next_id = 1

    def update(self, tracks, timestamp, motion):
        if motion.get('scene_cut'):
            self.person_links.clear()
            self.groups.clear()
        self.stationary.update(tracks, timestamp, motion)
        valid = [r for r in tracks if r['detection_age_seconds'] <= self.config['max_observation_age']
                 and r['position_source'] in ('detection', 'optical_flow', 'camera_motion')]
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
                stopped = row.get('stationary_info', {}).get('stationary_verified', False)
                # Đếm quan hệ ngay cả khi xe chưa đủ thời gian đứng yên; chỉ xuất sau khi đủ.
                add('người và xe máy đứng yên', [row], tid, stopped)
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
                else:
                    add('biển quảng cáo', [row], tid, False)

        # Một người chỉ ghép vào một nhóm gần nhất; không dùng người làm cầu nối cụm.
        next_links = {key: value for key, value in self.person_links.items()
                      if timestamp - value['last'] <= self.config['relation_grace_seconds']}
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
            # Người–xe được xét riêng, không để cụm hàng hoá chiếm mất quan hệ này.
            for bike_relation in (False, True):
                selected = [(distance, i) for distance, i in choices
                            if (candidates[i]['kind'] == 'người và xe máy đứng yên') == bike_relation]
                if not selected:
                    continue
                candidate = candidates[min(selected)[1]]
                key = (person['track_id'], candidate['kind'], candidate['anchor'])
                link = self.person_links.get(key)
                if link is None or timestamp - link['last'] > self.config['relation_grace_seconds']:
                    link = {'last': timestamp, 'seconds': 0.}
                else:
                    # Không cộng thời gian ở các frame không thấy quan hệ.
                    link = {'last': timestamp, 'seconds': link['seconds'] + (
                        timestamp - link['last'] if link.get('frame') == self.last_timestamp else 0.)}
                link['frame'] = timestamp
                next_links[key] = link
                if link['seconds'] >= self.config['person_confirm_seconds']:
                    candidate['rows'].append(person)
        self.person_links = next_links
        self.last_timestamp = timestamp

        previous = self.groups
        current, output = {}, []
        for candidate in candidates:
            rows = candidate['rows']
            if candidate['kind'] == 'người và xe máy đứng yên' and (not candidate['eligible'] or not any(r['label'] == 'người' for r in rows)):
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
