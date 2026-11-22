"""Tạo 5 nhóm theo quan hệ ảnh; xe máy đứng yên phải có detection và bù camera tin cậy."""

import cv2
import numpy as np
from shapely.geometry import Point
from grouping.stationary import StationaryState
from core.geometry import distance, diagonal, center, coverage, polygon, dimensions, shape


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
        people = [r for r in valid if r['label'] == 'Người']
        candidates = []
        def add(kind, rows, anchor, eligible=True):
            candidates.append({'kind': kind, 'rows': rows, 'anchor': anchor, 'eligible': eligible})

        # Cụm bàn ghế/hàng hoá: giới hạn đường kính để tránh nối dây chuyền cả khu vực.
        anchors = {'Bàn', 'Xe đẩy bán hàng'}
        remaining = sorted([r for r in valid if r['label'] in
                            {'Xô', 'Hàng hóa bày bán', 'Bàn', 'Ghế', 'Xe đẩy bán hàng'}],
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
                    diagonals = [diagonal(r) for r in members + [row]]
                    scale = max(float(np.median(diagonals)), 1)
                    def near(member):
                        other = np.asarray(member['bbox_xyxy'])
                        chair = 'Ghế' in (row['label'], member['label'])
                        ratio = self.config['chair_gap_ratio'] if chair else self.config['object_gap_ratio']
                        reference = np.asarray(anchor['bbox_xyxy']) if anchor is not None else None
                        local_scale = diagonal(anchor) if anchor is not None else scale
                        return distance(row, member) <= ratio * max(1, local_scale)
                    centers = np.asarray([center(r) for r in members + [row]])
                    if (any(near(m) for m in members) and (anchor is None or near(anchor)) and
                            np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max() <= self.config['cluster_span_ratio'] * scale):
                        members.append(row)
                        remaining.remove(row)
                        changed = True
            add('đồ dùng và hàng hoá', members, anchor['track_id'] if anchor else min(r['track_id'] for r in members))

        for row in valid:
            tid, label = row['track_id'], row['label']
            if label == 'Xe máy':
                stopped = row.get('stationary_info', {}).get('stationary_verified', False)
                # Đếm quan hệ ngay cả khi xe chưa đủ thời gian đứng yên; chỉ xuất sau khi đủ.
                add('người và xe máy đứng yên', [row], tid, stopped)
            elif label == 'Máy móc':
                add('Người và Máy móc', [row], tid)
            elif label == 'Phương tiện thu gom, vận chuyển rác':
                add('xe rác và người lái ứng viên', [row], tid, False)
            elif label == 'Vật liệu xây dựng':
                add('Vật liệu xây dựng', [row], tid)
            elif label == 'Biển quảng cáo':
                box = np.asarray(row['bbox_xyxy'])
                # Biển nằm phần lớn trong vùng bán hàng thì thuộc nhóm này, không tạo nhóm biển riêng.
                contained = []
                area = max(1., float(np.prod(np.maximum(0, box[2:] - box[:2]))))
                for candidate in candidates:
                    if candidate['kind'] != 'đồ dùng và hàng hoá':
                        continue
                    overlap = coverage(row, [r for r in candidate['rows'] if r['label'] != 'Biển quảng cáo'])
                    if overlap >= .5:
                        contained.append((overlap, candidate))
                if contained:
                    max(contained, key=lambda item: item[0])[1]['rows'].append(row)
                    continue
                supports = []
                for support in valid:
                    if support['label'] not in ('Thân cây', 'Cột điện'):
                        continue
                    other = np.asarray(support['bbox_xyxy'])
                    separation = distance(row, support) / diagonal(row)
                    if separation <= self.config['attachment_gap_ratio']:
                        supports.append((separation, support['track_id'], support))
                if supports:
                    support = min(supports, key=lambda item: item[:2])[2]
                    add('biển treo trên cây hoặc cột ứng viên', [row, support], tid, False)
                else:
                    add('Biển quảng cáo', [row], tid, False)

        # Một người chỉ ghép vào một nhóm gần nhất; không dùng người làm cầu nối cụm.
        next_links = {key: value for key, value in self.person_links.items()
                      if timestamp - value['last'] <= self.config['relation_grace_seconds']}
        for person in people:
            p = np.asarray(person['bbox_xyxy'])
            height = max(dimensions(person)) if "bbox_polygon" in person else max(p[3]-p[1],1)
            height = max(height,1)
            choices = []
            for i, candidate in enumerate(candidates):
                if candidate['kind'] not in ('đồ dùng và hàng hoá', 'người và xe máy đứng yên', 'xe rác và người lái ứng viên', 'Người và Máy móc'):
                    continue
                objects = [r for r in candidate['rows'] if r['label'] != 'Người']
                box = envelope(objects)
                separation = min(distance(person, r) for r in objects) / height
                feet_close = any(abs(p[3] - r['bbox_xyxy'][3]) / height <= self.config['person_foot_height_ratio'] for r in objects)
                if candidate['kind'] == 'xe rác và người lái ứng viên':
                    # Chỉ ứng viên khi tâm người nằm trong phần trên bbox xe, không khẳng định người lái.
                    person_center = polygon(person).mean(0)
                    feet_close = shape(objects[0]).covers(Point(person_center))
                    feet_close = feet_close and person_center[1] <= box[1] + .7*(box[3]-box[1])
                if separation <= self.config['person_gap_height_ratio'] and feet_close:
                    choices.append((separation, i))
            # Người–xe được xét riêng, không để cụm hàng hoá chiếm mất quan hệ này.
            for relation in ('objects', 'bike', 'machinery'):
                selected = [(distance, i) for distance, i in choices
                            if ('bike' if candidates[i]['kind'] == 'người và xe máy đứng yên' else
                                'machinery' if candidates[i]['kind'] == 'Người và Máy móc' else 'objects') == relation]
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
        current = {key: value for key, value in previous.items()
                   if timestamp - value.get('last', timestamp) <= self.config['relation_grace_seconds']}
        output, used_ids = [], set()
        candidates.sort(key=lambda c: -len(c['rows']))
        for candidate in candidates:
            rows = candidate['rows']
            if candidate['kind'] == 'Người và Máy móc' and not any(r['label'] == 'Người' for r in rows):
                continue
            if candidate['kind'] == 'người và xe máy đứng yên' and (not candidate['eligible'] or not any(r['label'] == 'Người' for r in rows)):
                continue
            key = (candidate['kind'], candidate['anchor'])
            ids = {r['track_id'] for r in rows}
            old = current.get(key)
            if old is not None and old['id'] in used_ids:
                old = None
            if old is None:
                def family(kind):
                    return 'sign' if kind in ('Biển quảng cáo', 'biển treo trên cây hoặc cột ứng viên') else kind
                matches = [(len(ids & value.get('members', set())), value)
                           for (kind, _), value in current.items()
                           if value['id'] not in used_ids and family(kind) == family(candidate['kind'])
                           and ids & value.get('members', set())]
                if matches:
                    old = max(matches, key=lambda item: item[0])[1]
            if old is None:
                old = {'id': self.next_id, 'first': timestamp}
                self.next_id += 1
            used_ids.add(old['id'])
            current = {k: v for k, v in current.items() if v['id'] != old['id']}
            old = {**old, 'last': timestamp, 'members': ids}
            current[key] = old
            confirmed = candidate['eligible'] and timestamp - old['first'] >= self.config['confirm_seconds']
            output.append({'group_id': old['id'], 'kind': candidate['kind'], 'track_ids': sorted(r['track_id'] for r in rows),
                           'bbox_xyxy': envelope(rows).tolist(),
                           'status': 'confirmed' if confirmed else 'candidate',
                           'stationary_verified': candidate['eligible'] if candidate['kind'] == 'người và xe máy đứng yên' else None})
            if any('bbox_polygon' in r for r in rows):
                # Box nhóm bao trọn các polygon; crop phân loại vẫn dùng AABB bao ngoài.
                points = np.concatenate([polygon(r) for r in rows]).astype(np.float32)
                output[-1]['bbox_polygon'] = cv2.boxPoints(cv2.minAreaRect(points)).tolist()
        self.groups = current
        return output
