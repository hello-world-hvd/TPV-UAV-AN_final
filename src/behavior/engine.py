"""Gán tên cảnh báo nghi vấn theo nhóm; chỉ MobileNetV3 được dùng để phân loại sửa/rửa xe."""

from pathlib import Path
import math
import yaml

from behavior.classifier import ActivityClassifier, CLASSES


class BehaviorEngine:
    def __init__(self, config_path, device):
        path = Path(config_path).resolve()
        self.config = yaml.safe_load(path.read_text(encoding='utf-8'))
        c = self.config
        if not isinstance(c, dict) or type(c.get('enabled')) is not bool:
            raise ValueError('behavior.enabled phải là boolean.')
        for key in ('confidence_threshold', 'min_margin', 'sample_interval_seconds', 'max_result_age_seconds', 'crop_padding_ratio'):
            value = c.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Tham số hành vi không hợp lệ: {key}')
        if not 0 < c['confidence_threshold'] <= 1 or not 0 <= c['min_margin'] <= 1 or not 0 <= c['crop_padding_ratio'] <= .5:
            raise ValueError('Ngưỡng xác suất/margin/padding không hợp lệ.')
        if not 0 < c['sample_interval_seconds'] <= c['max_result_age_seconds']:
            raise ValueError('Chu kỳ phân loại phải >0 và <= tuổi kết quả.')
        self.weights = (path.parent / c['weights']).resolve()
        self.metadata = (path.parent / c['metadata']).resolve()
        if c['enabled'] and (not self.weights.is_file() or not self.metadata.is_file()):
            raise FileNotFoundError(f'Thiếu checkpoint/metadata hành vi: {self.weights}')
        self.device = device
        self.classifier = None
        self.cache = {}

    def update(self, frame, tracks, groups, timestamp):
        if not self.config['enabled']:
            return
        by_id = {r['track_id']: r for r in tracks}
        self.cache = {key: value for key, value in self.cache.items()
                      if timestamp - value['timestamp'] <= self.config['max_result_age_seconds']}
        pending = []
        for group in groups:
            members = [by_id[i] for i in group['track_ids'] if i in by_id]
            labels = {r['label'] for r in members}
            names = []
            if group['kind'] == 'đồ dùng và hàng hoá':
                names.append('hoạt động kinh doanh trên vỉa hè')
                if {'xô chậu', 'người'} <= labels:
                    names.append('đổ nước tẩy rửa ra vỉa hè')
            elif group['kind'] == 'xe rác và người lái ứng viên':
                names.append('tập kết rác sai quy định')
            elif group['kind'] == 'vật liệu xây dựng':
                names.append('tập kết vật liệu xây dựng sai quy định')
            elif group['kind'] in ('biển quảng cáo', 'biển treo trên cây hoặc cột ứng viên'):
                names.append('đặt biển báo sai quy định')
            elif 'xe chở rác' in labels:
                names.append('tập kết rác sai quy định')
            elif 'vật liệu xây dựng' in labels:
                names.append('tập kết vật liệu xây dựng sai quy định')
            elif 'biển quảng cáo' in labels:
                names.append('đặt biển báo sai quy định')
            group['behaviors'] = [dict(name=name, source='group_rule', status='suspected',
                                       reason='Chỉ suy ra từ thành phần nhóm; chưa xác minh hành động/vị trí/quy định.') for name in names]
            if (group['kind'] == 'người và xe máy đứng yên' and group['status'] == 'confirmed'
                    and group.get('stationary_verified') and {'người', 'xe máy'} <= labels):
                key = (group['group_id'], tuple(group['track_ids']))
                cached = self.cache.get(key)
                # Chỉ lấy crop mới khi cả người và xe có quan sát tin cậy, không phân loại prediction cũ.
                fresh = all(r['position_source'] == 'detection' for r in members)
                if fresh and (cached is None or timestamp - cached['timestamp'] >= self.config['sample_interval_seconds']):
                    pending.append((key, group))
                elif cached is not None:
                    self._attach(group, cached)
        if pending:
            if self.classifier is None:
                self.classifier = ActivityClassifier(self.weights, self.metadata, self.device)
            for start in range(0, len(pending), 8):
                chunk = pending[start:start+8]
                probabilities = self.classifier.predict(frame, [g['bbox_xyxy'] for _, g in chunk], self.config['crop_padding_ratio'])
                for (key, group), scores in zip(chunk, probabilities):
                    result = {'timestamp': timestamp, 'probabilities': dict(zip(CLASSES, scores))}
                    self.cache[key] = result
                    self._attach(group, result)

    def _attach(self, group, result):
        scores = result['probabilities']
        ranked = sorted(scores, key=scores.get, reverse=True)
        label = ranked[0]
        accepted = (label != 'other_activity' and scores[label] >= self.config['confidence_threshold']
                    and scores[label] - scores[ranked[1]] >= self.config['min_margin'])
        group['activity_classification'] = {**result, 'class': label, 'accepted': accepted}
        if accepted:
            name = {'repairing_vehicle': 'sửa xe', 'washing_vehicle': 'rửa xe'}[label]
            group['behaviors'].append(dict(name=name, source='mobilenetv3', confidence=scores[label],
                                          status='suspected', sampled_at=result['timestamp']))
