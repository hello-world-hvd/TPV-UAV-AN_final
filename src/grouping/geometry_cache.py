"""Cache hình học trong một frame; lọc xa bảo thủ bằng AABB trước khi đo polygon chính xác."""
import numpy as np
from shapely.ops import unary_union
from core.geometry import polygon, shape


class FrameGeometry:
    def __init__(self):
        self.rows = {}
        self.distances = {}
        self.regions = {}

    def data(self, row):
        key = id(row)
        if key not in self.rows:
            p = polygon(row)
            sizes = (np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[1]))
            self.rows[key] = dict(polygon=p, dimensions=sizes, center=p.mean(0),
                diagonal=max(1., float(np.hypot(*sizes))), bounds=np.r_[p.min(0), p.max(0)])
        return self.rows[key]

    def polygon(self, row):
        return self.data(row)['polygon']

    def dimensions(self, row):
        return self.data(row)['dimensions']

    def diagonal(self, row):
        return self.data(row)['diagonal']

    def center(self, row):
        return self.data(row)['center']

    def shape(self, row):
        data = self.data(row)
        if 'shape' not in data:
            data['shape'] = shape(row)
        return data['shape']

    def distance(self, a, b, limit=None):
        key = tuple(sorted((id(a), id(b))))
        if key in self.distances:
            return self.distances[key]
        if limit is not None:
            x, y = self.data(a)['bounds'], self.data(b)['bounds']
            # shape() làm tròn lưới 0.001px: nới mỗi mép 0.001px để không loại nhầm sát ngưỡng.
            delta = np.maximum(0., np.maximum(x[:2]-y[2:], y[:2]-x[2:])-.002)
            if np.isfinite(delta).all() and np.linalg.norm(delta) > limit:
                return float('inf')  # Không cache kết quả lọc: lần sau có thể dùng ngưỡng lớn hơn.
        value = float(self.shape(a).distance(self.shape(b)))
        self.distances[key] = value
        return value

    def coverage(self, row, rows):
        key = tuple(sorted(id(r) for r in rows))
        if key not in self.regions:
            self.regions[key] = unary_union([self.shape(r) for r in rows]).convex_hull
        region, geometry = self.regions[key], self.shape(row)
        return geometry.intersection(region).area / max(geometry.area, 1e-6)
