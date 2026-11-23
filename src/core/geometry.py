"""Hình học dùng chung AABB/OBB: polygon, khoảng cách mép, giao diện tích, kích thước và biến đổi camera."""
import cv2
import numpy as np
from shapely import set_precision
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union


def polygon(row):
    if 'bbox_polygon' in row:
        return np.asarray(row['bbox_polygon'], np.float32)
    x1,y1,x2,y2 = row['bbox_xyxy']
    return np.asarray([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float32)


def shape(row):
    # Track dự đoán có thể bị ép rất mỏng hoặc sai thứ tự đỉnh sau bù camera.
    # Bao lồi sắp lại đỉnh; lưới 0.001 pixel loại sai số topo, không làm đổi rule.
    points = np.asarray(polygon(row), dtype=float).reshape(-1,2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return Polygon()
    geometry = MultiPoint(points).convex_hull
    if geometry.geom_type != 'Polygon':
        return Polygon()
    return set_precision(geometry, grid_size=.001)


def dimensions(row):
    p = polygon(row)
    return np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[1])


def diagonal(row):
    return max(1., float(np.hypot(*dimensions(row))))


def center(row):
    return polygon(row).mean(0)


def distance(a, b):
    return float(shape(a).distance(shape(b)))


def coverage(row, rows):
    # Vùng nhóm là bao lồi của polygon thành viên, tránh góc trống của AABB bao ngoài.
    region = unary_union([shape(r) for r in rows]).convex_hull
    s = shape(row)
    return s.intersection(region).area / max(s.area, 1e-6)


def transform(points, matrix):
    return np.asarray(points) @ matrix[:, :2].T + matrix[:, 2]


def attach_polygon(row, points):
    p = np.asarray(points, np.float32)
    row['bbox_polygon'] = p.tolist()
    row['bbox_xyxy'] = np.r_[p.min(0), p.max(0)].tolist()
    (x,y),(w,h),angle = cv2.minAreaRect(p)
    row['bbox_xywhr'] = [x,y,w,h,float(np.deg2rad(angle))]
    return row
