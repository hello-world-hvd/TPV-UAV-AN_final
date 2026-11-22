"""Nạp model YOLO detection theo cấu hình, lọc đúng nhãn yêu cầu và gộp các box trùng cùng nhãn."""

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.model import Model

from detection.labels import PHRASE_LABELS
from detection.predictor import BoxesOnlyPredictor


class Detector:
    def __init__(self, config):
        inference = config["inference"]
        self.device = inference["device"]
        self.confidence = inference["confidence"]
        self.iou = inference["nms_iou"]
        self.merge_iou = inference["merge_iou"]
        self.merge_ios = inference.get("merge_ios", 0.9)
        self.imgsz = inference["imgsz"]
        self.rect = inference.get("rect", True)
        self.max_detections = inference["max_detections"]
        self.thresholds = config["thresholds"]
        self.bbox_limits = config.get('bbox_limits', {})
        self.minimum_confidence = min([self.confidence, *self.thresholds.values()])
        self.models = []
        self.parallel_models = inference.get("parallel_models", False)
        self.parallel = None
        self.lock = Lock()
        if self.parallel_models and (not (str(self.device).isdigit() or str(self.device).startswith('cuda'))
                                     or any(Path(p).suffix != '.engine' for p in config['models'])):
            raise ValueError('parallel_models chỉ hỗ trợ engine TensorRT trên CUDA.')
        for filename in config["models"]:
            path = Path(filename)
            if not path.is_file():
                raise FileNotFoundError(f"Không tìm thấy model: {path}")
            # Engine giữ nguyên metadata tên lớp và dùng hậu xử lý box giống bản .pt.
            model = YOLO(str(path), task=config.get("task", "detect"))
            if model.task == "obb" and self.parallel_models:
                raise ValueError("OBB chỉ dùng một model cho chế độ test detection.")
            names = model.names
            names = names if isinstance(names, dict) else dict(enumerate(names))
            mapping = {int(index): PHRASE_LABELS[name] for index, name in names.items()
                       if name in PHRASE_LABELS}
            missing = set(PHRASE_LABELS.values()) - set(mapping.values())
            if missing:
                raise ValueError(f'Checkpoint thiếu nhãn yêu cầu: {sorted(missing)}')
            if not mapping:
                raise ValueError(f"Model không có nhãn phù hợp: {path}")
            self.models.append((model, mapping))

    def _predict_one(self, model, mapping, frame):
        kwargs = {"predictor": BoxesOnlyPredictor} if model.task == "segment" else {}
        result = Model.predict(model,
            source=frame, device=self.device, imgsz=self.imgsz,
            rect=self.rect,
            conf=self.minimum_confidence, iou=self.iou, classes=list(mapping),
            max_det=self.max_detections,
            agnostic_nms=False, verbose=False, **kwargs,
        )[0]
        if result.obb is not None:
            boxes = result.obb
            return [dict(bbox_xyxy=box, bbox_polygon=polygon, bbox_xywhr=rotated,
                         confidence=score, class_index=index)
                    for box, polygon, rotated, score, index in zip(boxes.xyxy.cpu().tolist(),
                        boxes.xyxyxyxy.cpu().tolist(), boxes.xywhr.cpu().tolist(),
                        boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())]
        return result.boxes.data.cpu().tolist() if result.boxes is not None else []

    def detect(self, frame):
        """Nhận ảnh BGR; chờ model hoàn tất rồi mới lọc/gộp box."""
        with self.lock:
            if self.parallel_models and self.parallel is None:
                # Khởi tạo predictor/warmup tuần tự một lần, trước khi dùng context riêng.
                for model, mapping in self.models:
                    self._predict_one(model, mapping, frame)
                from detection.parallel import ParallelEngines
                self.parallel = ParallelEngines(self.models, self.device)
            outputs = (self.parallel.predict(frame) if self.parallel_models else
                       [self._predict_one(model, mapping, frame) for model, mapping in self.models])
            detections = []
            for (_, mapping), rows in zip(self.models, outputs):
                for row in rows:
                    extra = {}
                    if isinstance(row, dict):
                        x1, y1, x2, y2 = row['bbox_xyxy']
                        score, index = row['confidence'], row['class_index']
                        extra = {key: row[key] for key in ('bbox_polygon', 'bbox_xywhr')}
                    else:
                        x1, y1, x2, y2, score, index = row
                    label = mapping.get(int(index))
                    if (label is not None and score >= self.thresholds.get(label, self.confidence)
                            and self._valid_box(label, [x1, y1, x2, y2], frame.shape, extra.get("bbox_polygon"))):
                        detections.append({"label": label, "confidence": float(score),
                                           "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)], **extra})
            return self._merge(detections)

    def _valid_box(self, label, box, shape, polygon=None):
        """Lọc box theo tỷ lệ ảnh gốc trước NMS/tracking; không áp dụng cho bbox nhóm."""
        if not np.isfinite(box).all():
            return False
        height, width = shape[:2]
        x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
        w, h = (x2 - x1) / width, (y2 - y1) / height
        area = w*h
        if polygon is not None:
            points = np.asarray(polygon, np.float32)
            if points.shape != (4,2) or not np.isfinite(points).all():
                return False
            # Chuẩn hoá cạnh trong hệ toạ độ ảnh để góc xoay không làm tăng kích thước giả.
            edges = np.diff(np.vstack([points,points[0]]), axis=0)
            lengths = np.linalg.norm(edges,axis=1)
            w,h = float(min(lengths))/width,float(max(lengths))/height
            area = cv2.contourArea(points)/(width*height)
        rule = self.bbox_limits.get(label, {})
        return (w > 0 and h > 0 and w <= rule.get('max_width_ratio', 1)
                and h <= rule.get('max_height_ratio', 1) and area <= rule.get('max_area_ratio', 1))

    def close(self):
        if self.parallel is not None:
            self.parallel.close()

    def _merge(self, detections):
        # NMS cùng nhãn theo IoU hoặc độ chứa IoS; ưu tiên confidence cao hơn.
        kept = []
        for candidate in sorted(detections, key=lambda row: row["confidence"], reverse=True):
            box = np.asarray(candidate["bbox_xyxy"])
            duplicate = False
            for previous in kept:
                if candidate["label"] != previous["label"]:
                    continue
                other = np.asarray(previous["bbox_xyxy"])
                overlap = np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]))
                intersection = float(np.prod(overlap))
                area = float(np.prod(np.maximum(0, box[2:] - box[:2])))
                other_area = float(np.prod(np.maximum(0, other[2:] - other[:2])))
                if 'bbox_polygon' in candidate and 'bbox_polygon' in previous:
                    # OBB phải so giao đa giác, không dùng diện tích AABB để loại nhầm box xoay.
                    polygon = np.asarray(candidate['bbox_polygon'], np.float32)
                    other_polygon = np.asarray(previous['bbox_polygon'], np.float32)
                    intersection = float(cv2.intersectConvexConvex(polygon, other_polygon)[0])
                    area, other_area = cv2.contourArea(polygon), cv2.contourArea(other_polygon)
                union = area + other_area - intersection
                smaller_area = min(area, other_area)
                contained = (self.merge_ios is not None and smaller_area > 0
                             and intersection / smaller_area >= self.merge_ios)
                if (union > 0 and intersection / union > self.merge_iou) or contained:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept
