"""Nạp hai model YOLOE có sẵn, lọc đúng nhãn yêu cầu và gộp các box trùng cùng nhãn."""

from pathlib import Path

import numpy as np
from ultralytics import YOLOE, YOLO
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
        self.imgsz = inference["imgsz"]
        self.rect = inference.get("rect", True)
        self.max_detections = inference["max_detections"]
        self.thresholds = config["thresholds"]
        self.minimum_confidence = min([self.confidence, *self.thresholds.values()])
        self.models = []
        for filename in config["models"]:
            path = Path(filename)
            if not path.is_file():
                raise FileNotFoundError(f"Không tìm thấy model: {path}")
            # Engine giữ nguyên metadata tên lớp và dùng hậu xử lý box giống bản .pt.
            model = YOLO(str(path), task="detect") if path.suffix == ".engine" else YOLOE(str(path))
            names = model.names
            names = names if isinstance(names, dict) else dict(enumerate(names))
            mapping = {int(index): PHRASE_LABELS[name] for index, name in names.items()
                       if name in PHRASE_LABELS}
            if not mapping:
                raise ValueError(f"Model không có nhãn phù hợp: {path}")
            self.models.append((model, mapping))

    def detect(self, frame):
        """Nhận ảnh BGR từ OpenCV; trả nhãn, độ tin cậy và box theo pixel ảnh gốc."""
        detections = []
        for model, mapping in self.models:
            kwargs = {"predictor": BoxesOnlyPredictor} if model.task == "segment" else {}
            result = Model.predict(model,
                source=frame, device=self.device, imgsz=self.imgsz,
                rect=self.rect,
                conf=self.minimum_confidence, iou=self.iou, classes=list(mapping),
                max_det=self.max_detections,
                agnostic_nms=False, verbose=False, **kwargs,
            )[0]
            if result.boxes is None:
                continue
            for x1, y1, x2, y2, score, index in result.boxes.data.cpu().tolist():
                label = mapping.get(int(index))
                if label is not None and score >= self.thresholds.get(label, self.confidence):
                    detections.append({"label": label, "confidence": float(score),
                                       "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]})
        return self._merge(detections)

    def _merge(self, detections):
        # NMS theo nhãn tiếng Việt vì nhiều cụm từ/model có thể nhận cùng một vật.
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
                union = float(np.prod(box[2:] - box[:2]) + np.prod(other[2:] - other[:2]) - intersection)
                if union > 0 and intersection / union > self.merge_iou:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept
