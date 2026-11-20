"""Nạp hai model YOLOE có sẵn, lọc đúng nhãn yêu cầu và gộp các box trùng cùng nhãn."""

from pathlib import Path
from threading import Lock

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
        self.merge_ios = inference.get("merge_ios", 0.9)
        self.imgsz = inference["imgsz"]
        self.rect = inference.get("rect", True)
        self.max_detections = inference["max_detections"]
        self.thresholds = config["thresholds"]
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
            model = YOLO(str(path), task="detect") if path.suffix == ".engine" else YOLOE(str(path))
            names = model.names
            names = names if isinstance(names, dict) else dict(enumerate(names))
            mapping = {int(index): PHRASE_LABELS[name] for index, name in names.items()
                       if name in PHRASE_LABELS}
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
        return result.boxes.data.cpu().tolist() if result.boxes is not None else []

    def detect(self, frame):
        """Nhận ảnh BGR; chờ cả hai model hoàn tất rồi mới lọc/gộp box."""
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
                for x1, y1, x2, y2, score, index in rows:
                    label = mapping.get(int(index))
                    if label is not None and score >= self.thresholds.get(label, self.confidence):
                        detections.append({"label": label, "confidence": float(score),
                                           "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]})
            return self._merge(detections)

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
