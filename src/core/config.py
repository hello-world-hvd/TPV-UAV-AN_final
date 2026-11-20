"""Đọc YAML trong configs, kiểm tra tham số và quy đổi đường dẫn theo vị trí file cấu hình."""

import math
from pathlib import Path

import yaml

from detection.labels import LABELS


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Cấu hình phải là một mapping YAML.")
    for key in ("inference", "thresholds", "output"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Cấu hình {key} phải là mapping.")

    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Đường dẫn cấu hình phải là chuỗi không rỗng.")
        value = Path(value).expanduser()
        return value.resolve() if value.is_absolute() else (path.parent / value).resolve()

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models phải là danh sách đường dẫn model không rỗng.")
    config["models"] = [resolve(value) for value in models]
    inference = config["inference"]
    detect_fps = inference.get("detect_fps")
    if detect_fps is not None and (isinstance(detect_fps, bool)
            or not isinstance(detect_fps, (int, float))
            or not math.isfinite(detect_fps) or detect_fps <= 0):
        raise ValueError("detect_fps phải là số dương hữu hạn hoặc null.")
    inference["detect_fps"] = detect_fps
    for name in ("confidence", "nms_iou", "merge_iou"):
        _threshold(inference.get(name), name)
    for name in ("imgsz", "max_detections"):
        value = inference.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} phải là số nguyên dương.")
    if inference["imgsz"] % 32:
        raise ValueError("imgsz phải là bội số của 32.")
    if not isinstance(inference.get("device"), (str, int)) or isinstance(inference["device"], bool):
        raise ValueError("device phải là tên thiết bị hoặc số GPU.")
    inference["device"] = str(inference["device"])
    for label, value in config["thresholds"].items():
        if label not in LABELS.values():
            raise ValueError(f"Nhãn không được hỗ trợ: {label}")
        _threshold(value, label)
    config["output"]["directory"] = resolve(config["output"].get("directory"))
    font = config["output"].get("font")
    config["output"]["font"] = resolve(font) if font is not None else None
    return config


def _threshold(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} phải là số thuộc khoảng (0, 1].")
