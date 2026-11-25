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

    runtime = config.setdefault('runtime', {})
    if not isinstance(runtime, dict):
        raise ValueError('runtime phải là mapping.')
    for name, default in [('cv_threads', 4), ('torch_threads', 2)]:
        runtime.setdefault(name, default)
        if type(runtime[name]) is not int or runtime[name] < 1:
            raise ValueError(f'runtime.{name} phải là số nguyên dương.')

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models phải là danh sách đường dẫn model không rỗng.")
    config["models"] = [resolve(value) for value in models]
    inference = config["inference"]
    inference.setdefault('parallel_models', False)
    if type(inference['parallel_models']) is not bool:
        raise ValueError('parallel_models phải là boolean.')
    detect_fps = inference.get("detect_fps")
    if detect_fps is not None and (isinstance(detect_fps, bool)
            or not isinstance(detect_fps, (int, float))
            or not math.isfinite(detect_fps) or detect_fps <= 0):
        raise ValueError("detect_fps phải là số dương hữu hạn hoặc null.")
    inference["detect_fps"] = detect_fps
    for name in ("confidence", "nms_iou", "merge_iou"):
        _threshold(inference.get(name), name)
    inference.setdefault("merge_ios", 0.9)
    if inference["merge_ios"] is not None:
        _threshold(inference["merge_ios"], "merge_ios")
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
    # File riêng dùng chung local/server, tránh thay cấu hình thiết bị khi chỉnh kích thước box.
    limits_path = path.parent / 'bbox_limits.yaml'
    limits = yaml.safe_load(limits_path.read_text(encoding='utf-8')) if limits_path.is_file() else {}
    if not isinstance(limits, dict):
        raise ValueError('bbox_limits.yaml phải là mapping theo nhãn.')
    for label, rule in limits.items():
        if label not in LABELS.values() or not isinstance(rule, dict):
            raise ValueError(f'Cấu hình giới hạn bbox không hợp lệ: {label}')
        for name, value in rule.items():
            if name not in ('max_width_ratio', 'max_height_ratio', 'max_area_ratio'):
                raise ValueError(f'Tham số giới hạn bbox không hỗ trợ: {name}')
            _threshold(value, f'{label}.{name}')
    config['bbox_limits'] = limits
    config["output"]["directory"] = resolve(config["output"].get("directory"))
    font = config["output"].get("font")
    config["output"]["font"] = resolve(font) if font is not None else None
    video = config["output"].setdefault("video", {})
    if not isinstance(video, dict):
        raise ValueError("output.video phải là mapping.")
    defaults = {"encoder": "opencv", "async_encode": False, "queue_size": 2, "preset": "p1", "cq": 20}
    for key, value in defaults.items():
        video.setdefault(key, value)
    if video["encoder"] not in ("opencv", "nvenc"):
        raise ValueError("encoder phải là opencv hoặc nvenc.")
    if type(video["async_encode"]) is not bool or type(video["queue_size"]) is not int or not 1 <= video["queue_size"] <= 8:
        raise ValueError("async_encode phải là boolean; queue_size phải thuộc 1..8.")
    if video["preset"] not in {f"p{i}" for i in range(1, 8)}:
        raise ValueError("NVENC preset phải thuộc p1..p7.")
    if type(video["cq"]) is not int or not 0 <= video["cq"] <= 51:
        raise ValueError("NVENC cq phải là số nguyên thuộc 0..51.")
    if video.get("ffmpeg") is not None:
        video["ffmpeg"] = str(resolve(video["ffmpeg"]))
    return config


def _threshold(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} phải là số thuộc khoảng (0, 1].")


def load_tracking_config(path):
    """Đọc cấu hình tracking riêng và kiểm tra các giới hạn thời gian/ghép box."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Cấu hình tracking phải là mapping.")
    for name in ("enabled", "camera_motion"):
        if type(config.get(name)) is not bool:
            raise ValueError(f"{name} phải là boolean.")
    for name in ("min_hits", "motion_max_side", "flow_refresh_interval"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} phải là số nguyên dương.")
    for name in ("max_lost_seconds", "max_prediction_seconds", "max_center_distance"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} phải là số dương hữu hạn.")
    _threshold(config.get("min_iou"), "min_iou")
    if config["max_prediction_seconds"] > config["max_lost_seconds"]:
        raise ValueError("max_prediction_seconds phải <= max_lost_seconds.")
    return config


def apply_cpu_threads(config):
    """Đặt một lần trước model/worker; không giới hạn tổng thread của chương trình hay CUDA/FFmpeg."""
    import cv2
    import torch
    runtime = config['runtime']
    cv2.setNumThreads(runtime['cv_threads'])
    torch.set_num_threads(runtime['torch_threads'])
    actual = dict(cv_threads=cv2.getNumThreads(), torch_threads=torch.get_num_threads())
    print(f"CPU threads: OpenCV={actual['cv_threads']} | PyTorch={actual['torch_threads']}", flush=True)
    return actual
