"""Đo thời gian toàn bộ detector, đồng bộ CUDA/MPS và thống kê riêng lượt khởi động."""

import time

import torch


class DetectorTimer:
    def __init__(self, device):
        self.device = str(device)
        self.count = 0
        self.total_seconds = 0.0
        self.first_seconds = None

    def synchronize(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.isdigit() or self.device.startswith("cuda"):
            device = f"cuda:{self.device}" if self.device.isdigit() else self.device
            torch.cuda.synchronize(device)

    def detect(self, detector, frame):
        self.synchronize()
        started = time.perf_counter()
        detections = detector.detect(frame)
        self.synchronize()
        seconds = time.perf_counter() - started
        self.count += 1
        self.total_seconds += seconds
        if self.first_seconds is None:
            self.first_seconds = seconds
        return detections, seconds

    def summary(self):
        average = self.total_seconds / self.count if self.count else None
        steady = ((self.total_seconds - self.first_seconds) / (self.count - 1)
                  if self.count > 1 else None)
        return {
            "device": self.device,
            "detect_calls": self.count,
            "total_detect_seconds": self.total_seconds,
            "first_detect_ms": self.first_seconds * 1000 if self.first_seconds is not None else None,
            "average_detect_ms": average * 1000 if average is not None else None,
            "detector_fps": 1 / average if average else None,
            "average_detect_ms_excluding_first": steady * 1000 if steady is not None else None,
        }
