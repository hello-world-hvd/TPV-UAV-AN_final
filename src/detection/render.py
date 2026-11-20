"""Vẽ nhanh trên frame BGR bằng OpenCV; cache vùng chữ tiếng Việt nhỏ tạo bằng PIL."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(font_path=None):
    candidates = [font_path] if font_path else [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(str(path), 20)
    raise FileNotFoundError("Không tìm thấy font Unicode. Truyền --font đường/dẫn/font.ttf")


@lru_cache(maxsize=1024)
def _label_patch(text, font):
    """Tạo vùng chữ nền đen một lần cho mỗi nội dung/font, giới hạn cache 1024 mục."""
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("RGB", (max(1, right - left + 4), max(1, bottom - top + 4)), (0, 0, 0))
    ImageDraw.Draw(image).text((2 - left, 2 - top), text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_detections(frame, detections, font, groups=()):
    # Chỉ sao chép frame BGR; không chuyển toàn ảnh sang RGB/PIL.
    result = frame.copy()
    height, width = result.shape[:2]
    for row in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in row["bbox_xyxy"])
        text = f'{row["label"]} {row["confidence"]:.2f}'
        if "track_id" in row:
            text = f'#{row["track_id"]} {text}'
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        patch = _label_patch(text, font)
        x = max(0, min(x1, width - 1))
        y = max(0, min(y1 - patch.shape[0], height - 1))
        patch_height = min(patch.shape[0], height - y)
        patch_width = min(patch.shape[1], width - x)
        result[y:y + patch_height, x:x + patch_width] = patch[:patch_height, :patch_width]
    for group in groups:
        x1, y1, x2, y2 = (int(round(value)) for value in group['bbox_xyxy'])
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 180, 0), 2)
        text = f"G{group['group_id']} {group['kind']} ({group['status']})"
        patch = _label_patch(text, font)
        x, y = max(0, min(x1, width - 1)), max(0, min(y2 + 2, height - 1))
        h, w = min(patch.shape[0], height - y), min(patch.shape[1], width - x)
        result[y:y+h, x:x+w] = patch[:h, :w]
    return result
