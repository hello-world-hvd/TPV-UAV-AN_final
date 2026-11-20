"""Vẽ hộp nhận diện và nhãn tiếng Việt lên ảnh, dùng font Unicode để giữ dấu."""

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


def draw_detections(frame, detections, font):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for row in detections:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        text = f'{row["label"]} {row["confidence"]:.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
        position = (max(0, x1), max(0, y1 - 25))
        draw.rectangle(draw.textbbox(position, text, font=font), fill=(0, 0, 0))
        draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
