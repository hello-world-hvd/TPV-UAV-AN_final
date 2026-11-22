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
def _label_patch(text, font, color=(0, 255, 0)):
    """Tạo vùng chữ nền đen một lần cho mỗi nội dung/font, giới hạn cache 1024 mục."""
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("RGB", (max(1, right - left + 4), max(1, bottom - top + 4)), (0, 0, 0))
    ImageDraw.Draw(image).text((2 - left, 2 - top), text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_detections(frame, detections, font, groups=(), group_ids_only=False):
    # Chỉ sao chép frame BGR; không chuyển toàn ảnh sang RGB/PIL.
    result = frame.copy()
    height, width = result.shape[:2]
    for row in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in row["bbox_xyxy"])
        text = f'{row["label"]} {row["confidence"]:.2f}'
        if "track_id" in row:
            text = f'#{row["track_id"]} {text}'
        info = row.get('stationary_info')
        if info:
            states = {'stationary': 'ĐỨNG YÊN', 'pending': 'ĐANG XÁC NHẬN',
                      'unknown': 'CHƯA RÕ', 'motion_threshold_exceeded': 'VƯỢT NGƯỠNG CHUYỂN ĐỘNG'}
            reasons = {'camera_unreliable': 'camera không tin cậy', 'observation_unavailable': 'thiếu quan sát mới',
                       'waiting_detection': 'chờ detect', 'anchor_displacement': 'lệch neo',
                       'step_displacement': 'dịch bước lớn', 'accumulated_path': 'đường đi tích luỹ',
                       'first_observation': 'quan sát đầu'}
            text += f" | {states[info['status']]} {info['observed_seconds']:.1f}/{info['required_seconds']:g}s"
            detail = info['reason'] if info['status'] != 'pending' else info['last_reset_reason']
            if info['reason'] in reasons:
                text += f" | {reasons[info['reason']]}"
        if 'bbox_polygon' in row:
            cv2.polylines(result, [np.rint(row['bbox_polygon']).astype(np.int32)], True, (0,255,0), 2)
        else:
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        patch = _label_patch(text, font)
        x = max(0, min(x1, width - 1))
        y = max(0, min(y1 - patch.shape[0], height - 1))
        patch_height = min(patch.shape[0], height - y)
        patch_width = min(patch.shape[1], width - x)
        result[y:y + patch_height, x:x + patch_width] = patch[:patch_height, :patch_width]
    for group in groups:
        x1, y1, x2, y2 = (int(round(value)) for value in group['bbox_xyxy'])
        behaviors = group.get('behaviors', [])
        color = (0,0,255) if behaviors else (255,180,0)
        if 'bbox_polygon' in group:
            cv2.polylines(result, [np.rint(group['bbox_polygon']).astype(np.int32)], True, color, 2)
        else:
            cv2.rectangle(result, (x1,y1), (x2,y2), color, 2)
        description = '; '.join(r['name'] for r in behaviors) if behaviors else group['kind']
        text = f"G{group['group_id']} {description}" + (f" ({group['status']})" if not behaviors else "")
        if group_ids_only:
            # Một bbox/nhóm: G khi chưa có hành vi; B khi đã gán cảnh báo hành vi.
            text = (f"B{group['group_id']} {description}" if behaviors
                    else f"G{group['group_id']}" + (": Người và Máy móc" if group['kind'] == 'Người và Máy móc' else ''))
        patch = _label_patch(text, font, (255, 0, 0) if behaviors else (0, 255, 0))
        x, y = max(0, min(x1, width - 1)), max(0, min(y2 + 2, max(0, height - patch.shape[0])))
        h, w = min(patch.shape[0], height - y), min(patch.shape[1], width - x)
        result[y:y+h, x:x+w] = patch[:h, :w]
    return result
