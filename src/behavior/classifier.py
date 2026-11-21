"""Nạp MobileNetV3 Small 3 lớp, kiểm tra metadata và tiền xử lý đúng checkpoint train."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small

CLASSES = ['other_activity', 'repairing_vehicle', 'washing_vehicle']
PREPROCESSING = dict(size=224, base_letterbox_size=224, resize='contain_and_pad',
                     fill=[114, 114, 114], mean=[.485, .456, .406], std=[.229, .224, .225])


class ActivityClassifier:
    def __init__(self, weights, metadata, device):
        metadata = json.loads(Path(metadata).read_text())
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        if (metadata.get('classes') != CLASSES or checkpoint.get('classes') != CLASSES
                or checkpoint.get('class_to_idx') != {name: i for i, name in enumerate(CLASSES)}):
            raise ValueError('Thứ tự lớp MobileNetV3 không khớp metadata.')
        if metadata.get('preprocessing') != PREPROCESSING or checkpoint.get('config', {}).get('preprocessing') != PREPROCESSING:
            raise ValueError('Tiền xử lý checkpoint không khớp RGB letterbox 224/ImageNet.')
        self.device = torch.device(f'cuda:{device}' if str(device).isdigit() else device)
        self.model = mobilenet_v3_small(weights=None)
        self.model.classifier[3] = torch.nn.Linear(self.model.classifier[3].in_features, 3)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.to(self.device).eval()

    @staticmethod
    def preprocess(frame, box, padding):
        height, width = frame.shape[:2]
        box = np.asarray(box, float)
        margin = (box[2:] - box[:2]) * padding
        x1, y1 = np.maximum(0, np.floor(box[:2] - margin)).astype(int)
        x2, y2 = np.minimum([width, height], np.ceil(box[2:] + margin)).astype(int)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Crop hành vi rỗng.')
        image = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        ratio = 224 / max(image.size)
        image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (224, 224), (114, 114, 114))
        canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
        array = (np.asarray(canvas, np.float32) / 255 - np.array(PREPROCESSING['mean'], np.float32)) / np.array(PREPROCESSING['std'], np.float32)
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    @torch.inference_mode()
    def predict(self, frame, boxes, padding):
        batch = torch.stack([self.preprocess(frame, box, padding) for box in boxes]).to(self.device)
        return self.model(batch).softmax(-1).cpu().tolist()
