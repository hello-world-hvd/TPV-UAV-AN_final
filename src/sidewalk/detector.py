"""Mask2Former phân vùng vỉa hè: BGR đầu vào, mask/nhãn/polygon theo ảnh gốc đầu ra; nạp model một lần."""
from contextlib import nullcontext
from pathlib import Path
from core.step_timing import StepTiming

import cv2
import numpy as np
import yaml
from shapely import make_valid
from shapely.geometry import Polygon

MODEL = 'facebook/mask2former-swin-large-mapillary-vistas-semantic'
SIDEWALK = {'sidewalk', 'pedestrian area', 'curb cut'}
ROAD = {'road', 'service lane', 'bike lane', 'parking', 'crosswalk - plain',
        'lane marking - crosswalk', 'lane marking - general'}


def class_groups(names):
    groups = [[], [], []]
    for index, name in names.items():
        name = ' '.join(name.casefold().replace('_', ' ').split())
        groups[2 if name in SIDEWALK else 1 if name in ROAD else 0].append(int(index))
    if not groups[2]:
        raise ValueError('Model không có nhãn vỉa hè.')
    return groups


def postprocess(scores, config):
    """Nhãn 0=khác, 1=đường, 2=vỉa hè, 255=chưa chắc; mask uint8 0/255, giữ lỗ."""
    ranked = np.sort(scores, axis=0)
    entropy = -(scores*np.log(np.maximum(scores, 1e-8))).sum(0)/np.log(3)
    accepted = ((ranked[-1] >= config['confidence']) &
                (ranked[-1]-ranked[-2] >= config['margin']) & (entropy <= config['max_entropy']))
    labels = scores.argmax(0).astype(np.uint8)
    labels[~accepted] = 255
    minimum = max(config['min_area_pixels'], scores.shape[1]*scores.shape[2]*config['min_area_ratio'])
    binary = (labels == 2).astype(np.uint8)
    kernel = config.get('close_kernel', 3)
    if kernel:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel)))
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= minimum)*255
    mask = keep[components]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            if hierarchy[0,index,3] != -1 or len(contour) < 3:
                continue
            holes = []
            child = hierarchy[0,index,2]
            while child != -1:
                if (len(contours[child]) >= 3 and cv2.contourArea(contours[child]) >=
                        config.get('min_hole_pixels', 100)):
                    holes.append(contours[child].reshape(-1,2))
                child = hierarchy[0,child,0]
            geometry = make_valid(Polygon(contour.reshape(-1,2), holes))
            geometry = geometry.simplify(geometry.length*config['simplify_ratio'], preserve_topology=True)
            parts = [geometry] if geometry.geom_type == 'Polygon' else getattr(geometry, 'geoms', ())
            for part in parts:
                if part.geom_type == 'Polygon' and part.area > 0:
                    polygons.append(dict(exterior=np.asarray(part.exterior.coords).tolist(),
                        holes=[np.asarray(ring.coords).tolist() for ring in part.interiors], area_pixels=part.area))
    # Dùng chính polygon đã làm mượt để tô màu, tránh viền và vùng tô khác nhau.
    mask.fill(0)
    for item in polygons:
        part = np.zeros_like(mask)
        cv2.fillPoly(part,[np.rint(item['exterior']).astype(np.int32)],255)
        for hole in item['holes']:
            cv2.fillPoly(part,[np.rint(hole).astype(np.int32)],0)
        cv2.bitwise_or(mask,part,dst=mask)
    labels[(labels == 2) & (mask == 0)] = 255
    labels[mask != 0] = 2
    return dict(labels=labels, mask=mask, polygons=polygons)


def restore_result(result, size):
    """Phóng mask/nhãn bằng nearest; polygon và diện tích đổi về hệ tọa độ ảnh gốc."""
    width, height = size
    h, w = result['mask'].shape
    scale = np.array([width/w, height/h])
    for item in result['polygons']:
        item['exterior'] = (np.asarray(item['exterior'])*scale).tolist()
        item['holes'] = [(np.asarray(ring)*scale).tolist() for ring in item['holes']]
        item['area_pixels'] *= float(scale.prod())
    for name in ('mask','labels'):
        result[name] = cv2.resize(result[name],size,interpolation=cv2.INTER_NEAREST)
    return result


class SidewalkDetector:
    def __init__(self, config_path):
        import torch
        from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
        path = Path(config_path).resolve()
        self.timings = StepTiming()
        self.config = c = yaml.safe_load(path.read_text(encoding='utf-8'))
        if c['max_side'] < 32 or c['precision'] not in ('fp32', 'fp16', 'bf16'):
            raise ValueError('max_side >=32, precision thuộc fp32/fp16/bf16.')
        if any(not np.isfinite(c[k]) or not 0 <= c[k] <= 1 for k in
               ('confidence', 'margin', 'max_entropy', 'min_area_ratio', 'simplify_ratio')) or c['min_area_pixels'] < 0:
            raise ValueError('Ngưỡng phân vùng không hợp lệ.')
        kernel = c.get('close_kernel', 3)
        if type(kernel) is not int or kernel < 0 or (kernel != 0 and kernel % 2 == 0):
            raise ValueError('close_kernel phải là 0 hoặc số nguyên lẻ dương.')
        if not np.isfinite(c.get('min_hole_pixels',100)) or c.get('min_hole_pixels',100) < 0:
            raise ValueError('min_hole_pixels phải không âm hữu hạn.')
        requested = str(c['device'])
        if requested == 'auto':
            requested = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device('cuda:'+requested if requested.isdigit() else requested)
        self.torch = torch
        self.precision = c['precision'] if self.device.type == 'cuda' else 'fp32'
        cache = (path.parent/c['cache_dir']).resolve()
        options = dict(cache_dir=str(cache), local_files_only=c['local_files_only'])
        processor, _ = Mask2FormerImageProcessor.get_image_processor_dict(c['model'], **options)
        self.processor = Mask2FormerImageProcessor.from_dict(
            {k:v for k,v in processor.items() if k not in ('_max_size','reduce_labels')})
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(c['model'], **options).to(self.device).eval()
        self.groups = class_groups(self.model.config.id2label)
        print(f'Mask2Former: {self.device} | {self.precision}', flush=True)

    def sync(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == 'mps':
            self.torch.mps.synchronize()

    def predict(self, frame):
        """Nhận BGR uint8; hậu xử lý nhỏ, trả mask/nhãn/polygon theo ảnh gốc."""
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError('Đầu vào phải là ảnh BGR uint8 H×W×3 không rỗng.')
        torch = self.torch
        self.sync()
        with self.timings.measure('preprocess'):
            height, width = frame.shape[:2]
            ratio = min(1., self.config['max_side']/max(height,width))
            h, w = max(1,round(height*ratio)), max(1,round(width*ratio))
            image = cv2.cvtColor(cv2.resize(frame,(w,h)),cv2.COLOR_BGR2RGB)
            inputs = {k:v.to(self.device) for k,v in self.processor(images=image,return_tensors='pt',do_resize=False).items()}
            self.sync()
        context = nullcontext() if self.precision == 'fp32' else torch.autocast('cuda',
            dtype=torch.float16 if self.precision == 'fp16' else torch.bfloat16)
        with torch.inference_mode(), context:
            with self.timings.measure('model'):
                output = self.model(**inputs)
                self.sync()  # GPU chạy bất đồng bộ: đợi xong để thời gian không dồn sang hậu xử lý.
            with self.timings.measure('postprocess'):
                classes = output.class_queries_logits.softmax(-1)[...,:-1].float()
                grouped = torch.stack([classes[...,ids].sum(-1) for ids in self.groups],dim=-1)
                scores = torch.einsum('bqg,bqhw->bghw',grouped,output.masks_queries_logits.sigmoid().float())
                scores = torch.nn.functional.interpolate(scores,size=inputs['pixel_values'].shape[-2:],
                    mode='bilinear',align_corners=False)[0,:,:h,:w]
                scores = scores/scores.sum(0,keepdim=True).clamp_min(1e-6)
                scores = scores.cpu().numpy()
                # Ngưỡng pixel cấu hình ở ảnh gốc; đổi sang ảnh hậu xử lý để giữ ý nghĩa diện tích.
                config = {**self.config, 'min_area_pixels': self.config['min_area_pixels']*(w*h)/(width*height)}
                config['min_hole_pixels'] = self.config.get('min_hole_pixels',100)*(w*h)/(width*height)
                result = postprocess(scores,config)
                result = restore_result(result,(width,height))
        return result
