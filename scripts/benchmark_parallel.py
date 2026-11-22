"""Đo luân phiên tuần tự/song song sau warmup và đối chiếu kết quả trên cùng ảnh."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from core.config import load_config
from core.timing import DetectorTimer
from detection.detector import Detector


def main():
    config = load_config(ROOT / 'configs/detection.server.yaml')
    if len(config['models']) < 2:
        raise ValueError('Cấu hình mới chỉ có một detector; không cần chạy hai model song song.')
    config['inference']['parallel_models'] = False
    images = [cv2.imread(str(p)) for p in sorted((ROOT/'data/datatest').glob('*.png'))]
    detector = Detector(config)
    results, reports = {}, []
    try:
        detector.detect(images[0])
        for parallel in (False, True, False, True):
            detector.parallel_models = parallel
            for _ in range(3): detector.detect(images[0])
            timer = DetectorTimer(config['inference']['device'])
            outputs = []
            for _ in range(3):
                for image in images:
                    rows, _ = timer.detect(detector, image)
                    outputs.append(rows)
            if not results:
                results = outputs
            else:
                for a, b in zip(results, outputs):
                    assert len(a)==len(b), 'Số box khác nhau'
                    for x,y in zip(a,b):
                        assert x['label']==y['label']
                        np.testing.assert_allclose(x['bbox_xyxy'], y['bbox_xyxy'], atol=.1)
                        assert abs(x['confidence']-y['confidence'])<.001
            report = {'parallel_models':parallel, **timer.summary()}
            reports.append(report)
            print(report, flush=True)
    finally:
        detector.close()
    path=ROOT/'outputs/parallel_benchmark.json'
    path.write_text(json.dumps(reports,indent=2))
    print('Đối chiếu kết quả cả 4 lượt: đạt', flush=True)


if __name__=='__main__': main()
