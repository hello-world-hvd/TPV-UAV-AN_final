"""Kiểm tra vòng đời ID, ghép box một-một và optical flow trên ảnh dịch chuyển."""
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from core.config import load_tracking_config
from tracking.tracker import Tracker
from tracking.motion import CameraMotion


def main():
    config = load_tracking_config(ROOT / 'configs/tracking.yaml')
    config['camera_motion'] = False
    frame = np.zeros((240, 320, 3), np.uint8)
    def detection(x):
        return {'label': 'Xe máy', 'confidence': .8, 'bbox_xyxy': [x, 50, x+30, 100]}
    tracker = Tracker(config)
    assert not tracker.update(frame, 0, [detection(20)])
    assert tracker.update(frame, .1, [detection(25)])[0]['track_id'] == 1
    assert tracker.update(frame, .2, None)[0]['track_id'] == 1
    assert not tracker.update(frame, 1, None)
    assert tracker.update(frame, 1.1, [detection(26)])[0]['track_id'] == 1
    assert not tracker.update(frame, 3, None)
    tracker.update(frame, 3.1, [detection(30)])
    assert tracker.update(frame, 3.2, [detection(31)])[0]['track_id'] == 2
    tracker = Tracker(config)
    tracker.update(frame, 0, [detection(20), detection(200)])
    assert {r['track_id'] for r in tracker.update(frame, .1, [detection(202), detection(22)])} == {1, 2}
    rng = np.random.default_rng(1)
    textured = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(textured, np.float32([[1, 0, 4], [0, 1, 3]]), (320, 240))
    motion = CameraMotion(320, 1)
    motion.update(textured, {1: [70, 70, 130, 130]})
    result = motion.update(shifted, {1: [70, 70, 130, 130]})
    assert result['reliable']
    np.testing.assert_allclose(result['matrix'][:, 2], [4, 3], atol=.6)
    np.testing.assert_allclose(result['local_boxes'][1], [74, 73, 134, 133], atol=1)
    print('Đạt: xác nhận ID, dự đoán, mất dấu/ghép lại, hết hạn, ghép một-một và optical flow.')


if __name__ == '__main__':
    main()
