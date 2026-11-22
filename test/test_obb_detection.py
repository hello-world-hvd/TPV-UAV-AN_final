"""Test model OBB trên ảnh/video, vẽ bbox xoay; tắt tracking/grouping để xem detection gốc."""
import sys
from pathlib import Path
from test_obj_detection import main

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--config', str(ROOT/'configs/detection.obb.yaml'),
                '--tracking-config', str(ROOT/'configs/tracking.detection_only.yaml'), *sys.argv[1:]]
    main()
