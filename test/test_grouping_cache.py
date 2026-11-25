"""Kiểm tra cache theo frame, ngưỡng khoảng cách và vùng bao nhóm dùng lại."""
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from core.geometry import distance, coverage, shape
from grouping.geometry_cache import FrameGeometry


def main():
    a = {'bbox_xyxy': [0,0,10,10]}
    b = {'bbox_xyxy': [12,0,22,10]}
    far = {'bbox_xyxy': [100,0,110,10]}
    cache = FrameGeometry()
    with patch('grouping.geometry_cache.shape', wraps=shape) as build:
        assert cache.distance(a, far, limit=2) == float('inf')
        assert build.call_count == 0
        assert cache.distance(a,b,limit=2) == distance(a,b) == 2
        assert cache.distance(b,a) == 2 and build.call_count == 2
        assert cache.distance(a,far,limit=100) == 90 and build.call_count == 3
        for rows in ([a,b], [b,a]):
            assert np.isclose(cache.coverage(a,rows), coverage(a,rows))
        assert len(cache.regions) == 1
    b['bbox_xyxy'] = [13,0,23,10]
    assert FrameGeometry().distance(a,b) == 3
    # Sát ngưỡng sau làm tròn lưới polygon vẫn phải đo chính xác.
    b['bbox_xyxy'] = [12.0004,0,22,10]
    assert FrameGeometry().distance(a,b,limit=2) == distance(a,b) == 2
    print('Đạt: cache cặp đối xứng, loại xa không dựng shape, đổi ngưỡng/frame và cache vùng nhóm.')


if __name__ == '__main__':
    main()
