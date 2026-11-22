"""Kiểm tra alert phát ngay khi đủ cửa sổ, màu đỏ, upload BE giả lập và chỉ dọn file khi ready."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from alerts.manager import AlertManager
from alerts.uploader import AlertDelivery
from detection.render import draw_detections, load_font
from pipeline.stream import StreamReader


def main():
    with patch.object(StreamReader, '_run', return_value=None):
        reader = StreamReader('http://camera.invalid/live/stream/', dict(timeout_ms=1000))
        assert reader.url == 'http://camera.invalid/live/stream/index.m3u8'
        reader.close()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = yaml.safe_load((ROOT/'configs/alerts.yaml').read_text())
        config.update(confirmation_seconds=0, evidence_min_seconds=1.5)
        path = root/'alerts.yaml'
        path.write_text(yaml.safe_dump(config))
        reports = []
        manager = AlertManager(root, path, False, on_package=lambda folder, report: reports.append((folder, report)))
        group = dict(group_id=1, track_ids=[1,2], status='confirmed', bbox_xyxy=[5,5,40,40],
                     behaviors=[dict(name='kinh doanh', source='group_rule')])
        frame = draw_detections(np.zeros((100,400,3), np.uint8), [], load_font(), [group], True)
        assert tuple(frame[5,5]) == (0,0,255)
        assert frame[44:,:,2].max() > 0 and frame[44:,:,:2].max() == 0
        for i in range(90):
            manager.update(frame, i, 10, [group])
        assert len(reports) == 1 and not manager.packages
        assert reports[0][1]['evidence'][0]['behavior_names'] == ['kinh doanh']
        manager.finalize(None, 10, 90)
        assert len(reports) == 1  # Không phát lại callback đã xử lý.

        # Mốc 37/30 + 8 bị lệch float: không được ghi thêm frame thứ 278.
        reports.clear()
        config['evidence_min_seconds'] = 0
        path.write_text(yaml.safe_dump(config))
        manager = AlertManager(root/'rounding', path, True,
            on_package=lambda folder, report: reports.append(report))
        for i in range(300):
            manager.update(frame, i, 30, [group] if i >= 37 else [])
        manager.finalize(None, 30, 300)
        assert len(reports) == 1
        assert reports[0]['stop_frame'] == 277 and reports[0]['video_frames'] == 277

        cfg = root/'backend.yaml'
        cfg.write_text(yaml.safe_dump(dict(enabled=False, api_url='https://test.invalid',
                                          location='21, 105', api_key_env='TEST_UPLOAD_KEY')))
        folder = root/'upload'
        folder.mkdir()
        (folder/'clip.mp4').write_bytes(b'abcdef')
        (folder/'group.jpg').write_bytes(b'jpeg')
        report = dict(package_id=1, trigger_event_id=1, clip='clip.mp4', behavior_names=['kinh doanh'],
                      events=[dict(event_id=1, alert_time=2)], evidence=[dict(snapshot='group.jpg')])
        calls = []
        def api(self, http, endpoint, payload=None):
            calls.append((endpoint,payload))
            if endpoint.endswith('/initiate'):
                assert payload['behavior_name'] == 'kinh doanh' and payload['violators'] is None
                return dict(violation_id='v1', part_size=3, total_parts=2,
                            images=[dict(upload_url='https://storage.invalid/image')])
            if endpoint.endswith('/part-urls'):
                return dict(parts=[dict(part_number=i, url=f'https://storage.invalid/{i}') for i in [1,2]])
            return dict(success=True, status='ready', violation_id='v1')
        with patch.dict('os.environ', {'TEST_UPLOAD_KEY':'test-only'}), patch.object(AlertDelivery, '_api', api), \
             patch.object(AlertDelivery, '_put', return_value='etag'):
            delivery = AlertDelivery(cfg, datetime.now(timezone.utc), True)
            delivery.submit(folder, report)
            delivery.close()
            assert delivery.counts['ready'] == 1 and not folder.exists()
            assert calls[-1][0].endswith('/complete')
        folder.mkdir()
        (folder/'clip.mp4').write_bytes(b'abcdef')
        (folder/'group.jpg').write_bytes(b'jpeg')
        with patch.dict('os.environ', {'TEST_UPLOAD_KEY':'test-only'}), \
             patch.object(AlertDelivery, '_api', side_effect=TimeoutError):
            delivery = AlertDelivery(cfg, datetime.now(timezone.utc), True)
            delivery.submit(folder, report)
            delivery.close()
            assert delivery.counts['failed'] == 1 and (folder/'clip.mp4').exists()
            assert json.loads((folder/'upload.json').read_text())['state'] == 'initiating'
    print('Đạt: callback realtime, không phát trùng, viz đỏ, multipart BE giả lập và giữ file khi lỗi.')


if __name__ == '__main__':
    main()
