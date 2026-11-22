"""Gửi alert tới BE bằng multipart/presigned URL; chỉ xoá file sau xác nhận ready, lỗi giữ tại workspace."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import math
import mimetypes
import os
from pathlib import Path
import queue
import shutil
import threading
import time

import requests
import yaml


class AlertDelivery:
    def __init__(self, config_path, started_at, enabled=None):
        self.config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        self.enabled = self.config['enabled'] if enabled is None else enabled
        self.started_at = started_at
        self.key = os.environ.get(self.config.get('api_key_env', 'TPV_UPLOAD_API_KEY'), '')
        self.url = self.config['api_url'].rstrip('/')
        if self.enabled and (not self.key or not self.url.startswith('https://') or not self.config.get('location')):
            raise ValueError('Bật upload cần HTTPS, biến môi trường API key và location GPS trong backend.yaml.')
        if self.enabled:
            try:
                latitude, longitude = map(float, self.config['location'].split(','))
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    raise ValueError
            except (AttributeError, TypeError, ValueError):
                raise ValueError('location phải có GPS hợp lệ dạng "latitude, longitude".') from None
        self.jobs = queue.Queue(maxsize=2)
        self.counts = dict(local=0, ready=0, failed=0)
        self.worker = None
        if self.enabled:
            self.worker = threading.Thread(target=self._run, name='alert-upload')
            self.worker.start()

    def submit(self, folder, report):
        if not self.enabled:
            self.counts['local'] += 1
            print(f"Alert {report['package_id']}: đã lưu {folder}", flush=True)
            return
        moment = self.started_at + timedelta(seconds=next(e['alert_time'] for e in report['events']
            if e['event_id'] == report['trigger_event_id']))
        self.jobs.put((Path(folder), {**report, '_detected_at': moment.isoformat()}))

    def close(self):
        if self.worker is not None:
            self.jobs.put(None)
            self.worker.join()
            self.worker = None

    def _api(self, http, endpoint, payload=None):
        result = http.post(self.url + endpoint, headers={'x-api-key': self.key}, json=payload, timeout=(15,120))
        result.raise_for_status()
        return result.json()

    @staticmethod
    def _put(url, data, content_type=None):
        for attempt in range(3):
            try:
                response = requests.put(url, data=data, headers={'Content-Type': content_type} if content_type else {},
                                        timeout=(15,300), allow_redirects=False)
                if not 200 <= response.status_code < 300:
                    raise RuntimeError('Storage không xác nhận HTTP 2xx.')
                return response.headers.get('ETag')
            except (requests.RequestException, RuntimeError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def _upload(self, folder, report):
        video = folder / report['clip']
        images = [folder / name for name in dict.fromkeys(e['snapshot'] for e in report['evidence'])]
        descriptions = [dict(filename=p.name, file_size=p.stat().st_size,
                             content_type=mimetypes.guess_type(p.name)[0]) for p in images]
        payload = dict(filename=video.name, file_size=video.stat().st_size, content_type='video/mp4',
            detected_at=report['_detected_at'],
            location=self.config['location'], behavior_name='; '.join(report['behavior_names']),
            camera_id=self.config.get('camera_id'), confidence=None, model_version=self.config.get('model_version'),
            violators=None, images=descriptions)
        state = {'state': 'initiating'}
        def save():
            (folder/'upload.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        save()
        with requests.Session() as http:
            initiated = self._api(http, '/api/uploads/initiate', payload)
            vid = initiated['violation_id']
            state.update(state='uploading', violation_id=vid)
            save()
            size, count = initiated['part_size'], initiated['total_parts']
            if type(size) is not int or size <= 0 or type(count) is not int or count != math.ceil(video.stat().st_size/size):
                raise ValueError('BE trả kế hoạch multipart không hợp lệ.')
            targets = initiated.get('images', [])
            if len(targets) != len(images):
                raise ValueError('BE trả số URL ảnh không khớp.')
            for path, item, target in zip(images, descriptions, targets):
                if target.get('filename', path.name) != path.name or target.get('content_type', item['content_type']) != item['content_type']:
                    raise ValueError('BE trả metadata ảnh không khớp.')
                self._put(target['upload_url'], path.read_bytes(), item['content_type'])
            parts = []
            def send(item):
                number = item['part_number']
                with video.open('rb') as source:
                    source.seek((number-1)*size)
                    etag = self._put(item['url'], source.read(size))
                if not etag:
                    raise ValueError('Storage không trả ETag.')
                return dict(part_number=number, etag=etag)
            for start in range(1, count+1, 100):
                numbers = list(range(start, min(count+1, start+100)))
                urls = self._api(http, f'/api/uploads/{vid}/part-urls', dict(part_numbers=numbers))['parts']
                if sorted(item['part_number'] for item in urls) != numbers:
                    raise ValueError('BE trả danh sách part không khớp.')
                with ThreadPoolExecutor(max_workers=2) as pool:
                    parts.extend(pool.map(send, urls))
            state['state'] = 'completing'
            save()
            result = self._api(http, f'/api/uploads/{vid}/complete', dict(parts=sorted(parts, key=lambda p:p['part_number'])))
            if result.get('success') is not True or result.get('status') != 'ready' or result.get('violation_id') != vid:
                raise ValueError('BE chưa xác nhận đúng alert ở trạng thái ready.')
        # Không tự retry initiate/complete: timeout có thể đã được BE xử lý, tránh tạo bản trùng.
        shutil.rmtree(folder)

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            folder, report = job
            try:
                self._upload(folder, report)
                self.counts['ready'] += 1
                print(f"Alert {report['package_id']}: BE xác nhận ready, đã dọn file tạm.", flush=True)
            except Exception as error:
                self.counts['failed'] += 1
                # Chỉ ghi loại lỗi; không lộ URL ký sẵn hoặc key qua exception HTTP.
                state_path = folder/'upload.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                state.update(error=type(error).__name__, outcome='uncertain_or_failed')
                state_path.write_text(json.dumps(state), encoding='utf-8')
                print(f"Upload lỗi ({type(error).__name__}); giữ alert tại {folder}", flush=True)
