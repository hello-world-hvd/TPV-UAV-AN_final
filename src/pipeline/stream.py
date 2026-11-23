"""Đọc RTMP/HLS ở luồng nền; giữ frame mới nhất, timeout và tự nối lại khi mất tín hiệu."""

import os
from collections import deque
import math
import queue
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import cv2


class StreamReader:
    def __init__(self, url, config):
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
        parsed = urlsplit(url)
        # Link HLS dạng thư mục là trang player; OpenCV cần playlist của luồng.
        if parsed.scheme in ('http', 'https') and parsed.path.endswith('/'):
            url = urlunsplit(parsed._replace(path=parsed.path+'index.m3u8'))
        self.url, self.config = url, config
        self.received_times = deque()
        self.source_fps = None
        self.connected = False
        self.last_frame_time = None
        self.frames = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._run, name='camera-reader', daemon=True)
        self.worker.start()

    def _run(self):
        epoch = 0
        while not self.stop.is_set():
            capture = None
            try:
                capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config['timeout_ms'],
                     cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config['timeout_ms']])
                if not capture.isOpened():
                    raise RuntimeError('Camera chưa kết nối được.')
                reported = capture.get(cv2.CAP_PROP_FPS)
                self.source_fps = reported if math.isfinite(reported) and reported > 0 else None
                self.connected = True
                self.received_times.clear()
                epoch += 1
                print(f'Camera đã kết nối, phiên {epoch}.', flush=True)
                while not self.stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        break
                    now = time.monotonic()
                    self.last_frame_time = now
                    self.received_times.append(now)
                    while len(self.received_times)>1 and now-self.received_times[0]>5:
                        self.received_times.popleft()
                    item = (epoch, now, frame)
                    try:
                        self.frames.put_nowait(item)
                    except queue.Full:
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                        self.frames.put_nowait(item)
            except (cv2.error, RuntimeError):
                pass
            finally:
                self.connected = False
                if capture is not None:
                    capture.release()
            if not self.stop.is_set():
                print('Camera gián đoạn; đang kết nối lại.', flush=True)
                self.stop.wait(self.config['reconnect_seconds'])

    def fps_status(self):
        times = list(self.received_times)
        now = time.monotonic()
        receive = ((len(times)-1)/(times[-1]-times[0])
                   if len(times)>1 and now-times[-1]<2 and times[-1]>times[0] else 0.)
        return self.source_fps, receive

    def read(self, max_age_seconds=None, poll_seconds=None):
        while not self.stop.is_set():
            try:
                item = self.frames.get(timeout=.5)
                while True:
                    try:
                        item = self.frames.get_nowait()
                    except queue.Empty:
                        break
                if max_age_seconds is not None and time.monotonic()-item[1] > max_age_seconds:
                    continue
                return item
            except queue.Empty:
                if poll_seconds is not None:
                    return None
                continue
        return None

    def close(self):
        self.stop.set()
        self.worker.join(timeout=2*self.config['timeout_ms']/1000 + 2)
