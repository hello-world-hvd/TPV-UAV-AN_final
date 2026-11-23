"""Phát frame đã viz lên RTMP bằng worker riêng; queue 2, bỏ frame cũ khi nghẽn, tự nối lại và không log URL có key."""
import os
import queue
import select
import subprocess
import threading
import time
from urllib.parse import urlsplit

import numpy as np


class RTMPPublisher:
    def __init__(self, url, fps, executable, encoder='nvenc'):
        if urlsplit(url).scheme not in ('rtmp', 'rtmps') or not 0 < fps <= 120:
            raise ValueError('Đích publish cần RTMP/RTMPS và FPS trong (0,120].')
        self.url, self.fps, self.executable, self.encoder = url, fps, executable, encoder
        self.frames = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.stats = dict(submitted=0, sent_to_encoder=0, dropped=0, failures=0, status='waiting')
        self.worker = threading.Thread(target=self._run, name='viz-publisher', daemon=True)
        self.worker.start()

    def write(self, frame):
        self.stats['submitted'] += 1
        item = np.ascontiguousarray(frame).copy()
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            try:
                self.frames.get_nowait()
                self.stats['dropped'] += 1
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(item)
            except queue.Full:
                self.stats['dropped'] += 1

    def command(self, shape):
        h,w = shape[:2]
        codec = (['-c:v','h264_nvenc','-preset','p1','-tune','ll','-rc','vbr','-cq','23','-b:v','0']
                 if self.encoder == 'nvenc' else
                 ['-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-crf','23','-threads','2'])
        return [str(self.executable),'-nostdin','-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pixel_format','bgr24','-video_size',f'{w}x{h}',
            '-framerate',str(self.fps),'-i','pipe:0','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
            *codec,'-pix_fmt','yuv420p','-g',str(max(1,round(self.fps))),'-bf','0',
            '-flvflags','no_duration_filesize','-rw_timeout','3000000','-f','flv',self.url]

    def _send(self, process, frame):
        data = memoryview(frame).cast('B')
        deadline = time.monotonic()+3
        while data and not self.stop.is_set():
            if process.poll() is not None or time.monotonic() > deadline:
                raise OSError('Encoder publish không phản hồi.')
            _,ready,_ = select.select([], [process.stdin.fileno()], [], .1)
            if ready:
                try:
                    written = os.write(process.stdin.fileno(),data[:65536])
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise OSError('Pipe publish đóng.')
                data = data[written:]
        return not data

    @staticmethod
    def _close(process):
        if process is None:
            return
        try:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        finally:
            process.stdin.close()

    def _run(self):
        process, shape, latest = None, None, None
        received, next_tick = 0., time.monotonic()
        try:
            while not self.stop.is_set():
                try:
                    latest = self.frames.get(timeout=.1) if latest is None else self.frames.get_nowait()
                    received = time.monotonic()
                    while True:
                        try:
                            latest = self.frames.get_nowait()
                            received = time.monotonic()
                            self.stats['dropped'] += 1
                        except queue.Empty:
                            break
                except queue.Empty:
                    pass
                if latest is None:
                    continue
                if time.monotonic()-received > 2:
                    # Ngừng phát hình cũ kéo dài khi nguồn/AI ngừng; không giả vờ video còn mới.
                    self._close(process)
                    process, shape, latest = None, None, None
                    self.stats['status'] = 'waiting'
                    continue
                if self.stop.wait(max(0,next_tick-time.monotonic())):
                    break
                try:
                    if process is None or shape != latest.shape:
                        self._close(process)
                        process = None
                        shape = latest.shape
                        process = subprocess.Popen(self.command(shape),stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,bufsize=0)
                        os.set_blocking(process.stdin.fileno(),False)
                    if self._send(process,latest):
                        self.stats['sent_to_encoder'] += 1
                        self.stats['status'] = 'sending'
                    next_tick = max(next_tick+1/self.fps,time.monotonic())
                except (OSError,ValueError):
                    self.stats['failures'] += 1
                    self.stats['status'] = 'reconnecting'
                    self._close(process)
                    process, shape, latest = None, None, None
                    print('Viz publish chưa kết nối được; sẽ thử lại, xử lý alert vẫn tiếp tục.',flush=True)
                    self.stop.wait(3)
                    next_tick = time.monotonic()
        finally:
            self._close(process)
            self.stats['status'] = 'stopped'

    def close(self):
        self.stop.set()
        self.worker.join(timeout=10)
        if self.worker.is_alive():
            raise RuntimeError('Worker publish chưa dừng.')
