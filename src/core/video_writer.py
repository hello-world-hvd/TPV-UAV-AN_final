"""Ghi video OpenCV hoặc NVENC; worker có hàng đợi giới hạn và chờ ghi hết khi đóng."""

from collections import deque
import queue
import subprocess
import threading
import time
from pathlib import Path

import cv2
import imageio_ffmpeg


class NVENCWriter:
    def __init__(self, path, fps, size, preset, cq, executable=None):
        width, height = size
        if width % 2 or height % 2:
            raise ValueError("NVENC yuv420p cần chiều rộng/cao chẵn để giữ nguyên độ phân giải.")
        self.size = size
        self.closed = False
        self.log_path = Path(path).with_suffix(".encoder.log")
        self.log = self.log_path.open("w+b")
        command = [str(executable or imageio_ffmpeg.get_ffmpeg_exe()), "-nostdin", "-hide_banner", "-loglevel", "error",
                   "-y", "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
                   "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "h264_nvenc",
                   "-preset", preset, "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
                   "-pix_fmt", "yuv420p", str(path)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def write(self, frame):
        if frame.shape[:2] != (self.size[1], self.size[0]):
            raise ValueError("Kích thước frame thay đổi trong video.")
        try:
            # Truyền trực tiếp bộ nhớ liên tục, không tạo thêm bytes bằng tobytes().
            if not frame.flags.c_contiguous:
                frame = frame.copy()
            data = memoryview(frame).cast('B')
            while data:
                written = self.process.stdin.write(data)
                if not written:
                    raise BrokenPipeError('Pipe encoder không nhận dữ liệu.')
                data = data[written:]
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"NVENC ghi thất bại; xem {self.log_path}") from error

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("FFmpeg quá thời gian hoàn tất video.")
            if code:
                raise RuntimeError(f"FFmpeg thoát với mã {code}; xem {self.log_path}")
        finally:
            self.log.close()


class ThreadedWriter:
    def __init__(self, writer, queue_size, initial_frames=(), copy_frames=True):
        self.writer = writer
        self.initial_frames = deque(initial_frames)
        self.copy_frames = copy_frames
        self.initial_frames_written = 0
        self.queue = queue.Queue(maxsize=queue_size)
        self.error = None
        self.closed = False
        self.write_seconds = 0.0
        self.frames_written = 0
        self.worker = threading.Thread(target=self._run, name="video-encoder", daemon=True)
        self.worker.start()

    def _run(self):
        try:
            if callable(self.writer):
                self.writer = self.writer()
            # Pre-roll chỉ giao một danh sách tham chiếu, worker ghi theo đúng thứ tự.
            while self.initial_frames:
                self._write(self.initial_frames.popleft())
                self.initial_frames_written += 1
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                self._write(frame)
        except BaseException as error:
            self.error = error
        finally:
            self.initial_frames.clear()
            try:
                if not callable(self.writer):
                    self.writer.release()
            except BaseException as error:
                self.error = self.error or error

    def _write(self, frame):
        started = time.perf_counter()
        self.writer.write(frame)
        self.write_seconds += time.perf_counter() - started
        self.frames_written += 1

    def _check(self):
        if self.error is not None:
            raise RuntimeError(f"Worker ghi video thất bại: {self.error}") from self.error

    def _put(self, item):
        while True:
            self._check()
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def write(self, frame):
        if self.closed:
            raise RuntimeError("Bộ ghi đã đóng.")
        if not self.copy_frames and frame.flags.writeable:
            raise ValueError('Ghi không copy chỉ nhận frame bất biến, không được sửa sau enqueue.')
        self._put(frame.copy() if self.copy_frames else frame)

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._put(None)
        finally:
            self.worker.join()
        self._check()


def open_writer(path, fps, size, config, initial_frames=None, copy_frames=True):
    """Clip alert có pre-roll sẽ mở encoder/ghi pre-roll ở worker; các writer cũ giữ mặc định copy."""
    def create():
        if config["encoder"] == "nvenc":
            return NVENCWriter(path, fps, size, config["preset"], config["cq"], config.get("ffmpeg"))
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Không mở được bộ ghi: {path}")
        return writer
    if config["async_encode"]:
        # Chỉ trì hoãn mở encoder cho clip alert; video tổng vẫn kiểm tra mở ngay như trước.
        return ThreadedWriter(create if initial_frames is not None else create(), config["queue_size"],
                              initial_frames or (), copy_frames)
    writer = create()
    for frame in initial_frames or ():
        writer.write(frame)
    return writer
