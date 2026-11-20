"""Ghi video OpenCV hoặc NVENC; worker có hàng đợi giới hạn và chờ ghi hết khi đóng."""

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
            self.process.stdin.write(frame.tobytes())
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
    def __init__(self, writer, queue_size):
        self.writer = writer
        self.queue = queue.Queue(maxsize=queue_size)
        self.error = None
        self.closed = False
        self.write_seconds = 0.0
        self.frames_written = 0
        self.worker = threading.Thread(target=self._run, name="video-encoder", daemon=True)
        self.worker.start()

    def _run(self):
        try:
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                started = time.perf_counter()
                self.writer.write(frame)
                self.write_seconds += time.perf_counter() - started
                self.frames_written += 1
        except BaseException as error:
            self.error = error
        finally:
            try:
                self.writer.release()
            except BaseException as error:
                self.error = self.error or error

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
        self._put(frame.copy())

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._put(None)
        finally:
            self.worker.join()
        self._check()


def open_writer(path, fps, size, config):
    if config["encoder"] == "nvenc":
        writer = NVENCWriter(path, fps, size, config["preset"], config["cq"], config.get("ffmpeg"))
    else:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Không mở được bộ ghi: {path}")
    return ThreadedWriter(writer, config["queue_size"]) if config["async_encode"] else writer
