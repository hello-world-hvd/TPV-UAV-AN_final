"""Ghi video OpenCV hoặc NVENC; worker có hàng đợi giới hạn và chờ ghi hết khi đóng."""

from collections import deque
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import imageio_ffmpeg


class NVENCWriter:
    def __init__(self, path, fps, size, preset, cq, executable=None, profile=False, input_pixel_format="bgr24"):
        width, height = size
        if width % 2 or height % 2:
            raise ValueError("NVENC yuv420p cần chiều rộng/cao chẵn để giữ nguyên độ phân giải.")
        if input_pixel_format not in ("bgr24", "yuv420p"):
            raise ValueError("input_pixel_format phải là bgr24 hoặc yuv420p.")
        self.input_pixel_format = input_pixel_format
        if input_pixel_format not in ("bgr24", "yuv420p"):
            raise ValueError("Đầu vào NVENC phải là bgr24 hoặc yuv420p/I420.")
        self.input_pixel_format = input_pixel_format
        self.profile = profile
        self.size = size
        self.closed = False
        self.log_path = Path(path).with_suffix(".encoder.log")
        self.log = self.log_path.open("w+b")
        command = [str(executable or imageio_ffmpeg.get_ffmpeg_exe()), "-nostdin", "-hide_banner", "-loglevel", "error",
                   "-y", "-f", "rawvideo", "-pixel_format", input_pixel_format, "-video_size", f"{width}x{height}",
                   "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "h264_nvenc",
                   "-preset", preset, "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
                   "-pix_fmt", "yuv420p", str(path)]
        if profile:
            command[command.index('-loglevel')+1] = 'info'
            # bench đo filter chuyển màu; benchmark_all đo lời gọi decoder/encoder CPU wall.
            command[1:1] = ['-benchmark_all']
            available = subprocess.run([command[0],'-hide_banner','-filters'],capture_output=True,text=True,check=True).stdout
            self.bench_available = bool(re.search(r'(?m)^.*\sbench\s',available))
            if self.bench_available:
                position = command.index('-c:v')
                command[position:position] = ['-vf','bench=start,format=yuv420p,bench=stop']
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def write(self, frame):
        expected = (self.size[1]*3//2,self.size[0]) if self.input_pixel_format == "yuv420p" else (self.size[1],self.size[0],3)
        if frame.shape != expected or frame.dtype.name != "uint8":
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

    def profile_summary(self):
        """Log chẩn đoán, không coi encode_video là thời gian kernel GPU hay cộng vào wall tổng."""
        if not self.profile:
            return {}
        text = self.log_path.read_text(errors='replace')
        stages = {}
        for user, system, real, name in re.findall(
                r'(\d+) user\s+(\d+) sys\s+(\d+) real\s+(decode_video|encode_video|flush_video)',text):
            step = stages.setdefault(name,dict(calls=0,user_seconds=0.,system_seconds=0.,wall_seconds=0.))
            step['calls'] += 1
            step['user_seconds'] += int(user)/1e6
            step['system_seconds'] += int(system)/1e6
            step['wall_seconds'] += int(real)/1e6
        values = [float(value) for value in re.findall(r'\bt:([0-9.]+)',text)]
        stages['bgr_to_yuv_filter'] = dict(available=self.bench_available,calls=len(values),seconds=sum(values) if values else None,
            average_ms=sum(values)*1000/max(len(values),1))
        return dict(log=str(self.log_path),stages=stages,
            note='decode_video không gồm chờ đọc pipe; encode_video/flush_video là thời gian gọi encoder, không phải GPU kernel; bench đo chuỗi filter đổi màu nếu FFmpeg hỗ trợ; available=false nghĩa là chưa đo được chuyển màu. Không cộng các mục vào wall pipeline.')

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
        self.copy_seconds = 0.0
        self.queue_put_seconds = 0.0
        self.frames_enqueued = 0
        self.frames_copied = 0
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
        if self.copy_frames:
            started = time.perf_counter()
            frame = frame.copy()
            self.copy_seconds += time.perf_counter()-started
            self.frames_copied += 1
        started = time.perf_counter()
        self._put(frame)
        self.queue_put_seconds += time.perf_counter()-started
        self.frames_enqueued += 1

    def timing_summary(self):
        """Đọc sau release; queue_put gồm enqueue và chờ chỗ trống, không tính sentinel đóng."""
        return {
            'copy_frame': dict(seconds=self.copy_seconds, calls=self.frames_copied,
                average_ms=self.copy_seconds*1000/max(self.frames_copied,1)),
            'queue_put_wait': dict(seconds=self.queue_put_seconds, calls=self.frames_enqueued,
                average_ms=self.queue_put_seconds*1000/max(self.frames_enqueued,1)),
            'worker_write': dict(seconds=self.write_seconds, calls=self.frames_written,
                average_ms=self.write_seconds*1000/max(self.frames_written,1)),
        }

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
            return NVENCWriter(path, fps, size, config["preset"], config["cq"], config.get("ffmpeg"), config.get("profile_ffmpeg", False), config.get("input_pixel_format", "bgr24"))
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
