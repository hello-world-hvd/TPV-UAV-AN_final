"""Worker chốt clip/ảnh/metadata với queue giới hạn; truyền lỗi về luồng chính và chờ đủ khi đóng."""
import queue
import threading
import time


class EvidenceWorker:
    def __init__(self, pending_jobs=1):
        # Một job đang chạy + tối đa pending_jobs chờ, không giữ vô hạn clip/metadata.
        self.queue = queue.Queue(maxsize=pending_jobs)
        self.error = None
        self.closed = False
        self.wait_seconds = 0.
        self.run_seconds = 0.
        self.thread = threading.Thread(target=self._run, name='alert-evidence', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f'Worker minh chứng thất bại: {self.error}') from self.error

    def submit(self, job):
        if self.closed:
            raise RuntimeError('Worker minh chứng đã đóng.')
        started = time.perf_counter()
        while True:
            self.check()
            try:
                self.queue.put(job, timeout=.1)
                self.wait_seconds += time.perf_counter()-started
                return
            except queue.Full:
                continue

    def _run(self):
        while True:
            job = self.queue.get()
            try:
                if job is None:
                    return
                started = time.perf_counter()
                try:
                    job()
                finally:
                    self.run_seconds += time.perf_counter()-started
            except BaseException as error:
                # Tiếp tục drain các job đã nhận để đóng hết encoder, không treo queue.join().
                self.error = self.error or error
            finally:
                self.queue.task_done()

    def wait(self):
        self.queue.join()
        self.check()

    def close(self):
        if self.closed:
            self.check()
            return
        self.closed = True
        try:
            self.queue.join()
        finally:
            self.queue.put(None)
            self.thread.join()
        self.check()
