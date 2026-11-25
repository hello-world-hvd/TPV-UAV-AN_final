"""Kiểm tra chốt minh chứng chạy nền, truyền lỗi và pre-roll giữ đúng thứ tự không copy lại."""
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from alerts.manager import AlertManager
from alerts.worker import EvidenceWorker
from core.video_writer import ThreadedWriter


def main():
    class Writer:
        def __init__(self):
            self.frames = []
            self.closed = False
        def write(self, frame):
            self.frames.append(frame)
        def release(self):
            self.closed = True
    frames = [np.full((4,4,3), i, np.uint8) for i in range(6)]
    for frame in frames:
        frame.setflags(write=False)
    target = Writer()
    writer = ThreadedWriter(lambda: target, 2, frames[:3], copy_frames=False)
    for frame in frames[3:]:
        writer.write(frame)
    writer.release()
    assert target.closed and writer.frames_written == 6
    assert all(a is b for a, b in zip(target.frames, frames))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = yaml.safe_load((ROOT/'configs/alerts.yaml').read_text())
        config.update(confirmation_seconds=0, post_seconds=.2, evidence_min_seconds=0)
        path = root/'alerts.yaml'
        path.write_text(yaml.safe_dump(config))
        entered, proceed = threading.Event(), threading.Event()
        reports = []
        manager = AlertManager(root, path, False, on_package=lambda folder, report: reports.append(report))
        original = manager._seal
        def slow_seal(*args):
            entered.set()
            assert proceed.wait(5)
            original(*args)
        group = dict(group_id=1, track_ids=[1], status='confirmed',
                     behaviors=[dict(name='test', source='group_rule')])
        try:
            with patch.object(manager, '_seal', side_effect=slow_seal):
                for index in range(3):
                    manager.update(frames[0], index, 10, [group])
                assert entered.wait(2) and not reports
                # Chốt ảnh đang bị giữ ở worker nhưng luồng chính vẫn nhận frame mới.
                manager.update(frames[0], 3, 10, [group])
                proceed.set()
                manager.close()
                manager.finalize(None, 10, 4)
                assert len(reports) == 1
        finally:
            proceed.set()
            manager.close()

    worker = EvidenceWorker()
    worker.submit(lambda: (_ for _ in ()).throw(ValueError('test lỗi')))
    try:
        worker.close()
    except RuntimeError as error:
        assert isinstance(error.__cause__, ValueError)
    else:
        raise AssertionError('Worker phải truyền lỗi về luồng chính.')
    assert not worker.thread.is_alive()
    print('Đạt: pre-roll đúng thứ tự/chung bộ nhớ, chốt nền, đóng đủ và truyền lỗi.')


if __name__ == '__main__':
    main()
