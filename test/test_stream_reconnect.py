"""Kiểm tra reconnect giữ model/classifier, reset lịch sử và bỏ frame stream quá cũ; không gọi mạng/GPU."""
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch, MagicMock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pipeline.runner import AlertPipeline
from pipeline.stream import StreamReader
from alerts.manager import AlertManager
import yaml
import numpy as np


def main():
    with tempfile.TemporaryDirectory() as directory, \
         patch('pipeline.runner.Detector') as detector, \
         patch('pipeline.runner.BehaviorEngine') as behavior, \
         patch('pipeline.runner.Tracker'), patch('pipeline.runner.Grouper'), \
         patch('pipeline.runner.AlertManager') as alerts:
        alerts.return_value.finalize.return_value=dict(event_count=0,package_count=0)
        config=dict(inference=dict(device='cpu',detect_fps=None),output=dict(font=None,video={}))
        delivery=MagicMock()
        p=AlertPipeline(config,ROOT/'configs/alerts.yaml',Path(directory)/'first',30,delivery)
        classifier=object()
        p.behavior.classifier=classifier
        original_detector=p.detector
        for i in range(3):
            p.reset_session(Path(directory)/str(i))
            assert p.detector is original_detector and p.behavior.classifier is classifier
            assert p.previous is None and p.index==0 and p.last_slot==-1
        assert detector.call_count==1 and behavior.call_count==1
        assert not original_detector.close.called
        assert p.behavior.cache.clear.call_count==4
        # Đồng hồ bằng chứng không nhảy 5 giây khi resume; slot ghi clip vẫn theo media time.
        p.tracker.tracks = []
        p.tracker.update.return_value = []
        p.tracker.last_motion = dict(scene_cut=False)
        p.grouper.update.return_value = []
        p.detector.detect.return_value = []
        with patch.object(p.timer, 'detect', return_value=([],0)):
            import numpy as np
            frame=np.zeros((100,100,3),np.uint8)
            p.process(frame,0)
            first=p.observation_time
            p.process(frame,150,resume=True)
            assert abs(p.observation_time-first-1/30) < 1e-8
            repeated=[c for c in p.alerts.update.call_args_list if c.kwargs.get('observed') is False]
            assert len(repeated)==149
        p.close()
        original_detector.close.assert_called_once()
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        config=yaml.safe_load((ROOT/'configs/alerts.yaml').read_text())
        config.update(confirmation_seconds=.5,evidence_min_seconds=1.5)
        path=root/'alerts.yaml'; path.write_text(yaml.safe_dump(config))
        manager=AlertManager(root,path,False)
        frame=np.zeros((100,100,3),np.uint8)
        group=dict(group_id=1,track_ids=[1],status='confirmed',behaviors=[dict(name='test',source='rule')])
        manager.update(frame,0,10,[group],observation_time=0)
        for i in range(1,50):
            manager.update(frame,i,10,[group],observed=False,observation_time=0)
        assert manager.event_count==0  # Frame lặp không tạo cảnh báo.
        for i in range(50,56):
            manager.update(frame,i,10,[group],observation_time=(i-49)*.1,observation_seconds=.1)
        assert manager.event_count==1
        report=manager.finalize(None,10,56)
        assert report['packages'][0]['evidence']==[]  # Tổng frame thật chưa đủ 1.5s.
    with patch.object(StreamReader,'_run',return_value=None):
        reader=StreamReader('rtmp://example.invalid/test',dict(timeout_ms=10))
        reader.frames.put((1,time.monotonic()-10,'old'))
        def next_item(timeout):
            reader.stop.set()
            return (2,time.monotonic(),'fresh')
        # Item cũ bị bỏ; reader phải lấy tiếp thay vì trả item quá hạn.
        original_get=reader.frames.get
        calls=[]
        def get(timeout=None, block=True):
            if not block:
                return original_get(block=False)
            calls.append(1)
            return original_get(timeout=timeout) if len(calls)==1 else next_item(timeout)
        with patch.object(reader.frames,'get',side_effect=get):
            assert reader.read(max_age_seconds=2)[2]=='fresh'
        reader.close()
    print('Đạt: reconnect không nạp lại model, giữ classifier, reset lịch sử và bỏ frame cũ.')

if __name__=='__main__':main()
