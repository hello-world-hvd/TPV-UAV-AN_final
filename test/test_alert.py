"""Test cảnh báo: unit kiểm tra ảnh/clip/cửa sổ; video chạy toàn pipeline, hỗ trợ --no-save-video."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from alerts.manager import AlertManager


def check_rules():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = yaml.safe_load((ROOT/'configs/alerts.yaml').read_text())
        config['confirmation_seconds'] = 0
        path = root/'alerts.yaml'
        path.write_text(yaml.safe_dump(config))
        fps = 10
        video = root/'viz.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (96,64))
        managers = [AlertManager(root/'on', path, True), AlertManager(root/'off', path, False)]
        def group(gid, name):
            return dict(group_id=gid,track_ids=[gid],status='confirmed',behaviors=[dict(name=name,source='group_rule',status='suspected')])
        for index in range(150):
            frame = np.zeros((64,96,3),np.uint8)
            cv2.rectangle(frame,(10,10),(70,50),(0,255,0),2)
            writer.write(frame)
            groups = []
            if 10 <= index < 100: groups.append(group(1,'rửa xe'))
            if 80 <= index < 90: groups.append(group(2,'sửa xe'))
            if index >= 95: groups.append(group(3,'đặt biển báo sai quy định'))
            for manager in managers: manager.update(frame,index,fps,groups)
        writer.release()
        for manager in managers:
            report = manager.finalize(video,fps,150)
            assert report['event_count']==3 and report['package_count']==2
            first = json.loads((manager.output/'alert_000001/metadata.json').read_text())
            assert first['behavior_names']==['rửa xe','sửa xe']
            assert first['missing_pre_seconds']==1
            assert first['actual_start']==0 and first['actual_end']==9
            assert len(first['events'])==2
            for event in first['events']:
                image=cv2.imread(str(manager.output/'alert_000001'/event['snapshot']))
                assert image is not None and image[10,10,1]>150
            if manager.save_video:
                capture=cv2.VideoCapture(str(manager.output/'alert_000001/clip.mp4'))
                assert capture.get(cv2.CAP_PROP_FRAME_COUNT)==90
                assert capture.get(cv2.CAP_PROP_FPS)==fps
                capture.release()
            else:
                assert not list(manager.output.rglob('*.mp4'))
        second=json.loads((managers[0].output/'alert_000002/metadata.json').read_text())
        assert second['behavior_names']==['sửa xe','đặt biển báo sai quy định']
        assert second['actual_start']==7.5 and second['missing_post_seconds']==2.5
        # Mất hành vi đủ thời gian mới phát lại; không phát lại khi vẫn duy trì.
        manager=AlertManager(root/'rearm',path,False)
        frame=np.zeros((64,96,3),np.uint8)
        for i in range(45):
            manager.update(frame,i,fps,[group(1,'rửa xe')] if i<5 or i>=35 else [])
        assert len(manager.events)==2
    print('Đạt: chống lặp, đa hành vi/ảnh, clip -2/+8s, cắt biên video và tắt lưu clip.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['unit','video'],default='unit')
    args,remaining=parser.parse_known_args()
    if args.mode=='unit':
        if remaining: parser.error('Unit không nhận tham số video.')
        check_rules()
    else:
        from test_obj_detection import main as run_video
        sys.argv=[sys.argv[0],'--mode','video',*remaining,'--alerts','--visualization','group_ids']
        run_video()


if __name__=='__main__': main()
