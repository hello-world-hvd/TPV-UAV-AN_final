"""Test hành vi: unit kiểm tra quy tắc; video chỉ vẽ bbox nhóm G và nhóm có hành vi B."""
import argparse
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from behavior.engine import BehaviorEngine
from behavior.classifier import ActivityClassifier


def check_rules():
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder)
        (p/'weights.pt').touch()
        (p/'metadata.json').touch()
        config = yaml.safe_load((ROOT/'configs/behavior.yaml').read_text())
        config.update(weights='weights.pt', metadata='metadata.json')
        (p/'behavior.yaml').write_text(yaml.safe_dump(config))
        engine = BehaviorEngine(p/'behavior.yaml', 'cpu')
        def row(i, label):
            return dict(track_id=i, label=label, bbox_xyxy=[10,10,80,90], position_source='detection')
        tracks = [row(1,'xô chậu'), row(2,'người'), row(3,'xe máy')]
        def group(i, kind, ids, **extra):
            return dict(group_id=i, kind=kind, track_ids=ids, bbox_xyxy=[10,10,80,90], status='confirmed', **extra)
        groups = [group(1,'đồ dùng và hàng hoá',[1,2]),
                  group(2,'người và xe máy đứng yên',[2,3],stationary_verified=True),
                  group(3,'vật liệu xây dựng',[]), group(4,'biển quảng cáo',[]),
                  group(5,'xe rác và người lái ứng viên',[])]
        frame=np.zeros((100,150,3),np.uint8)
        with patch('behavior.engine.ActivityClassifier') as factory:
            factory.return_value.predict.return_value=[[.05,.05,.9]]
            engine.update(frame,tracks,groups,0)
            assert groups[1]['behaviors'][0]['name']=='rửa xe'
            assert len(groups[0]['behaviors'])==2
            assert groups[2]['behaviors'][0]['name']=='tập kết vật liệu xây dựng sai quy định'
            assert groups[3]['behaviors'][0]['name']=='đặt biển báo sai quy định'
            assert groups[4]['behaviors'][0]['name']=='tập kết rác sai quy định'
            factory.return_value.predict.return_value=[[.9,.05,.05]]
            engine.update(frame,tracks,groups,1)
            assert not groups[1]['behaviors']
            factory.return_value.predict.return_value=[[.05,.9,.05]]
            engine.update(frame,tracks,groups,2)
            assert groups[1]['behaviors'][0]['name']=='sửa xe'
        tensor=ActivityClassifier.preprocess(frame,[10,10,80,90],.1)
        assert tensor.shape==(3,224,224) and np.isfinite(tensor.numpy()).all()
    print('Đạt: nhãn quy tắc, rửa/sửa xe, other_activity và crop RGB letterbox 224.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['unit', 'video'], default='unit')
    args, remaining = parser.parse_known_args()
    if args.mode == 'unit':
        if remaining:
            parser.error('Chế độ unit không nhận tham số video.')
        check_rules()
    else:
        from test_obj_detection import main as run_video
        sys.argv = [sys.argv[0], '--mode', 'video', *remaining, '--visualization', 'group_ids']
        run_video()


if __name__=='__main__': main()
