"""Kiểm tra ưu tiên biển treo cây/cột, chân biển trên vỉa hè, mask warp/reset và không gọi model khi không cần."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sidewalk.signs import SignVerifier
from behavior.engine import BehaviorEngine


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'sidewalk.yaml'
        config = yaml.safe_load((ROOT/'configs/sidewalk.yaml').read_text())
        config['signs']['confirm_seconds'] = .4
        path.write_text(yaml.safe_dump(config))
        def row(i,label,box):
            return dict(track_id=i,label=label,bbox_xyxy=box,position_source='detection',detection_age_seconds=0)
        sign = row(1,'Biển quảng cáo',[30,20,60,50])
        group = dict(group_id=7,kind='biển treo trên cây hoặc cột ứng viên',track_ids=[1,2],status='candidate')
        frame = np.zeros((100,100,3),np.uint8)
        motion = dict(reliable=True,matrix=np.eye(2,3),scene_cut=False)
        with patch('sidewalk.signs.SidewalkDetector') as factory:
            for label in ('Cột điện','Thân cây'):
                verifier = SignVerifier(path)
                tracks = [sign,row(2,label,[45,0,50,95])]
                assert not verifier.update(frame,tracks,[group],0,motion)
                assert verifier.update(frame,tracks,[group],.4,motion) == {7:'Treo biển báo'}
            factory.assert_not_called()
            # Biển đặt: Mask2Former chạy theo lịch, không theo số biển hoặc mọi frame.
            group = dict(group_id=7,kind='Biển quảng cáo',track_ids=[1],status='candidate')
            labels = np.zeros((100,100),np.uint8)
            labels[49:65,20:70] = 2
            factory.return_value.predict.return_value = dict(labels=labels)
            verifier = SignVerifier(path)
            assert not verifier.update(frame,[sign],[group],0,motion)
            assert verifier.update(frame,[sign],[group],.4,motion) == {7:'Đặt biển báo'}
            assert factory.return_value.predict.call_count == 1
            # Warp 5px phải đưa mask sang vị trí mới, không làm đổi group ID.
            moved = row(1,'Biển quảng cáo',[35,20,65,50])
            shift = {**motion,'matrix':np.array([[1,0,5],[0,1,0]])}
            assert verifier.update(frame,[moved],[group],.45,shift) == {7:'Đặt biển báo'}
            assert verifier.labels[55,74] == 2
            # Mất độ tin cậy trước lượt AI kế: dừng xác nhận, không dùng mask lệch.
            assert not verifier.update(frame,[moved],[group],.46,dict(reliable=False))
            assert verifier.labels is None
            # Ngoài vỉa hè hoặc unknown: không hành vi.
            for value in (0,255):
                factory.return_value.predict.return_value = dict(labels=np.full((100,100),value,np.uint8))
                verifier = SignVerifier(path)
                verifier.update(frame,[sign],[group],0,motion)
                assert not verifier.update(frame,[sign],[group],.4,motion)
            # Resume xoá mask và lịch xác nhận nhưng giữ model.
            model = verifier.model
            verifier.update(frame,[sign],[group],.5,motion,resume=True)
            assert verifier.model is model
            # Engine không còn tự suy ra đặt biển chỉ từ label.
            engine = BehaviorEngine.__new__(BehaviorEngine)
            engine.config = dict(enabled=True,max_result_age_seconds=1)
            engine.cache = {}
            engine.signs = SignVerifier(path)
            engine.update(frame,[sign],[group],1,motion)
            assert group['behaviors'] == []
            hung_group = dict(group_id=8,kind='biển treo trên cây hoặc cột ứng viên',track_ids=[1,2],status='candidate')
            pole = row(2,'Thân cây',[45,0,50,95])
            engine.update(frame,[sign,pole],[hung_group],2,motion)
            engine.update(frame,[sign,pole],[hung_group],2.4,motion)
            assert hung_group['behaviors'][0]['name'] == 'Treo biển báo'
            engine.signs.reset()
            factory.return_value.predict.return_value = dict(labels=labels)
            engine.update(frame,[sign],[group],3,motion)
            engine.update(frame,[sign],[group],3.4,motion)
            assert group['behaviors'][0]['name'] == 'Đặt biển báo'
    print('Đạt: treo cây/cột ưu tiên, đặt trên vỉa hè, ngoài/unknown bỏ, cache/warp/reset, giữ ID/model.')


if __name__ == '__main__':
    main()
