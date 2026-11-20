"""Test grouping: kiểm tra quy tắc bằng dữ liệu tổng hợp hoặc chạy video có bbox nhóm và JSONL."""
import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from grouping.groups import Grouper


def check_rules():
    config = yaml.safe_load((ROOT / 'configs/grouping.yaml').read_text())
    grouper = Grouper(config)
    def row(i, label, box):
        return dict(track_id=i, label=label, bbox_xyxy=box, position_source='detection', detection_age_seconds=0)
    rows = [row(1, 'xe máy', [0,0,50,100]), row(2, 'người', [40,0,70,100]),
            row(3, 'bàn nhựa', [200,0,250,100]), row(4, 'ghế nhựa', [240,0,270,80]),
            row(5, 'biển quảng cáo', [400,0,430,40]), row(6, 'cột điện', [420,0,440,150]),
            row(7, 'vật liệu xây dựng', [500,0,600,100]), row(8, 'xe chở rác', [700,0,800,100]),
            row(9, 'người', [720,0,750,60])]
    motion = dict(reliable=True, scene_cut=False, matrix=np.eye(2,3))
    # Người phải ở gần đủ lâu; xe đang xét đứng yên không được xuất candidate sớm.
    first = grouper.update(rows, 0, motion)
    assert not any(g['kind'] == 'người và xe máy đứng yên' for g in first)
    grouper.update(rows, 1, motion)
    assert not any(g['kind'] == 'người và xe máy đứng yên' for g in grouper.update(rows, 2.1, motion))
    confirmed = grouper.update(rows, 3.2, motion)
    assert any(g['stationary_verified'] is True for g in confirmed)
    assert all(g['status'] == 'candidate' for g in confirmed if 'ứng viên' in g['kind'])
    grouper.update(rows, 4, {**motion, 'reliable': False})
    assert not any(g['stationary_verified'] is True for g in grouper.update(rows, 4.1, motion))
    # Ghế cách xa hơn được nối vào bàn; không gộp hai bàn qua chuỗi ghế.
    furniture = [row(20,'bàn nhựa',[0,0,60,60]), row(21,'ghế nhựa',[100,0,120,30]),
                 row(22,'ghế nhựa',[150,0,170,30]), row(23,'người',[60,0,90,60])]
    g = Grouper(config)
    grouped = g.update(furniture,0,motion)
    assert len(grouped)==1 and 23 not in grouped[0]['track_ids']
    assert 23 in g.update(furniture,1.1,motion)[0]['track_ids']
    # Người đi ngang một frame không được ghép.
    assert 23 not in Grouper(config).update(furniture,0,motion)[0]['track_ids']
    # Xe chạy tích luỹ đủ xa phải liên tục reset trạng thái đứng yên.
    moving = Grouper(config)
    for i in range(40):
        x = i * 3
        current = [row(1,'xe máy',[x,0,x+50,100]), row(2,'người',[x+40,0,x+70,100])]
        assert not any(g['stationary_verified'] for g in moving.update(current,i*.1,motion))
    print('Đạt: xe chạy không tạo nhóm, xe dừng có dwell-time, ghế mở rộng và người ở gần đủ lâu.')



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['unit', 'video'], default='unit')
    args, remaining = parser.parse_known_args()
    if args.mode == 'unit':
        if remaining:
            parser.error('Chế độ unit không nhận tham số video.')
        check_rules()
    else:
        # Dùng chung runner để không lặp lại decode, detect, tracking và ghi video.
        from test_obj_detection import main as run_video
        sys.argv = [sys.argv[0], '--mode', 'video', *remaining]
        run_video()


if __name__=='__main__':
    main()
