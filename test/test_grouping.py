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
    def row(i, label, box):
        return dict(track_id=i, label=label, bbox_xyxy=box, position_source='detection', detection_age_seconds=0)
    motion = dict(reliable=True, scene_cut=False, matrix=np.eye(2,3))
    def sample(i, moving=False):
        x = i * 5 if moving else (.4 if i % 2 else -.4)
        return [row(1,'Xe máy',[x,0,x+50,100]), row(2,'Người',[x+40,0,x+70,100]),
                row(3,'Bàn',[60,0,120,100])]
    g = Grouper(dict(config))
    for i in range(40):
        rows = sample(i)
        groups = g.update(rows, i*.1, motion)
    assert rows[0]['stationary_info']['stationary_verified']
    assert any(x['kind']=='người và xe máy đứng yên' and 2 in x['track_ids'] for x in groups)
    # Người có thể ở gần cả hàng hoá và xe, quan hệ xe không bị cụm đồ dùng chặn.
    assert any(x['kind']=='đồ dùng và hàng hoá' and 2 in x['track_ids'] for x in groups)
    rows=sample(40);g.update(rows,4,{**motion,'reliable':False})
    assert rows[0]['stationary_info']['stationary_verified']
    rows=sample(41);g.update(rows,4 + config['relation_grace_seconds'] + .1,{**motion,'reliable':False})
    assert not rows[0]['stationary_info']['stationary_verified']
    moving = Grouper(dict(config))
    for i in range(40):
        rows=sample(i,True)
        assert not any(x['kind']=='người và xe máy đứng yên' for x in moving.update(rows,6+(i-40)*.1,motion))
    # Xe đã dừng rồi chạy phải được thu hồi sau thời gian kiểm tra chuyển động.
    for i in range(40,80):
        rows=sample(i-40,True);g.update(rows,6+(i-40)*.1,motion)
    assert not rows[0]['stationary_info']['stationary_verified']
    stable = Grouper(dict(config))
    chairs = [row(10, 'Ghế', [0,0,20,30]), row(11, 'Ghế', [22,0,42,30])]
    initial = stable.update(chairs, 0, motion)[0]['group_id']
    # Mất vật làm mốc rồi thêm lại vẫn là nhóm cũ.
    assert stable.update(chairs[1:], .1, motion)[0]['group_id'] == initial
    assert stable.update(chairs, .2, motion)[0]['group_id'] == initial
    assert stable.update([], .3, motion) == []
    assert stable.update(chairs, .4, motion)[0]['group_id'] == initial
    signage = Grouper(dict(config))
    rows = [row(20, 'Bàn', [0,0,100,100]), row(21, 'Biển quảng cáo', [20,20,60,60]),
            row(22, 'Biển quảng cáo', [200,200,230,230])]
    grouped = signage.update(rows, 0, motion)
    goods = next(r for r in grouped if r['kind'] == 'đồ dùng và hàng hoá')
    assert goods['track_ids'] == [20, 21]
    assert sum(21 in r['track_ids'] for r in grouped) == 1
    assert any(r['kind'] == 'Biển quảng cáo' and r['track_ids'] == [22] for r in grouped)
    print('Đạt: bbox rung, hysteresis, camera gián đoạn, xe di chuyển và ghép người–xe song song đồ dùng.')


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
