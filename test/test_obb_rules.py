"""Kiểm tra OBB: khoảng cách thật khi AABB chồng, ghép ID qua đổi góc, giữ polygon khi prediction/camera và đứng yên."""
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from core.geometry import attach_polygon, distance, coverage, transform
from core.config import load_tracking_config
from tracking.tracker import Tracker
from grouping.groups import Grouper


def row(tid, label, center, angle=45):
    r=dict(track_id=tid,label=label,confidence=.9,position_source='detection',detection_age_seconds=0)
    return attach_polygon(r,cv2.boxPoints((center,(100,10),angle)))


def main():
    a=row(1,'Ghế',(100,100))
    b=row(2,'Ghế',(100,140))
    assert distance(a,b)>15 and coverage(b,[a])==0
    config=load_tracking_config(ROOT/'configs/tracking.yaml')
    config.update(min_hits=1,camera_motion=False)
    tracker=Tracker(config)
    frame=np.zeros((400,400,3),np.uint8)
    first=tracker.update(frame,0,[a])[0]
    second=tracker.update(frame,.1,[row(1,'Ghế',(100,100),55)])[0]
    assert first['track_id']==second['track_id'] and 'bbox_polygon' in second
    predicted=tracker.update(frame,.2,None)[0]
    assert 'bbox_polygon' in predicted and 'observed_bbox_polygon' in predicted
    matrix=np.array([[0.,-1.,300.],[1.,0.,0.]])
    tracker.motion.update=lambda *args,**kwargs:dict(reliable=True,scene_cut=False,matrix=matrix,local_boxes={})
    tracker.config['camera_motion']=True
    previous=np.asarray(predicted['bbox_polygon'])
    warped=tracker.update(frame,.3,None)[0]
    assert np.allclose(warped['bbox_polygon'],transform(previous,matrix),atol=1)
    c=yaml.safe_load((ROOT/'configs/grouping.yaml').read_text())
    g=Grouper(c)
    motion=dict(reliable=True,scene_cut=False,matrix=np.eye(2,3))
    for i in range(50):
        bike=row(1,'Xe máy',(150,150),45+(i%2)*10)
        bike['observed_bbox_polygon']=bike['bbox_polygon']
        g.update([bike],i*.1,motion)
    assert bike['stationary_info']['stationary_verified']
    # Biển ở phần góc rỗng của AABB không bị hút vào bàn xoay.
    table=row(3,'Bàn',(100,100))
    sign=attach_polygon(dict(track_id=4,label='Biển quảng cáo',confidence=.9,
        position_source='detection',detection_age_seconds=0),[[65,125],[75,125],[75,135],[65,135]])
    groups=Grouper(dict(c)).update([table,sign],0,motion)
    assert any(x['kind']=='Biển quảng cáo' for x in groups)
    print('Đạt: hình học polygon, ID đổi góc, giữ OBB qua prediction/camera, đứng yên và không gộp góc trống.')

if __name__=='__main__':main()
