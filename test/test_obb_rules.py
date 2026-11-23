"""Kiểm tra OBB: khoảng cách thật khi AABB chồng, ghép ID qua đổi góc, giữ polygon khi prediction/camera và đứng yên."""
import sys
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from core.geometry import attach_polygon, distance, coverage, transform, shape
from core.config import load_tracking_config
from tracking.tracker import Tracker, association_costs
from grouping.groups import Grouper


def row(tid, label, center, angle=45):
    r=dict(track_id=tid,label=label,confidence=.9,position_source='detection',detection_age_seconds=0)
    return attach_polygon(r,cv2.boxPoints((center,(100,10),angle)))


def main():
    # Đỉnh chéo kiểu bow-tie và box gần suy biến không được gây TopologyException.
    malformed = dict(bbox_polygon=[[0,0],[20,20],[0,20],[20,0]])
    assert shape(malformed).is_valid and shape(malformed).area == 400
    collapsed = dict(bbox_polygon=[[3781.98617,771.18768],[3781.98618,771.18768],
                                 [3781.98618,771.18769],[3781.98617,771.18769]])
    assert shape(collapsed).is_empty
    t = dict(x=np.array([10.,10.,20.,20.,0,0,0,0]), label='Ghế', polygon=malformed['bbox_polygon'])
    d = attach_polygon(dict(label='Ghế'), [[0,0],[20,0],[20,20],[0,20]])
    costs=association_costs([t],[d],dict(min_iou=.1,max_center_distance=1.))
    assert np.isfinite(costs).all() and costs[0,0] < 1e5
    t['polygon']=collapsed['bbox_polygon']
    assert np.isfinite(association_costs([t],[d],dict(min_iou=.1,max_center_distance=1.))).all()
    # So sánh cost với công thức Shapely cũ trên box xoay ngẫu nhiên.
    rng=np.random.default_rng(42)
    tracks, detections=[],[]
    for i in range(20):
        r=row(i,'Ghế',tuple(rng.uniform(50,300,2)),float(rng.uniform(0,90)))
        box=np.asarray(r['bbox_xyxy'])
        tracks.append(dict(x=np.r_[(box[:2]+box[2:])/2,box[2:]-box[:2],np.zeros(4)],
                           label=r['label'],polygon=r['bbox_polygon']))
        detections.append(row(i,'Ghế',tuple(rng.uniform(50,300,2)),float(rng.uniform(0,90))))
    cfg=dict(min_iou=.1,max_center_distance=1.)
    import tracking.tracker as module
    with patch.object(module,'_obb_geometry',wraps=module._obb_geometry) as prepare:
        costs=association_costs(tracks,detections,cfg)
        assert prepare.call_count == len(tracks)+len(detections)
    for i,t in enumerate(tracks):
        p=np.asarray(t['polygon'])
        aa=shape(dict(bbox_polygon=p))
        size_a=np.sort([np.linalg.norm(p[1]-p[0]),np.linalg.norm(p[2]-p[1])])
        for j,r in enumerate(detections):
            q=np.asarray(r['bbox_polygon']);bb=shape(r)
            inter=aa.intersection(bb).area
            iou=inter/(aa.area+bb.area-inter)
            size_b=np.sort([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[1])])
            delta=np.abs(np.log(np.maximum(size_b,1)/np.maximum(size_a,1))).max()
            dist=np.linalg.norm(q.mean(0)-p.mean(0))/max(np.linalg.norm(size_a),1)
            expected=1-iou+.2*dist if delta<.8 and (iou>=cfg['min_iou'] or dist<=cfg['max_center_distance']) else 1e6
            assert np.isclose(costs[i,j],expected,atol=1e-4), (i,j,costs[i,j],expected,iou,dist)
    # Hai box xa: không gọi phép giao OpenCV.
    distant=row(99,'Ghế',(10000,10000))
    with patch.object(module.cv2,'intersectConvexConvex',wraps=cv2.intersectConvexConvex) as intersect:
        association_costs(tracks,[distant],cfg)
        assert intersect.call_count == 0
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
