"""Test module vỉa hè độc lập: unit không tải model; ảnh/video xuất viz, mask và polygon; chưa ghép pipeline chính."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sidewalk.detector import SidewalkDetector, class_groups, postprocess, restore_result
from core.step_timing import StepTiming
from core.video_writer import open_writer


def prepare_overlay(mask, polygons=()):
    """Chuẩn bị ROI, ảnh màu và contour một lần mỗi kết quả phân vùng."""
    roi = cv2.boundingRect(mask)
    x,y,w,h = roi
    color = np.empty((h,w,3),np.uint8)
    color[:] = (0,255,255)
    contours = [np.rint(ring).astype(np.int32) for item in polygons
                for ring in [item['exterior'], *item['holes']] if len(ring) >= 3]
    return roi, np.ascontiguousarray(mask[y:y+h,x:x+w]), color, contours


def overlay(frame, mask, polygons=(), prepared=None):
    viz = frame.copy()
    (x,y,w,h), roi_mask, color, contours = (
        prepare_overlay(mask,polygons) if prepared is None else prepared)
    if w and h:
        target = viz[y:y+h,x:x+w]
        blended = cv2.addWeighted(target,.55,color,.45,0)
        cv2.copyTo(blended,roi_mask,target)
    if contours:
        cv2.polylines(viz,contours,True,(0,255,255),2,cv2.LINE_AA)
    return viz


def run_video(args, detector):
    stages = StepTiming()
    capture = cv2.VideoCapture(str(args.input))
    writer = None
    index = calls = 0
    detect_seconds = 0.
    started = time.perf_counter()
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Không mở được video: {args.input}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video không có FPS hợp lệ.')
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        segment_fps = min(args.segment_fps or fps, fps)
        print(f'FPS video/output: {fps:g} | FPS phân vùng: {segment_fps:g}', flush=True)
        args.output.mkdir(parents=True,exist_ok=True)
        last_log = started
        with (args.output/'polygons.jsonl').open('w',encoding='utf-8') as records:
            while not args.max_frames or index < args.max_frames:
                with stages.measure('decode'):
                    ok, frame = capture.read()
                if not ok:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    config = yaml.safe_load(args.config.read_text())
                    video = dict(config['video'])
                    video['profile_ffmpeg'] = args.profile_ffmpeg
                    video['input_pixel_format'] = args.input_pixel_format
                    if args.input_pixel_format == 'yuv420p' and (args.encoder or video['encoder']) != 'nvenc':
                        raise ValueError('Đầu vào yuv420p chỉ dành cho NVENC.')
                    if args.encoder:
                        video['encoder'] = args.encoder
                    if video.get('ffmpeg'):
                        video['ffmpeg'] = str((args.config.resolve().parent/video['ffmpeg']).resolve())
                    with stages.measure('open_writer'):
                        writer = open_writer(args.output/'viz.mp4',fps,(width,height),video,copy_frames=False)
                # Lịch theo timestamp nguồn, tránh sai số làm tròn bước frame khi FPS không chia hết.
                if calls == 0 or index/fps + 1e-8 >= calls/segment_fps:
                    step = time.perf_counter()
                    result = detector.predict(frame)
                    detect_seconds += time.perf_counter()-step
                    calls += 1
                    with stages.measure('overlay_prepare'):
                        prepared = prepare_overlay(result['mask'],result['polygons'])
                    records.write(json.dumps(dict(frame=index,time=index/fps,polygons=result['polygons']),ensure_ascii=False)+'\n')
                with stages.measure('overlay'):
                    viz = overlay(frame,result['mask'],prepared=prepared)
                if args.input_pixel_format == 'yuv420p':
                    with stages.measure('bgr_to_i420'):
                        viz = cv2.cvtColor(viz,cv2.COLOR_BGR2YUV_I420)
                with stages.measure('encode_enqueue'):
                    viz.setflags(write=False)  # Chuyển quyền dùng frame mới cho worker, không sửa lại.
                    writer.write(viz)
                index += 1
                now = time.perf_counter()
                if now-last_log >= 1:
                    print(f'{index}/{total} frame | Phân vùng: {calls} lượt | Xử lý: {index/(now-started):.2f} FPS',flush=True)
                    last_log = now
    finally:
        capture.release()
        if writer is not None:
            with stages.measure('finalize_video'):
                writer.release()
    elapsed = time.perf_counter()-started
    stats = dict(frames=index,segment_calls=calls,source_fps=fps,segment_fps=segment_fps,
                 elapsed_seconds=elapsed,processing_fps=index/max(elapsed,1e-8),
                 average_segment_ms=detect_seconds*1000/max(calls,1))
    stats['stages'] = {**stages.summary(), **detector.timings.summary()}
    stats['encoder_worker'] = writer.timing_summary() if writer else {}
    if writer is not None and args.profile_ffmpeg and hasattr(writer.writer,'profile_summary'):
        stats['ffmpeg_profile'] = writer.writer.profile_summary()
        print('FFmpeg profile:',json.dumps(stats['ffmpeg_profile'],ensure_ascii=False))
    print('Chi tiết writer (worker_write = ghi pipe FFmpeg khi dùng NVENC):')
    for name, step in stats['encoder_worker'].items():
        print(f"  {name:18s}: {step['seconds']:.3f}s | {step['average_ms']:.3f} ms/lượt | {step['calls']} lượt")
    if writer is not None and hasattr(writer.writer,'profile_summary'):
        stats['ffmpeg_profile'] = writer.writer.profile_summary()
        if stats['ffmpeg_profile']:
            print(f"Chi tiết FFmpeg: {stats['ffmpeg_profile']}",flush=True)
    stages.show('Đọc/viz/ghi video')
    detector.timings.show('Phân vùng (model đã đồng bộ GPU)')
    print('encode_enqueue gồm copy/đẩy/chờ queue; finalize_video chờ ghi hết. Thời gian worker chạy chồng, không cộng vào tổng.')
    (args.output/'timing.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(f'Hoàn tất: {index} frame | {elapsed:.2f}s | {stats["processing_fps"]:.2f} FPS | {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['unit', 'image', 'video'], default='unit')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sidewalk.yaml')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/sidewalk')
    parser.add_argument('--segment-fps', type=float, default=1., help='Lượt phân vùng/giây nguồn; 0: mọi frame.')
    parser.add_argument('--input-pixel-format',choices=['bgr24','yuv420p'],default='bgr24',help='Thử I420 giảm một nửa dữ liệu pipe NVENC; mặc định giữ BGR.')
    parser.add_argument('--encoder',choices=['nvenc','opencv'],help='Ghi đè encoder trong YAML; local không NVIDIA chọn opencv.')
    parser.add_argument('--profile-ffmpeg',action='store_true',help='Đo filter chuyển màu và lời gọi encoder FFmpeg; log có overhead, chỉ bật khi chẩn đoán.')
    parser.add_argument('--max-frames', type=int, default=0, help='0: toàn bộ video.')
    args = parser.parse_args()
    if not np.isfinite(args.segment_fps) or args.segment_fps < 0 or args.max_frames < 0:
        parser.error('segment-fps/max-frames phải không âm.')
    config = yaml.safe_load(args.config.read_text())
    if args.mode == 'unit':
        assert class_groups({0:'road',1:'sidewalk',2:'pedestrian_area',3:'car'}) == [[3],[0],[1,2]]
        scores = np.zeros((3,100,100), np.float32)
        scores[:] = np.array([.02,.02,.96])[:,None,None]
        scores[:,30:60,30:60] = np.array([.96,.02,.02])[:,None,None]
        scores[:,:5,:] = 1/3
        result = postprocess(scores, config)
        labels, mask, polygons = (result[k] for k in ('labels','mask','polygons'))
        assert (labels[:5] == 255).all() and not mask[40,40] and mask[80,80] == 255
        assert len(polygons) == 1 and len(polygons[0]['holes']) == 1
        before = np.asarray(result['polygons'][0]['exterior']).copy()
        restored = restore_result(result,(200,300))
        assert restored['mask'].shape == (300,200)
        assert np.allclose(restored['polygons'][0]['exterior'],before*np.array([2,3]))
        assert set(np.unique(restored['mask'])) <= {0,255}
        assert overlay(np.zeros((300,200,3),np.uint8),restored['mask'],restored['polygons']).any()
        scores[:] = np.array([.02,.96,.02])[:,None,None]
        assert not postprocess(scores,config)['mask'].any()
        scores[:] = np.array([.02,.02,.96])[:,None,None]
        scores[:,40:60,40:60] = np.array([.96,.02,.02])[:,None,None]
        scores[:,15:18,15:18] = np.array([.96,.02,.02])[:,None,None]
        smooth = postprocess(scores,config)
        assert smooth['mask'][16,16] == 255 and smooth['mask'][50,50] == 0
        assert (smooth['labels'][smooth['mask'] != 0] == 2).all()
        assert len(smooth['polygons'][0]['holes']) == 1
        frame = np.full((40,60,3),101,np.uint8)
        mask = np.zeros((40,60),np.uint8)
        mask[10:30,15:45] = 255
        mask[18:22,25:30] = 0
        prepared = prepare_overlay(mask)
        viz = overlay(frame,mask,prepared=prepared)
        assert prepared[0] == (15,10,30,20)
        assert np.array_equal(viz[mask == 0],frame[mask == 0])
        expected = cv2.addWeighted(frame,.55,np.full_like(frame,(0,255,255)),.45,0)
        assert np.array_equal(viz[mask != 0],expected[mask != 0])
        assert np.array_equal(overlay(frame,np.zeros_like(mask)),frame)
        assert (frame == 101).all()  # Không sửa ảnh nguồn đang dùng cho bước khác.
        print('Đạt: ROI/blend uint8/giữ lỗ và nền/cache overlay; taxonomy, mask/nhãn, vùng chưa chắc, loại đường và giữ lỗ polygon.')
        return
    if args.input is None:
        parser.error('--mode image/video cần --input')
    from core.config import apply_cpu_threads
    apply_cpu_threads({'runtime': config.get('runtime',{'cv_threads':4,'torch_threads':2})})
    if args.mode == 'video':
        run_video(args, SidewalkDetector(args.config))
        return
    frame = cv2.imread(str(args.input))
    if frame is None:
        parser.error(f'Không đọc được ảnh {args.input}')
    detector = SidewalkDetector(args.config)
    started = time.perf_counter()
    result = detector.predict(frame)
    elapsed = time.perf_counter()-started
    args.output.mkdir(parents=True,exist_ok=True)
    for name in ('mask','labels'):
        if not cv2.imwrite(str(args.output/f'{name}.png'),result[name]):
            raise RuntimeError(f'Không ghi được {name}')
    viz = overlay(frame,result['mask'],result['polygons'])
    if not cv2.imwrite(str(args.output/'viz.jpg'),viz):
        raise RuntimeError('Không ghi được viz')
    (args.output/'polygons.json').write_text(json.dumps(result['polygons'],ensure_ascii=False),encoding='utf-8')
    print(f'Suy luận + hậu xử lý: {elapsed:.3f}s | {len(result["polygons"])} polygon | {args.output}')


if __name__ == '__main__':
    main()
