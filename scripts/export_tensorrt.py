"""Export YOLO26m fine-tune 13 nhãn sử dụng sang TensorRT FP16 trên GPU server; checkpoint giữ đủ lớp gốc."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imgsz', type=int, default=960)
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / 'models/yolo26m_finetune_detect.pt'
    engine = YOLO(str(path)).export(format='engine', device=0, imgsz=args.imgsz,
                                    half=True, batch=1, dynamic=False, workspace=4, nms=False)
    print(f'Đã export: {engine}')


if __name__ == '__main__': main()
