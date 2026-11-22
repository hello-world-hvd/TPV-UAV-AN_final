"""Export model YOLO detect/OBB sang TensorRT FP16 trên server; giữ metadata lớp và task của checkpoint."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1]/'models/yolo26m_finetune_detect.pt')
    parser.add_argument('--imgsz', type=int, default=960)
    args = parser.parse_args()
    path = args.model.expanduser().resolve()
    engine = YOLO(str(path)).export(format='engine', device=0, imgsz=args.imgsz,
                                    half=True, batch=1, dynamic=False, workspace=4, nms=False)
    print(f'Đã export: {engine}')


if __name__ == '__main__': main()
