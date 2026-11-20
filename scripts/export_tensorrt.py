"""Export phần box YOLOE sang TensorRT FP16, giữ metadata nhãn và bỏ đầu ra mask."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLOE
from ultralytics.utils.export.engine import torch2onnx, onnx2engine


class BoxesExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.nc = len(model.names)
        self.end2end = getattr(model.model[-1], "end2end", False)

    def forward(self, image):
        result = self.model(image)
        if isinstance(result, tuple):
            result = result[0]
        return result[..., :6] if self.end2end else result[:, :4 + self.nc, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.cuda.set_device(0)
    for filename in ("yoloe-v8s-single.pt", "yoloe-sidewalk-extra-v5.pt"):
        path = root / "models" / filename
        model = YOLOE(str(path)).model.to("cuda").float().eval()
        model.fuse()
        head = model.model[-1]
        head.export, head.dynamic, head.format = True, False, "onnx"
        head.shape = None
        wrapper = BoxesExport(model).eval()
        image = torch.zeros(1, 3, args.imgsz, args.imgsz, device="cuda")
        with torch.no_grad():
            output = wrapper(image)
            print(filename, "box output:", tuple(output.shape), flush=True)
            onnx = torch2onnx(wrapper, image, path.with_suffix(".onnx"), opset=17)
        metadata = {"task": "detect", "names": model.names, "stride": int(model.stride.max()),
                    "batch": 1, "imgsz": [args.imgsz, args.imgsz],
                    "end2end": wrapper.end2end, "args": {"half": True, "nms": False, "dynamic": False}}
        engine = onnx2engine(onnx, path.with_suffix(".engine"), workspace=4, half=True,
                             shape=tuple(image.shape), metadata=metadata)
        print(f"Đã export: {engine}", flush=True)


if __name__ == "__main__":
    main()
