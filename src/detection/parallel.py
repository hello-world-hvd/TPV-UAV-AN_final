"""Chạy hai engine TensorRT độc lập trên CUDA stream riêng, không sửa thư viện Ultralytics."""

from concurrent.futures import ThreadPoolExecutor

import torch


class ParallelEngines:
    def __init__(self, models, device):
        self.device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
        self.predictors = [model.predictor for model, _ in models]
        self.streams = [torch.cuda.Stream(device=self.device) for _ in models]
        for predictor in self.predictors:
            backend = predictor.model.backend
            if backend.dynamic or not backend.is_trt10:
                raise ValueError("Chạy song song yêu cầu engine TensorRT 10+ có kích thước cố định.")
        torch.cuda.synchronize(self.device)
        self.pool = ThreadPoolExecutor(max_workers=len(models), thread_name_prefix="trt-model")

    def _run(self, index, frame):
        predictor = self.predictors[index]
        backend = predictor.model.backend
        stream = self.streams[index]
        with torch.cuda.device(self.device), torch.cuda.stream(stream), torch.inference_mode():
            image = predictor.preprocess([frame])
            if backend.fp16 and image.dtype != torch.float16:
                image = image.half()
            if tuple(image.shape) != tuple(backend.bindings['images'].shape):
                raise ValueError("Kích thước đầu vào không khớp engine TensorRT.")
            for name, binding in backend.bindings.items():
                pointer = image.data_ptr() if name == 'images' else binding.data.data_ptr()
                if not backend.context.set_tensor_address(name, pointer):
                    raise RuntimeError(f"Không gắn được buffer TensorRT: {name}")
            if not backend.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 thất bại.")
            outputs = [backend.bindings[name].data for name in sorted(backend.output_names)]
            predictions = outputs[0] if len(outputs) == 1 else outputs
            result = predictor.postprocess(predictions, image, [frame])[0]
            rows = result.boxes.data.cpu().tolist() if result.boxes is not None else []
            stream.synchronize()
            return rows

    def predict(self, frame):
        futures = [self.pool.submit(self._run, index, frame) for index in range(len(self.predictors))]
        # Chờ hết cả hai kể cả khi một nhánh lỗi, tránh tái sử dụng context còn đang chạy.
        results, error = [], None
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:
                error = error or exc
        if error:
            raise error
        return results

    def close(self):
        self.pool.shutdown(wait=True)
