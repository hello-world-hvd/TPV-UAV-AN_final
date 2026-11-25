"""Đo tích luỹ bước con bằng đồng hồ thực; báo tổng, số lượt và thời gian trung bình."""
from contextlib import contextmanager
from time import perf_counter


class StepTiming:
    def __init__(self):
        self.steps = {}

    @contextmanager
    def measure(self, name):
        started = perf_counter()
        try:
            yield
        finally:
            step = self.steps.setdefault(name, {'seconds': 0., 'calls': 0})
            step['seconds'] += perf_counter()-started
            step['calls'] += 1

    def summary(self):
        return {name: {**step, 'average_ms': step['seconds']*1000/step['calls']}
                for name, step in self.steps.items()}

    def show(self, title):
        print(f'{title}:', flush=True)
        for name, step in sorted(self.summary().items(), key=lambda item: -item[1]['seconds']):
            print(f"  {name:18s}: {step['seconds']:.3f}s | {step['average_ms']:.3f} ms/lượt | {step['calls']} lượt", flush=True)
