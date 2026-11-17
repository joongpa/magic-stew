"""Benchmark native CPU runtimes and produce shared inputs for /tests/cpu-benchmark.html."""

from pathlib import Path
import os

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))
import time, json, statistics
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
model = (
    YOLO(str(root / "runs/pose/runs/26m-pose/weights/best.pt"))
    .model.float()
    .eval()
    .fuse(verbose=False)
)
model.end2end = False
rows = []
for size in [160, 256]:
    x = np.random.default_rng(17).random((1, 3, size, size), dtype=np.float32)
    x.astype("<f4").tofile(root / f"web/public/models/cpu-bench-input-{size}.f32")
    for threads in [1, 4]:
        torch.set_num_threads(threads)
        tensor = torch.from_numpy(x)
        with torch.inference_mode():
            for _ in range(3):
                model(tensor)
            times = []
            for _ in range(10):
                start = time.perf_counter()
                model(tensor)
                times.append((time.perf_counter() - start) * 1000)
        rows.append(
            dict(
                runtime="pytorch",
                size=size,
                threads=threads,
                medianMs=statistics.median(times),
                timesMs=times,
            )
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(root / f"web/public/models/cube-pose-{size}.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        feed = {session.get_inputs()[0].name: x}
        for _ in range(3):
            session.run(None, feed)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            session.run(None, feed)
            times.append((time.perf_counter() - start) * 1000)
        rows.append(
            dict(
                runtime="onnx-native",
                size=size,
                threads=threads,
                medianMs=statistics.median(times),
                timesMs=times,
            )
        )
print(json.dumps(rows, indent=2))
(root / "web/public/models/native-cpu-benchmark.json").write_text(json.dumps(rows, indent=2) + "\n")
