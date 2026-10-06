"""Computational cost (CPU, 2 threads, batch 1, 640x640): parameters, GFLOPs of one full forward pass
(PyTorch FlopCounterMode), median inference time of 10 runs (after 1 warm-up)."""
import json, time, statistics, torch
from torch.utils.flop_counter import FlopCounterMode
import detectors as Dt
torch.set_num_threads(2)
x = torch.rand(3, 640, 640); res = {}

def timeit(f, n=10):
    f(); ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000

def flops(f):
    with FlopCounterMode(display=False) as fc: f()
    return fc.get_total_flops() / 1e9

with torch.no_grad():
    for name in ['frcnn', 'ssd', 'ssdlite']:
        m = Dt.build_torchvision(name).eval()
        res[name] = dict(params_M=round(sum(q.numel() for q in m.parameters()) / 1e6, 2),
                         GFLOPs=round(flops(lambda: m([x])), 1), ms=round(timeit(lambda: m([x]))))
    from ultralytics import YOLO
    y = YOLO('/home/claude/work/weights/yolov8s.pt'); net = y.model.eval(); xb = x[None]
    res['yolo'] = dict(params_M=round(sum(q.numel() for q in net.parameters()) / 1e6, 2),
                       GFLOPs=round(flops(lambda: net(xb)), 1),
                       ms=round(timeit(lambda: y.predict(xb, imgsz=640, device='cpu', verbose=False))))
res['ensemble'] = dict(params_M=round(sum(v['params_M'] for v in res.values()), 2),
                       GFLOPs=round(sum(v['GFLOPs'] for v in res.values()), 1), ms=sum(v['ms'] for v in res.values()))
json.dump(res, open('results_cost.json', 'w'), indent=1); print(res)
