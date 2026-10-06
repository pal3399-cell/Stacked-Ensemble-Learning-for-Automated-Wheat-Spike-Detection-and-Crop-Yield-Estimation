"""Train and run the four base detectors on CPU.

Every detector: COCO-pretrained, fine-tuned for one class (wheat spike),
SGD (momentum 0.9, weight decay 1e-4), batch 8 / 4, lr 0.01, cosine schedule.
Augmentation is applied on the fly to TRAINING images only:
horizontal flip, vertical flip, 90-degree rotation, scale jitter (0.8-1.2).
"""
import json, math, os, random, shutil, time
from functools import partial
import numpy as np, torch, torchvision
from PIL import Image
from torchvision.models import detection as D
from torchvision.transforms import functional as TF

IMG_DIR = '/home/claude/work/data/images'
torch.set_num_threads(2)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GT = None


def gt():
    global GT
    if GT is None:
        GT = json.load(open('/home/claude/work/exp/gt.json'))
    return GT


# ------------------------------------------------------------------ models
def build_torchvision(name, score_thresh=0.01, max_det=300):
    if name == 'frcnn':
        m = D.fasterrcnn_resnet50_fpn(weights='DEFAULT', min_size=640, max_size=640,
                                      box_detections_per_img=max_det, box_score_thresh=score_thresh)
        inf = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = D.faster_rcnn.FastRCNNPredictor(inf, 2)
    elif name == 'ssd':
        m = D.ssd300_vgg16(weights='DEFAULT', detections_per_img=max_det, score_thresh=score_thresh)
        ch = D._utils.retrieve_out_channels(m.backbone, (300, 300))
        na = m.anchor_generator.num_anchors_per_location()
        m.head.classification_head = D.ssd.SSDClassificationHead(ch, na, 2)
    elif name == 'ssdlite':
        m = D.ssdlite320_mobilenet_v3_large(weights='DEFAULT', detections_per_img=max_det,
                                            score_thresh=score_thresh)
        ch = D._utils.retrieve_out_channels(m.backbone, (320, 320))
        na = m.anchor_generator.num_anchors_per_location()
        norm = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        m.head.classification_head = D.ssdlite.SSDLiteClassificationHead(ch, na, 2, norm)
    else:
        raise ValueError(name)
    if name in ('ssd', 'ssdlite'):
        # Same 640x640 input as the other detectors. Default-box scales are set from the
        # training-label size distribution (wheat heads: 4-12% of the image side; the
        # torchvision defaults start at 7% (SSD) and 20% (SSDlite), too large for spikes).
        # Aspect ratios are unchanged, so the pretrained heads keep their shapes.
        m.transform.fixed_size = (640, 640); m.transform.min_size = (640,); m.transform.max_size = 640
        ar = m.anchor_generator.aspect_ratios
        m.anchor_generator = D.anchor_utils.DefaultBoxGenerator(
            ar, scales=[0.04, 0.07, 0.12, 0.20, 0.35, 0.60, 0.80], steps=None)
    return m


def load_img(name):
    return TF.to_tensor(Image.open(os.path.join(IMG_DIR, name)).convert('RGB'))


def augment(img, boxes):
    """img: 3xHxW tensor, boxes: Nx4 (x1,y1,x2,y2). Training images only."""
    _, H, W = img.shape
    if random.random() < 0.5:
        img = img.flip(-1); boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([W, 0, W, 0])
    if random.random() < 0.5:
        img = img.flip(-2); boxes = boxes[:, [0, 3, 2, 1]] * torch.tensor([1, -1, 1, -1]) + torch.tensor([0, H, 0, H])
    if random.random() < 0.5:  # rotate 90 degrees counter-clockwise
        img = torch.rot90(img, 1, (1, 2))
        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack([y1, W - x2, y2, W - x1], 1)
    s = random.uniform(0.8, 1.2)
    if abs(s - 1) > 0.02:
        n = int(round(640 * s))
        img = TF.resize(img, [n, n], antialias=True); boxes = boxes * (n / 640)
        if n > 640:  # random crop back to 640
            ox, oy = random.randint(0, n - 640), random.randint(0, n - 640)
            img = img[:, oy:oy + 640, ox:ox + 640]
            boxes = boxes - torch.tensor([ox, oy, ox, oy])
            boxes = boxes.clamp(0, 640)
        else:  # pad to 640
            pad = torch.zeros(3, 640, 640); pad[:, :n, :n] = img; img = pad
    keep = ((boxes[:, 2] - boxes[:, 0]) > 2) & ((boxes[:, 3] - boxes[:, 1]) > 2)
    return img, boxes[keep]


def train_torchvision(name, train_imgs, epochs, lr=0.01, batch=4, seed=0, log=None, ckpt=None):
    random.seed(seed); torch.manual_seed(seed)
    m = build_torchvision(name).to(DEV); m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    steps = epochs * math.ceil(len(train_imgs) / batch); warm = min(50, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    G = gt(); cache = {n: (load_img(n) * 255).round().to(torch.uint8) for n in train_imgs}  # uint8 cache: 4x less RAM
    start = 0
    if ckpt and os.path.exists(ckpt):  # resume an interrupted job from its last finished epoch
        c = torch.load(ckpt, map_location=DEV, weights_only=False)
        m.load_state_dict(c['model']); opt.load_state_dict(c['opt']); sched.load_state_dict(c['sched'])
        random.setstate(c['py_rng']); torch.set_rng_state(c['torch_rng']); start = c['epoch']
        if log: log(f'resumed {name} at epoch {start}/{epochs}')
    for ep in range(start, epochs):
        order = random.sample(train_imgs, len(train_imgs)); tot = 0; t0 = time.time()
        starts = list(range(0, len(order), batch))
        if len(order) - starts[-1] == 1 and len(starts) > 1:
            starts = starts[:-1]  # merge a leftover single image into the previous batch (BatchNorm needs >1)
        for si, i in enumerate(starts):
            end = starts[si + 1] if si + 1 < len(starts) else len(order)
            ims, tg = [], []
            for n in order[i:end]:
                b = torch.tensor(G[n], dtype=torch.float32).reshape(-1, 4)
                im, b = augment(cache[n].float() / 255, b)
                ims.append(im.to(DEV)); tg.append({'boxes': b.to(DEV), 'labels': torch.ones(len(b), dtype=torch.int64, device=DEV)})
            loss = sum(m(ims, tg).values())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step(); sched.step()
            tot += loss.item()
        if log: log(f'{name} epoch {ep + 1}/{epochs} loss {tot / math.ceil(len(order) / batch):.4f} ({time.time() - t0:.0f}s)')
        if ckpt:
            torch.save({'model': m.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'py_rng': random.getstate(), 'torch_rng': torch.get_rng_state(), 'epoch': ep + 1}, ckpt + '.tmp')
            os.replace(ckpt + '.tmp', ckpt)
    m.eval()
    return m


@torch.no_grad()
def predict_torchvision(m, imgs, bs=4):
    m.eval(); out = {}
    for i in range(0, len(imgs), bs):
        names = imgs[i:i + bs]
        res = m([load_img(n).to(DEV) for n in names])
        for n, r in zip(names, res):
            out[n] = {'boxes': r['boxes'].cpu().round(decimals=2).tolist(), 'scores': r['scores'].cpu().round(decimals=4).tolist()}
    return out


# ------------------------------------------------------------------ YOLOv8
def _yolo_dataset(root, train_imgs, val_imgs):
    G = gt()
    for sub, imgs in [('train', train_imgs), ('val', val_imgs)]:
        os.makedirs(f'{root}/images/{sub}', exist_ok=True); os.makedirs(f'{root}/labels/{sub}', exist_ok=True)
        for n in imgs:
            dst = f'{root}/images/{sub}/{n}'
            if not os.path.exists(dst): os.symlink(os.path.join(IMG_DIR, n), dst)
            with open(f'{root}/labels/{sub}/{n[:-4]}.txt', 'w') as f:
                for x1, y1, x2, y2 in G[n]:
                    f.write(f'0 {(x1 + x2) / 1280:.6f} {(y1 + y2) / 1280:.6f} {(x2 - x1) / 640:.6f} {(y2 - y1) / 640:.6f}\n')
    with open(f'{root}/data.yaml', 'w') as f:
        f.write(f'path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: spike\n')
    return f'{root}/data.yaml'


def train_yolo(train_imgs, epochs, workdir, lr=0.01, batch=8, seed=0, weights='yolov8s.pt'):
    from ultralytics import YOLO
    last = f'{workdir}/run/weights/last.pt'
    if os.path.exists(last) and not os.path.exists(f'{workdir}/finished'):
        try:  # resume an interrupted run from its last saved epoch
            YOLO(last).train(resume=True)
            open(f'{workdir}/finished', 'w').close()
            return YOLO(last)
        except Exception as e:
            print('YOLO resume failed, restarting job:', e)
    if os.path.exists(workdir): shutil.rmtree(workdir)
    # Ultralytics needs a 'val' set for its own logging; we pass a training image so
    # no held-out image is ever touched during training.
    yaml = _yolo_dataset(workdir + '/ds', train_imgs, train_imgs[:2])
    m = YOLO(f'/home/claude/work/weights/{weights}')
    m.train(data=yaml, epochs=epochs, imgsz=640, batch=batch, device=0 if DEV.type == 'cuda' else 'cpu', workers=0, optimizer='SGD',
            lr0=lr, momentum=0.9, weight_decay=1e-4, cos_lr=True, seed=seed, deterministic=False,
            fliplr=0.5, flipud=0.5, degrees=0.0, scale=0.2, mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            translate=0.0, val=False, plots=False, project=workdir, name='run', exist_ok=True, verbose=False,
            amp=False, max_det=300)
    open(f'{workdir}/finished', 'w').close()
    return YOLO(f'{workdir}/run/weights/last.pt')


def predict_yolo(m, imgs, conf=0.01):
    out = {}
    for i in range(0, len(imgs), 8):
        names = imgs[i:i + 8]
        res = m.predict([os.path.join(IMG_DIR, n) for n in names], imgsz=640, conf=conf, iou=0.5,
                        max_det=300, device=0 if DEV.type == 'cuda' else 'cpu', verbose=False)
        for n, r in zip(names, res):
            out[n] = {'boxes': r.boxes.xyxy.cpu().numpy().round(2).tolist(), 'scores': r.boxes.conf.cpu().numpy().round(4).tolist()}
    return out
