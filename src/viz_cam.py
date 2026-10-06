"""EigenCAM heatmaps (gradient-free class activation maps) for the four refitted detectors of a
primary split, on held-out test images with overlapping spikes, plus the fused ensemble output."""
import json, sys
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import detectors as Dt, stack

r = int(sys.argv[1]) if len(sys.argv) > 1 else 0
P = json.load(open('primary.json')); rep = P['reps'][r]
G = stack.GT
# pick the 2 test images with the most overlapping ground-truth pairs (IoU > 0.3)
def n_overlap(n):
    g = np.array(G[n]); M = stack.iou_mat(g, g); np.fill_diagonal(M, 0); return int((M > 0.3).sum() // 2)
imgs = sorted(rep['test'], key=n_overlap, reverse=True)[:2]

class TVWrap(torch.nn.Module):          # expose backbone features for torchvision detectors
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m.backbone(self.m.transform(list(x))[0].tensors)

def tv_cam(name, x):
    m = Dt.build_torchvision(name); m.load_state_dict(torch.load(f'preds/rep{r}/{name}_final.pt', map_location='cpu')); m.eval()
    w = TVWrap(m)
    if name == 'frcnn':
        layer = [m.backbone.body.layer4]
        w.forward = lambda x: m.backbone.body(m.transform(list(x))[0].tensors)['3']
    else:
        layer = [m.backbone.features[-1]] if name == 'ssd' else [m.backbone.features[0][-1]]
        w.forward = lambda x: list(m.backbone(m.transform(list(x))[0].tensors).values())[0]
    cam = EigenCAM(model=w, target_layers=layer)
    return cam(x[None], targets=None, eigen_smooth=False)[0]

def yolo_cam(x):
    from ultralytics import YOLO
    y = YOLO(f'/tmp/claude-0/yolo_rep{r}_yolo_final.json/run/weights/last.pt').model.eval()
    class W(torch.nn.Module):
        def __init__(s): super().__init__(); s.y = y
        def forward(s, x): return s.y(x)[0]
    cam = EigenCAM(model=W(), target_layers=[y.model[-2]])
    return cam(x[None], targets=None)[0]

fin = {d: json.load(open(f'preds/rep{r}/{d}_final.json')) for d in stack.DETS}
oof = {d: {} for d in stack.DETS}
for d in stack.DETS:
    for k in range(5): oof[d].update(json.load(open(f'preds/rep{r}/{d}_fold{k}.json')))
st = stack.Stacker(stack.DETS).fit(oof, rep['dev'])
X, _, meta = stack.build(fin, imgs, stack.DETS, with_labels=False); p = st.model.predict_proba(X)[:, 1]

cols = ['Input + ground truth', 'Faster R-CNN', 'YOLOv8', 'SSD', 'MobileNet-SSD', 'Stacked ensemble']
fig, ax = plt.subplots(len(imgs), 6, figsize=(18, 3.2 * len(imgs)))
for i, n in enumerate(imgs):
    rgb = np.asarray(Image.open(f'{Dt.IMG_DIR}/{n}').convert('RGB')).astype(np.float32) / 255
    x = torch.tensor(rgb).permute(2, 0, 1)
    maps = [tv_cam('frcnn', x), yolo_cam(x), tv_cam('ssd', x), tv_cam('ssdlite', x)]
    ax[i, 0].imshow(rgb)
    for b in G[n]: ax[i, 0].add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec='white', lw=0.8))
    for j, mp in enumerate(maps):
        mp = np.array(Image.fromarray((mp * 255).astype(np.uint8)).resize((640, 640))) / 255
        ax[i, j + 1].imshow(show_cam_on_image(rgb, mp, use_rgb=True))
    ax[i, 5].imshow(rgb)
    nag = {}
    for (nn, b), s, f in zip(meta, p, X):
        if nn == n and s >= st.tau:
            c = {1: '#eda100', 2: '#eb6834', 3: '#2a78d6', 4: '#1baf7a'}[int(f[12])]
            ax[i, 5].add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=c, lw=1.2))
    for j in range(6):
        ax[i, j].axis('off')
        if i == 0: ax[i, j].set_title(cols[j], fontsize=11)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=c, lw=2) for c in ['#eda100', '#eb6834', '#2a78d6', '#1baf7a']],
           ['1 detector', '2 detectors', '3 detectors', '4 detectors'], loc='lower right', ncol=4, frameon=False,
           title='Ensemble boxes: number of agreeing detectors')
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(f'fig_cam_rep{r}.png', dpi=200); plt.savefig(f'fig_cam_rep{r}.pdf')
print('saved', imgs, [n_overlap(n) for n in imgs])
