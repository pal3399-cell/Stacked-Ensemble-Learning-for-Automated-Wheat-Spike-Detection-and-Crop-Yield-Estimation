"""Training curves of the four refitted detectors (split 2 final models) and Table 6 rows (YOLOv8)."""
import csv, re
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

log = open('run.log').read().splitlines()
curves = {}
for mdl in ['frcnn', 'ssd', 'ssdlite']:
    start = max(i for i, l in enumerate(log) if f'START rep1/{mdl}_final.json' in l)
    ys = []
    for l in log[start + 1:]:
        m = re.search(rf'{mdl} epoch (\d+)/(\d+) loss ([\d.]+)', l)
        if m: ys.append(float(m.group(3)))
        if 'DONE' in l or 'START' in l: break
    curves[mdl] = ys
rows = list(csv.DictReader(open('/tmp/claude-0/yolo_rep1_yolo_final.json/run/results.csv')))
yb = [float(r['train/box_loss']) for r in rows]; yc = [float(r['train/cls_loss']) for r in rows]; yd = [float(r['train/dfl_loss']) for r in rows]

C = {'frcnn': '#2a78d6', 'yolo': '#eb6834', 'ssd': '#1baf7a', 'ssdlite': '#eda100'}
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
for a, (mdl, title) in zip(ax[[0, 2, 3]], [('frcnn', 'Faster R-CNN (RPN + head loss)'), ('ssd', 'SSD (conf. + loc. loss)'), ('ssdlite', 'MobileNet-SSD (conf. + loc. loss)')]):
    a.plot(range(1, len(curves[mdl]) + 1), curves[mdl], color=C[mdl], lw=2, marker='o', ms=3); a.set_title(title, fontsize=10)
ax[1].plot(range(1, len(yb) + 1), yb, color='#eb6834', lw=2, label='box (CIoU)')
ax[1].plot(range(1, len(yd) + 1), yd, color='#4a3aa7', lw=2, label='DFL')
ax[1].plot(range(1, len(yc) + 1), yc, color='#52514e', lw=2, label='classification')
ax[1].set_title('YOLOv8 (Eq. 5 terms)', fontsize=10); ax[1].legend(frameon=False, fontsize=8)
for a in ax:
    a.set_xlabel('Epoch'); a.set_ylabel('Training loss'); a.grid(axis='y', color='#e6e5e1', lw=0.6)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig('fig_training.pdf'); plt.savefig('fig_training.png', dpi=200)
for e in [1, 5, 10, 20, 30, 40]:
    print(f'{e} & {yb[e-1]:.3f} & {yd[e-1]:.3f} & {yc[e-1]:.3f} \\\\')
