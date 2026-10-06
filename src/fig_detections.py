"""Fig. 5: detections of each model and of the stacked ensemble on one held-out test image of split 1,
with the predicted count and the annotated count."""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
import stack, detectors as Dt

P = json.load(open('primary.json')); rep = P['reps'][0]
fin = {d: json.load(open(f'preds/rep0/{d}_final.json')) for d in stack.DETS}
oof = {d: {} for d in stack.DETS}
for d in stack.DETS:
    for k in range(5): oof[d].update(json.load(open(f'preds/rep0/{d}_fold{k}.json')))
taus = {d: stack.best_tau(oof[d], rep['dev']) for d in stack.DETS}
st = stack.Stacker(stack.DETS).fit(oof, rep['dev']); ens = st.predict(fin, rep['test'])
dens = P['density']
n = [i for i in rep['test'] if dens[i] == 'dense'][0]           # a dense test image
gt = stack.GT[n]; rgb = np.asarray(Image.open(f'{Dt.IMG_DIR}/{n}').convert('RGB'))
panels = [('Ground truth', gt, None)] + \
         [(name, fin[d][n], taus[d]) for name, d in [('Faster R-CNN', 'frcnn'), ('YOLOv8', 'yolo'), ('SSD', 'ssd'), ('MobileNet-SSD', 'ssdlite')]] + \
         [('Stacked ensemble', ens[n], st.tau)]
fig, ax = plt.subplots(2, 3, figsize=(13.5, 9.6))
for a, (title, det, tau) in zip(ax.ravel(), panels):
    a.imshow(rgb); a.axis('off')
    if tau is None:
        boxes = gt; col = '#ffffff'
        a.set_title(f'{title}: {len(boxes)} heads', fontsize=12)
    else:
        s = np.asarray(det['scores']); boxes = np.asarray(det['boxes']).reshape(-1, 4)[s >= tau]; col = '#eda100'
        tp, _ = stack.match(boxes, s[s >= tau], gt)
        a.set_title(f'{title}: {len(boxes)} detected ({int(tp.sum())} correct)', fontsize=12)
    for b in boxes:
        a.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=1.1))
plt.tight_layout(); plt.savefig('fig_detections_rep0.png', dpi=200); plt.savefig('fig_detections_rep0.pdf')
print(n, len(gt))
