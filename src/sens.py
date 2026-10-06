"""Hyperparameter sensitivity (development data only: inner folds 0 and 1 of split 1).
Part A (retraining): learning rate and epochs for YOLOv8 and MobileNet-SSD.
Part B (no retraining): pre-filter tau0, cluster IoU and counting threshold for the ensemble.
Resumable: each finished training job writes its prediction file."""
import json, os, sys, time
import numpy as np
from detectors import train_torchvision, predict_torchvision, train_yolo, predict_yolo
import stack

P = json.load(open('/home/claude/work/exp/primary.json')); rep = P['reps'][0]
OUT = '/home/claude/work/exp/preds/sens'; os.makedirs(OUT, exist_ok=True)
FOLDS = [0, 1]
GRID = {'yolo': {'base': (0.01, 25)}, 'ssdlite': {'base': (0.01, 25)}}
LRS = [0.001, 0.005, 0.02]; EPS = {'yolo': [10, 40], 'ssdlite': [10, 40]}


def log(m):
    line = time.strftime('%m-%d %H:%M:%S ') + 'SENS ' + m
    print(line, flush=True); open('/home/claude/work/exp/run.log', 'a').write(line + '\n')


def job(mdl, lr, ep, k):
    path = f'{OUT}/{mdl}_lr{lr}_ep{ep}_fold{k}.json'
    if os.path.exists(path): return
    tr = [i for j, f in enumerate(rep['folds']) if j != k for i in f]
    t = time.time(); log(f'START {os.path.basename(path)}')
    if mdl == 'yolo':
        m = train_yolo(tr, ep, f'/tmp/claude-0/sens_{mdl}_{lr}_{ep}_{k}', lr=lr); p = predict_yolo(m, rep['folds'][k])
    else:
        m = train_torchvision(mdl, tr, ep, lr=lr); p = predict_torchvision(m, rep['folds'][k])
    json.dump(p, open(path, 'w')); log(f'DONE  {os.path.basename(path)} {(time.time()-t)/60:.1f} min')


if __name__ == '__main__' and sys.argv[1:] == ['train']:
    for mdl in ['ssdlite', 'yolo']:
        configs = [(lr, 25) for lr in LRS] + [(0.01, e) for e in EPS[mdl]]
        for lr, ep in configs:
            for k in FOLDS:
                job(mdl, lr, ep, k)
    log('all sensitivity training finished')

if __name__ == '__main__' and sys.argv[1:] == ['eval']:
    imgs = [i for k in FOLDS for i in rep['folds'][k]]
    res = {'A': {}, 'B': {}}
    for mdl in ['ssdlite', 'yolo']:
        base = {}
        for k in FOLDS: base.update(json.load(open(f'/home/claude/work/exp/preds_v1_25ep/rep0/{mdl}_fold{k}.json')))
        res['A'][f'{mdl} lr=0.01 ep=25 (initial setting)'] = round(stack.average_precision(base, imgs), 3)
        for lr, ep in [(lr, 25) for lr in LRS] + [(0.01, e) for e in EPS[mdl]]:
            d = {}
            for k in FOLDS: d.update(json.load(open(f'{OUT}/{mdl}_lr{lr}_ep{ep}_fold{k}.json')))
            res['A'][f'{mdl} lr={lr} ep={ep}'] = round(stack.average_precision(d, imgs), 3)
    # Part B (after retraining at 40 epochs): ensemble thresholds on the full out-of-fold set of split 1 (40 dev images, nested CV)
    dev = rep['dev']; oof = {d: {} for d in stack.DETS}
    for d in stack.DETS:
        for k in range(5): oof[d].update(json.load(open(f'/home/claude/work/exp/preds/rep0/{d}_fold{k}.json')))
    def cv_ensemble():   # leave-one-fold-out over the 5 inner folds: fit meta on 4 folds' OOF, score the 5th
        scores = []
        for k in range(5):
            tr = [i for j, f in enumerate(rep['folds']) if j != k for i in f]; te = rep['folds'][k]
            st = stack.Stacker(stack.DETS).fit(oof, tr); scores.append(stack.average_precision(st.predict(oof, te), te))
        return round(float(np.mean(scores)), 3), round(float(np.std(scores)), 3)
    for t0 in [0.01, 0.05, 0.10, 0.20]:
        stack.TAU0 = t0; res['B'][f'tau0={t0}'] = cv_ensemble()
    stack.TAU0 = 0.05
    for ti in [0.40, 0.50, 0.55, 0.60, 0.70]:
        stack.TAU_IOU = ti; res['B'][f'tau_iou={ti}'] = cv_ensemble()
    stack.TAU_IOU = 0.55
    st = stack.Stacker(stack.DETS).fit(oof, dev); o = st.predict(oof, dev)
    res['B']['count_threshold'] = {str(t): {k: round(v, 3) for k, v in stack.prf_count(o, dev, t).items() if k in ('F1', 'MAE')}
                                   for t in [0.25, 0.35, 0.45, 0.55, 0.65]}
    res['B']['selected_tau_meta'] = st.tau
    json.dump(res, open('/home/claude/work/exp/results_sens.json', 'w'), indent=1); print(json.dumps(res, indent=1))
