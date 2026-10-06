"""Evaluate one primary split (rep): single detectors, stacked ensemble for every detector
subset, fixed fusion rules, occlusion-stratified recall. Thresholds are chosen on the
out-of-fold (development) predictions only; the 8 test images are used once, at the end."""
import itertools, json, sys
import numpy as np
import stack
from stack import DETS, SHORT, GT

r = int(sys.argv[1])
P = json.load(open('/home/claude/work/exp/primary.json')); rep = P['reps'][r]
D = f'/home/claude/work/exp/preds/rep{r}'
dev, test = rep['dev'], rep['test']

oof, fin = {}, {}
for d in DETS:
    oof[d] = {}
    for k in range(5):
        oof[d].update(json.load(open(f'{D}/{d}_fold{k}.json')))
    fin[d] = json.load(open(f'{D}/{d}_final.json'))

res = {'single': {}, 'subsets': {}, 'fusion': {}, 'occlusion': {}, 'coef': {}}


def rnd(x):
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in x.items()}


# ---- single detectors
for d in DETS:
    tau = stack.best_tau(oof[d], dev)
    res['single'][SHORT[d]] = rnd(stack.evaluate(fin[d], test, tau))

# ---- stacked ensemble for every subset of size >= 2
for k in (2, 3, 4):
    for sub in itertools.combinations(DETS, k):
        st = stack.Stacker(list(sub)).fit(oof, dev)
        out = st.predict(fin, test)
        key = '+'.join(SHORT[d] for d in sub)
        res['subsets'][key] = rnd(stack.evaluate(out, test, st.tau))
        if k == 4:
            res['coef'] = st.coefficients(); ens_out, ens_tau = out, st.tau

# ---- fixed fusion rules on all four detectors (threshold chosen on OOF)
for name, fn in [('union_nms', stack.union_nms), ('vote2_wbf', stack.vote_wbf), ('wbf_equal', stack.wbf_equal)]:
    tau = stack.best_tau(fn(oof, dev, DETS), dev)
    res['fusion'][name] = rnd(stack.evaluate(fn(fin, test, DETS), test, tau))

# ---- occlusion-stratified recall (IoU with the most-overlapping other GT box)
strata = {}
for n in test:
    g = np.array(GT[n]).reshape(-1, 4); M = stack.iou_mat(g, g); np.fill_diagonal(M, 0)
    mx = M.max(1) if len(g) > 1 else np.zeros(len(g))
    strata[n] = np.where(mx < 0.1, 'isolated', np.where(mx <= 0.3, 'touching', 'overlapping'))


def strat_recall(dets, tau):
    hit = {'isolated': [0, 0], 'touching': [0, 0], 'overlapping': [0, 0]}
    for n in test:
        sc = np.asarray(dets[n]['scores']); keep = sc >= tau
        b = np.asarray(dets[n]['boxes']).reshape(-1, 4)[keep]; g = np.array(GT[n]).reshape(-1, 4)
        found = np.zeros(len(g), bool)
        if len(b) and len(g):
            M = stack.iou_mat(b[np.argsort(-sc[keep])], g); used = np.zeros(len(g), bool)
            for i in range(len(M)):
                m = np.where(used, -1, M[i]); j = int(np.argmax(m))
                if m[j] >= 0.5: used[j] = True
            found = used
        for s, f in zip(strata[n], found):
            hit[s][0] += int(f); hit[s][1] += 1
    return {k: (round(a / b, 4) if b else None, b) for k, (a, b) in hit.items()}


for d in DETS:
    res['occlusion'][SHORT[d]] = strat_recall(fin[d], res['single'][SHORT[d]]['tau'])
res['occlusion']['Ensemble'] = strat_recall(ens_out, ens_tau)

json.dump(res, open(f'/home/claude/work/exp/results_rep{r}.json', 'w'), indent=1)
print(json.dumps({'single': {k: (v['mAP50'], v['F1'], v['MAE']) for k, v in res['single'].items()},
                  'ensemble': {k: (v['mAP50'], v['F1'], v['MAE']) for k, v in res['subsets'].items()},
                  'fusion': {k: (v['mAP50'], v['F1'], v['MAE']) for k, v in res['fusion'].items()},
                  'occlusion': res['occlusion']}, indent=0))
