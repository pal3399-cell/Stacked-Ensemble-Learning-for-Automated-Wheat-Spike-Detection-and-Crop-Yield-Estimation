"""Cross-domain evaluation on the official GWHD 2021 split.
Detectors were trained on the 492-image training subset (European sessions).
The meta-learner and all thresholds are fitted on the validation subset (268 images,
other sessions); the 1,382 test images (unseen sessions in 6 countries) are used once."""
import itertools, json
import numpy as np
import stack
from stack import DETS, SHORT

C = json.load(open('/home/claude/work/exp/crossdomain.json'))
info = json.load(open('/home/claude/work/exp/info.json'))
pred = {d: json.load(open(f'/home/claude/work/exp/preds/cross/{d}.json')) for d in DETS}
val, test = C['val'], C['test']
countries = sorted({info[n]['country'] for n in test})
groups = {c: [n for n in test if info[n]['country'] == c] for c in countries}
groups['All test'] = test


def rnd(x):
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in x.items()}


def per_group(dets, tau):
    return {g: rnd(dict(n=len(imgs), mAP50=stack.average_precision(dets, imgs), **{k: v for k, v in stack.prf_count(dets, imgs, tau).items()}))
            for g, imgs in groups.items()}


res = {'single': {}, 'ensemble': {}, 'subsets': {}, 'fusion': {}}
for d in DETS:
    tau = stack.best_tau(pred[d], val)
    res['single'][SHORT[d]] = per_group(pred[d], tau)
    res['single'][SHORT[d]]['All test']['mAP5095'] = stack.map_5095(pred[d], test)

st = stack.Stacker(DETS).fit(pred, val)
ens = st.predict(pred, test)
res['ensemble'] = per_group(ens, st.tau); res['ensemble']['All test']['mAP5095'] = stack.map_5095(ens, test)
res['coef'] = st.coefficients()

for k in (2, 3):
    for sub in itertools.combinations(DETS, k):
        s = stack.Stacker(list(sub)).fit(pred, val); o = s.predict(pred, test)
        res['subsets']['+'.join(SHORT[d] for d in sub)] = rnd(dict(mAP50=stack.average_precision(o, test), **stack.prf_count(o, test, s.tau)))
for name, fn in [('union_nms', stack.union_nms), ('vote2_wbf', stack.vote_wbf), ('wbf_equal', stack.wbf_equal)]:
    tau = stack.best_tau(fn(pred, val, DETS), val); o = fn(pred, test, DETS)
    res['fusion'][name] = rnd(dict(mAP50=stack.average_precision(o, test), **stack.prf_count(o, test, tau)))

json.dump(res, open('/home/claude/work/exp/results_cross.json', 'w'), indent=1)
for g in groups:
    print(g.ljust(10), ' '.join(f"{k}:{res['single'][k][g]['mAP50']:.3f}/{res['single'][k][g]['MAE']:.1f}" for k in 'FYSM'),
          f"ENS:{res['ensemble'][g]['mAP50']:.3f}/{res['ensemble'][g]['MAE']:.1f}")
print({k: round(v['mAP50'], 3) for k, v in res['fusion'].items()})
