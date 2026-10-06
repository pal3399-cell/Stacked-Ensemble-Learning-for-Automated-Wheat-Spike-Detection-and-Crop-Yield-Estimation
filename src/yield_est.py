"""Count -> spike density -> yield, on the 24 held-out test images of the three primary splits.
Footprint per image from the GSD of the released GWHD patches:
 - sessions documented in the GWHD paper (Table 3, harmonised GSD): exact value;
 - other sessions: 0.2-0.4 mm/px (GWHD 2021 acquisition range), central 0.3.
Yield = SD x GPS x TGW / 100 (kg/ha); GPS 45 (33-71), TGW 38 g (30-61 g) [Philipp et al. 2018].
Compares the yield implied by the ensemble count with the yield implied by the annotated count,
which isolates the error that the detector adds."""
import json
import numpy as np
import stack

GSD_DOC = {'ethz_1': 0.55, 'rres_1': 0.33, 'usask_1': 0.45, 'inrae_1': 0.28, 'utokyo_1': 0.43,
           'utokyo_2': 0.30, 'nau_1': 0.21, 'uq_1': 0.40}
GPS, GPS_R = 45, (33, 71)
TGW, TGW_R = 38, (30, 61)
P = json.load(open('/home/claude/work/exp/primary.json'))
info = json.load(open('/home/claude/work/exp/info.json'))
rows = []
import os
for r, rep in enumerate(P['reps']):
    if not os.path.exists(f'/home/claude/work/exp/preds/rep{r}/frcnn_final.json'): continue
    D = f'/home/claude/work/exp/preds/rep{r}'
    oof = {d: {} for d in stack.DETS}; fin = {}
    for d in stack.DETS:
        for k in range(5): oof[d].update(json.load(open(f'{D}/{d}_fold{k}.json')))
        fin[d] = json.load(open(f'{D}/{d}_final.json'))
    st = stack.Stacker(stack.DETS).fit(oof, rep['dev']); ens = st.predict(fin, rep['test'])
    for n in rep['test']:
        dom = info[n]['domain'].lower()
        g, lo, hi = (GSD_DOC[dom],) * 3 if dom in GSD_DOC else (0.30, 0.20, 0.40)
        area = lambda gsd: (1024 * gsd / 1000) ** 2           # m2 covered by the 1024x1024 patch
        pred = int((np.asarray(ens[n]['scores']) >= st.tau).sum()); true = len(stack.GT[n])
        rows.append(dict(img=n, rep=r, domain=info[n]['domain'], gsd=g, documented=dom in GSD_DOC, area=area(g),
                         area_lo=area(lo), area_hi=area(hi), count_pred=pred, count_true=true))

def yld(sd, gps=GPS, tgw=TGW): return sd * gps * tgw / 100

sd_pred = np.array([x['count_pred'] / x['area'] for x in rows]); sd_true = np.array([x['count_true'] / x['area'] for x in rows])
sd_pred_lo = np.array([x['count_pred'] / x['area_hi'] for x in rows]); sd_pred_hi = np.array([x['count_pred'] / x['area_lo'] for x in rows])
out = dict(
    n_images=len(rows), n_documented=int(sum(x['documented'] for x in rows)),
    mean_area_m2=round(float(np.mean([x['area'] for x in rows])), 3),
    SD_pred=round(float(sd_pred.mean()), 1), SD_true=round(float(sd_true.mean()), 1),
    SD_pred_range=[round(float(sd_pred.min()), 1), round(float(sd_pred.max()), 1)],
    yield_pred=round(float(yld(sd_pred.mean()))), yield_true=round(float(yld(sd_true.mean()))),
    yield_range_components=[round(float(yld(sd_pred.mean(), GPS_R[0], TGW_R[0]))), round(float(yld(sd_pred.mean(), GPS_R[1], TGW_R[1])))],
    yield_range_all=[round(float(yld(sd_pred_lo.mean(), GPS_R[0], TGW_R[0]))), round(float(yld(sd_pred_hi.mean(), GPS_R[1], TGW_R[1])))],
    count_error_pct=round(float((sum(x['count_pred'] for x in rows) / sum(x['count_true'] for x in rows) - 1) * 100), 1),
    per_image_yield_mape=round(float(np.mean(np.abs(sd_pred - sd_true) / sd_true) * 100), 1),
)
json.dump(dict(summary=out, images=rows), open('/home/claude/work/exp/results_yield.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
