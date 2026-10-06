"""Table: test results of the initial (25-epoch YOLOv8/MobileNet-SSD) vs final (40-epoch) configuration."""
import json
o = json.load(open('preds_v1_25ep/summary.json')); n = json.load(open('summary.json'))
oc = json.load(open('preds_v1_25ep/results_cross.json')); nc = json.load(open('results_cross.json'))
NM = {'F': 'Faster R-CNN', 'Y': 'YOLOv8', 'S': 'SSD', 'M': 'MobileNet-SSD', 'E': 'Stacked ensemble'}
def prim(s, k): return s['table7']['F+Y+S+M' if k == 'E' else k]['mAP50']
def cross(c, k): return (c['ensemble'] if k == 'E' else c['single'][k])['All test']['mAP50']
rows = []
for k in 'FYSME':
    rows.append(f"{NM[k]} & {prim(o, k):.3f} & {prim(n, k):.3f} & {cross(oc, k):.3f} & {cross(nc, k):.3f} \\\\")
rows.append(f"Ensemble gain over best single (pp) & {o['gain_mean_pp']:.1f} & {n['gain_mean_pp']:.1f} & {o['cross_gain_pp']:.1f} & {n['cross_gain_pp']:.1f} \\\\")
open('table_epochs_v1.tex', 'w').write('\n'.join(rows) + '\n'); print('\n'.join(rows))
