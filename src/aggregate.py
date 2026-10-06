"""Combine the three primary splits (mean +- s.d.) and the cross-domain results into LaTeX table
bodies that are pasted into the manuscript. Writes tables.tex and summary.json."""
import json
import numpy as np

R = [json.load(open(f'/home/claude/work/exp/results_rep{r}.json')) for r in range(3)]
X = json.load(open('/home/claude/work/exp/results_cross.json'))
NAME = {'F': 'Faster R-CNN', 'Y': 'YOLOv8', 'S': 'SSD', 'M': 'MobileNet-SSD'}


def ms(vals, d=3):
    v = np.array(vals, float)
    return f'{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}'


def get(res, key):  # key: 'F' or subset like 'F+Y+S+M'
    return res['single'][key] if key in res['single'] else res['subsets'][key]


out = {}
# ---- Table 7: single detectors + ensemble
rows = []
for k in ['S', 'M', 'Y', 'F', 'F+Y+S+M']:
    vals = {m: [get(r, k)[m] for r in R] for m in ['mAP50', 'mAP5095', 'P', 'R', 'F1', 'MAE', 'RMSE']}
    nm = NAME.get(k, r'\textbf{Stacked Ensemble (Proposed)}')
    cells = [ms(vals[m]) for m in ['mAP50', 'mAP5095', 'P', 'R', 'F1']] + [ms(vals['MAE'], 1), ms(vals['RMSE'], 1)]
    rows.append(nm + ' & ' + ' & '.join(cells) + r' \\' + '\n\\hline')
    out[k] = {m: float(np.mean(v)) for m, v in vals.items()}
t7 = '\n'.join(rows)

# ---- Table 8: subsets
order = ['S', 'M', 'Y', 'F', 'F+Y', 'F+S', 'F+M', 'Y+S', 'Y+M', 'S+M', 'F+Y+S', 'F+Y+M', 'F+S+M', 'Y+S+M', 'F+Y+S+M']
full = np.array([get(r, 'F+Y+S+M')['mAP50'] for r in R])
rows = []
for k in order:
    v = {m: [get(r, k)[m] for r in R] for m in ['mAP50', 'mAP5095', 'F1', 'MAE']}
    size = k.count('+') + 1
    lab = k + {'F+Y+S': ' (without M)', 'F+Y+M': ' (without S)', 'F+S+M': ' (without Y)', 'Y+S+M': ' (without F)', 'F+Y+S+M': ' (proposed)'}.get(k, '')
    cross = X['single'][k]['All test']['mAP50'] if k in X['single'] else (X['ensemble']['All test']['mAP50'] if k == 'F+Y+S+M' else X['subsets'][k]['mAP50'])
    rows.append(f"{size} & {lab} & {ms(v['mAP50'])} & {ms(v['mAP5095'])} & {ms(v['F1'])} & {ms(v['MAE'], 1)} & {cross:.3f} \\\\")
    if k in ('F', 'S+M', 'Y+S+M'):
        rows.append(r'\midrule')
t8 = '\n'.join(rows)
loo = {k: float(full.mean() - np.mean([get(r, k)['mAP50'] for r in R])) for k in ['F+Y+S', 'F+Y+M', 'F+S+M', 'Y+S+M']}

# ---- Table 9: fusion rules
rows = []
for k, lab in [('union_nms', 'Union of all boxes + NMS'), ('vote2_wbf', r'Majority vote ($n_c\ge2$) + WBF'),
               ('wbf_equal', r'WBF with equal weights \cite{solovyev2021wbf}')]:
    v = {m: [r['fusion'][k][m] for r in R] for m in ['mAP50', 'F1', 'MAE']}
    rows.append(f"{lab} & {ms(v['mAP50'])} & {ms(v['F1'])} & {ms(v['MAE'], 1)} & {X['fusion'][k]['mAP50']:.3f} \\\\")
v = {m: [get(r, 'F+Y+S+M')[m] for r in R] for m in ['mAP50', 'F1', 'MAE']}
rows.append(f"Stacked meta-learner (proposed) & {ms(v['mAP50'])} & {ms(v['F1'])} & {ms(v['MAE'], 1)} & {X['ensemble']['All test']['mAP50']:.3f} \\\\")
t9 = '\n'.join(rows)

# ---- Table 11: cross-domain per country (mAP50 / MAE)
rows = []
for g in ['Australia', 'China', 'Japan', 'Mexico', 'Sudan', 'US', 'All test']:
    n = X['ensemble'][g]['n']
    cells = [f"{X['single'][k][g]['mAP50']:.3f} / {X['single'][k][g]['MAE']:.1f}" for k in 'FYSM']
    best = max(X['single'][k][g]['mAP50'] for k in 'FYSM'); e = X['ensemble'][g]['mAP50']
    ens = f"{e:.3f} / {X['ensemble'][g]['MAE']:.1f}"
    ens = r'\textbf{' + ens + '}' if e >= best else ens
    lab = {'US': 'USA', 'All test': r'\midrule All test'}.get(g, g)
    rows.append(f"{lab} & {n:,} & " .replace(',', '{,}') + ' & '.join(cells) + f' & {ens} \\\\')
t11 = '\n'.join(rows).replace(r'\midrule All test', r'\midrule' + '\nAll test')

# ---- Table 12: occlusion recall, pooled over the 24 test images
rows = []
pooled = {}
for s in ['isolated', 'touching', 'overlapping']:
    n = sum(r['occlusion']['F'][s][1] for r in R)
    cells = []
    for k in ['F', 'Y', 'S', 'M', 'Ensemble']:
        hit = sum(r['occlusion'][k][s][0] * r['occlusion'][k][s][1] for r in R) / n
        pooled.setdefault(k, {})[s] = hit; cells.append(f'{hit:.3f}')
    rows.append(f"{s.capitalize()} ({n}) & " + ' & '.join(cells) + r' \\')
t12 = '\n'.join(rows)

gains = [get(r, 'F+Y+S+M')['mAP50'] - max(r['single'][k]['mAP50'] for k in 'FYSM') for r in R]
summary = dict(table7=out, gain_over_best_single_pp=[round(g * 100, 1) for g in gains],
               gain_mean_pp=round(float(np.mean(gains)) * 100, 1), leave_one_out_drop=loo,
               occlusion_pooled=pooled, cross_gain_pp=round((X['ensemble']['All test']['mAP50'] - max(X['single'][k]['All test']['mAP50'] for k in 'FYSM')) * 100, 1),
               coef=[r['coef'] for r in R], cross_coef=X['coef'])
open('/home/claude/work/exp/tables.tex', 'w').write(
    '%% T7\n' + t7 + '\n\n%% T8\n' + t8 + '\n\n%% T9\n' + t9 + '\n\n%% T11\n' + t11 + '\n\n%% T12\n' + t12 + '\n')
json.dump(summary, open('/home/claude/work/exp/summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k not in ('coef',)}, indent=1))
