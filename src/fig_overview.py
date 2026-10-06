"""Fig. 6: (a) mAP@0.5 per model and split, (b) predicted vs annotated count on the 24 test images,
(c) cross-domain mAP@0.5 per test country."""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import stack

R = [json.load(open(f'results_rep{r}.json')) for r in range(3)]; X = json.load(open('results_cross.json'))
P = json.load(open('primary.json'))
C = {'F': '#2a78d6', 'Y': '#eb6834', 'S': '#1baf7a', 'M': '#eda100', 'E': '#4a3aa7'}
NM = {'F': 'Faster R-CNN', 'Y': 'YOLOv8', 'S': 'SSD', 'M': 'MobileNet-SSD', 'E': 'Ensemble'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={'width_ratios': [1, 1, 1.5]})

# (a)
keys = ['M', 'Y', 'S', 'F', 'E']
for i, k in enumerate(keys):
    v = [(r['subsets']['F+Y+S+M'] if k == 'E' else r['single'][k])['mAP50'] for r in R]
    ax[0].bar(i, np.mean(v), 0.6, color=C[k], alpha=0.35, edgecolor='white')
    ax[0].errorbar(i, np.mean(v), yerr=np.std(v, ddof=1), color='#0b0b0b', capsize=4, lw=1.2)
    ax[0].scatter([i] * 3, v, color=C[k], s=28, zorder=3, edgecolor='white', lw=0.8)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels([NM[k].replace('MobileNet-SSD', 'MobileNet\n-SSD').replace('Faster R-CNN', 'Faster\nR-CNN') for k in keys])
ax[0].set_ylim(0.5, 0.95); ax[0].set_ylabel('mAP@0.5'); ax[0].set_title('(a) Held-out test images, 3 splits', loc='left')
ax[0].grid(axis='y', color='#e6e5e1', lw=0.6); ax[0].set_axisbelow(True)

# (b) counts on the 24 test images: ensemble and best single detector (SSD)
true, ens, ssd = [], [], []
for r, rep in enumerate(P['reps']):
    D = f'preds/rep{r}'
    oof = {d: {} for d in stack.DETS}; fin = {}
    for d in stack.DETS:
        for k in range(5): oof[d].update(json.load(open(f'{D}/{d}_fold{k}.json')))
        fin[d] = json.load(open(f'{D}/{d}_final.json'))
    st = stack.Stacker(stack.DETS).fit(oof, rep['dev']); e = st.predict(fin, rep['test'])
    ts = R[r]['single']['S']['tau']
    for n in rep['test']:
        true.append(len(stack.GT[n])); ens.append(int((np.asarray(e[n]['scores']) >= st.tau).sum()))
        ssd.append(int((np.asarray(fin['ssd'][n]['scores']) >= ts).sum()))
true, ens, ssd = map(np.array, (true, ens, ssd))
lim = [0, max(true.max(), ens.max(), ssd.max()) + 5]
ax[1].plot(lim, lim, color='#52514e', lw=1, ls='--', label='1:1')
ax[1].scatter(true, ssd, s=30, color=C['S'], label=f'SSD (MAE {np.abs(ssd - true).mean():.1f})', edgecolor='white', lw=0.6)
ax[1].scatter(true, ens, s=30, color=C['E'], marker='D', label=f'Ensemble (MAE {np.abs(ens - true).mean():.1f})', edgecolor='white', lw=0.6)
ax[1].set_xlim(lim); ax[1].set_ylim(lim); ax[1].set_xlabel('Annotated heads per image'); ax[1].set_ylabel('Detected heads per image')
ax[1].set_title('(b) Counts on the 24 test images', loc='left'); ax[1].legend(frameon=False, loc='upper left')
ax[1].grid(color='#e6e5e1', lw=0.6); ax[1].set_axisbelow(True)

# (c) cross-domain per country
G = ['Australia', 'China', 'Japan', 'Mexico', 'Sudan', 'US', 'All test']
w = 0.16; x = np.arange(len(G))
for i, k in enumerate(['F', 'Y', 'S', 'M', 'E']):
    v = [(X['ensemble'][g] if k == 'E' else X['single'][k][g])['mAP50'] for g in G]
    ax[2].bar(x + (i - 2) * w, v, w, color=C[k], label=NM[k], edgecolor='white', lw=0.6)
ax[2].set_xticks(x); ax[2].set_xticklabels([g.replace('US', 'USA').replace('All test', 'All (1,382)') for g in G])
ax[2].set_ylabel('mAP@0.5'); ax[2].set_ylim(0, 1.12); ax[2].set_yticks(np.arange(0, 1.01, 0.2)); ax[2].set_title('(c) Unseen countries (GWHD 2021 test split)', loc='left')
ax[2].legend(frameon=False, ncol=5, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.0))
ax[2].grid(axis='y', color='#e6e5e1', lw=0.6); ax[2].set_axisbelow(True)
plt.tight_layout(); plt.savefig('fig_overview.pdf'); plt.savefig('fig_overview.png', dpi=200)
print('MAE ens', np.abs(ens - true).mean(), 'ssd', np.abs(ssd - true).mean(), 'total true', true.sum(), 'ens', ens.sum())
