"""Fig. 8: hyperparameter sensitivity on development data (inner folds of split 1).
(a) learning rate and (b) epochs for YOLOv8 / MobileNet-SSD (folds 0-1, mAP@0.5);
(c) cluster IoU and (d) counting threshold for the stacked ensemble (nested 5-fold CV)."""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

S = json.load(open('results_sens.json')); A, B = S['A'], S['B']
C = {'yolo': '#eb6834', 'ssdlite': '#eda100', 'E': '#4a3aa7'}
NM = {'yolo': 'YOLOv8', 'ssdlite': 'MobileNet-SSD'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))


def get(m, lr, ep):
    for k, v in A.items():
        if k.startswith(f'{m} lr={lr} ep={ep}'): return v


for m, mk in [('yolo', 'o'), ('ssdlite', 's')]:
    lrs = [0.001, 0.005, 0.01, 0.02]
    ax[0].plot(range(4), [get(m, l, 25) for l in lrs], marker=mk, color=C[m], lw=2, ms=6, label=NM[m])
    eps = [10, 25, 40]
    ax[1].plot(eps, [get(m, 0.01, e) for e in eps], marker=mk, color=C[m], lw=2, ms=6, label=NM[m])
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(['0.001', '0.005', '0.01', '0.02'])
ax[0].set_xlabel('Learning rate (25 epochs)'); ax[0].set_title('(a) Learning rate', loc='left')
ax[0].axvline(2, color='#a3a29e', lw=1, ls=':')
ax[1].set_xticks([10, 25, 40]); ax[1].set_xlabel('Epochs (lr 0.01)'); ax[1].set_title('(b) Training epochs', loc='left')
ax[1].axvline(25, color='#a3a29e', lw=1, ls=':'); ax[1].axvline(40, color='#0b0b0b', lw=1, ls='--')
ax[1].text(25, 0.03, ' initial', color='#52514e', fontsize=8); ax[1].text(40, 0.03, ' used', color='#0b0b0b', fontsize=8, ha='right')
for a in ax[:2]:
    a.set_ylabel('mAP@0.5 (dev. folds)'); a.set_ylim(0, 0.85); a.legend(frameon=False, loc='lower right' if a is ax[0] else 'upper left')

iou = [0.4, 0.5, 0.55, 0.6, 0.7]
mu = np.array([B[f'tau_iou={v}'][0] for v in iou]); sd = np.array([B[f'tau_iou={v}'][1] for v in iou])
ax[2].errorbar(iou, mu, yerr=sd, color=C['E'], marker='D', lw=2, ms=6, capsize=3)
ax[2].axvline(0.55, color='#0b0b0b', lw=1, ls='--')
ax[2].set_xlabel(r'Cluster IoU $\tau_{IoU}$'); ax[2].set_ylabel('Ensemble mAP@0.5 (nested CV)')
ax[2].set_ylim(0.7, 0.95); ax[2].set_title('(c) Ensemble: cluster IoU', loc='left')

ct = B['count_threshold']; th = sorted(ct, key=float)
ax[3].plot([float(t) for t in th], [ct[t]['MAE'] for t in th], color=C['E'], marker='D', lw=2, ms=6)
ax[3].axvline(B['selected_tau_meta'], color='#0b0b0b', lw=1, ls='--')
for t in th:
    ax[3].annotate(f"F1 {ct[t]['F1']:.3f}", (float(t), ct[t]['MAE']), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=7, color='#52514e')
ax[3].set_xlabel(r'Meta-learner threshold $\tau_{meta}$'); ax[3].set_ylabel('Count MAE (heads / image)')
ax[3].set_ylim(0, max(ct[t]['MAE'] for t in th) * 1.25); ax[3].set_title('(d) Ensemble: counting threshold', loc='left')
for a in ax:
    a.grid(axis='y', color='#e6e5e1', lw=0.6); a.set_axisbelow(True)
plt.tight_layout(); plt.savefig('fig_sensitivity.pdf'); plt.savefig('fig_sensitivity.png', dpi=200)
