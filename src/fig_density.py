import json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open('results_density.json'))['results']
names = list(R); short = ['Logistic\nRegression', 'Random\nForest', 'SVM', 'Gradient\nBoosting', 'Stacked\nensemble']
fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(2, 5, height_ratios=[1.1, 1])
ax = fig.add_subplot(gs[0, :])
mets = ['acc', 'F1', 'MCC', 'AUC']; lab = ['Accuracy', 'Macro F1', 'MCC', 'AUC (one-vs-rest)']
col = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']; w = 0.2; x = np.arange(len(names))
for i, (m, c, l) in enumerate(zip(mets, col, lab)):
    v = [R[n][m] for n in names]
    ax.bar(x + (i - 1.5) * w, v, w, color=c, label=l, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels([s.replace('\n', ' ') for s in short]); ax.set_ylim(0, 1)
ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.18)); ax.set_ylabel('Score')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='y', color='#e6e5e1', lw=0.6); ax.set_axisbelow(True)
ax.set_title('(a) Density-class prediction, 48 images, stratified 5-fold CV', loc='left', fontsize=11, pad=28)
for j, n in enumerate(names):
    a = fig.add_subplot(gs[1, j]); cm = np.array(R[n]['cm'])
    a.imshow(cm, cmap='Blues', vmin=0, vmax=18)
    for r in range(3):
        for c in range(3):
            a.text(c, r, cm[r, c], ha='center', va='center', color='white' if cm[r, c] > 9 else '#0b0b0b', fontsize=11)
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(['S', 'M', 'D']); a.set_yticklabels(['S', 'M', 'D'])
    a.set_xlabel('Predicted'); a.set_ylabel('True' if j == 0 else '')
    a.set_title(('(b) ' if j == 0 else '') + short[j].replace('\n', ' ') + f"\n{R[n]['correct']}/48 correct", fontsize=10)
plt.tight_layout(); plt.savefig('fig_density.pdf'); plt.savefig('fig_density.png', dpi=200)
