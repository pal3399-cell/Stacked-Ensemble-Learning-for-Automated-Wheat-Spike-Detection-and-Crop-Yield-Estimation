"""Build all image lists and labels for the experiments (seed 42, reproducible)."""
import json, os, numpy as np, pandas as pd
rng = np.random.default_rng(42)
W = '/home/claude/work'
meta = pd.read_csv(f'{W}/77264cdf-metadata_dataset.csv', sep=';'); meta['key'] = meta.name.str.lower()
S = 640/1024
rows = []
for split, f in [('train','19410d63-competition_train.csv'),('val','2cb383b4-competition_val.csv'),('test','81e52d93-competition_test.csv')]:
    d = pd.read_csv(f'{W}/{f}'); d['split'] = split; rows.append(d)
df = pd.concat(rows, ignore_index=True)
df['key'] = df.domain.str.lower(); df = df.merge(meta[['key','country','development_stage']], on='key')
def boxes(s):
    if pd.isna(s) or str(s).strip()=='no_box': return []
    out=[]
    for b in str(s).split(';'):
        b=b.strip()
        if not b: continue
        x1,y1,x2,y2=[float(v)*S for v in b.split()]
        if x2-x1>=1 and y2-y1>=1: out.append([round(x1,2),round(y1,2),round(x2,2),round(y2,2)])
    return out
df['boxes'] = df.BoxesString.apply(boxes); df['n'] = df.boxes.apply(len)
df['img'] = df.image_name.str.replace('.png','.jpg')
# ground truth for every image
gt = {r.img: r.boxes for r in df.itertuples()}
info = {r.img: dict(split=r.split, domain=r.domain, country=r.country, stage=r.development_stage, n=r.n) for r in df.itertuples()}
json.dump(gt, open('gt.json','w')); json.dump(info, open('info.json','w'))

# ---------- primary experiment: 48 images, density-balanced ----------
pool = df[(df.split!='test') & (df.n>0)].copy()
pool['density'] = np.where(pool.n<30,'sparse',np.where(pool.n<=51,'moderate','dense'))
want = {'dense':18,'moderate':16,'sparse':14}
sel = pd.concat([pool[pool.density==k].sample(v, random_state=42) for k,v in want.items()])
dens = dict(zip(sel.img, sel.density))
# three disjoint stratified 8-image test sets: 3 dense, 3 moderate, 2 sparse each
reps = []
by = {k: list(rng.permutation(sel[sel.density==k].img.values)) for k in want}
for r in range(3):
    test = by['dense'][3*r:3*r+3] + by['moderate'][3*r:3*r+3] + by['sparse'][2*r:2*r+2]
    dev = [i for i in sel.img if i not in test]
    # 5 stratified inner folds over the 40 development images
    folds = [[] for _ in range(5)]
    for k in want:
        imgs = list(rng.permutation([i for i in dev if dens[i]==k]))
        for j,i in enumerate(imgs): folds[j%5].append(i)
    reps.append(dict(test=test, dev=dev, folds=folds))
json.dump(dict(images=list(sel.img), density=dens, reps=reps), open('primary.json','w'), indent=1)
print('primary 48:', sel.density.value_counts().to_dict(), 'countries', sel.country.nunique(), 'domains', sel.domain.nunique(), 'stages', sel.development_stage.nunique())

# ---------- cross-domain: 500 train / 300 val / all test ----------
def strat(d, n, col):
    per = int(np.ceil(n / d[col].nunique()))
    s = pd.concat([g.sample(min(len(g), per), random_state=42) for _, g in d.groupby(col)])
    return s.sample(min(n, len(s)), random_state=42)
tr = strat(df[df.split=='train'], 500, 'domain'); va = strat(df[df.split=='val'], 300, 'domain')
te = df[df.split=='test']
json.dump(dict(train=list(tr.img), val=list(va.img), test=list(te.img)), open('crossdomain.json','w'))
print('cross-domain:', len(tr), len(va), len(te), 'train domains', tr.domain.nunique(), 'val domains', va.domain.nunique())
sel[['img','domain','country','development_stage','n','density']].to_csv('primary_48_images.csv', index=False)
