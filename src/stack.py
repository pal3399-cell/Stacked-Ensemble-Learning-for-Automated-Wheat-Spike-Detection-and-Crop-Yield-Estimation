"""Stacked ensemble: box alignment, meta-features, meta-learner, fusion baselines, evaluation."""
import json, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DETS = ['frcnn', 'yolo', 'ssd', 'ssdlite']          # F, Y, S, M
SHORT = {'frcnn': 'F', 'yolo': 'Y', 'ssd': 'S', 'ssdlite': 'M'}
TAU0 = 0.05        # pre-filter confidence
TAU_IOU = 0.55     # cluster IoU threshold
GT = json.load(open('/home/claude/work/exp/gt.json'))


# ------------------------------------------------------------------ geometry
def iou_mat(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


# ------------------------------------------------------------------ matching / metrics
def match(boxes, scores, gt, thr=0.5):
    """Greedy matching in descending score order. Returns tp flags (sorted order) and sorted scores."""
    order = np.argsort(-np.asarray(scores)); boxes = np.asarray(boxes, float).reshape(-1, 4)[order]
    s = np.asarray(scores, float)[order]
    tp = np.zeros(len(s), bool)
    if len(gt) and len(s):
        M = iou_mat(boxes, gt); used = np.zeros(len(gt), bool)
        for i in range(len(s)):
            m = np.where(used, -1, M[i]); j = int(np.argmax(m))
            if m[j] >= thr:
                tp[i] = True; used[j] = True
    return tp, s


def average_precision(dets, imgs, thr=0.5):
    """COCO-style 101-point interpolated AP over a set of images (one class)."""
    allt, alls, npos = [], [], 0
    for n in imgs:
        tp, s = match(dets[n]['boxes'], dets[n]['scores'], GT[n], thr)
        allt.append(tp); alls.append(s); npos += len(GT[n])
    tp = np.concatenate(allt); s = np.concatenate(alls)
    o = np.argsort(-s, kind='stable'); tp = tp[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(~tp)
    rec = ctp / max(npos, 1); prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side='left')
        ap += prec[k] if k < len(prec) else 0.0
    return ap / 101


def map_5095(dets, imgs):
    return float(np.mean([average_precision(dets, imgs, t) for t in np.arange(0.5, 0.96, 0.05)]))


def prf_count(dets, imgs, tau):
    TP = FP = FN = 0; err = []
    for n in imgs:
        sc = np.asarray(dets[n]['scores']); keep = sc >= tau
        b = np.asarray(dets[n]['boxes']).reshape(-1, 4)[keep]
        tp, _ = match(b, sc[keep], GT[n]); t = int(tp.sum())
        TP += t; FP += int(keep.sum()) - t; FN += len(GT[n]) - t
        err.append(int(keep.sum()) - len(GT[n]))
    P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1); F = 2 * P * R / max(P + R, 1e-9)
    err = np.array(err)
    return dict(P=P, R=R, F1=F, MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err ** 2).mean())),
                count=int(sum(int((np.asarray(dets[n]['scores']) >= tau).sum()) for n in imgs)),
                gt=int(sum(len(GT[n]) for n in imgs)))


def best_tau(dets, imgs):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f = [prf_count(dets, imgs, t)['F1'] for t in grid]
    return float(grid[int(np.argmax(f))])


def evaluate(dets, imgs, tau):
    r = prf_count(dets, imgs, tau)
    r.update(mAP50=average_precision(dets, imgs, 0.5), mAP5095=map_5095(dets, imgs), tau=tau)
    return r


# ------------------------------------------------------------------ clustering + meta-features
def cluster_image(per_det, dets):
    """per_det: {det: {'boxes','scores'}} for one image. Returns list of clusters,
    each {det: (box, score)}. Greedy, highest score first, at most one box per detector."""
    pool = []
    for d in dets:
        b = np.asarray(per_det[d]['boxes'], float).reshape(-1, 4); s = np.asarray(per_det[d]['scores'], float)
        k = s >= TAU0
        for bb, ss in zip(b[k], s[k]):
            pool.append((ss, d, bb))
    pool.sort(key=lambda x: -x[0])
    clusters, heads = [], []          # heads: box of each cluster's highest-score member
    for s, d, b in pool:
        placed = False
        if heads:
            ious = iou_mat([b], heads)[0]
            for c in np.argsort(-ious):
                if ious[c] < TAU_IOU:
                    break
                if d not in clusters[c]:
                    clusters[c][d] = (b, s); placed = True
                    break
        if not placed:
            clusters.append({d: (b, s)}); heads.append(b)
    return clusters


def fuse(cl):
    w = np.array([s for _, s in cl.values()]); b = np.array([bb for bb, _ in cl.values()])
    return (w[:, None] * b).sum(0) / w.sum()


def features(cl, dets):
    cons = fuse(cl); f = []
    for d in dets:
        if d in cl:
            b, s = cl[d]; f += [s, 1.0, float(iou_mat([b], [cons])[0, 0])]
        else:
            f += [0.0, 0.0, 0.0]
    f += [len(cl), (cons[2] - cons[0]) / 640, (cons[3] - cons[1]) / 640]
    return f, cons


def build(preds, imgs, dets, with_labels=True):
    X, y, meta = [], [], []
    for n in imgs:
        cls = cluster_image({d: preds[d][n] for d in dets}, dets)
        feats = [features(c, dets) for c in cls]
        if not feats:
            continue
        F = np.array([f for f, _ in feats]); B = np.array([b for _, b in feats])
        lab = np.zeros(len(F), int)
        if with_labels and len(GT[n]):
            order = np.argsort(-F[:, [3 * i for i in range(len(dets))]].max(1))
            M = iou_mat(B, GT[n]); used = np.zeros(len(GT[n]), bool)
            for i in order:
                m = np.where(used, -1, M[i]); j = int(np.argmax(m))
                if m[j] >= 0.5:
                    lab[i] = 1; used[j] = True
        X.append(F); y.append(lab); meta += [(n, b) for b in B]
    return np.vstack(X), np.concatenate(y), meta


def to_dets(meta, scores, imgs):
    out = {n: {'boxes': [], 'scores': []} for n in imgs}
    for (n, b), s in zip(meta, scores):
        out[n]['boxes'].append(list(b)); out[n]['scores'].append(float(s))
    return out


class Stacker:
    def __init__(self, dets, C=1.0):
        self.dets = dets
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, oof_preds, imgs):
        X, y, _ = build(oof_preds, imgs, self.dets)
        self.model.fit(X, y)
        self.tau = best_tau(self.predict(oof_preds, imgs), imgs)
        return self

    def predict(self, preds, imgs):
        X, _, meta = build(preds, imgs, self.dets, with_labels=False)
        return to_dets(meta, self.model.predict_proba(X)[:, 1], imgs)

    def coefficients(self):
        lr = self.model[-1]; sc = self.model[0]
        names = [f'{k}_{SHORT[d]}' for d in self.dets for k in ('conf', 'present', 'iou')] + ['n_agree', 'w', 'h']
        return dict(zip(names, (lr.coef_[0]).round(3).tolist()))


# ------------------------------------------------------------------ fixed fusion rules (baselines)
def union_nms(preds, imgs, dets, iou=0.5):
    import torch, torchvision
    out = {}
    for n in imgs:
        b = np.vstack([np.asarray(preds[d][n]['boxes'], float).reshape(-1, 4) for d in dets])
        s = np.concatenate([np.asarray(preds[d][n]['scores'], float) for d in dets])
        k = torchvision.ops.nms(torch.tensor(b), torch.tensor(s), iou).numpy() if len(s) else []
        out[n] = {'boxes': b[k].tolist(), 'scores': s[k].tolist()}
    return out


def vote_wbf(preds, imgs, dets, min_votes=2):
    out = {}
    for n in imgs:
        cls = cluster_image({d: preds[d][n] for d in dets}, dets)
        bb, ss = [], []
        for c in cls:
            if len(c) >= min_votes:
                bb.append(fuse(c).tolist()); ss.append(float(np.mean([s for _, s in c.values()]) * len(c) / len(dets)))
        out[n] = {'boxes': bb, 'scores': ss}
    return out


def wbf_equal(preds, imgs, dets):
    from ensemble_boxes import weighted_boxes_fusion
    out = {}
    for n in imgs:
        bl = [np.clip(np.asarray(preds[d][n]['boxes'], float).reshape(-1, 4) / 640, 0, 1).tolist() for d in dets]
        sl = [preds[d][n]['scores'] for d in dets]; ll = [[0] * len(s) for s in sl]
        b, s, _ = weighted_boxes_fusion(bl, sl, ll, iou_thr=TAU_IOU, skip_box_thr=TAU0)
        out[n] = {'boxes': (np.asarray(b) * 640).tolist(), 'scores': np.asarray(s).tolist()}
    return out
