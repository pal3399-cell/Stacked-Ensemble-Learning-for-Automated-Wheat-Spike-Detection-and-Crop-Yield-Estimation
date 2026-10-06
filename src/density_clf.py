"""Secondary experiment: predict density class (sparse/moderate/dense) from detection-derived
features. Features for each of the 48 images come from a YOLOv8 model that did not see the image
(split-1 out-of-fold predictions for the 40 development images, the refitted model for the 8 test
images). Classifiers are evaluated with stratified 5-fold CV (out-of-fold predictions)."""
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score, confusion_matrix
import stack

P = json.load(open('primary.json')); rep = P['reps'][0]
pred = {}
for k in range(5): pred.update(json.load(open(f'preds/rep0/yolo_fold{k}.json')))
pred.update(json.load(open('preds/rep0/yolo_final.json')))
tau = stack.best_tau({n: pred[n] for n in rep['dev']}, rep['dev'])
imgs = P['images']; cls = {'sparse': 0, 'moderate': 1, 'dense': 2}
X, y = [], []
for n in imgs:
    s = np.asarray(pred[n]['scores']); b = np.asarray(pred[n]['boxes']).reshape(-1, 4)[s >= tau]
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) if len(b) else np.zeros(1)
    X.append([len(b), a.mean(), a.std(), len(b) / (640 * 640) * 1e4]); y.append(cls[P['density'][n]])
X, y = np.array(X), np.array(y)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
base = {'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'Random Forest': RandomForestClassifier(300, random_state=42),
        'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)}
models = dict(base); models['Stacked ensemble'] = StackingClassifier(list(base.items()), final_estimator=LogisticRegression(max_iter=2000), cv=5)
res = {}
for name, m in models.items():
    prob = cross_val_predict(m, X, y, cv=cv, method='predict_proba'); p = prob.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    res[name] = dict(acc=round(accuracy_score(y, p), 3), P=round(pr, 3), R=round(rc, 3), F1=round(f1, 3),
                     MCC=round(matthews_corrcoef(y, p), 3), AUC=round(roc_auc_score(y, prob, multi_class='ovr'), 3),
                     cm=confusion_matrix(y, p).tolist(), correct=int((p == y).sum()))
json.dump(dict(n=len(y), results=res), open('results_density.json', 'w'), indent=1)
for k, v in res.items(): print(k.ljust(20), {kk: vv for kk, vv in v.items() if kk != 'cm'})
