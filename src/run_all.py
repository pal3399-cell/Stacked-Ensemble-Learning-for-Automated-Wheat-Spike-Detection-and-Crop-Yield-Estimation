"""Run every base-detector training/prediction job in order. Resumable:
a job whose prediction file exists is skipped, so the script can be restarted."""
import json, os, sys, time, traceback
from detectors import train_torchvision, predict_torchvision, train_yolo, predict_yolo

EPOCHS = {'frcnn': 12, 'ssd': 20, 'ssdlite': 40, 'yolo': 40}   # v2: YOLOv8/SSDlite 25 -> 40 after sensitivity analysis
CROSS_EPOCHS = {'frcnn': 10, 'ssd': 10, 'ssdlite': 16, 'yolo': 16}   # same 1.6x increase for YOLOv8/SSDlite
MODELS = ['ssdlite', 'yolo', 'ssd', 'frcnn']
ROOT = '/home/claude/work/exp/preds'
P = json.load(open('/home/claude/work/exp/primary.json'))
C = json.load(open('/home/claude/work/exp/crossdomain.json'))


def log(msg):
    line = time.strftime('%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    open('/home/claude/work/exp/run.log', 'a').write(line + '\n')


def jobs():
    """(output file, model, train images, images to predict)"""
    J = []
    order = [0, 1, 2, 'cross']
    for r in order:
        for mdl in MODELS:
            if r == 'cross':
                J.append((f'cross/{mdl}.json', mdl, C['train'], C['val'] + C['test']))
                continue
            rep = P['reps'][r]
            for k in range(5):  # out-of-fold predictions for the meta-learner
                tr = [i for j, f in enumerate(rep['folds']) if j != k for i in f]
                J.append((f'rep{r}/{mdl}_fold{k}.json', mdl, tr, rep['folds'][k]))
            J.append((f'rep{r}/{mdl}_final.json', mdl, rep['dev'], rep['test']))  # refit on all 40
    return J


def run(out, mdl, tr, pr):
    path = f'{ROOT}/{out}'
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = time.time()
    ep = CROSS_EPOCHS[mdl] if out.startswith('cross') else EPOCHS[mdl]
    log(f'START {out}  train={len(tr)} predict={len(pr)} epochs={ep}')
    if mdl == 'yolo':
        m = train_yolo(tr, ep, f'/tmp/claude-0/yolo_{out.replace("/", "_")}')
        preds = predict_yolo(m, pr)
    else:
        ck = f'/home/claude/work/exp/ckpt/{out.replace("/", "_")}.ckpt'; os.makedirs(os.path.dirname(ck), exist_ok=True)
        m = train_torchvision(mdl, tr, ep, log=log, ckpt=ck)
        preds = predict_torchvision(m, pr)
        if out.endswith('_final.json') or out.startswith('cross'):
            import torch
            torch.save(m.state_dict(), path.replace('.json', '.pt'))
    json.dump(preds, open(path + '.tmp', 'w'))
    os.replace(path + '.tmp', path)
    if mdl != 'yolo' and os.path.exists(ck): os.remove(ck)
    log(f'DONE  {out}  {(time.time() - t) / 60:.1f} min')


if __name__ == '__main__':
    J = jobs()
    if len(sys.argv) == 3 and sys.argv[1] == '--job':      # child process: one job, then exit (frees all memory)
        run(*J[int(sys.argv[2])]); sys.exit(0)
    import subprocess
    log(f'--- runner started, {len(J)} jobs, {sum(os.path.exists(f"{ROOT}/{j[0]}") for j in J)} already done')
    for i, j in enumerate(J):
        if os.path.exists(f'{ROOT}/{j[0]}'):
            continue
        rc = subprocess.run([sys.executable, 'run_all.py', '--job', str(i)], cwd='/home/claude/work/exp').returncode
        if rc != 0:
            log(f'FAILED {j[0]} (exit code {rc}); will retry on next runner start')
    log('--- all jobs finished v2')
