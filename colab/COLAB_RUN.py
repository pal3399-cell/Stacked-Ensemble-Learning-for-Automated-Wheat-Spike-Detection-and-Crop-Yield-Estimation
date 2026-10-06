# =====================================================================
# BACKUP: run all detector training on a free Colab GPU (a few hours).
# 1) Runtime -> Change runtime type -> T4 GPU
# 2) Upload experiment_code.zip to Colab (folder icon on the left)
# 3) Paste this whole file into one cell and run it.
# 4) When it finishes, download /content/predictions.zip and attach it
#    in the Claude chat (it is small, a few tens of MB).
# =====================================================================
import os, subprocess, zipfile, shutil
def sh(c): print('$', c); subprocess.run(c, shell=True, check=True)

sh('pip -q install ultralytics ensemble-boxes')
for d in ['/home/claude/work/exp', '/home/claude/work/data/images', '/home/claude/work/weights', '/tmp/claude-0']:
    os.makedirs(d, exist_ok=True)
zipfile.ZipFile('/content/experiment_code.zip').extractall('/home/claude/work/exp')

REL = ('https://github.com/pal3399-cell/Stacked-Ensemble-Learning-for-Automated-Wheat-Spike-Detection-'
       'and-Crop-Yield-Estimation/releases/download/V1_Data/gwhd_all_640.zip')
if len(os.listdir('/home/claude/work/data/images')) < 6515:
    sh(f'wget -q -O /content/gwhd.zip "{REL}"')
    zipfile.ZipFile('/content/gwhd.zip').extractall('/home/claude/work/data/images')
sh('wget -q -O /home/claude/work/weights/yolov8s.pt '
   'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt')

import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - switch runtime to T4 GPU!')
os.chdir('/home/claude/work/exp')
sh('python run_all.py')

shutil.make_archive('/content/predictions', 'zip', '/home/claude/work/exp', 'preds')
print('Done. Download /content/predictions.zip and attach it in the Claude chat.')
