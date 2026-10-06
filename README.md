# Stacked Ensemble Learning for Automated Wheat Spike Detection and Crop Yield Estimation

Code, image lists and results for the revised manuscript. Every number and figure in the paper
was produced by the scripts in `src/`, and the repository contains the per-image predictions
behind every table.

## Data

All images come from the **Global Wheat Head Detection 2021** dataset (David et al., *Plant
Phenomics* 2021, doi:10.34133/2021/9846158), available at <https://zenodo.org/records/5092309>.
Images were resized from 1024 × 1024 to 640 × 640 px (bilinear, JPEG quality 90; script
`colab/colab_shrink_all.py`). Box labels are taken from the official GWHD 2021 CSV files and
rescaled by 0.625.

| File | Content |
|---|---|
| `data/primary_48_images.csv` | The 48 images of the primary experiment (session, country, stage, head count, density class) |
| `data/primary.json` | The 3 outer 40/8 splits (disjoint test sets) and the 5 inner folds of each |
| `data/crossdomain.json` | Cross-domain image lists: 492 train, 268 validation, 1,382 test |

## Method in brief

* **Base detectors** (all COCO-pretrained, 640 × 640 input, SGD lr 0.01, momentum 0.9, wd 1e-4,
  cosine schedule): Faster R-CNN ResNet-50-FPN (12 epochs), YOLOv8s (40), SSD300-VGG16 (20),
  SSDlite320-MobileNetV3 (40). YOLOv8 and SSDlite were first run with 25 epochs; the sensitivity
  analysis on development folds showed they were under-trained, so they were retrained with 40
  (see `results/initial_25ep/` and `results/table_epochs_v1.tex` for the initial results). SSD default-box scales are set from the label size distribution
  (0.04 … 0.80 of the image side).
* **Augmentation** on the fly, training images only, after splitting: flips, 90° rotation, scale 0.8–1.2.
* **Stacking** (`src/stack.py`): boxes of all detectors (confidence ≥ 0.05) are clustered by IoU ≥ 0.55;
  each cluster gets 15 features (per detector: confidence, presence, IoU with consensus; plus number of
  agreeing detectors, width, height); an L2 logistic regression trained on **out-of-fold** predictions
  (5 inner folds) scores clusters; kept clusters are fused by confidence-weighted averaging. All
  thresholds are chosen by maximum out-of-fold F1; test images are used once.
* **Cross-domain**: detectors trained on 492 European images (10 epochs), meta-learner fitted on 268
  validation images (blending); YOLOv8 and SSDlite 16 epochs there, tested on all 1,382 test images from 6 unseen countries.
* **Yield**: spike density from each image's ground footprint (GWHD GSD), then
  Y = SD × grains/spike × TGW / 100 kg/ha with literature ranges for grains/spike and TGW.

## Main results

| Model | mAP@0.5 (3 splits) | mAP@0.5 unseen countries |
|---|---|---|
| Faster R-CNN | 0.814 ± 0.038 | 0.568 |
| SSD | 0.812 ± 0.003 | 0.529 |
| YOLOv8 | 0.840 ± 0.036 | 0.616 |
| MobileNet-SSD | 0.699 ± 0.061 | 0.422 |
| **Stacked ensemble** | **0.883 ± 0.022** | **0.622** |

Full tables: `results/tables.tex`, `results/summary.json`; per-split details in `results/results_rep*.json`.

## Reproducing

```bash
pip install -r requirements.txt
bash setup_paths.sh /path/to/gwhd_640_images /path/to/gwhd_csv_folder
cd /home/claude/work/exp
python3 prep_data.py      # image selection and splits (seed 42) -> primary.json, crossdomain.json
python3 run_all.py        # trains/predicts all 76 detector jobs (resumable; GPU used if available)
python3 eval_rep.py 0; python3 eval_rep.py 1; python3 eval_rep.py 2
python3 eval_cross.py; python3 aggregate.py           # Tables 7-9, 12, 13
python3 sens.py train; python3 sens.py eval           # Table 10 (sensitivity; part A reads the
                                                      #   initial 25-epoch predictions from exp/preds_v1_25ep/)
python3 compare_v1.py                                 # Table 11 (initial vs final budgets)
python3 cost.py; python3 yield_est.py; python3 density_clf.py   # Table 16, Section 4.9, Table 14
python3 fig_overview.py; python3 fig_detections.py; python3 fig_training.py; python3 fig_density.py; python3 fig_sensitivity.py; python3 viz_cam.py 0
```

`colab/COLAB_RUN.py` runs `run_all.py` on a free Colab GPU (a few hours instead of ~3 days on CPU).
The per-image predictions used in the paper are in `results/predictions.zip`
(`rep{0,1,2}/<detector>_fold{k}.json`, `<detector>_final.json`, `cross/<detector>.json`, `sens/`);
unzip them into `exp/` (they extract to `exp/preds/`) to recompute every table without retraining.
`results/predictions_initial_25ep.zip` holds the YOLOv8/SSDlite predictions of the initial 25-epoch
runs; unzip into `exp/preds_v1_25ep/` for `sens.py eval` and `compare_v1.py`.

## Licence

[Choose a licence, e.g. MIT, and add a LICENSE file.]
