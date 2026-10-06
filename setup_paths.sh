#!/bin/bash
# The scripts in src/ are the exact files used for the paper; they read and write fixed
# paths under /home/claude/work. This script recreates that layout with symbolic links.
# Usage:  bash setup_paths.sh /path/to/gwhd_640_images /path/to/gwhd_csv_folder
set -e
IMAGES=${1:?folder with the 6,515 GWHD 2021 images resized to 640x640 JPEG}
CSV=${2:?folder with competition_train.csv, competition_val.csv, competition_test.csv, metadata_dataset.csv}
REPO=$(cd "$(dirname "$0")" && pwd)
mkdir -p /home/claude/work/exp /home/claude/work/data /home/claude/work/weights /tmp/claude-0
ln -sfn "$IMAGES" /home/claude/work/data/images
ln -sf "$CSV/competition_train.csv"  /home/claude/work/19410d63-competition_train.csv
ln -sf "$CSV/competition_val.csv"    /home/claude/work/2cb383b4-competition_val.csv
ln -sf "$CSV/competition_test.csv"   /home/claude/work/81e52d93-competition_test.csv
ln -sf "$CSV/metadata_dataset.csv"   /home/claude/work/77264cdf-metadata_dataset.csv
for f in "$REPO"/src/*.py; do ln -sf "$f" /home/claude/work/exp/; done
cp -n "$REPO"/data/*.json /home/claude/work/exp/ 2>/dev/null || true
wget -q -O /home/claude/work/weights/yolov8s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
echo "Layout ready. Run:  cd /home/claude/work/exp && python3 prep_data.py && python3 run_all.py"
