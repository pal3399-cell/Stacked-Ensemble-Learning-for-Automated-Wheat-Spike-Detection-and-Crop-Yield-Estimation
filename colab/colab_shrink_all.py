# ================================================================
#  Run this in Google Colab:  Runtime -> Run all
#  It downloads GWHD 2021 from Zenodo, shrinks all images to
#  640x640 JPEG, packages the detector weights, and puts the
#  files to upload in /content/upload/
#  Time: about 15-30 minutes.  No GPU needed.
# ================================================================
import os, glob, zipfile, subprocess, requests
from PIL import Image

OUT = "/content/upload"
os.makedirs(OUT, exist_ok=True)

# ---------- STEP 1: download GWHD 2021 from Zenodo ----------
# (If the images are already in your Google Drive, skip this step and set
#  SRC in step 2 to that folder after mounting Drive.)
rec = requests.get("https://zenodo.org/api/records/5092309").json()
for f in rec["files"]:
    print("Zenodo file:", f["key"], round(f["size"] / 1e9, 2), "GB")
    if f["key"].endswith(".zip"):
        path = f"/content/{f['key']}"
        if not os.path.exists(path):
            subprocess.run(["wget", "-q", "-O", path, f["links"]["self"]], check=True)
        zipfile.ZipFile(path).extractall("/content/gwhd_raw")

# find the folder that holds the .png images
pngs = glob.glob("/content/gwhd_raw/**/*.png", recursive=True)
SRC = os.path.dirname(pngs[0])
print("Found", len(pngs), "images in", SRC)

# ---------- STEP 2: shrink all images to 640x640 JPEG ----------
with zipfile.ZipFile(f"{OUT}/gwhd_all_640.zip", "w", zipfile.ZIP_STORED) as z:
    for i, p in enumerate(sorted(pngs)):
        im = Image.open(p).convert("RGB").resize((640, 640), Image.BILINEAR)
        im.save("/tmp/t.jpg", quality=90)
        z.write("/tmp/t.jpg", os.path.basename(p)[:-4] + ".jpg")
        if i % 500 == 0:
            print("shrunk", i, "/", len(pngs))

# ---------- STEP 3: detector weights (COCO-pretrained) ----------
import torchvision
torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
ck = os.path.expanduser("~/.cache/torch/hub/checkpoints")
with zipfile.ZipFile(f"{OUT}/detector_weights.zip", "w", zipfile.ZIP_STORED) as z:
    for f in glob.glob(ck + "/*.pth"):
        z.write(f, os.path.basename(f))

# ---------- done ----------
print("\nFiles to upload:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}: {os.path.getsize(os.path.join(OUT, f)) // 1_000_000} MB")

# Optional: copy the results to your Google Drive so they are not lost
# from google.colab import drive; drive.mount('/content/drive')
# !cp /content/upload/* /content/drive/MyDrive/
