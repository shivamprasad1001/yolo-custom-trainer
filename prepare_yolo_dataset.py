import os
import random
import shutil
import zipfile
import yaml
from pathlib import Path

# === CONFIGURATION ===
ZIP_PATH = "data.zip"            # Path to your zip file
TRAIN_SPLIT = 0.8                # Ratio of training data
EXTRACT_DIR = "temp_extracted"  # Temporary extraction folder
OUTPUT_DIR = "data"             # Final YOLO-ready dataset folder

# === Step 1: Unzip ===
if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(f"ZIP file not found: {ZIP_PATH}")

print(f"📦 Unzipping {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
print(f"✅ Extracted to {EXTRACT_DIR}/")

# === Step 2: Paths ===
images_dir = os.path.join(EXTRACT_DIR, "images")
labels_dir = os.path.join(EXTRACT_DIR, "labels")
classes_file = os.path.join(EXTRACT_DIR, "classes.txt")

if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
    raise FileNotFoundError("❌ 'images/' or 'labels/' folder not found inside ZIP.")
if not os.path.exists(classes_file):
    raise FileNotFoundError("❌ 'classes.txt' not found inside ZIP.")

# === Step 3: Create Output Folder Structure ===
subdirs = [
    "train/images", "train/labels",
    "validation/images", "validation/labels"
]
for subdir in subdirs:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

# === Step 4: Split Dataset ===
image_files = list(Path(images_dir).rglob('*.[jp][pn]g'))  # .jpg/.png
random.shuffle(image_files)

split_index = int(len(image_files) * TRAIN_SPLIT)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def move_files(file_list, img_dst, lbl_dst):
    for img_path in file_list:
        img_name = img_path.name
        base_name = img_path.stem
        label_name = base_name + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        shutil.copy(img_path, os.path.join(img_dst, img_name))

        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(lbl_dst, label_name))
        else:
            print(f"⚠️ Missing label for {img_name}, skipping label.")

move_files(train_files, os.path.join(OUTPUT_DIR, "train/images"), os.path.join(OUTPUT_DIR, "train/labels"))
move_files(val_files, os.path.join(OUTPUT_DIR, "validation/images"), os.path.join(OUTPUT_DIR, "validation/labels"))

print(f"✅ Data split complete. Train: {len(train_files)} | Val: {len(val_files)}")

# === Step 5: Generate data.yaml ===
with open(classes_file, 'r') as f:
    class_names = [line.strip() for line in f if line.strip()]

if not class_names:
    raise ValueError("❌ 'classes.txt' is empty or invalid.")

yaml_dict = {
    "path": OUTPUT_DIR,
    "train": "train/images",
    "val": "validation/images",
    "nc": len(class_names),
    "names": class_names
}

yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(yaml_dict, f, default_flow_style=False)

print(f"✅ data.yaml created at {yaml_path}")
print("🎉 Dataset preparation complete!")
