# 🧠 YOLO Custom Object Detection Model & Tools

This directory contains the custom-trained YOLO object detection model and the helper scripts to test, run, and document the project's lifecycle.

---

## 📂 Directory Contents

* **`my_model.pt` / `train/weights/best.pt`**: The custom-trained YOLOv8 model weights capable of detecting 7 classes: `book`, `chess`, `colour`, `favicol`, `glue`, `mobile`, `pen`.
* **`yolo_detect.py`**: The inference script to test the model on real-world inputs across 3 different modes.
* **`requirements.txt`**: Minimal dependencies required for the model inference environment.

---

## 🚀 Model Testing & Inference (`yolo_detect.py`)

`yolo_detect.py` supports 3 main modes of operation to run detection on real data.

### 📋 Prerequisites

Install required libraries in your environment:
```bash
pip install -r requirements.txt
```

### 1️⃣ Mode 1: Image / Folder Inference
Detects objects in a single image or a folder of images. If a folder is specified, the script processes all supported images inside it.
* **Single Image**:
  ```bash
  python yolo_detect.py --model my_model.pt --source test.jpg
  ```
* **Folder of Images**:
  ```bash
  python yolo_detect.py --model my_model.pt --source ./images/
  ```
* **Controls**: Press `q` to quit, `s` to pause, `p` to save a prediction capture.

### 2️⃣ Mode 2: Video File Inference
Processes and displays detections from a local video file.
```bash
python yolo_detect.py --model my_model.pt --source test-vdo.mp4
```

### 3️⃣ Mode 3: Webcam & Camera Inference
Performs real-time object detection using a connected USB webcam or Raspberry Pi camera.
* **Run on USB Camera**:
  ```bash
  python yolo_detect.py --model my_model.pt --source usb0 --resolution 640x480
  ```
* **Run and Record Output** (saves as `demo1.avi`):
  ```bash
  python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720 --record
  ```

---

## 📌 Controls During Inference

* Press **`q`** → Quit
* Press **`s`** → Pause inference
* Press **`p`** → Save current frame as `capture.png`

---

## 📌 Project History & Training Scripts Reference

Below is a reference guide for the training, preparation, and configuration files previously pushed to the repository (representing the model creation pipeline):

### 🛠️ 1. Dataset Preparation (`prepare_yolo_dataset.py`)
Used to preprocess and structure raw dataset files before training.
* **Purpose**: Automatically splits raw images and labels into `train`, `val`, and (optional) `test` folders according to specified ratios (e.g., 80% train, 20% validation). It ensures labels align properly with images in the standard YOLO folder structure:
  ```
  dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
  ```

### ⚙️ 2. Dataset Configuration (`config.yaml`)
A YAML config file that links YOLO to the dataset and class definitions.
* **Format**:
  ```yaml
  path: ../data  # dataset root dir
  train: images/train
  val: images/val

  # Classes
  names:
    0: book
    1: chess
    2: colour
    3: favicol
    4: glue
    5: mobile
    6: pen
  ```

### 🏋️ 3. Model Training (`train.py`)
Initializes and launches the model training process on the custom dataset.
* **Purpose**: Loads pre-trained YOLO weights (e.g., `yolov8n.pt`) and trains them on the dataset specified in `config.yaml`.
* **Example Usage**:
  ```python
  from ultralytics import YOLO

  # Load a model
  model = YOLO("yolov8n.pt")  # load a pre-trained model

  # Train the model
  results = model.train(
      data="config.yaml",
      epochs=100,
      imgsz=640,
      device="0"  # Use GPU 0, or "cpu"
  )
  ```

### 📊 4. Model Evaluation (`evaluate.py`)
Validates model performance on the validation/test dataset.
* **Purpose**: Runs post-training validation to generate performance metrics including Mean Average Precision (mAP50, mAP50-95), precision, recall, and confusion matrices.
* **Example Usage**:
  ```python
  from ultralytics import YOLO

  # Load the trained model
  model = YOLO("runs/detect/train/weights/best.pt")

  # Validate the model
  metrics = model.val(data="config.yaml")
  print("mAP50-95:", metrics.box.map)
  print("mAP50:", metrics.box.map50)
  ```

---

## 🆕 Latest Updates & Migration Details (June 2026)

The project codebase has been updated to support modern python environments and python-telegram-bot v22.x:

1. **python-telegram-bot v22.x Migration**:
   - Upgraded `object_detection_TG-Bot/bot.py` from deprecated v13 callbacks/Updater classes to the modern async `ApplicationBuilder` flow.
   - Converted all message handlers and callbacks to `async def` and integrated `ContextTypes.DEFAULT_TYPE`.
2. **Python 3.13 Compatibility**:
   - Added a timezone conversion shim to resolve `TypeError: Only timezones from the pytz library are supported` caused by `apscheduler` constraints on Python 3.13.
3. **YOLO Custom Weights Loading Fix**:
   - Resolved `AttributeError: Can't get attribute 'C3k2'` when loading weights trained on newer architectures via older packages.
   - Dynamic class injection added to both `bot.py` and `yolo_detect.py`.
4. **Git Tracking Cleanups**:
   - Configured `.gitignore` to correctly ignore heavy `*.pt` weights, environment folders (`.venv`), and local logs/media outputs while keeping demo files tracked.

---

## ⚠️ Important Developer Notes & Compatibility Patches

### 1. YOLOv11 Architecture Backwards Compatibility
* **Issue**: The trained weights (`best.pt` / `my_model.pt`) contain block layer definitions (`C3k2`, `C3k`, `PSABlock`, `Attention`, `C2PSA`, `SCDown`) that are not present in older `ultralytics` package releases (such as `8.2.103`). Attempting to load the model throws `AttributeError: Can't get attribute 'C3k2' on <module 'ultralytics.nn.modules.block'>`.
* **Solution**: A runtime monkey-patch is injected at the top of `yolo_detect.py` and `bot.py`. It dynamically registers these classes in `sys.modules['ultralytics.nn.modules.block']` prior to loading the model. Developers must retain this patch block to run the model on older environments.

### 2. Timezone Conflicts in python-telegram-bot
* **Issue**: The `apscheduler` scheduler package used internally by `python-telegram-bot` (v20.x+) expects timezone objects from the `pytz` library. When using newer Python versions (3.13+) and PTB configurations, standard `datetime.timezone.utc` objects are used, causing `TypeError: Only timezones from the pytz library are supported`.
* **Solution**: A timezone monkeypatch is injected at the top of `bot.py` before importing `telegram.ext`. It overrides `tzlocal.get_localzone` and `apscheduler.util.astimezone` to safely coerce standard Python timezone objects into `pytz` objects.

### 3. Headless Environment Warnings
* **Issue**: `cv2.imshow` and `cv2.waitKey` in `yolo_detect.py` require a graphical GUI window system (X11 server). Running the script on a headless linux VM, server, or Docker container will fail or hang.
* **Solution**: Run on a headful environment, set up a virtual display server (like `xvfb-run`), or modify the script to run offline inference and write files directly without GUI components.

---

## 👨‍💻 Author

**Shivam Prasad**
* 🌐 **Portfolio**: [shivamprasad1001.in](https://shivamprasad1001.in)
* 🔗 **LinkedIn**: [shivamprasad1001](https://www.linkedin.com/in/shivamprasad1001)

