
<div align="center">

# 🎯 Custom YOLOv8 Object Detection Pipeline  
**End-to-End Deep Learning for Real-World Applications**

A complete object detection solution using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) with custom dataset support, training, and evaluation.



</div>

---

<p align="center">
  <!-- Core Technologies -->
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black" />
  
  <!-- Data Processing -->
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  
  <!-- Annotation & Visualization -->
  <img src="https://img.shields.io/badge/Label_Studio-FF6B6B?style=for-the-badge&logo=label-studio&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-5C8EBC?style=for-the-badge" />
  
  <!-- Deployment & Monitoring -->
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black" />
  
  <!-- Performance -->
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorRT-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  
  <!-- Project Info -->
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/contributions-welcome-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/release-v1.0.0-blue?style=for-the-badge" />
</p>

---

## 📜 Project Overview

This pipeline provides a complete workflow for custom object detection:

1. **Dataset Collection**: Smartphone-captured images
2. **Annotation**: Label Studio for precise bounding boxes
3. **Preparation**: Auto-conversion to YOLO format
4. **Training**: Configurable YOLOv8 model training
5. **Evaluation**: Comprehensive metrics (mAP, PR curves)
6. **Deployment**: Ready-to-use model output

---
## 📸 Sample Outputs

<p align="center">
  <img src="demo/yolo-custom-trainer%20fig-01.png" alt="Sample Image" width="45%" />
</p>

---

## 🌟 Key Features

- 📸 **Custom Dataset Support** - Use your own images
- 🏷️ **Label Studio Integration** - Streamlined annotation
- ⚙️ **Configurable Training** - Edit `config.yaml` for different models
- 📊 **Visual Evaluation** - PR curves, confusion matrices
- 🔄 **Reproducible** - Version controlled with requirements

---

## 📂 Project Structure

```mermaid
graph TD
    A[Data Collection] --> B[Label Studio Annotation]
    B --> C[prepare_yolo_dataset.py]
    C --> D[YOLO-formatted Dataset]
    D --> E[train.py]
    E --> F[Model Training]
    F --> G[evaluate.py]
    G --> H[Performance Metrics]
```

```text
project-root/
├── config.yaml                  # Training configuration
├── prepare_yolo_dataset.py      # Dataset preparation
├── training/
│   ├── train.py                 # Training script
│   └── my_model.pt             # Output model
├── evaluate.py                  # Evaluation script
├── data/                       # Organized dataset
│   ├── train/images/           # Training images
│   ├── train/labels/           # Training labels
│   ├── val/images/             # Validation images
│   ├── val/labels/             # Validation labels
│   └── data.yaml               # Dataset config
└── runs/                       # Training outputs
```

---

## 🛠️ Tech Stack

| Component       | Technology |
|----------------|------------|
| Framework      | YOLOv8 (PyTorch) |
| Annotation     | Label Studio |
| Data Processing | OpenCV, Pandas |
| Configuration  | YAML |
| Version Control | Git |

---

```bash
conda activate yolo-env1
pip install ultralytics opencv-python numpy
```

---

## 📌 Script Arguments

The script takes the following arguments:

| Argument       | Required | Description                                                                                                   |
| -------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `--model`      | ✅ Yes    | Path to YOLO model file (`.pt`), e.g., `runs/detect/train/weights/best.pt`                                    |
| `--source`     | ✅ Yes    | Input source: image file (`test.jpg`), folder (`./images/`), video file (`video.mp4`), or USB webcam (`usb0`) |
| `--thresh`     | ❌ No     | Minimum confidence threshold for detections (default: `0.5`)                                                  |
| `--resolution` | ❌ No     | Output resolution in `WxH` format (e.g., `640x480`). Default is source resolution.                            |
| `--record`     | ❌ No     | Record results (only works with video or webcam). Saves to `demo1.avi`. Requires `--resolution`.              |

---

## 📌 Usage Examples

### 1. Run on a single image

```bash
python yolo_detect.py --model runs/detect/train/weights/best.pt --source test.jpg
```

### 2. Run on a folder of images

```bash
python yolo_detect.py --model runs/detect/train/weights/best.pt --source ./images/
```

### 3. Run on a video file

```bash
python yolo_detect.py --model runs/detect/train/weights/best.pt --source video.mp4
```

### 4. Run on a USB webcam

```bash
python yolo_detect.py --model runs/detect/train/weights/best.pt --source usb0 --resolution 640x480
```

### 5. Record results from webcam

```bash
python yolo_detect.py --model runs/detect/train/weights/best.pt --source usb0 --resolution 640x480 --record
```

---

## 📌 Controls During Inference

* Press **`q`** → Quit
* Press **`s`** → Pause inference
* Press **`p`** → Save current frame as `capture.png`

---



## 🚀 Getting Started

### ✅ Prerequisites
- Python 3.8+
- Ultralytics YOLOv8 (`pip install ultralytics`)
- Label Studio (for annotation)

### ⚙️ Setup
```bash
git clone https://github.com/shivamprasad1001/yolo-project.git
cd yolo-project
pip install -r requirements.txt
```

### � Dataset Preparation
1. Annotate images in Label Studio (YOLO format)
2. Export as `data.zip`
3. Run:
```bash
python prepare_yolo_dataset.py
```

### 🏋️ Training
Edit `config.yaml` then:
```bash
python training/train.py
```

### 📊 Evaluation
```bash
python evaluate.py
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
# Model configuration
model: yolov8n.pt        # yolov8n/s/m/l/x
data: data/data.yaml      # Dataset config
epochs: 50               # Training epochs
imgsz: 640               # Image size
batch: 16                # Batch size
project: runs/train      # Output directory
name: custom             # Run name
```

---

## 🔐 Security & Best Practices

- All training data remains local
- Model weights can be encrypted for deployment
- Git ignores sensitive training outputs

---

## 🚧 Future Roadmap

- [ ] TensorRT optimization for deployment
- [ ] Web-based annotation interface
- [ ] Automated hyperparameter tuning
- [ ] Docker support for easy setup

---

## 👨‍💻 Author

**Shivam Prasad**  
[GitHub](https://github.com/shivamprasad1001) | 
[LinkedIn](https://www.linkedin.com/in/shivamprasad1001)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.
