
<div align="center">

# 🎯 Custom YOLOv8 Object Detection Pipeline  
**End-to-End Deep Learning for Real-World Applications**

A complete object detection solution using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) with custom dataset support, training, and evaluation.



</div>

---

<p align="center">
  <!-- Framework -->
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-00FFFF?style=for-the-badge" />

  <!-- Annotation -->
  <img src="https://img.shields.io/badge/Label%20Studio-Data%20Annotation-FF6B6B?style=for-the-badge&logo=label-studio&logoColor=white" />

  <!-- Processing -->
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Handling-150458?style=for-the-badge&logo=pandas&logoColor=white" />

  <!-- Deployment -->
  <img src="https://img.shields.io/badge/Git-Version%20Control-F05032?style=for-the-badge&logo=git&logoColor=white" />
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

## 📸 Sample Outputs

<p align="center">
  <img src="assets/training_curve.png" alt="Training Metrics" width="45%" />
  <img src="assets/detection_example.png" alt="Detection Example" width="45%" />
</p>

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
