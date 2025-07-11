# Custom YOLOv8 Object Detection Pipeline

A full end-to-end object detection pipeline using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), complete with:

- Custom dataset collected via smartphone
- Image annotation using [Label Studio](https://labelstud.io/)
- Auto-preparation of YOLOv8 data format
- Model training with `YOLOv8`
- Evaluation with mAP, PR curves, confusion matrix
- Configurable via `config.yaml`

Built and maintained by [shivamprasad1001](https://github.com/shivamprasad1001).

---

##  Project Structure

```

project-root/
├── config.yaml                  # Training configuration (editable)
├── .gitignore                   # Ignores runs/, models, caches, etc.
├── requirements.txt             # Dependencies list
├── prepare\_yolo\_dataset.py      # Auto-prepare YOLO dataset from data.zip
├── training/
│   ├── train.py                 # Trains YOLOv8 using config.yaml
│   └── my\_model.pt              # Output model (after training)
├── evaluate.py                  # Evaluate model on validation set
├── data/
│   ├── train/images/
│   ├── train/labels/
│   ├── validation/images/
│   ├── validation/labels/
│   └── data.yaml                # YOLO dataset config
└── runs/                        # YOLO training/eval outputs (auto-generated)

````

---

##  Dataset Pipeline

This project uses a **custom dataset**:

- Images were captured using a **smartphone camera**
- Annotated via [Label Studio](https://labelstud.io/) in **YOLO format**
- Exported as a zip file (`data.zip`) with:
  - `images/`
  - `labels/`
  - optional `meta.json` (if exported)

###  Dataset Preparation Script: `prepare_yolo_dataset.py`

Automatically:
- Extracts `data.zip` into `temp_extracted/`
- Organizes files into `data/train/` and `data/validation/` (80/20 split)
- Generates a valid `data/data.yaml` for YOLO training

---

## ⚙️ Features

- **Modular training with `train.py`** (reads `config.yaml`)
- **Evaluation with `evaluate.py`**
- Clean structure, portable scripts
- Professional `.gitignore` and reproducibility via `requirements.txt`

---

##  Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

### 2. Prepare dataset

```bash
python prepare_yolo_dataset.py
```

> This will create the full `data/` folder structure and generate `data.yaml`.

### 3. Train the model

```bash
python training/train.py
```

### 4. Evaluate performance

```bash
python evaluate.py
```

---

## 🔧 Config (`config.yaml`)

```yaml
model: yolov8n.pt
data: data/data.yaml
epochs: 50
img_size: 640
save_dir: runs/train
run_name: custom
output_model: my_model.pt
```

---

##  Outputs

After training & evaluation:

* Model: `training/my_model.pt`
* Curves, metrics: `runs/train/custom/` and `runs/val/exp/`
* Visuals: PR curves, confusion matrix, predictions

---

##  Tech Stack

* Python 3.8+
* Ultralytics YOLOv8
* Label Studio (annotation)
* PyYAML
* tqdm (optional)
* Git for version control

---

##  Author

Made with ❤️ by [Shivam Prasad](https://github.com/shivamprasad1001)

> Feel free to ⭐ the repo and share your results!

---

## 📝 License

MIT License – use freely, just give credit 🙂
