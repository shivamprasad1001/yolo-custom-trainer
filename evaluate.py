# evaluate.py

from ultralytics import YOLO
import os

MODEL_PATH = "my_model.pt"   # Trained model path
DATA_YAML = "./data/data.yaml"          # Dataset config (with val path inside)

def evaluate():
    print("📊 Evaluating YOLOv8 model on validation set...")

    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATA_YAML,
        save=True,         # Save results in 'runs/val/'
        save_txt=False,    # Save label predictions in .txt
        save_conf=True     # Save confidence scores
    )

    print("Evaluation complete!")
    print(metrics)

if __name__ == "__main__":
    evaluate()
