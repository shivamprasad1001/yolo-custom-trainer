import os
import yaml
from ultralytics import YOLO

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(config):
    print(f"Starting YOLOv8 training with config: {config}....")

    model = YOLO(config["model"])

    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["img_size"],
        project=config["save_dir"],
        name=config["run_name"],
        exist_ok=True
    )

    trained_model_path = os.path.join(config["save_dir"], config["run_name"], "weights", "best.pt")
    if os.path.exists(trained_model_path):
        os.rename(trained_model_path, config["output_model"])
        print(f"Training complete. Model saved as: {config['output_model']}")
    else:
        print("Warning: Model training finished, but best.pt not found.")

if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
