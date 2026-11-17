
from pathlib import Path
from ultralytics import YOLO

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_DIR / "data.yaml"

# Start from the existing improved model
MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "xray_detector_improved"
    / "weights"
    / "best.pt"
)

# Separate experiment output
OUTPUT_DIR = PROJECT_DIR / "results"
EXPERIMENT_NAME = "xray_detector_exp_recall"


def train_experiment():

    print("=" * 55)
    print("X-RAY THREAT DETECTION - RECALL IMPROVEMENT EXPERIMENT")
    print("=" * 55)

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    experiment_path = OUTPUT_DIR / EXPERIMENT_NAME

    if experiment_path.exists():
        raise FileExistsError(
            f"Experiment folder already exists: {experiment_path}\n"
            "Choose a new experiment name before running."
        )

    print("\nLoading improved checkpoint...")
    model = YOLO(str(MODEL_PATH))

    print("\nStarting CPU fine-tuning...")
    print("Image size: 416")
    print("Batch size: 2")
    print("Epochs: 20")

    model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=416,
        batch=2,
        device="cpu",
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=1,
        patience=7,
        workers=2,
        seed=42,
        deterministic=True,
        cache=False,
        amp=False,
        close_mosaic=5,
        plots=True,
        save=True,
        save_period=5,
        project=str(OUTPUT_DIR),
        name=EXPERIMENT_NAME,
        exist_ok=False
    )

    print("\nExperiment training completed!")
    print("Experiment folder:", experiment_path)
    print("Best weights:", experiment_path / "weights" / "best.pt")


if __name__ == "__main__":
    train_experiment()
