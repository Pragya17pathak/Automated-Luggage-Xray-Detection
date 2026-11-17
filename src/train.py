
from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_DIR / "data.yaml"

MODEL_NAME = str(
    PROJECT_DIR / "results" / "xray_detector" / "weights" / "best.pt"
)

EPOCHS = 30
IMAGE_SIZE = 416
BATCH_SIZE = 2

def train_model():

    print("=" * 50)
    print("AUTOMATED LUGGAGE X-RAY THREAT DETECTION")
    print("IMPROVED TRAINING")
    print("=" * 50)

    if not DATA_YAML.exists():
        raise FileNotFoundError("data.yaml not found!")

    if not Path(MODEL_NAME).exists():
        raise FileNotFoundError("Pretrained model weights not found!")

    print("\nLoading previous best model...")
    model = YOLO(MODEL_NAME)

    print("\nStarting improved CPU training...")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
        optimizer="AdamW",
        lr0=0.001,
        patience=10,
        workers=2,
        seed=42,
        project=str(PROJECT_DIR / "results"),
        name="xray_detector_improved",
        plots=True,
        save=True
    )

    print("\nImproved training completed!")
    print(
        "Results:",
        PROJECT_DIR / "results" / "xray_detector_improved"
    )

if __name__ == "__main__":
    train_model()
