
from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "results" / "xray_detector_improved" / "weights" / "best.pt"
IMAGE_PATH = Path(r"C:\Users\hp\Downloads\archive (13)\sixray_v3\test\images")
OUTPUT_DIR = PROJECT_DIR / "results" / "predictions"

def detect_objects():
    print("Loading trained X-ray detection model...")

    model = YOLO(str(MODEL_PATH))

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Test image folder not found: {IMAGE_PATH}")

    print("Running threat detection...")

    model.predict(
        source=str(IMAGE_PATH),
        imgsz=416,
        conf=0.25,
        device="cpu",
        save=True,
        project=str(OUTPUT_DIR),
        name="test_detection"
    )

    print("Detection completed!")
    print("Check results/predictions/test_detection")

if __name__ == "__main__":
    detect_objects()
