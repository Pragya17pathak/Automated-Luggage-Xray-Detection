
from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "xray_detector_improved"
    / "weights"
    / "best.pt"
)

DATA_YAML = PROJECT_DIR / "data.yaml"


def evaluate_model():
    print("=" * 50)
    print("X-RAY THREAT DETECTION - TEST EVALUATION")
    print("=" * 50)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=416,
        batch=2,
        device="cpu",
        workers=2,
        plots=True,
        project=str(PROJECT_DIR / "results"),
        name="final_test_evaluation"
    )

    print("\nFINAL TEST RESULTS")
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    print("\nResults saved in:")
    print(PROJECT_DIR / "results" / "final_test_evaluation")


if __name__ == "__main__":
    evaluate_model()
