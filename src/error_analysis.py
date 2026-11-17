from pathlib import Path
import csv
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"C:\Users\hp\Downloads\archive (13)\sixray_v3")

MODEL_PATH = PROJECT_DIR / "results" / "xray_detector_exp_recall" / "weights" / "best.pt"
IMAGE_DIR = DATASET_DIR / "test" / "images"
LABEL_DIR = DATASET_DIR / "test" / "labels"

CLASS_NAMES = ["Gun", "Knife", "Pliers", "Scissors", "Wrench"]
CONFIDENCE = 0.10
NMS_IOU = 0.45
MATCH_IOU = 0.50


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def read_labels(label_path):
    objects = []

    if not label_path.exists():
        return objects

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()
            if len(values) != 5:
                continue

            cls = int(float(values[0]))
            box = list(map(float, values[1:]))

            if 0 <= cls < len(CLASS_NAMES):
                objects.append((cls, xywh_to_xyxy(box)))

    return objects


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError("Test images or labels folder not found.")

    model = YOLO(str(MODEL_PATH))

    counts = {
        name: {"TP": 0, "FP": 0, "FN": 0}
        for name in CLASS_NAMES
    }

    error_rows = []
    image_count = 0

    print("Starting class-wise error analysis...")
    print("Model:", MODEL_PATH)
    print("Confidence:", CONFIDENCE)
    print("Matching IoU:", MATCH_IOU)

    results = model.predict(
        source=str(IMAGE_DIR),
        imgsz=416,
        conf=CONFIDENCE,
        iou=NMS_IOU,
        device="cpu",
        max_det=100,
        stream=True,
        verbose=False
    )

    for result in results:
        image_count += 1
        image_name = Path(result.path).name
        label_path = LABEL_DIR / (Path(result.path).stem + ".txt")

        ground_truth = read_labels(label_path)
        predictions = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxyn.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, confidence in zip(boxes, classes, confidences):
                if 0 <= cls < len(CLASS_NAMES):
                    predictions.append(
                        (cls, box.tolist(), float(confidence))
                    )

        candidates = []

        for pred_index, (pred_cls, pred_box, confidence) in enumerate(predictions):
            for gt_index, (gt_cls, gt_box) in enumerate(ground_truth):
                if pred_cls == gt_cls:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou >= MATCH_IOU:
                        candidates.append((iou, pred_index, gt_index))

        candidates.sort(reverse=True)

        matched_predictions = set()
        matched_ground_truth = set()

        for iou, pred_index, gt_index in candidates:
            if pred_index not in matched_predictions and gt_index not in matched_ground_truth:
                matched_predictions.add(pred_index)
                matched_ground_truth.add(gt_index)

        for index, (cls, box, confidence) in enumerate(predictions):
            name = CLASS_NAMES[cls]

            if index in matched_predictions:
                counts[name]["TP"] += 1
            else:
                counts[name]["FP"] += 1
                error_rows.append([
                    image_name, "False Positive", name,
                    round(confidence, 4), ""
                ])

        for index, (cls, box) in enumerate(ground_truth):
            if index not in matched_ground_truth:
                name = CLASS_NAMES[cls]
                counts[name]["FN"] += 1
                error_rows.append([
                    image_name, "Missed Object", name, "",
                    "Ground-truth object not detected"
                ])

        if image_count % 100 == 0:
            print(f"Processed {image_count} images...")

    output_dir = PROJECT_DIR / "results" / "xray_error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "detection_errors.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image", "Error Type", "Class",
            "Confidence", "Details"
        ])
        writer.writerows(error_rows)

    print("\n" + "=" * 65)
    print("CLASS-WISE ERROR ANALYSIS")
    print("=" * 65)
    print(f"{'Class':<12}{'TP':>8}{'FP':>8}{'FN':>8}{'Precision':>13}{'Recall':>12}")

    total_tp = total_fp = total_fn = 0

    for name, values in counts.items():
        tp, fp, fn = values["TP"], values["FP"], values["FN"]
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            f"{name:<12}{tp:>8}{fp:>8}{fn:>8}"
            f"{precision * 100:>12.2f}%{recall * 100:>11.2f}%"
        )

    overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0

    print("-" * 65)
    print(f"Images processed: {image_count}")
    print(f"Overall TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
    print(f"Overall precision: {overall_precision * 100:.2f}%")
    print(f"Overall recall: {overall_recall * 100:.2f}%")
    print("\nDetailed errors saved to:")
    print(csv_path)


if __name__ == "__main__":
    main()
