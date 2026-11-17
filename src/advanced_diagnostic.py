from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(r"C:\Users\hp\Downloads\archive (13)\sixray_v3")

MODEL_PATH = ROOT / "results" / "xray_detector_exp_recall" / "weights" / "best.pt"
IMAGE_DIR = DATA / "test" / "images"
LABEL_DIR = DATA / "test" / "labels"

NAMES = ["Gun", "Knife", "Pliers", "Scissors", "Wrench"]

CONF = 0.10
NMS_IOU = 0.45
MATCH_IOU = 0.50


def to_xyxy(box):
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def read_gt(path):
    objects = []
    if not path.exists():
        return objects

    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) != 5:
            continue

        cls = int(float(values[0]))
        if 0 <= cls < len(NAMES):
            box = list(map(float, values[1:]))
            objects.append({"cls": cls, "box": to_xyxy(box)})

    return objects


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    totals = defaultdict(int)
    confusion = np.zeros((len(NAMES), len(NAMES)), dtype=int)
    rows = []
    image_count = 0

    print("Starting advanced diagnostic...")
    print("Confidence:", CONF)
    print("Matching IoU:", MATCH_IOU)

    results = model.predict(
        source=str(IMAGE_DIR),
        imgsz=416,
        conf=CONF,
        iou=NMS_IOU,
        device="cpu",
        max_det=100,
        stream=True,
        verbose=False
    )

    for result in results:
        image_count += 1
        filename = Path(result.path).name
        label_path = LABEL_DIR / (Path(result.path).stem + ".txt")

        gt = read_gt(label_path)
        pred = []

        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxyn.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            for box, cls, score in zip(boxes, classes, scores):
                if 0 <= cls < len(NAMES):
                    pred.append({
                        "cls": int(cls),
                        "box": box.tolist(),
                        "score": float(score)
                    })

        candidates = []
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gt):
                overlap = iou(p["box"], g["box"])
                if overlap >= MATCH_IOU:
                    candidates.append((overlap, pi, gi))

        candidates.sort(reverse=True)

        matched_p = set()
        matched_g = set()
        matches = []

        for overlap, pi, gi in candidates:
            if pi not in matched_p and gi not in matched_g:
                matched_p.add(pi)
                matched_g.add(gi)
                matches.append((pi, gi, overlap))

        for pi, gi, overlap in matches:
            p = pred[pi]
            g = gt[gi]
            actual = g["cls"]
            predicted = p["cls"]

            if actual == predicted:
                totals["Correct class and box"] += 1
                category = "Correct detection"
            else:
                totals["Class confusion"] += 1
                confusion[actual, predicted] += 1
                category = "Wrong class, matched box"

            rows.append([
                filename, category, NAMES[actual],
                NAMES[predicted], round(overlap, 4),
                round(p["score"], 4)
            ])

        # Unmatched predictions that overlap an already matched
        # ground-truth object may be duplicate detections.
        for pi, p in enumerate(pred):
            if pi in matched_p:
                continue

            overlaps = [
                (iou(p["box"], g["box"]), gi)
                for gi, g in enumerate(gt)
            ]

            best_overlap, best_gi = max(overlaps, default=(0.0, -1))

            if best_overlap >= MATCH_IOU and best_gi in matched_g:
                totals["Duplicate detection"] += 1
                category = "Duplicate detection"
                actual_name = NAMES[gt[best_gi]["cls"]]
            else:
                totals["Unmatched prediction"] += 1
                category = "Unmatched prediction"
                actual_name = ""

            rows.append([
                filename, category, actual_name,
                NAMES[p["cls"]], round(best_overlap, 4),
                round(p["score"], 4)
            ])

        for gi, g in enumerate(gt):
            if gi not in matched_g:
                totals["Missed object"] += 1
                rows.append([
                    filename, "Missed object",
                    NAMES[g["cls"]], "", "", ""
                ])

        if image_count % 100 == 0:
            print(f"Processed {image_count} images...")

    out_dir = ROOT / "results" / "xray_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "advanced_diagnostic.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image", "Diagnostic", "Actual Class",
            "Predicted Class", "IoU", "Confidence"
        ])
        writer.writerows(rows)

    print("\n" + "=" * 65)
    print("ADVANCED DIAGNOSTIC SUMMARY")
    print("=" * 65)

    for category in [
        "Correct class and box",
        "Class confusion",
        "Duplicate detection",
        "Unmatched prediction",
        "Missed object"
    ]:
        print(f"{category:<30}: {totals[category]}")

    print("\nCLASS CONFUSION MATRIX")
    print("Rows = actual class | Columns = predicted class")
    print(f"{'Actual/Pred':<12}" + "".join(f"{n[:8]:>10}" for n in NAMES))

    for i, name in enumerate(NAMES):
        print(f"{name:<12}" + "".join(f"{confusion[i, j]:>10}" for j in range(len(NAMES))))

    print("\nImages processed:", image_count)
    print("Detailed results:", csv_path)


if __name__ == "__main__":
    main()
