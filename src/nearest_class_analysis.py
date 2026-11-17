from pathlib import Path
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(r"C:\Users\hp\Downloads\archive (13)\sixray_v3")
CSV_PATH = ROOT / "results" / "xray_error_analysis" / "advanced_diagnostic.csv"

NAMES = ["Gun", "Knife", "Pliers", "Scissors", "Wrench"]

def convert(box):
    x, y, w, h = box
    return [x-w/2, y-h/2, x+w/2, y+h/2]

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows = [
    r for r in rows
    if r["Diagnostic"] == "Unmatched prediction"
    and r["IoU"] and 0.25 <= float(r["IoU"]) < 0.50
]

rows.sort(key=lambda r: float(r["IoU"]), reverse=True)
rows = rows[:20]

print("\nNEAREST GROUND-TRUTH CLASS")
print("=" * 105)
print(f"{'Image':<48}{'Predicted':<12}{'Actual':<12}{'IoU':>8}{'Confidence':>12}")

confusion = Counter()

for row in rows:
    image = row["Image"]
    label_path = DATA / "test" / "labels" / (Path(image).stem + ".txt")
    image_path = DATA / "test" / "images" / image

    # Recover this prediction's box by rerunning inference on its image.
    from ultralytics import YOLO
    if "model" not in globals():
        model_path = ROOT / "results" / "xray_detector_exp_recall" / "weights" / "best.pt"
        model = YOLO(str(model_path))

    result = model.predict(
        source=str(image_path), imgsz=416, conf=0.10,
        iou=0.45, device="cpu", max_det=100, verbose=False
    )[0]

    boxes = result.boxes.xyxyn.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()

    # Locate the prediction using its class and confidence.
    target_cls = NAMES.index(row["Predicted Class"])
    target_conf = float(row["Confidence"])

    choices = [
        i for i, (cls, score) in enumerate(zip(classes, scores))
        if cls == target_cls and abs(float(score) - target_conf) < 0.0002
    ]

    if not choices:
        continue

    pred_box = boxes[choices[0]].tolist()
    best_iou, best_class = 0.0, "No overlap"

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            v = line.split()
            if len(v) != 5:
                continue
            cls = int(float(v[0]))
            if not 0 <= cls < len(NAMES):
                continue
            overlap = iou(pred_box, convert(list(map(float, v[1:]))))
            if overlap > best_iou:
                best_iou, best_class = overlap, NAMES[cls]

    confusion[(best_class, row["Predicted Class"])] += 1

    print(
        f"{image[:47]:<48}{row['Predicted Class']:<12}"
        f"{best_class:<12}{best_iou:>8.4f}{target_conf:>12}"
    )

print("\nNEAREST-CLASS SUMMARY")
for (actual, predicted), count in confusion.most_common():
    print(f"Actual: {actual:<10} Predicted: {predicted:<10} Count: {count}")
